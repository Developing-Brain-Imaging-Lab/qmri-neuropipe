import subprocess, shlex, logging
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    console = Console()
except ImportError:
    console = None

log = logging.getLogger("qmri-neuropipe")

def run_cmd(cmd: str, *, label: str | None = None, dry_run: bool = False) -> None:
    """Exec a CLI; log command at DEBUG and emit progress lines at INFO."""
    label = label or "cmd"
    
    if console:
        log.debug(f"[dim][CMD][/dim] [bold cyan]{cmd}[/bold cyan]", extra={"markup": True})
    else:
        log.debug(f"[CMD] {cmd}")

    if dry_run:
        return

    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, errors='replace')
    
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        if console:
            console.print(f"[bold red]Command failed:[/bold red] {label}")
            console.print(f"[red]{tail}[/red]")
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stderr}")
    
    if proc.stdout.strip():
        # Avoid spamming huge outputs
        lines = proc.stdout.strip().splitlines()
        if len(lines) > 20:
             log.debug(f"{lines[0]} ... ({len(lines)} lines)")
        else:
             log.debug(proc.stdout.strip())