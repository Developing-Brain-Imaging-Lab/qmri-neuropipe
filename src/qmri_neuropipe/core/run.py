import subprocess, shlex, logging
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    console = Console()
except ImportError:
    console = None

log = logging.getLogger("qmri-neuropipe")

def run_cmd(cmd: str, *, label: str | None = None, dry_run: bool = False, env: dict = None, n_threads: int = None) -> None:
    """Exec a CLI; log command at DEBUG and emit progress lines at INFO."""
    label = label or "cmd"
    
    if console:
        log.debug(f"[dim][CMD][/dim] [bold cyan]{cmd}[/bold cyan]", extra={"markup": True})
    else:
        log.debug(f"[CMD] {cmd}")

    if dry_run:
        return

    # Prepare environment
    import os
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    
    if n_threads:
        cmd_env.update({
            "OMP_NUM_THREADS": str(n_threads),
            "MKL_NUM_THREADS": str(n_threads),
            "OPENBLAS_NUM_THREADS": str(n_threads),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(n_threads),
        })

    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, errors='replace', env=cmd_env)
    
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