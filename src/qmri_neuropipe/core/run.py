import os
import subprocess, shlex, logging
from datetime import datetime

try:
    from qmri_neuropipe.core.ui import console
except ImportError:
    console = None

log = logging.getLogger("qmri-neuropipe")
_COMMAND_HISTORY: list[dict] = []

def _writable_tmpdir() -> str:
    """Return a writable temp directory that should also exist in containers."""
    candidates = (
        os.environ.get("TMPDIR"),
        os.environ.get("TEMP"),
        os.environ.get("TMP"),
        "/tmp",
    )
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return "."


def run_cmd(cmd: str, *, label: str | None = None, dry_run: bool = False, env: dict = None, n_threads: int = None, cwd: str | None = None) -> None:
    """Exec a CLI; log command at DEBUG and emit progress lines at INFO."""
    label = label or "cmd"
    command_record = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "command": cmd,
        "cwd": str(cwd) if cwd else None,
        "dry_run": bool(dry_run),
        "env_overrides": dict(env or {}),
        "returncode": None,
    }
    _COMMAND_HISTORY.append(command_record)
    
    if console:
        log.debug(f"[dim][CMD][/dim] [bold cyan]{cmd}[/bold cyan]", extra={"markup": True})
    else:
        log.debug(f"[CMD] {cmd}")

    if dry_run:
        command_record["returncode"] = 0
        return

    # Prepare environment
    cmd_env = os.environ.copy()
    cmd_env.setdefault("MRTRIX_TMPFILE_DIR", _writable_tmpdir())
    if env:
        log.debug(f"[ENV] Overrides: {env}")
        cmd_env.update(env)
    
    if n_threads:
        cmd_env.update({
            "OMP_NUM_THREADS": str(n_threads),
            # A hard ceiling even when a program calls omp_set_num_threads().
            # TORTOISEProcess does this during its own initialization.
            "OMP_THREAD_LIMIT": str(n_threads),
            "MKL_NUM_THREADS": str(n_threads),
            "OPENBLAS_NUM_THREADS": str(n_threads),
            "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(n_threads),
        })

    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, errors='replace', env=cmd_env, cwd=cwd)
    command_record["returncode"] = proc.returncode
    
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        out_tail = "\n".join(proc.stdout.splitlines()[-20:])
        if console:
            console.print(f"[bold red]Command failed:[/bold red] {label}")
            console.print(f"[red]STDERR:\n{tail}[/red]")
            console.print(f"[yellow]STDOUT:\n{out_tail}[/yellow]")
            
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    
    if proc.stdout.strip():
        # Avoid spamming huge outputs
        lines = proc.stdout.strip().splitlines()
        if len(lines) > 20:
             log.debug(f"{lines[0]} ... ({len(lines)} lines)")
        else:
             log.debug(proc.stdout.strip())


def get_command_history(start: int = 0) -> list[dict]:
    """Return command records captured by ``run_cmd`` in this process."""
    return [dict(item) for item in _COMMAND_HISTORY[start:]]


def command_history_len() -> int:
    """Return the number of captured command records."""
    return len(_COMMAND_HISTORY)
