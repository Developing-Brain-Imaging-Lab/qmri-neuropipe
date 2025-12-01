import subprocess, shlex, logging
#from .progress import get_progress

log = logging.getLogger("qmri-neuropipe")

def run_cmd(cmd: str, *, label: str | None = None, dry_run: bool = False) -> None:
    """Exec a CLI; log command at DEBUG and emit progress lines at INFO."""
    #p = get_progress()
    label = label or "cmd"
    log.debug(f"[CMD] {cmd}")
    #p.emit({"phase": "exec", "task": label, "msg": f"⧗ {label} running"})
    if dry_run:
        #p.emit({"phase": "exec", "task": label, "msg": f"[DRY-RUN] {cmd}"})
        return
    proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        # include a trimmed stderr in progress for quick diagnosis
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        #p.emit({"phase": "stderr", "task": label, "msg": tail})
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stderr}")
    # optional: capture brief stdout snippets at DEBUG
    if proc.stdout.strip():
        log.debug(proc.stdout.strip())
    #p.emit({"phase": "exec_done", "task": label, "msg": f"✓ {label} done"})