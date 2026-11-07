"""Command construction for optional, externally installed anatomical tools.

No model packages are imported into qmri-neuropipe's Python environment.
"""
from pathlib import Path
import os
import shlex
import shutil

from ..core.run import run_cmd


def execute_anatomical(args, settings, mounts, *, label, gpu=False):
    """Run natively or bind absolute paths into Docker/Apptainer unchanged."""
    runtime = settings.get("runtime", "native")
    if runtime not in {"native", "docker", "apptainer", "singularity"}:
        raise ValueError(f"Unsupported anatomical runtime: {runtime}")
    if runtime == "native":
        executable = str(settings.get("executable") or args[0])
        if not shutil.which(executable):
            raise FileNotFoundError(f"{executable} is unavailable; install {label} or configure a container runtime.")
        command = [executable, *args[1:]]
    else:
        if not shutil.which(runtime):
            raise FileNotFoundError(f"Container runtime {runtime!r} is unavailable")
        image = settings.get("image")
        if not image:
            raise ValueError(f"{label}: image is required for {runtime}")
        if runtime == "docker":
            command = [runtime, "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"]
            if gpu:
                command += ["--gpus", "all"]
            for path, readonly in mounts:
                command += ["-v", f"{path}:{path}" + (":ro" if readonly else "")]
        else:
            command = [runtime, "run"]
            if gpu:
                command += ["--nv"]
            for path, readonly in mounts:
                command += ["--bind", f"{path}:{path}" + (":ro" if readonly else "")]
        # Both official images expose the tool via their container entrypoint.
        command += [str(image), *args[1:]]
    run_cmd(shlex.join(command), label=label)
    return command
