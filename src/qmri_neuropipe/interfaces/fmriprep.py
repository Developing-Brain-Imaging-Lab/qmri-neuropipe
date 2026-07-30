"""
fMRIPrep interface wrapper.
Handles the execution of the fmriprep container (Singularity or Docker).
"""

from pathlib import Path
from typing import Optional, List
from ..core.run import run_cmd
import logging

def run_fmriprep(
    bids_dir: Path,
    output_dir: Path,
    participant_label: str,
    container_path: Optional[Path] = None,
    docker_image: Optional[str] = None,
    fs_license_file: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    nthreads: int = 1,
    omp_nthreads: int = 1,
    custom_args: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Executes the fMRIPrep BIDS app via Singularity or Docker.
    """
    if not logger:
        logger = logging.getLogger(__name__)

    if not container_path and not docker_image:
        raise ValueError("Must specify either container_path (for Singularity) or docker_image (for Docker).")

    # Determine runtime engine
    engine = "singularity" if container_path else "docker"

    cmd = []
    
    if engine == "singularity":
        cmd.extend(["singularity", "run", "--cleanenv"])
        # Bind mounts
        binds = [f"{bids_dir}:/data:ro", f"{output_dir}:/out"]
        if work_dir:
            binds.append(f"{work_dir}:/work")
        if fs_license_file:
            binds.append(f"{fs_license_file}:/opt/freesurfer/license.txt:ro")
        
        cmd.append("-B")
        cmd.append(",".join(binds))
        cmd.append(str(container_path))
    else:
        cmd.extend(["docker", "run", "--rm", "-it"])
        # Bind mounts
        cmd.extend(["-v", f"{bids_dir}:/data:ro"])
        cmd.extend(["-v", f"{output_dir}:/out"])
        if work_dir:
            cmd.extend(["-v", f"{work_dir}:/work"])
        if fs_license_file:
            cmd.extend(["-v", f"{fs_license_file}:/opt/freesurfer/license.txt:ro"])
        cmd.append(docker_image)

    # Standard fMRIPrep positional args
    cmd.extend(["/data", "/out", "participant"])

    # Standard flags
    cmd.extend(["--participant-label", participant_label])
    cmd.extend(["--nprocs", str(nthreads)])
    cmd.extend(["--omp-nthreads", str(omp_nthreads)])
    
    # FreeSurfer license
    if fs_license_file:
        cmd.extend(["--fs-license-file", "/opt/freesurfer/license.txt"])
        
    if work_dir:
        cmd.extend(["-w", "/work"])
        
    # Custom additional arguments
    if custom_args:
        # Avoid duplicating participant label or things we've handled
        cmd.extend(custom_args)

    cmd_str = " ".join(cmd)
    logger.info(f"Executing fMRIPrep via {engine}: {cmd_str}")
    
    run_cmd(cmd_str, label="fmriprep")
