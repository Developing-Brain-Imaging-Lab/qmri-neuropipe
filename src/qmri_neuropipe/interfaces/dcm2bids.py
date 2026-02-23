
from pathlib import Path
from typing import Optional, Union, List
from ..core.run import run_cmd
from ..core import ensure_dir

def dcm2bids(
    dicom_dir: Path,
    participant_id: str,
    config_file: Path,
    output_dir: Path,
    session_id: Optional[str] = None,
    clobber: bool = False,
    force_dccm: bool = False,
    extra_args: str = ""
) -> Path:
    """
    Run dcm2bids to convert DICOMs to BIDS structure.
    
    Args:
        dicom_dir: Input directory containing DICOM files.
        participant_id: Participant ID (e.g. '001').
        config_file: Path to dcm2bids configuration JSON.
        output_dir: Path to the BIDS project directory.
        session_id: Optional session ID.
        clobber: Overwrite existing files.
        force_dccm: Use dccm instead of dcm2niix (rare).
        extra_args: Additional command line arguments.
        
    Returns:
        Path to the output directory.
    """
    dicom_dir = Path(dicom_dir)
    config_file = Path(config_file)
    output_dir = ensure_dir(output_dir)
    
    cmd = ["dcm2bids"]
    
    cmd.append(f"-d {dicom_dir}")
    cmd.append(f"-p {participant_id}")
    cmd.append(f"-c {config_file}")
    cmd.append(f"-o {output_dir}")
    
    if session_id:
        cmd.append(f"-s {session_id}")
        
    if clobber:
        cmd.append("--clobber")
        
    if force_dccm:
        cmd.append("--forceDccm")
        
    if extra_args:
        cmd.append(extra_args)
        
    run_cmd(" ".join(cmd), label="dcm2bids")
    
    return output_dir

def dcm2bids_scaffold(output_dir: Path):
    """
    Create a new BIDS project scaffold using dcm2bids_scaffold.
    """
    output_dir = ensure_dir(output_dir)
    run_cmd(f"dcm2bids_scaffold -o {output_dir}", label="dcm2bids_scaffold")
    return output_dir
