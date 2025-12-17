from pathlib import Path
from typing import Optional, List, Dict, Any

from ..core.run import run_cmd
from ..core.utils import ensure_dir

def run_mriqc(
    bids_dir: Path,
    output_dir: Path,
    participant_label: str,
    session_id: Optional[str] = None,
    n_procs: int = 1,
    mem_gb: Optional[float] = None,
    modalities: Optional[List[str]] = None,
    verbose_reports: bool = False,
    extra_args: str = ""
) -> Path:
    """
    Run MRIQC on a specific participant.
    
    Args:
        bids_dir: Path to BIDS dataset root.
        output_dir: Path to directory where MRIQC outputs should be stored.
        participant_label: Participant ID (without 'sub-' prefix).
        session_id: Optional session ID (without 'ses-' prefix).
        n_procs: Number of threads/processors to use.
        mem_gb: Maximum memory to use in GB.
        modalities: List of modalities to run (e.g. ['T1w', 'bold']). MRIQC mostly auto-detects.
        verbose_reports: Enable verbose reporting in MRIQC.
        extra_args: Additional command line arguments.
        
    Returns:
        Path to the output directory.
    """
    out_p = ensure_dir(output_dir)
    bids_p = Path(bids_dir).resolve()
    
    # Construct base command
    # mriqc <bids_dir> <output_dir> participant
    cmd = [
        "mriqc",
        str(bids_p),
        str(out_p),
        "participant"
    ]
    
    # Add participant label (remove sub- prefix if present for cleaner passing, 
    # though mriqc often handles both. Standard BIDS apps usually expect just the label)
    p_label = participant_label.replace("sub-", "")
    cmd.append(f"--participant-label {p_label}")
    
    if session_id:
        s_label = session_id.replace("ses-", "")
        cmd.append(f"--session-id {s_label}")
        
    if n_procs > 1:
        cmd.append(f"--n_procs {n_procs}")
        
    if mem_gb:
        cmd.append(f"--mem_gb {mem_gb}")
        
    if verbose_reports:
        # Check actual flag, usually -v or --verbose-reports
        cmd.append("--verbose-reports")
        
    if modalities:
        # MRIQC uses -m for modality, e.g. -m T1w bold
        mods = " ".join(modalities)
        cmd.append(f"-m {mods}")
        
    if extra_args:
        cmd.append(extra_args)
        
    # Join command
    full_cmd = " ".join(cmd)
    
    # Run
    # Note: MRIQC might take a while.
    run_cmd(full_cmd, label="mriqc")
    
    return out_p
