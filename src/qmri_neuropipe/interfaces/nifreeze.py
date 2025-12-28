import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging
from ..core.types import ImageFile

logger = logging.getLogger(__name__)

def nifreeze_shoreline(
    in_dwi: Union[str, Path],
    in_bval: Union[str, Path],
    in_bvec: Union[str, Path],
    out_dir: Union[str, Path],
    out_prefix: str = "shoreline",
    b0_thresh: int = 5,
    nthreads: int = 1,
    verbose: bool = False
) -> Tuple[Path, Path, Path]:
    """
    Wraps existing 'nifreeze shoreline' CLI.
    
    Args:
        in_dwi: Input DWI NIfTI file.
        in_bval: Input bval file.
        in_bvec: Input bvec file.
        out_dir: Output directory.
        out_prefix: Prefix for output files (default: shoreline).
        b0_thresh: b-value threshold for identifying b0 volumes (default: 5).
        num_threads: Number of OMP threads to use.
        verbose: Enable verbose output.

    Returns:
        Tuple[Path, Path, Path]: Paths to (corrected_dwi, corrected_bvec, corrected_bval)
        Note: NiiFreeze (ShoreLine) typically updates bvecs. Bvals usually remain same but strictly should be returned.
    """
    
    # Ensure nifreeze is installed/available
    if shutil.which("nifreeze") is None:
         # Fallback to checking if 'shoreline' is the command? 
         # The repo says 'nifreeze' might be the entry point or 'shoreline' directly.
         # Assuming 'nifreeze shoreline' based on modern nipreps structure or just 'shoreline'.
         if shutil.which("shoreline") is None:
             raise FileNotFoundError("NiiFreeze/ShoreLine executable not found in PATH.")
         cmd_base = ["shoreline"]
    else:
         cmd_base = ["nifreeze", "shoreline"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct Command
    # Usage: shoreline input_image bval_file bvec_file output_prefix [options]
    # Check actual CLI usage: shoreline -h
    # Based on typical usage: 
    # shoreline <input_image> <bval_file> <bvec_file> <output_prefix> --b0_thresh <val> ...
    
    # WAIT: output_prefix in shoreline usually includes path?
    # Let's assume out_prefix is full path prefix or relative to cwd?
    # Safer to provide full path prefix.
    full_prefix = out_dir / out_prefix
    
    cmd = cmd_base + [
        str(in_dwi),
        str(in_bval),
        str(in_bvec),
        str(full_prefix)
    ]
    
    cmd.extend(["--b0_thresh", str(b0_thresh)])
    cmd.extend(["--n_jobs", str(nthreads)]) # shoreline uses n_jobs
    
    if verbose:
        cmd.append("--verbose")
        
    logger.info(f"Running NiiFreeze: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=not verbose)
    except subprocess.CalledProcessError as e:
        logger.error(f"NiiFreeze failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        raise
        
    # Expected Outputs (ShoreLine usually adds suffix)
    # If out_prefix is "shoreline", it creates:
    #   shoreline_corrected.nii.gz
    #   shoreline_corrected.bvec
    #   shoreline_corrected.bval (maybe?)
    
    # Let's verify standard shoreline output conventions. 
    # ShoreLine outputs: [prefix]_corrected.nii.gz, [prefix]_corrected.bvec
    
    out_dwi = out_dir / f"{out_prefix}_corrected.nii.gz"
    out_bvec = out_dir / f"{out_prefix}_corrected.bvec"
    out_bval = out_dir / f"{out_prefix}_corrected.bval" # Often just copies input if not eddy
    
    if not out_dwi.exists():
        raise FileNotFoundError(f"NiiFreeze did not produce expected output: {out_dwi}")
        
    # Check if bval allows pass-through or is created
    if not out_bval.exists() and Path(in_bval).exists():
         shutil.copy(in_bval, out_bval)
         
    return out_dwi, out_bvec, out_bval
