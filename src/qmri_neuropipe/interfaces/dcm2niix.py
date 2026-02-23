
from pathlib import Path
from typing import Optional, Union, List
from ..core.run import run_cmd
from ..core import ensure_dir

def dcm2niix(
    in_dir: Path,
    out_dir: Path,
    filename: Optional[str] = None,
    compress: bool = True,
    bids: bool = True,
    verbose: bool = False,
    extra_args: str = ""
) -> Path:
    """
    Run dcm2niix on a directory of DICOM files.
    
    Args:
        in_dir: Input directory containing DICOMs.
        out_dir: Output directory for NIfTI files.
        filename: Output filename template (e.g. '%p_%s').
        compress: Compress output (results in .nii.gz).
        bids: Generate BIDS sidecar (.json).
        verbose: Enable verbose output.
        extra_args: Additional command line arguments.
        
    Returns:
        Path to the output directory.
    """
    in_dir = Path(in_dir)
    out_dir = ensure_dir(out_dir)
    
    cmd = ["dcm2niix"]
    
    # Compression: y=yes, n=no, i=internal
    z_val = "y" if compress else "n"
    cmd.append(f"-z {z_val}")
    
    # BIDS sidecar: y=yes, n=no
    b_val = "y" if bids else "n"
    cmd.append(f"-b {b_val}")
    
    # Filename template
    if filename:
        cmd.append(f"-f {filename}")
        
    # Verbosity
    if verbose:
        cmd.append("-v y")
        
    # Output directory
    cmd.append(f"-o {out_dir}")
    
    # Additional args
    if extra_args:
        cmd.append(extra_args)
        
    # Input directory (last argument)
    cmd.append(str(in_dir))
    
    run_cmd(" ".join(cmd), label="dcm2niix")
    
    return out_dir
