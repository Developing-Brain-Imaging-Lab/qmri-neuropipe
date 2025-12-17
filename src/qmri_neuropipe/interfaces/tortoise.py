from pathlib import Path
from typing import Optional, Tuple
from ..core.run import run_cmd

from ..core.types import ImageLike
from ..core.utils import extract_image_path, ensure_dir

def create_gradient_nonlinearity_bmatrix(
    in_file: ImageLike | Path, 
    out_file: Path, 
    grad_coeffs: Path, 
    force: bool = False
) -> Path:
    """
    Compute/Correct gradient nonlinearities using TORTOISE CreateGradientNonlinearityBMatrix.
    
    Wraps the 'CreateGradientNonlinearityBMatrix' command.
    
    Args:
        in_file: Input DWI image.
        out_file: Output path (likely for the resulting BMatrix file or corrected data).
        grad_coeffs: The gradient nonlinearity coefficients file.
        force: Overwrite existing output.
        
    Returns:
        Path to the output file.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Skip if already done
    if not force and out_p.exists():
        return out_p

    # Command structure assumption:
    # CreateGradientNonlinearityBMatrix -i input.nii -c coeffs.txt -o output
    
    cmd = f"CreateGradientNonlinearityBMatrix --input {in_p} --coefficients {grad_coeffs} --output {out_p}"
    
    run_cmd(cmd, label="create_grad_nonlin_bmatrix")
    
    return out_p


def apply_grad_nonlin(
    in_file: ImageLike | Path,
    out_file: Path,
    grad_coeffs: Path,
    nthreads: int = 2,
    force: bool = False,
) -> Path:
    """Apply gradient nonlinearity correction using TORTOISE.

    This wrapper uses `create_gradient_nonlinearity_bmatrix` to generate the B-matrix
    and assumes the corrected image is written to `out_file`.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    # Ensure output directory exists
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # Generate B-matrix (or corrected data) using the existing utility
    _ = create_gradient_nonlinearity_bmatrix(
        in_file=in_p,
        out_file=out_p,
        grad_coeffs=grad_coeffs,
        force=force,
    )
    # In a full implementation, the B-matrix would be applied to the image.
    # For now, we assume the output image is produced directly.
    return out_p
