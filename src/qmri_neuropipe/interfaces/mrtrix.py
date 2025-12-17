from pathlib import Path
from typing import Optional, Tuple
from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.utils import ensure_path, ensure_dir, extract_image_path

# Standardize path utility
# Or import: from ..interfaces.fsl import _as_path if we want to share. 
# But fsl.py is not really a "common" util.
# For now, importing from fsl is easiest, or duplicating.
# Let's import to avoid code duplication if fsl is always present.
# But circular imports? fsl imports types.
# Actually, let's redefine _as_path here to be safe and independent.


def dwidenoise(in_file: ImageLike | Path, out_file: Path, nthreads: int=2, mask: Optional[Path]=None, noise_map: Optional[Path]=None, force: bool=False) -> Tuple[Path, Optional[Path]]:
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    nm_p = ensure_dir(noise_map) if noise_map else None

    # Skip if already done (unless force)
    if not force and out_p.exists() and (not nm_p or nm_p.exists()):
        return str(out_p), (str(nm_p) if nm_p else None)
    
    mask_arg  = f"-mask {mask}" if mask else ""
    noise_arg = f"-noise {noise_map}" if noise_map else ""
    force_arg = f"-force" if force else ""
    cmd = f"dwidenoise {in_p} {out_p} {mask_arg} {noise_arg} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="dwidenoise")
   
    return out_p, (nm_p if nm_p else None)
    
def dwibiascorrect(in_file: ImageLike | Path, in_bvec: Path, in_bval: Path, out_file: Path, method: str = "ants", mask: Optional[Path]=None, bias_field: Optional[Path]=None, nthreads: int = 2, force: bool = False):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    diff_arg  = f"-fslgrad {in_bvec} {in_bval}"
    mask_arg  = f"-mask {mask}" if mask else ""
    bias_arg  = f"-bias {bias_field}" if bias_field else ""
    force_arg = f"-force" if force else ""
    cmd = f"dwibiascorrect {method} {in_p} {diff_arg} {out_p} {mask_arg} {bias_arg} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="dwibiascorrect")
    
    return out_p

def mrdegibbs(in_file: ImageLike | Path, out_file: Path, nthreads: int = 2, force: bool = False):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    force_arg = f"-force" if force else ""
    cmd = f"mrdegibbs {in_p} {out_p} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="mrdegibbs")

    return out_p

def dwi2mask(in_file: ImageLike | Path, out_file: Path, nthreads: int = 2, force: bool = False):
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    if not force and out_p.exists():
        return out_p
    cmd = f"dwi2mask {in_p} {out_p} -nthreads {nthreads} -quiet" + (" -force" if force else "")
    run_cmd(cmd, label="dwi2mask")
    return out_p
    run_cmd(cmd, label="dwi2mask")
    return out_p


def dwigradcheck(in_file: ImageLike | Path, in_bvec: Path = None, in_bval: Path = None, export_grad_fsl: Tuple[Path, Path] = None, nthreads: int = 2, force: bool = False):
    """
    Wrapper for dwigradcheck.
    
    Args:
        in_file: Input DWI image
        in_bvec: Input bvec file
        in_bval: Input bval file
        export_grad_fsl: Optional tuple (bvec_path, bval_path) to export corrected gradients
        nthreads: Number of threads
        force: Force overwrite
    """
    in_p = extract_image_path(in_file)
    
    # Check if outputs exist (only gradients now)
    outputs_exist = True
    if export_grad_fsl:
        outputs_exist = export_grad_fsl[0].exists() and export_grad_fsl[1].exists()
    else:
        # If no export requested, we can't really skip based on output existence unless we just rely on run logging?
        # But usually we call this TO export.
        outputs_exist = False

    if not force and outputs_exist:
        return

    cmd_parts = [
        "dwigradcheck",
        str(in_p),
        "-nthreads", str(nthreads),
        "-quiet"
    ]
    
    if in_bvec and in_bval:
        cmd_parts.extend(["-fslgrad", str(in_bvec), str(in_bval)])
    
    if force:
        cmd_parts.append("-force")
        
    if export_grad_fsl:
        bvec, bval = export_grad_fsl
        cmd_parts.extend(["-export_grad_fsl", str(bvec), str(bval)])
        
    run_cmd(" ".join(cmd_parts), label="dwigradcheck")

