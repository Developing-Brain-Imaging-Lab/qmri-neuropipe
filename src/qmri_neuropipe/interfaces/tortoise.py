from pathlib import Path
from typing import Optional, List, Union
from ..core.run import run_cmd
from ..core.types import ImageLike, DWIFile, ImageFile
from ..core.utils import extract_image_path, ensure_dir

def diffprep(
    dwi_file: Union[Path, DWIFile],
    out_dir: Path,
    structural_file: Optional[Path] = None,
    bvecs: Optional[Path] = None, 
    bvals: Optional[Path] = None,
    phase: str = 'vertical', # 'vertical' or 'horizontal'
    will_be_drbuddied: bool = True,
    do_gibbs: bool = True,
    do_denoise: bool = True,
    settings_file: Optional[Path] = None,
    nthreads: int = 1,
    force: bool = False
) -> Path:
    """
    Wrapper for TORTOISE DIFFPREP (v3).
    
    Args:
        dwi_file: Input DWI NIfTI.
        out_dir: Output Directory.
        structural_file: Anatomical reference (T2w usually preferred).
        bvecs: Path to bvecs (if not in dwi_file object).
        bvals: Path to bvals (if not in dwi_file object).
        phase: Phase encoding direction ('vertical' (AP/PA) or 'horizontal' (RL/LR)).
        will_be_drbuddied: If True, optimized for subsequent DRBUDDI step.
        do_gibbs: Perform Gibbs unringing.
        do_denoise: Perform Denoising.
        
    Returns:
        Path to the relevant output file (list file or proc file).
    """
    out_dir = ensure_dir(out_dir)
    
    # Resolve Inputs
    if isinstance(dwi_file, DWIFile):
        in_img = dwi_file.img
        in_bvec = dwi_file.bvec
        in_bval = dwi_file.bval
    else:
        in_img = Path(dwi_file)
        in_bvec = bvecs
        in_bval = bvals
        
    if not in_bvec or not in_bval:
        raise ValueError("DIFFPREP requires bvecs/bvals.")
        
    # TORTOISE typically expects a list file or command line args.
    # v3.1.2: DIFFPREP --dwi image.nii --bvals bvals --bvecs bvecs --structural struct.nii --phase vertical ...
    
    cmd_parts = ["DIFFPREP"]
    cmd_parts.append(f"--dwi {in_img}")
    cmd_parts.append(f"--bvals {in_bval}")
    cmd_parts.append(f"--bvecs {in_bvec}")
    cmd_parts.append(f"--output_folder {out_dir}")
    
    if structural_file:
        cmd_parts.append(f"--structural {structural_file}")
        
    cmd_parts.append(f"--phase {phase}")
    
    if will_be_drbuddied:
        cmd_parts.append("--will_be_drbuddied 1")
    else:
        cmd_parts.append("--will_be_drbuddied 0")
        
    cmd_parts.append(f"--do_gibbs {1 if do_gibbs else 0}")
    cmd_parts.append(f"--do_denoise {1 if do_denoise else 0}")
    
    if settings_file:
        cmd_parts.append(f"--settings {settings_file}")
        
    # Check for completion (TORTOISE outputs complicated structure)
    # Usually <dwi_name>_proc directory or similar.
    # We rely on run_cmd to handle execution.
    
    run_cmd(" ".join(cmd_parts), label="DIFFPREP", n_threads=nthreads)
    
    return out_dir

def drbuddi(
    up_data: Path, # Blip up (or main)
    down_data: Path, # Blip down (or reverse)
    out_dir: Path,
    structural_file: Path,
    fieldmap: Optional[Path] = None,
    tensor_fit: bool = False,
    nthreads: int = 1
) -> Path:
    """
    Wrapper for TORTOISE DRBUDDI.
    
    Should be run after DIFFPREP with --will_be_drbuddied 1.
    Inputs are usually the _proc folders or list files from DIFFPREP.
    """
    ensure_dir(out_dir)
    
    cmd_parts = ["DRBUDDI"]
    cmd_parts.append(f"--up_data {up_data}")
    cmd_parts.append(f"--down_data {down_data}")
    cmd_parts.append(f"--output_folder {out_dir}")
    cmd_parts.append(f"--structural {structural_file}")
    
    if fieldmap:
        cmd_parts.append(f"--fieldmap {fieldmap}")
        
    if tensor_fit:
         cmd_parts.append("--tensor_fit 1")
         
    run_cmd(" ".join(cmd_parts), label="DRBUDDI", n_threads=nthreads)
    return out_dir

def apply_grad_nonlin(
    in_file: ImageLike | Path,
    out_file: Path,
    grad_coeffs: Path,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """
    Apply gradient nonlinearity correction.
    Uses 'ComputeGradientNonlinearityMap' and then applies it, or similar tool.
    Currently retaining the placeholder implementation which assumes 'CreateGradientNonlinearityBMatrix' usage logic,
    but updated to be cleaner.
    """
    # ... (existing logic or improvement)
    # The existing function calls 'CreateGradientNonlinearityBMatrix'
    # Real TORTOISE workflow often uses DIFFPREP to apply GNL if coeffs are provided.
    # Independent tool: 'FullDualSpaceGNLCorrection' or similar in internal builds?
    # Publicly: 'ImportDICOM' allows handling, or specific diff_prep settings.
    
    # We will stick to the existing signature compatibility.
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    ensure_dir(out_p.parent)
    
    if out_p.exists() and not force: return out_p
    
    # Note: This command name might vary by version. 
    # v3: CreateGradientNonlinearityBMatrix is correct for calculating changes.
    cmd = f"CreateGradientNonlinearityBMatrix --input {in_p} --coefficients {grad_coeffs} --output {out_p} --threads {nthreads}"
    run_cmd(cmd, label="TORTOISE_GNL")
    
    return out_p
