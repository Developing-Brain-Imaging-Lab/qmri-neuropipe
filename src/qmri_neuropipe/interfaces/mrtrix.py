from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from ..core.run import run_cmd
from ..core.types import ImageLike, DWIFile
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


def dwidenoise(in_file: ImageLike | Path, out_file: Path, nthreads: int=1, mask: Optional[Path]=None, noise_map: Optional[Path]=None, force: bool=False) -> Tuple[Path, Optional[Path]]:
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
    
def dwibiascorrect(in_file: ImageLike | Path, in_bvec: Path, in_bval: Path, out_file: Path, method: str = "ants", mask: Optional[Path]=None, bias_field: Optional[Path]=None, nthreads: int = 1, force: bool = False):
    
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

def mrdegibbs(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, force: bool = False):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    force_arg = f"-force" if force else ""
    cmd = f"mrdegibbs {in_p} {out_p} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="mrdegibbs")

    return out_p

def dwi2mask(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, force: bool = False):
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    if not force and out_p.exists():
        return out_p
    cmd = f"dwi2mask {in_p} {out_p} -nthreads {nthreads} -quiet" + (" -force" if force else "")
    run_cmd(cmd, label="dwi2mask")
    return out_p


def dwigradcheck(in_file: ImageLike | Path, in_bvec: Path = None, in_bval: Path = None, export_grad_fsl: Tuple[Path, Path] = None, nthreads: int = 1, force: bool = False):
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


def fit_dti(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    metrics: list[str] = ['fa', 'adc', 'ad', 'rd'],
    nthreads: int = 1
) -> Dict[str, Path]:
    """
    Fit DTI using MRtrix3.
    
    nthreads : int
        Number of threads (-nthreads).
    """
    import shutil
    import json
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path
    
    in_path = extract_image_path(in_file)
    
    if bval_file is None or bvec_file is None:
        if isinstance(in_file, DWIFile):
            bval_file = bval_file or in_file.bval
            bvec_file = bvec_file or in_file.bvec
            
    if not bval_file or not bvec_file:
         raise ValueError("Gradient files (bval/bvec) are required but not provided or found in input DWIFile.")
         
    ent_base = get_entities_from_path(in_path)
    if 'desc' in ent_base: del ent_base['desc']
    ent_base['model'] = 'DTI'
    
    # 1. dwi2tensor
    # Output tensor file
    tensor_out_name = build_bids_name({**ent_base, 'suffix': 'tensor'})
    dt_out = out_dir / tensor_out_name
    
    cmd = ["dwi2tensor", str(in_path), str(dt_out), "-fslgrad", str(bvec_file), str(bval_file)]
    
    if mask_file:
        cmd.extend(["-mask", str(mask_file)])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    
    cmd.extend(["-quiet", "-force"])
    run_cmd(" ".join(cmd), label="dwi2tensor")
    
    # Save sidecar for tensor
    sidecar = {
        "ModelName": "Diffusion Tensor Imaging",
        "FittingSoftware": "MRtrix3",
        "InputData": in_path.name,
        "FittingMethod": "dwi2tensor (iterative reweighted linear least squares)"
    }
    with open(str(dt_out).replace('.nii.gz', '.json'), 'w') as f:
         json.dump(sidecar, f, indent=4)
    
    # 2. tensor2metric
    cmd_metric = f"tensor2metric {dt_out} -quiet -force"
    if mask_file:
        cmd_metric += f" -mask {mask_file}"
        
    output_files = {}
    if 'tensor' in metrics:
         output_files['tensor'] = dt_out

    for m in metrics:
        if m == 'tensor': continue
        
        # Map metric name to BIDS suffix and MRtrix flag
        flag = f"-{m}"
        suffix = m.upper()
        if m == 'md': 
            flag = '-adc'
            suffix = 'MD'
        if m == 'adc': 
            flag = '-adc'
            suffix = 'MD'
            
        out_name = build_bids_name({**ent_base, 'suffix': suffix})
        out_path = out_dir / out_name
        
        cmd_metric += f" {flag} {out_path}"
        output_files[m] = out_path
        
        # Save sidecar
        with open(str(out_path).replace('.nii.gz', '.json'), 'w') as f:
             json.dump(sidecar, f, indent=4)
        
    run_cmd(cmd_metric, label="tensor2metric")
    
    return output_files


def mrconvert(
    in_file: Union[Path, ImageLike], 
    out_file: Path, 
    stride: str = None, 
    datatype: str = None,
    in_bvec: Path = None, 
    in_bval: Path = None, 
    export_grad_fsl: Tuple[Path, Path] = None, 
    json_import: Path = None, 
    json_export: Path = None, 
    nthreads: int = 1,
    force: bool = False
):
    """
    Wrapper for mrconvert.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    if not force and out_p.exists():
        if export_grad_fsl:
            if export_grad_fsl[0].exists() and export_grad_fsl[1].exists():
                return
        else:
            return

    cmd = ["mrconvert", str(in_p), str(out_p)]
    
    if stride:
        cmd.extend(["-stride", stride])
        
    if datatype:
        cmd.extend(["-datatype", datatype])
        
    if in_bvec and in_bval:
        cmd.extend(["-fslgrad", str(in_bvec), str(in_bval)])
        
    if export_grad_fsl:
        cmd.extend(["-export_grad_fsl", str(export_grad_fsl[0]), str(export_grad_fsl[1])])
        
    if json_import:
        cmd.extend(["-json_import", str(json_import)])
        
    if json_export:
        cmd.extend(["-json_export", str(json_export)])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.extend(["-quiet"])
    if force:
        cmd.append("-force")

    run_cmd(" ".join(cmd), label="mrconvert")
    
def dwi2response(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    in_bvec: Path = None,
    in_bval: Path = None,
    mask_file: Path = None,
    algorithm: str = "dhollander",
    nthreads: int = 1,
    force: bool = False
) -> Dict[str, Path]:
    """
    Wrapper for dwi2response.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_dir)

    # Output filenames depend on algorithm
    # dhollander produces wm, gm, csf responses
    # tournier produces response
    # We will assume standard output names for now and return a dict
    
    responses = {}
    if algorithm == 'dhollander':
        responses['wm'] = out_p / "response_wm.txt"
        responses['gm'] = out_p / "response_gm.txt"
        responses['csf'] = out_p / "response_csf.txt"
        all_exist = all(p.exists() for p in responses.values())
    else:
        # Default single response
        responses['response'] = out_p / "response.txt"
        all_exist = responses['response'].exists()

    if not force and all_exist:
        return responses

    cmd = ["dwi2response", algorithm, str(in_p)]
    
    if algorithm == 'dhollander':
        cmd.extend([str(responses['wm']), str(responses['gm']), str(responses['csf'])])
    else:
        cmd.append(str(responses['response']))
        
    if in_bvec and in_bval:
        cmd.extend(["-fslgrad", str(in_bvec), str(in_bval)])
        
    if mask_file:
        cmd.extend(["-mask", str(mask_file)])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.extend(["-quiet"])
    if force:
        cmd.append("-force")
    
    run_cmd(" ".join(cmd), label=f"dwi2response-{algorithm}")
    
    return responses

def dwi2fod(
    in_file: Union[Path, ImageLike],
    response_files: Dict[str, Path],
    out_dir: Path,
    in_bvec: Path = None,
    in_bval: Path = None,
    mask_file: Path = None,
    algorithm: str = "msmt_csd",
    lmax: Optional[Union[int, str]] = None,
    nthreads: int = 1,
    force: bool = False
) -> Dict[str, Path]:
    """
    Wrapper for dwi2fod.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_dir)
    
    fods = {}
    # Determine outputs based on inputs and algorithm
    # msmt_csd takes pairs of response, output
    
    cmd = ["dwi2fod", algorithm, str(in_p)]
    
    # Check what responses we have
    if 'wm' in response_files and 'gm' in response_files and 'csf' in response_files:
        # 3-tissue
        fods['wm'] = out_p / "wmfod.nii.gz"
        fods['gm'] = out_p / "gmfod.nii.gz"
        fods['csf'] = out_p / "csffod.nii.gz"
        
        cmd.extend([str(response_files['wm']), str(fods['wm'])])
        cmd.extend([str(response_files['gm']), str(fods['gm'])])
        cmd.extend([str(response_files['csf']), str(fods['csf'])])
        
    elif 'wm' in response_files and 'csf' in response_files:
        # 2-tissue
        fods['wm'] = out_p / "wmfod.nii.gz"
        fods['csf'] = out_p / "csffod.nii.gz"
        cmd.extend([str(response_files['wm']), str(fods['wm'])])
        cmd.extend([str(response_files['csf']), str(fods['csf'])])
        
    elif 'response' in response_files:
        # Single shell / single tissue
        fods['fod'] = out_p / "fod.nii.gz"
        cmd.extend([str(response_files['response']), str(fods['fod'])])
        
    else:
        # Try to infer from keys if specific names aren't used but count matches?
        # For now, require specific keys or 'response'
        pass

    # Check existence
    all_exist = all(p.exists() for p in fods.values())
    if not force and all_exist:
        return fods

    if in_bvec and in_bval:
        cmd.extend(["-fslgrad", str(in_bvec), str(in_bval)])
        
    if mask_file:
        cmd.extend(["-mask", str(mask_file)])
        
    if lmax:
        cmd.extend(["-lmax", str(lmax)])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.extend(["-quiet"])
    if force:
        cmd.append("-force")
    
    run_cmd(" ".join(cmd), label=f"dwi2fod-{algorithm}")
    
    return fods


def transformconvert(
    in_transform: Path,
    out_mrtrix_transform: Path,
    operation: str = 'flirt_import', 
    ref_image: Optional[Path] = None,
    in_image: Optional[Path] = None,
    force: bool = False
):
    """
    Wrapper for transformconvert.
    
    Args:
        in_transform: Input transform file (e.g. FSL mat, etc.)
        out_mrtrix_transform: Output MRTrix transform file.
        operation: 'flirt_import', 'itk_import', etc.
        ref_image: Reference image (required for flirt_import).
        in_image: Input moving image (required for flirt_import).
        force: Force overwrite.
    """
    out_p = ensure_dir(out_mrtrix_transform)
    
    if not force and out_p.exists():
        return out_p
    
    cmd = ["transformconvert", str(in_transform), operation]
    
    if operation == 'flirt_import':
        if not ref_image or not in_image:
             raise ValueError("transformconvert 'flirt_import' requires ref_image and in_image.")
        cmd.append(str(in_image))
        cmd.append(str(ref_image))
        
    cmd.append(str(out_p))
    
    if force:
        cmd.append("-force")
        
    cmd.append("-quiet")
    
    run_cmd(" ".join(cmd), label="transformconvert")
    return out_p


def mrtransform(
    in_file: Union[Path, ImageLike],
    out_file: Path,
    linear_transform: Optional[Path] = None,
    warp_image: Optional[Path] = None,
    template: Optional[Path] = None,
    interp: str = "cubic",
    datatype: str = None,
    nthreads: int = 1,
    force: bool = False,
    **kwargs
):
    """
    Wrapper for mrtransform.
    
    Args:
        in_file: Input image.
        out_file: Output transformed image.
        linear_transform: Linear transform (e.g. from transformconvert).
        warp_image: Nonlinear warp image.
        template: Template image (for regridding).
        interp: Interpolation ('nearest', 'linear', 'cubic', 'sinc'). Default 'cubic'.
        datatype: Output data type.
        nthreads: Number of threads.
        force: Force overwrite.
        **kwargs: Additional flags.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if not force and out_p.exists():
        return out_p
        
    cmd = ["mrtransform", str(in_p), str(out_p)]
    
    if linear_transform:
        cmd.extend(["-linear", str(linear_transform)])
        
    if warp_image:
        cmd.extend(["-warp", str(warp_image)])
        
    if template:
        cmd.extend(["-template", str(template)])
        
    if interp:
        cmd.extend(["-interp", interp])
        
    if datatype:
        cmd.extend(["-datatype", datatype])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    if force:
        cmd.append("-force")
        
    cmd.append("-quiet")
    
    run_cmd(" ".join(cmd), label="mrtransform")
    return out_p


def apply_mrtrix_transform(
    dwi_file: Union[Path, DWIFile],
    out_dwi: Path,
    transform_file: Path,
    transform_type: str = "flirt", # 'flirt' or 'mrtrix'
    ref_image: Optional[Path] = None,
    interp: str = "cubic",
    nthreads: int = 1,
    force: bool = False
) -> Tuple[Path, Path, Path]:
    """
    Apply a registration transform to a DWI file using MRTrix.
    
    Workflow:
    1. Convert NIfTI DWI -> .mif (embedding bvals/bvecs).
    2. Convert transform to MRTrix format if needed (e.g. FSL FLIRT mat).
    3. Apply mrtransform (handles gradient reorientation automatically).
    4. Convert .mif -> NIfTI (exporting rotated bvals/bvecs).
    
    Args:
        dwi_file: Input DWI file or object (must have bval/bvec if Path).
        out_dwi: Output NIfTI path.
        transform_file: Transform file (e.g. .mat).
        transform_type: Type of input transform ('flirt', 'mrtrix').
        ref_image: Reference target image (required for FLIRT import and regridding).
        interp: Interpolation method for mrtransform.
        nthreads: Logic threads.
        
    Returns:
        (out_dwi, out_bvec, out_bval) paths.
    """
    import os
    
    if isinstance(dwi_file, DWIFile):
        in_path = dwi_file.img
        in_bvec = dwi_file.bvec
        in_bval = dwi_file.bval
        if not in_bvec or not in_bval:
             raise ValueError("DWIFile must have bvec/bval for MRTrix transform.")
    else:
        # Assume path, try to find sidecars? Or require explict args? 
        # For simplicity in this function signature, let's assume user passes DWIFile mostly.
        # Check if we can find them next to image?
        in_path = Path(dwi_file)
        in_bvec = in_path.with_suffix("").with_suffix(".bvec")
        in_bval = in_path.with_suffix("").with_suffix(".bval")
        if not in_bvec.exists(): in_bvec = in_path.with_suffix(".bvec")
        if not in_bval.exists(): in_bval = in_path.with_suffix(".bval")
        
        if not in_bvec.exists() or not in_bval.exists():
             raise ValueError(f"Could not automatically find bvec/bval for {in_path}. Use DWIFile wrapper.")

    out_p = ensure_dir(out_dwi)
    out_bvec = out_p.with_suffix("").with_suffix(".bvec")
    out_bval = out_p.with_suffix("").with_suffix(".bval")
    
    if not force and out_p.exists() and out_bvec.exists():
        return out_p, out_bvec, out_bval

    # Create temporary directory for conversions
    temp_dir = out_p.parent / f"temp_mrtrix_trans_{out_p.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 1. NIfTI -> MIF
        temp_mif_in = temp_dir / "input.mif"
        mrconvert(
            in_file=in_path,
            out_file=temp_mif_in,
            in_bvec=in_bvec,
            in_bval=in_bval,
            nthreads=nthreads,
            force=True
        )
        
        # 2. Prepare Transform
        mrtrix_transform = transform_file
        if transform_type == 'flirt':
             # Convert FSL mat to MRTrix
             if not ref_image:
                 raise ValueError("ref_image is required for FLIRT transform conversion.")
                 
             mrtrix_transform = temp_dir / "transform_mrtrix.txt"
             transformconvert(
                 in_transform=transform_file,
                 out_mrtrix_transform=mrtrix_transform,
                 operation='flirt_import',
                 ref_image=ref_image,
                 in_image=in_path,
                 force=True
             )
        
        # 3. Apply Transform
        temp_mif_out = temp_dir / "output.mif"
        mrtransform(
            in_file=temp_mif_in,
            out_file=temp_mif_out,
            linear_transform=mrtrix_transform,
            template=ref_image, # Use ref image as template for grid
            interp=interp,
            nthreads=nthreads,
            force=True
        )
        
        # 4. MIF -> NIfTI + bvecs
        if out_bvec.exists(): out_bvec.unlink() # cleanup before export to ensure fresh
        if out_bval.exists(): out_bval.unlink()
        
        mrconvert(
            in_file=temp_mif_out,
            out_file=out_p,
            export_grad_fsl=(out_bvec, out_bval),
            nthreads=nthreads,
            force=True
        )
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    return out_p, out_bvec, out_bval
