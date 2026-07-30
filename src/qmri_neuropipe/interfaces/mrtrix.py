from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Dict, Union
from ..core.run import run_cmd, _writable_tmpdir
from ..core.types import ImageLike, DWIFile
from ..core.utils import ensure_dir, extract_image_path

# Standardize path utility
# Or import: from ..interfaces.fsl import _as_path if we want to share. 
# But fsl.py is not really a "common" util.
# For now, importing from fsl is easiest, or duplicating.
# Let's import to avoid code duplication if fsl is always present.
# But circular imports? fsl imports types.
# Actually, let's redefine _as_path here to be safe and independent.


def _mrtrix_config_args() -> list[str]:
    tmpdir = _writable_tmpdir()
    return ["-config", "TmpFileDir", tmpdir, "-config", "ScriptScratchDir", tmpdir]


_SCRIPT_COMMANDS_WITH_SELECTOR = {
    "5ttgen",
    "dwi2response",
    "dwibiascorrect",
}


def _global_option_index(parts: list[str]) -> int:
    """Return where MRtrix global options may be inserted safely."""
    if parts and Path(str(parts[0])).name in _SCRIPT_COMMANDS_WITH_SELECTOR:
        # These scripts require an algorithm/backend selector immediately
        # after the executable (for example, ``dwibiascorrect ants``).
        return min(2, len(parts))
    return min(1, len(parts))


def _mrtrix_cmd(parts: list[str]) -> str:
    if not parts:
        return ""
    index = _global_option_index(parts)
    configured = [
        *map(str, parts[:index]),
        *_mrtrix_config_args(),
        *map(str, parts[index:]),
    ]
    return " ".join(configured)


def _mrtrix_script_parts(parts: list[str]) -> list[str]:
    """Add MRtrix Python-script scratch directory without affecting compiled tools."""
    if any(str(part) == "-scratch" for part in parts):
        return parts
    index = _global_option_index(parts)
    return [
        *map(str, parts[:index]),
        "-scratch",
        _writable_tmpdir(),
        *map(str, parts[index:]),
    ]


def _run_mrtrix(parts: list[str], *, label: str, script: bool = False, n_threads: int = None) -> None:
    cmd_parts = _mrtrix_script_parts(parts) if script else parts
    run_cmd(
        _mrtrix_cmd(cmd_parts),
        label=label,
        n_threads=n_threads,
    )


def _run_mrtrix_str(cmd: str, *, label: str) -> None:
    parts = cmd.split(maxsplit=1)
    if not parts:
        return
    configured_cmd = " ".join([parts[0], *_mrtrix_config_args(), *(parts[1:] if len(parts) > 1 else [])])
    run_cmd(configured_cmd, label=label)


def dwidenoise(in_file: ImageLike | Path, out_file: Path, nthreads: int=1, mask: Optional[Path]=None, noise_map: Optional[Path]=None, force: bool=False) -> Tuple[Path, Optional[Path]]:
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    nm_p = ensure_dir(noise_map) if noise_map else None

    # Skip if already done (unless force)
    if not force and out_p.exists() and (not nm_p or nm_p.exists()):
        return str(out_p), (str(nm_p) if nm_p else None)
    
    mask_p = extract_image_path(mask) if mask else None
    mask_arg  = f"-mask {mask_p}" if mask_p else ""
    noise_arg = f"-noise {noise_map}" if noise_map else ""
    force_arg = f"-force" if force else ""
    cmd = ["dwidenoise", str(in_p), str(out_p)]
    if mask_p:
        cmd.extend(["-mask", str(mask_p)])
    if noise_map:
        cmd.extend(["-noise", str(noise_map)])
    cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="dwidenoise")
   
    return out_p, (nm_p if nm_p else None)
    
def dwibiascorrect(in_file: ImageLike | Path, out_file: Path, in_bvec: Path = None, in_bval: Path = None, method: str = "ants", mask: Optional[Path]=None, bias_field: Optional[Path]=None, nthreads: int = 1, force: bool = False):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    # Auto-extract gradient table if possible
    if in_bvec is None and in_bval is None and isinstance(in_file, DWIFile):
        in_bvec = in_file.bvec
        in_bval = in_file.bval

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    mask_p = extract_image_path(mask) if mask else None
    bf_p = extract_image_path(bias_field) if bias_field else None

    diff_arg  = f"-fslgrad {in_bvec} {in_bval}" if (in_bvec and in_bval) else ""
    mask_arg  = f"-mask {mask_p}" if mask_p else ""
    bias_arg  = f"-bias {bf_p}" if bf_p else ""
    force_arg = f"-force" if force else ""
    cmd = ["dwibiascorrect", method, str(in_p)]
    if in_bvec and in_bval:
        cmd.extend(["-fslgrad", str(in_bvec), str(in_bval)])
    cmd.append(str(out_p))
    if mask_p:
        cmd.extend(["-mask", str(mask_p)])
    if bf_p:
        cmd.extend(["-bias", str(bf_p)])
    # MRtrix passes -nthreads through to N4BiasFieldCorrection as -nt for
    # the ANTs backend, but some ANTs builds reject that flag. Use env
    # thread limits via run_cmd(n_threads=...) instead.
    if str(method).lower() != "ants":
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="dwibiascorrect", script=True, n_threads=nthreads)
    
    return out_p

def mrdegibbs(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, force: bool = False):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    force_arg = f"-force" if force else ""
    cmd = ["mrdegibbs", str(in_p), str(out_p), "-nthreads", str(nthreads)]
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="mrdegibbs")

    return out_p

def dwi2mask(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, force: bool = False):
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    if not force and out_p.exists():
        return out_p
    cmd = ["dwi2mask", str(in_p), str(out_p), "-nthreads", str(nthreads), "-quiet"]
    if force:
        cmd.append("-force")
    _run_mrtrix(cmd, label="dwi2mask")
    return out_p



def maskfilter(in_file: ImageLike | Path, out_file: Path, filter_type: str = 'dilate', npass: int = 1, nthreads: int = 1, force: bool = False):
    """
    Wrapper for maskfilter.
    
    Args:
        in_file: Input mask
        out_file: Output mask
        filter_type: 'dilate', 'erode', etc.
        npass: Number of passes
        nthreads: Number of threads
        force: Force overwrite
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if not force and out_p.exists():
        return out_p
        
    cmd = ["maskfilter", str(in_p), filter_type, str(out_p), "-npass", str(npass), "-nthreads", str(nthreads), "-quiet"]
    if force:
        cmd.append("-force")
        
    _run_mrtrix(cmd, label=f"maskfilter-{filter_type}")
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
        
    _run_mrtrix(cmd_parts, label="dwigradcheck", script=True)


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
    
    metrics_norm = [metric.strip().lower() for metric in metrics]

    def _write_sidecar(path: Path, output_metric: str, extras: Optional[dict] = None) -> None:
        payload = dict(sidecar)
        payload["OutputMetric"] = output_metric
        if extras:
            payload.update(extras)
        with open(str(path).replace('.nii.gz', '.json'), 'w') as f:
            json.dump(payload, f, indent=4)

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
    _run_mrtrix(cmd, label="dwi2tensor")
    
    # Save sidecar for tensor
    sidecar = {
        "ModelName": "Diffusion Tensor Imaging",
        "FittingSoftware": "MRtrix3",
        "InputData": in_path.name,
        "FittingMethod": "dwi2tensor (iterative reweighted linear least squares)"
    }
    _write_sidecar(dt_out, "tensor", extras={"TensorConvention": "MRtrix", "TensorBasis": "world"})
    
    # 2. tensor2metric
    base_cmd = f"tensor2metric {dt_out} -quiet -force"
    if mask_file:
        base_cmd += f" -mask {mask_file}"

    cmd_metric = base_cmd
    run_main_metric_cmd = False
    deferred_metric_cmds: list[tuple[str, str]] = []
    output_files = {}
    if 'tensor' in metrics_norm:
         output_files['tensor'] = dt_out

    def _metric_suffix(metric_name: str) -> str:
        mapping = {
            'md': 'MD',
            'adc': 'MD',
            'color_fa': 'DECFA',
            'l1': 'L1',
            'l2': 'L2',
            'l3': 'L3',
            'v1': 'V1',
            'v2': 'V2',
            'v3': 'V3',
        }
        return mapping.get(metric_name, metric_name.upper())

    for m in metrics_norm:
        if m == 'tensor': continue
        
        # Map metric name to BIDS suffix and MRtrix flag
        flag = f"-{m}"
        suffix = _metric_suffix(m)
        if m == 'md': 
            flag = '-adc'
        if m == 'adc': 
            flag = '-adc'
        if m == 'color_fa':
            flag = '-vector'
        if m in {'l1', 'l2', 'l3'}:
            flag = '-value'
        if m in {'v1', 'v2', 'v3'}:
            flag = '-vector'
        if m == 'evals':
            flag = '-value'
        if m == 'evecs':
            flag = '-vector'
            
        out_name = build_bids_name({**ent_base, 'suffix': suffix})
        out_path = out_dir / out_name

        if m == 'color_fa':
            cmd_metric += f" {flag} {out_path} -modulate FA"
            run_main_metric_cmd = True
        elif m in {'l1', 'l2', 'l3'}:
            deferred_metric_cmds.append((f"{base_cmd} {flag} {out_path} -num {int(m[-1])}", m))
        elif m in {'v1', 'v2', 'v3'}:
            deferred_metric_cmds.append((f"{base_cmd} {flag} {out_path} -num {int(m[-1])} -modulate none", m))
        elif m == 'evals':
            deferred_metric_cmds.append((f"{base_cmd} {flag} {out_path} -num 1,2,3", m))
        elif m == 'evecs':
            deferred_metric_cmds.append((f"{base_cmd} {flag} {out_path} -num 1,2,3 -modulate none", m))
        else:
            cmd_metric += f" {flag} {out_path}"
            run_main_metric_cmd = True
        output_files[m] = out_path
        extras = {"VectorConvention": "world"} if m in {'color_fa', 'v1', 'v2', 'v3', 'evecs'} else None
        _write_sidecar(out_path, suffix, extras=extras)
        
    if run_main_metric_cmd:
        _run_mrtrix_str(cmd_metric, label="tensor2metric")
    for metric_cmd, metric_name in deferred_metric_cmds:
        _run_mrtrix_str(metric_cmd, label=f"tensor2metric_{metric_name}")

    if 'tensor_mrtrix' in metrics_norm:
        tensor_mrtrix_out = out_dir / build_bids_name({**ent_base, 'suffix': 'tensorMRTRIX'})
        shutil.copyfile(dt_out, tensor_mrtrix_out)
        _write_sidecar(tensor_mrtrix_out, "tensorMRTRIX", extras={"TensorConvention": "MRtrix", "TensorBasis": "world"})
        output_files['tensor_mrtrix'] = tensor_mrtrix_out
    
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
        
    # Auto-extract gradient table
    if in_bvec is None and in_bval is None and isinstance(in_file, DWIFile):
        in_bvec = in_file.bvec
        in_bval = in_file.bval

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

    _run_mrtrix(cmd, label="mrconvert")
    
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
    
    _run_mrtrix(cmd, label=f"dwi2response-{algorithm}", script=True)
    
    return responses

def dwi2fod(
    in_file: Union[Path, ImageLike],
    response_files: Dict[str, Path],
    out_dir: Path,
    in_bvec: Path = None,
    in_bval: Path = None,
    mask_file: Path = None,
    algorithm: str = "msmt_csd",
    lmax: Optional[Union[int, str, list[int], tuple[int, ...]]] = None,
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
        
    if lmax is not None:
        if isinstance(lmax, (list, tuple)):
            lmax_str = ",".join(map(str, lmax))
        else:
            lmax_str = str(lmax)

            # MRtrix expects one lmax per input tissue for multi-tissue CSD.
            # A scalar configuration conventionally describes the anisotropic
            # WM compartment; the GM and CSF compartments are isotropic.
            if algorithm == "msmt_csd" and len(fods) > 1 and "," not in lmax_str:
                lmax_str = ",".join([lmax_str, *(["0"] * (len(fods) - 1))])
        cmd.extend(["-lmax", lmax_str])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.extend(["-quiet"])
    if force:
        cmd.append("-force")
    
    # dwi2fod is a compiled MRtrix binary, not a Python script. It accepts
    # -config temp-dir settings but not the script-only -scratch option.
    _run_mrtrix(cmd, label=f"dwi2fod-{algorithm}")
    
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
    
    cmd = ["transformconvert", str(in_transform)]
    
    if operation == 'flirt_import':
        if not ref_image or not in_image:
             raise ValueError("transformconvert 'flirt_import' requires ref_image and in_image.")
        cmd.append(str(in_image))
        cmd.append(str(ref_image))
        cmd.append(operation)
    else:
        cmd.append(operation)
        
    cmd.append(str(out_p))
    
    if force:
        cmd.append("-force")
        
    cmd.append("-quiet")
    
    _run_mrtrix(cmd, label="transformconvert")
    return out_p


def mrtransform(
    in_file: Union[Path, ImageLike],
    out_file: Path,
    linear_transform: Optional[Path] = None,
    warp_image: Optional[Path] = None,
    template: Optional[Path] = None,
    interp: str = "cubic",
    datatype: str = None,
    strides: Optional[Union[Path, str]] = None,
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
        strides: Set strides (logical orientation) of output image. Can be path or string (e.g. '-1,2,3').
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
    
    if strides:
        if isinstance(strides, (list, tuple)):
            strides_str = ",".join(map(str, strides))
        else:
            strides_str = str(strides)
        cmd.extend(["-strides", strides_str])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    if force:
        cmd.append("-force")
    
    cmd.append("-reorient_fod no")
    cmd.append("-quiet")
    
    _run_mrtrix(cmd, label="mrtransform")
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
    from ..core.utils import get_nifti_stem
    temp_dir = out_p.parent / f"temp_mrtrix_trans_{get_nifti_stem(out_p)}"
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


def dwiextract(
    in_file: Union[Path, ImageLike],
    out_file: Path,
    bzero: bool = False,
    no_bzero: bool = False,
    singleshell: bool = False,
    shells: Optional[list] = None,
    in_bvec: Path = None,
    in_bval: Path = None,
    export_grad_fsl: Tuple[Path, Path] = None,
    nthreads: int = 1,
    force: bool = False
):
    """
    Wrapper for dwiextract.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if not force and out_p.exists():
        if export_grad_fsl:
            if export_grad_fsl[0].exists() and export_grad_fsl[1].exists():
                return out_p
        else:
            return out_p

    cmd = ["dwiextract", str(in_p), str(out_p)]
    
    if bzero:
        cmd.append("-bzero")
    if no_bzero:
        cmd.append("-no_bzero")
    if singleshell:
        cmd.append("-singleshell")
        
    if shells:
        shell_str = ",".join(map(str, shells))
        cmd.extend(["-shells", shell_str])
        
    # Auto-extract gradient table
    if in_bvec is None and in_bval is None and isinstance(in_file, DWIFile):
        in_bvec = in_file.bvec
        in_bval = in_file.bval

    if in_bvec and in_bval:
        cmd.extend(["-fslgrad", str(in_bvec), str(in_bval)])
        
    if export_grad_fsl:
        cmd.extend(["-export_grad_fsl", str(export_grad_fsl[0]), str(export_grad_fsl[1])])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.append("-quiet")
    if force:
        cmd.append("-force")
        
    _run_mrtrix(cmd, label="dwiextract")
    return out_p


def mrcalc(
    in_file1: Union[Path, ImageLike],
    operation_or_file2: Union[str, Path, ImageLike],
    out_file: Path,
    *args,
    axis: int = None,
    nthreads: int = 1,
    force: bool = False
):
    """
    Wrapper for mrcalc.
    usage: mrcalc input1 input2 -add output
           mrcalc input1 0.5 -add output
           mrcalc input1 -max output
    """
    in_p = extract_image_path(in_file1)
    out_p = ensure_dir(out_file)
    
    if not force and out_p.exists():
        return out_p
        
    cmd = ["mrcalc", str(in_p)]
    
    # Handle second argument (operation or operand)
    op = str(operation_or_file2)
    # If op is a file/number, use it as operand. If it looks like an operation (text), prepend dash if missing
    # But mrcalc is purely stack based: operand operand op
    # If user passed "mean", it's unary: operand -mean
    # If "add", it's binary: operand operand -add
    
    # Heuristic: try to guess if op is a file or number
    is_path = "/" in op or Path(op).exists() or "." in op # weak check for extension
    
    if is_path: 
         cmd.append(op)
         # If binary, we expect the operation in *args?
         # Or maybe we assume default operation? No.
         # For GNL usage: mrcalc(file, "mean", output). "mean" is unary.
         # If binary, user must pass: mrcalc(file1, "add", out, args=(file2,))? 
         # Or mrcalc(file1, file2, out, args=("add",))?
         # Let's stick to the GNL usage first.
    else:
         # It's an operation string like "mean" -> "-mean"
         if not op.startswith("-"): op = f"-{op}"
         cmd.append(op)
    
    # Output is handled by ensure_dir result
    cmd.append(str(out_p))
    
    if axis is not None:
        cmd.extend(["-axis", str(axis)])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.append("-quiet")
    if force:
        cmd.append("-force")
        
    _run_mrtrix(cmd, label="mrcalc")
    return out_p


def mrmath(
    in_file: Union[Path, ImageLike],
    operation: str,
    out_file: Path,
    axis: int = 3,
    nthreads: int = 1,
    force: bool = False
):
    """
    Wrapper for mrmath.
    usage: mrmath input mean output -axis 3
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if not force and out_p.exists():
        return out_p
        
    cmd = ["mrmath", str(in_p), operation, str(out_p)]
    
    cmd.extend(["-axis", str(axis)])
        
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.append("-quiet")
    if force:
        cmd.append("-force")
        
    _run_mrtrix(cmd, label="mrmath")
    return out_p


def sh2peaks(
    in_file: Union[Path, ImageLike],
    out_file: Path,
    nthreads: int = 1,
    force: bool = False,
    num_peaks: int = 3,
    threshold: float = 0.1
):
    """
    Wrapper for sh2peaks.
    usage: sh2peaks fod.mif peaks.nii.gz -num 3 -threshold 0.1
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if not force and out_p.exists():
        return out_p
        
    cmd = ["sh2peaks", str(in_p), str(out_p)]
    
    if num_peaks:
        cmd.extend(["-num", str(num_peaks)])
        
    if threshold:
        cmd.extend(["-threshold", str(threshold)])
    
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
        
    cmd.append("-quiet")
    if force:
        cmd.append("-force")
        
    _run_mrtrix(cmd, label="sh2peaks")
    return out_p


def _append_options(cmd: list[str], options: Optional[Dict[str, Any]]) -> None:
    """Append MRtrix-style options from a mapping.

    ``True`` emits a flag, ``False``/``None`` omit it, and sequences emit one
    option followed by all their values. Keys may be written with or without a
    leading dash. This is intentionally small and is used only for documented
    MRtrix passthrough options.
    """
    for key, value in (options or {}).items():
        flag = str(key) if str(key).startswith("-") else f"-{key}"
        if value is True:
            cmd.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            cmd.extend([flag, *map(str, value)])
        else:
            cmd.extend([flag, str(value)])


def five_tt_gen(
    algorithm: str,
    in_file: ImageLike | Path,
    out_file: Path,
    *,
    options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Generate an ACT five-tissue-type image with ``5ttgen``."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["5ttgen", algorithm, str(extract_image_path(in_file)), str(out_p)]
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label=f"5ttgen-{algorithm}", script=True, n_threads=nthreads)
    return out_p


def five_tt_check(in_file: ImageLike | Path, *, nthreads: int = 1) -> Path:
    """Validate a five-tissue-type image using ``5ttcheck``."""
    in_p = extract_image_path(in_file)
    cmd = ["5ttcheck", str(in_p), "-quiet"]
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    _run_mrtrix(cmd, label="5ttcheck", n_threads=nthreads)
    return in_p


def five_tt_to_gmwmi(
    in_file: ImageLike | Path, out_file: Path, *, force: bool = False
) -> Path:
    """Create a grey-matter/white-matter interface seed mask."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["5tt2gmwmi", str(extract_image_path(in_file)), str(out_p)]
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="5tt2gmwmi")
    return out_p


def tckgen(
    in_file: ImageLike | Path,
    out_file: Path,
    *,
    algorithm: str = "iFOD2",
    select: int = 10_000_000,
    act: Optional[Path] = None,
    seed_gmwmi: Optional[Path] = None,
    seed_image: Optional[Path] = None,
    mask: Optional[Path] = None,
    include: Optional[Iterable[Path]] = None,
    exclude: Optional[Iterable[Path]] = None,
    options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Generate a tractogram using ``tckgen``."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tckgen", str(extract_image_path(in_file)), str(out_p), "-algorithm", algorithm]
    if select:
        cmd.extend(["-select", str(select)])
    if act:
        cmd.extend(["-act", str(act)])
    if seed_gmwmi:
        cmd.extend(["-seed_gmwmi", str(seed_gmwmi)])
    elif seed_image:
        cmd.extend(["-seed_image", str(seed_image)])
    if mask:
        cmd.extend(["-mask", str(mask)])
    for roi in include or []:
        cmd.extend(["-include", str(roi)])
    for roi in exclude or []:
        cmd.extend(["-exclude", str(roi)])
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label=f"tckgen-{algorithm}", n_threads=nthreads)
    return out_p


def tckedit(
    in_file: Path,
    out_file: Path,
    *,
    include: Optional[Iterable[Path]] = None,
    exclude: Optional[Iterable[Path]] = None,
    ends_only: bool = False,
    options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Extract or edit streamlines using ROI constraints."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tckedit", str(in_file), str(out_p)]
    for roi in include or []:
        cmd.extend(["-include", str(roi)])
    for roi in exclude or []:
        cmd.extend(["-exclude", str(roi)])
    if ends_only:
        cmd.append("-ends_only")
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tckedit", n_threads=nthreads)
    return out_p


def tcksift2(
    tracks: Path,
    fod: ImageLike | Path,
    out_weights: Path,
    *,
    act: Optional[Path] = None,
    options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Estimate per-streamline SIFT2 weights."""
    out_p = ensure_dir(out_weights)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tcksift2", str(tracks), str(extract_image_path(fod)), str(out_p)]
    if act:
        cmd.extend(["-act", str(act)])
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tcksift2", n_threads=nthreads)
    return out_p


def tcksift(
    tracks: Path,
    fod: ImageLike | Path,
    out_file: Path,
    *,
    act: Optional[Path] = None,
    term_number: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Filter a tractogram using SIFT."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tcksift", str(tracks), str(extract_image_path(fod)), str(out_p)]
    if act:
        cmd.extend(["-act", str(act)])
    if term_number:
        cmd.extend(["-term_number", str(term_number)])
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tcksift", n_threads=nthreads)
    return out_p


def tckmap(
    tracks: Path, out_file: Path, *, template: Optional[Path] = None,
    weights: Optional[Path] = None, options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1, force: bool = False,
) -> Path:
    """Create a track-density image."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tckmap", str(tracks), str(out_p)]
    if template:
        cmd.extend(["-template", str(template)])
    if weights:
        cmd.extend(["-tck_weights_in", str(weights)])
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tckmap", n_threads=nthreads)
    return out_p


def tcksample(
    tracks: Path, image: ImageLike | Path, out_file: Path, *,
    statistic_tck: Optional[str] = None, options: Optional[Dict[str, Any]] = None,
    nthreads: int = 1, force: bool = False,
) -> Path:
    """Sample an image along streamlines."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tcksample", str(tracks), str(extract_image_path(image)), str(out_p)]
    if statistic_tck:
        cmd.extend(["-stat_tck", statistic_tck])
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tcksample", n_threads=nthreads)
    return out_p


def tckresample(
    tracks: Path, out_file: Path, *, num_points: int = 100,
    nthreads: int = 1, force: bool = False,
) -> Path:
    """Resample every streamline to a fixed number of points."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tckresample", str(tracks), str(out_p), "-num_points", str(num_points)]
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tckresample", n_threads=nthreads)
    return out_p


def tck2connectome(
    tracks: Path, nodes: ImageLike | Path, out_file: Path, *,
    weights: Optional[Path] = None, scale_file: Optional[Path] = None,
    statistic: str = "sum", symmetric: bool = True, zero_diagonal: bool = True,
    options: Optional[Dict[str, Any]] = None, nthreads: int = 1,
    force: bool = False,
) -> Path:
    """Generate a connectome matrix from a tractogram and node image."""
    out_p = ensure_dir(out_file)
    if out_p.exists() and not force:
        return out_p
    cmd = ["tck2connectome", str(tracks), str(extract_image_path(nodes)), str(out_p)]
    if weights:
        cmd.extend(["-tck_weights_in", str(weights)])
    if scale_file:
        cmd.extend(["-scale_file", str(scale_file)])
    if statistic:
        cmd.extend(["-stat_edge", statistic])
    if symmetric:
        cmd.append("-symmetric")
    if zero_diagonal:
        cmd.append("-zero_diagonal")
    _append_options(cmd, options)
    if nthreads > 1:
        cmd.extend(["-nthreads", str(nthreads)])
    if force:
        cmd.append("-force")
    cmd.append("-quiet")
    _run_mrtrix(cmd, label="tck2connectome", n_threads=nthreads)
    return out_p
