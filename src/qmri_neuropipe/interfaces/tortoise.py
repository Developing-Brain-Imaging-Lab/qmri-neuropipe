from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union
import shutil
import shlex
import os
import nibabel as nib
from ..core.run import run_cmd
from ..core.types import DWIFile
from ..core.utils import ensure_dir
from ..core import ProcessingError


def _format_v4_option(name: str, value: Any) -> list[str]:
    """Format a TORTOISEProcess option without shell-specific ambiguity."""
    option = f"--{name}"
    if isinstance(value, bool):
        return [option, "1" if value else "0"]
    if isinstance(value, (list, tuple)):
        return [option, *(str(item) for item in value)]
    return [option, str(value)]


def build_tortoise_v4_command(
    dwi_file: DWIFile,
    out_file: Path,
    *,
    down_file: Optional[DWIFile] = None,
    structural_file: Optional[Path | Sequence[Path]] = None,
    reorientation_file: Optional[Path] = None,
    b0_id: int = -1,
    correction_mode: str = "quadratic",
    slice_to_volume: bool = False,
    repol: bool = False,
    niter: int = 3,
    denoising: str = "off",
    gibbs: bool = False,
    drift: str = "off",
    epi: str = "off",
    output_orientation: Optional[str] = None,
    output_res: Optional[Sequence[float]] = None,
    output_voxels: Optional[Sequence[int]] = None,
    output_data_combination: Optional[str] = None,
    output_signal_redist_method: Optional[str] = None,
    temp_folder: Optional[Path] = None,
    executable: str = "TORTOISEProcess",
    do_qc: bool = True,
    extra_options: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Build the TORTOISEV4 motion/eddy command for an existing DWI.

    All major TORTOISEV4 stages are explicit so the caller can choose either a
    correction-only invocation or a complete single-interpolation workflow.
    """
    if not dwi_file.bval or not dwi_file.bvec:
        raise ValueError("TORTOISEV4 requires bval and bvec sidecars")
    command = [
        executable,
        "--up_data", str(dwi_file.img),
        "--ub", str(dwi_file.bval),
        "--uv", str(dwi_file.bvec),
        "--output", str(out_file),
        "--denoising", str(denoising),
        "--gibbs", "1" if gibbs else "0",
        "--drift", str(drift),
        "--epi", str(epi),
        "--correction_mode", str(correction_mode),
        "--b0_id", str(int(b0_id)),
        "--s2v", "1" if slice_to_volume else "0",
        "--repol", "1" if repol else "0",
        "--niter", str(int(niter)),
        "--do_QC", "1" if do_qc else "0",
    ]
    if down_file:
        if not down_file.bval or not down_file.bvec:
            raise ValueError("TORTOISEV4 down_data requires bval and bvec sidecars")
        command.extend([
            "--down_data", str(down_file.img),
            "--db", str(down_file.bval),
            "--dv", str(down_file.bvec),
        ])
    if structural_file:
        structurals = (
            list(structural_file)
            if isinstance(structural_file, (list, tuple))
            else [structural_file]
        )
        command.extend(["--structural", *(str(path) for path in structurals)])
    if reorientation_file:
        command.extend(["--reorientation", str(reorientation_file)])
    if output_orientation:
        command.extend(["--output_orientation", str(output_orientation)])
    if temp_folder:
        command.extend(["--temp_folder", str(temp_folder)])
    if output_res:
        command.extend(["--output_res", *(str(float(value)) for value in output_res)])
    if output_voxels:
        command.extend(["--output_voxels", *(str(int(value)) for value in output_voxels)])
    if output_data_combination:
        command.extend(["--output_data_combination", str(output_data_combination)])
    if output_signal_redist_method:
        command.extend(["--output_signal_redist_method", str(output_signal_redist_method)])
    for name, value in (extra_options or {}).items():
        if value is None:
            continue
        command.extend(_format_v4_option(str(name).lstrip("-"), value))
    return command


def tortoise_v4_motion_eddy(
    dwi_file: DWIFile,
    out_file: Path,
    *,
    down_file: Optional[DWIFile] = None,
    structural_file: Optional[Path | Sequence[Path]] = None,
    reorientation_file: Optional[Path] = None,
    b0_id: int = -1,
    correction_mode: str = "quadratic",
    slice_to_volume: bool = False,
    repol: bool = False,
    niter: int = 3,
    denoising: str = "off",
    gibbs: bool = False,
    drift: str = "off",
    epi: str = "off",
    output_orientation: Optional[str] = None,
    output_res: Optional[Sequence[float]] = None,
    output_voxels: Optional[Sequence[int]] = None,
    output_data_combination: Optional[str] = None,
    output_signal_redist_method: Optional[str] = None,
    temp_folder: Optional[Path] = None,
    executable: Optional[str] = None,
    use_gpu: bool = False,
    nthreads: int = 1,
    do_qc: bool = True,
    extra_options: Optional[Mapping[str, Any]] = None,
) -> DWIFile:
    """Run TORTOISEV4 and return its corrected gradients to the pipeline."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    executable = executable or ("TORTOISEProcess_cuda" if use_gpu else "TORTOISEProcess")
    if shutil.which(executable) is None:
        raise ProcessingError(
            f"Required TORTOISEV4 executable '{executable}' was not found in PATH"
        )

    staging_dir = out_file.parent / "tortoise_inputs"
    staged_up = _stage_tortoise_v4_input(dwi_file, staging_dir, "up")
    staged_down = (
        _stage_tortoise_v4_input(down_file, staging_dir, "down") if down_file else None
    )
    command = build_tortoise_v4_command(
        staged_up,
        out_file,
        down_file=staged_down,
        structural_file=structural_file,
        reorientation_file=reorientation_file,
        b0_id=b0_id,
        correction_mode=correction_mode,
        slice_to_volume=slice_to_volume,
        repol=repol,
        niter=niter,
        denoising=denoising,
        gibbs=gibbs,
        drift=drift,
        epi=epi,
        output_orientation=output_orientation,
        output_res=output_res,
        output_voxels=output_voxels,
        output_data_combination=output_data_combination,
        output_signal_redist_method=output_signal_redist_method,
        temp_folder=temp_folder,
        executable=executable,
        do_qc=do_qc,
        extra_options=extra_options,
    )
    run_cmd(" ".join(shlex.quote(part) for part in command), label="TORTOISEV4", n_threads=nthreads)

    base = Path(str(out_file).split(".nii", 1)[0])
    generated_bvec = Path(f"{base}.bvecs")
    generated_bval = Path(f"{base}.bvals")
    if not out_file.exists() or not generated_bvec.exists() or not generated_bval.exists():
        raise ProcessingError(
            "TORTOISEV4 did not create the requested image and corrected gradient sidecars"
        )
    out_bvec = Path(f"{base}.bvec")
    out_bval = Path(f"{base}.bval")
    shutil.copy2(generated_bvec, out_bvec)
    shutil.copy2(generated_bval, out_bval)
    return DWIFile(
        entities=dict(dwi_file.entities),
        img=out_file,
        json=dwi_file.json,
        bval=out_bval,
        bvec=out_bvec,
        Delta=getattr(dwi_file, "Delta", None),
        delta=getattr(dwi_file, "delta", None),
    )


def _stage_tortoise_v4_input(dwi_file: DWIFile, staging_dir: Path, label: str) -> DWIFile:
    """Give TORTOISE basename-matched JSON without mutating pipeline inputs."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if str(dwi_file.img).endswith(".nii.gz") else ".nii"
    staged_img = staging_dir / f"{label}{suffix}"
    if staged_img.exists() or staged_img.is_symlink():
        staged_img.unlink()
    shape = nib.load(str(dwi_file.img)).shape
    if len(shape) == 3:
        # TORTOISE expands 3D reverse-PE b0 inputs to 4D in place.
        shutil.copy2(dwi_file.img, staged_img)
        staged_bval = staging_dir / f"{label}.bval"
        staged_bvec = staging_dir / f"{label}.bvec"
        staged_bval.write_text("0 0\n")
        staged_bvec.write_text("0 0\n0 0\n0 0\n")
    else:
        os.symlink(Path(dwi_file.img).resolve(), staged_img)
        staged_bval = dwi_file.bval
        staged_bvec = dwi_file.bvec

    staged_json = Path(str(staged_img).split(".nii", 1)[0] + ".json")
    if not dwi_file.json or not Path(dwi_file.json).exists():
        raise ProcessingError(f"TORTOISEV4 requires JSON metadata for {label}_data")
    shutil.copy2(dwi_file.json, staged_json)
    return DWIFile(
        entities=dict(dwi_file.entities),
        img=staged_img,
        json=staged_json,
        bval=staged_bval,
        bvec=staged_bvec,
        Delta=getattr(dwi_file, "Delta", None),
        delta=getattr(dwi_file, "delta", None),
    )

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
    initial_image: Optional[Path],
    final_image: Path,
    grad_coeffs: Path,
    nthreads: int = 1,
    force: bool = False,
    is_ge: bool = True,
    cwd: Optional[Path] = None
) -> None:
    """
    Apply gradient nonlinearity correction using CreateGradientNonlinearityBMatrix.
    
    Args:
        initial_image: Native space image (usually mean b0). Optional.
        final_image: Final space image (usually mean b0 of processing space).
        grad_coeffs: Coefficients file.
        nthreads: Number of threads.
        force: Force run.
        is_ge: Whether to apply GE specific corrections.
        cwd: Working directory for execution (output usually generated here).
    """
    
    final_image = Path(final_image).resolve()
    grad_coeffs = Path(grad_coeffs).resolve()
    initial_image = Path(initial_image).resolve() if initial_image else None

    if not final_image.exists():
        raise ProcessingError(f"TORTOISE final_image does not exist: {final_image}")
    if initial_image and not initial_image.exists():
        raise ProcessingError(f"TORTOISE initial_image does not exist: {initial_image}")
    if not grad_coeffs.exists():
        raise ProcessingError(f"TORTOISE nonlinearity coefficients file does not exist: {grad_coeffs}")

    # Updated command based on user feedback to use CreateGradientNonlinearityBMatrix with correct flags
    executable = "CreateGradientNonlinearityBMatrix"
    if shutil.which(executable) is None:
        raise ProcessingError(
            f"Required TORTOISE executable '{executable}' was not found in PATH. "
            "Install TORTOISE inside the container or use the native GE GNL backend "
            "by setting dmri.preprocessing.grad_nonlin.method: native_ge when the "
            "DWI sidecars include GE gradient nonlinearity metadata."
        )

    cmd_parts = [executable]
    
    if initial_image:
        cmd_parts.append(f"--initial_image {initial_image}")
        
    cmd_parts.append(f"--final_image {final_image}")
    cmd_parts.append(f"--nonlinearity {grad_coeffs}")
    
    if is_ge:
        cmd_parts.append("--isGE")
    
    # Run command
    # Output file (graddev_c.nii) is created in the CWD or typically implied by the input.
    # We rely on the caller to find and rename the output.
    
    run_cmd(" ".join(cmd_parts), label="TORTOISE_GNL", n_threads=nthreads, cwd=cwd)
