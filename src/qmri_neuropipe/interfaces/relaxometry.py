
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import shutil

from ..core.run import run_cmd
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, ImageFile

def _get_binary(name: str) -> str:
    """Find binary in PATH."""
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Binary '{name}' not found in PATH. Please ensure it is installed.")
    return path


def _append_cli_options(cmd_parts: list[str], options: Optional[Dict[str, Any]] = None) -> None:
    if not options:
        return
    for key, value in options.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd_parts.append(flag)
            continue
        cmd_parts.append(f"{flag}={value}")

def fit_despot1(
    spgr_file: ImageLike | Path,
    params_file: Path,
    out_dir: Path,
    irspgr_file: Optional[ImageLike | Path] = None,
    b1_file: Optional[ImageLike | Path] = None,
    mask_file: Optional[ImageLike | Path] = None,
    out_base: str = "despot1",
    algo: str = "lsq",
    log_json: Optional[Path] = None,
    nthreads: int = 1,
    verbose: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Wrapper for qmri_fit_despot1.
    """
    binary = _get_binary("qmri_fit_despot1")
    out_d = ensure_dir(out_dir)
    spgr_p = extract_image_path(spgr_file)
    
    cmd_parts = [
        binary,
        f"--spgr={spgr_p}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base}",
        f"--algo={algo}",
        f"--nthreads={nthreads}"
    ]
    
    if irspgr_file:
        cmd_parts.append(f"--irspgr={extract_image_path(irspgr_file)}")
        
    if b1_file:
        cmd_parts.append(f"--b1={extract_image_path(b1_file)}")
        
    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")
        
    if log_json:
        cmd_parts.append(f"--log-json={log_json}")
        
    if verbose:
        cmd_parts.append("--verbose")
    _append_cli_options(cmd_parts, extra_options)
        
    run_cmd(" ".join(cmd_parts), label="despot1_fit")
    
    # Expected outputs
    outputs = {
        "t1": out_d / f"{out_base}_T1.nii.gz",
        "m0": out_d / f"{out_base}_M0.nii.gz"
    }
    return outputs

def fit_despot1_hifi(
    spgr_file: ImageLike | Path,
    irspgr_file: ImageLike | Path,
    params_file: Path,
    out_dir: Path,
    mask_file: Optional[ImageLike | Path] = None,
    out_base: str = "despot1_hifi",
    algo: str = "lsq",
    nthreads: int = 1,
    verbose: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Wrapper for qmri_fit_despot1_hifi.
    """
    binary = _get_binary("qmri_fit_despot1_hifi")
    out_d = ensure_dir(out_dir)
    
    cmd_parts = [
        binary,
        f"--spgr={extract_image_path(spgr_file)}",
        f"--irspgr={extract_image_path(irspgr_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base}",
        f"--algo={algo}",
        f"--nthreads={nthreads}"
    ]
    
    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")
    
    if verbose:
        cmd_parts.append("--verbose")
    _append_cli_options(cmd_parts, extra_options)
        
    run_cmd(" ".join(cmd_parts), label="despot1_hifi_fit")
    
    outputs = {
        "t1": out_d / f"{out_base}_T1.nii.gz",
        "m0": out_d / f"{out_base}_M0.nii.gz",
        "b1": out_d / f"{out_base}_B1.nii.gz" # Assuming HIFI outputs B1 map
    }
    return outputs

def fit_despot2(
    ssfp_file: ImageLike | Path,
    t1_file: ImageLike | Path,
    b1_file: ImageLike | Path,
    params_file: Path,
    out_dir: Path,
    f0_file: Optional[ImageLike | Path] = None,
    mask_file: Optional[ImageLike | Path] = None,
    out_base: str = "despot2",
    algo: str = "lsq",
    nthreads: int = 1,
    verbose: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Wrapper for qmri_fit_despot2.
    """
    binary = _get_binary("qmri_fit_despot2")
    out_d = ensure_dir(out_dir)
    
    cmd_parts = [
        binary,
        f"--ssfp={extract_image_path(ssfp_file)}",
        f"--t1={extract_image_path(t1_file)}",
        f"--b1={extract_image_path(b1_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base}",
        f"--algo={algo}",
        f"--nthreads={nthreads}"
    ]
    
    if f0_file:
        cmd_parts.append(f"--f0={extract_image_path(f0_file)}")
        
    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

    if verbose:
        cmd_parts.append("--verbose")
    _append_cli_options(cmd_parts, extra_options)

    run_cmd(" ".join(cmd_parts), label="despot2_fit")
    
    outputs = {
        "t2": out_d / f"{out_base}_T2.nii.gz",
        "m0": out_d / f"{out_base}_M0.nii.gz",
        "f0": out_d / f"{out_base}_F0.nii.gz" # If estimated or passed through
    }
    return outputs

def fit_despot2_fm(
    ssfp_file: ImageLike | Path,
    t1_file: ImageLike | Path,
    b1_file: ImageLike | Path,
    params_file: Path,
    out_dir: Path,
    f0_file: Optional[ImageLike | Path] = None,
    mask_file: Optional[ImageLike | Path] = None,
    out_base: str = "despot2_fm",
    algo: str = "src", # SRC is specific to despot2_fm?
    nthreads: int = 1,
    verbose: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Wrapper for qmri_fit_despot2fm (mcDESPOT / 2-Component).
    """
    binary = _get_binary("qmri_fit_despot2fm")
    out_d = ensure_dir(out_dir)
    
    cmd_parts = [
        binary,
        f"--ssfp={extract_image_path(ssfp_file)}",
        f"--t1={extract_image_path(t1_file)}",
        f"--b1={extract_image_path(b1_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base}",
        f"--algo={algo}",
        f"--nthreads={nthreads}"
    ]
    
    if f0_file:
        cmd_parts.append(f"--f0={extract_image_path(f0_file)}")
        
    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

    if verbose:
        cmd_parts.append("--verbose")
    _append_cli_options(cmd_parts, extra_options)

    run_cmd(" ".join(cmd_parts), label="despot2fm_fit")
    
    outputs = {
        "mwf": out_d / f"{out_base}_MWF.nii.gz", # Myelin Water Fraction
        "t1_fast": out_d / f"{out_base}_T1_fast.nii.gz",
        "t1_slow": out_d / f"{out_base}_T1_slow.nii.gz",
        "t2_fast": out_d / f"{out_base}_T2_fast.nii.gz",
        "t2_slow": out_d / f"{out_base}_T2_slow.nii.gz",
        "tau": out_d / f"{out_base}_Tau.nii.gz" # Residence time / Exchange
    }
    return outputs


def fit_mcdespot(
    ssfp_file: ImageLike | Path,
    t1_file: ImageLike | Path,
    b1_file: ImageLike | Path,
    params_file: Path,
    out_dir: Path,
    f0_file: Optional[ImageLike | Path] = None,
    mask_file: Optional[ImageLike | Path] = None,
    out_base: str = "mcdespot",
    algo: str = "src",
    nthreads: int = 1,
    verbose: bool = False,
    cuda: bool = False,
    extra_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Wrapper for qmri_fit_mcdespot.

    This currently assumes the CLI contract matches the existing DESPOT2-FM style
    interface: SSFP stack plus T1, B1, params, optional F0 and mask, along with
    standard output base and threading flags.
    """
    binary = _get_binary("qmri_fit_mcdespot_cuda" if cuda else "qmri_fit_mcdespot")
    out_d = ensure_dir(out_dir)

    cmd_parts = [
        binary,
        f"--ssfp={extract_image_path(ssfp_file)}",
        f"--t1={extract_image_path(t1_file)}",
        f"--b1={extract_image_path(b1_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base}",
        f"--algo={algo}",
        f"--nthreads={nthreads}"
    ]

    if f0_file:
        cmd_parts.append(f"--f0={extract_image_path(f0_file)}")

    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

    if verbose:
        cmd_parts.append("--verbose")
    _append_cli_options(cmd_parts, extra_options)

    run_cmd(" ".join(cmd_parts), label="mcdespot_fit")

    outputs = {
        "mwf": out_d / f"{out_base}_MWF.nii.gz",
        "t1_fast": out_d / f"{out_base}_T1_fast.nii.gz",
        "t1_slow": out_d / f"{out_base}_T1_slow.nii.gz",
        "t2_fast": out_d / f"{out_base}_T2_fast.nii.gz",
        "t2_slow": out_d / f"{out_base}_T2_slow.nii.gz",
        "tau": out_d / f"{out_base}_Tau.nii.gz"
    }
    return outputs
