
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


def _get_qmri_fit_command(subcommand: str, legacy_binary: Optional[str] = None) -> list[str]:
    """Resolve the modern qmri-fit/qmri_fit CLI with optional legacy fallback."""
    for name in ("qmri-fit", "qmri_fit"):
        qmri_fit = shutil.which(name)
        if qmri_fit:
            return [qmri_fit, subcommand]
    if legacy_binary:
        return [_get_binary(legacy_binary)]
    raise RuntimeError(
        "Binary 'qmri-fit' or 'qmri_fit' not found in PATH. Please ensure it is installed."
    )


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


def _resolve_output_path(out_dir: Path, out_base: str, metric: str) -> Path:
    canonical = out_dir / f"{out_base}_{metric}.nii.gz"
    legacy = out_dir / f"{out_base}{metric}.nii.gz"

    if legacy.exists() and not canonical.exists():
        legacy.rename(canonical)
    return canonical


def _normalize_legacy_outputs(out_dir: Path, out_base: str, metrics: List[str]) -> None:
    for metric in metrics:
        _resolve_output_path(out_dir, out_base, metric)

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
    Wrapper for `qmri_fit despot1`.
    """
    out_d = ensure_dir(out_dir)
    spgr_p = extract_image_path(spgr_file)
    
    cmd_parts = _get_qmri_fit_command("despot1", legacy_binary="qmri_fit_despot1") + [
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
        "t1": _resolve_output_path(out_d, out_base, "T1"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
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
    Wrapper for `qmri_fit despot1_hifi`.
    """
    out_d = ensure_dir(out_dir)
    
    cmd_parts = _get_qmri_fit_command("despot1_hifi", legacy_binary="qmri_fit_despot1_hifi") + [
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
        "t1": _resolve_output_path(out_d, out_base, "T1"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
        "b1": _resolve_output_path(out_d, out_base, "B1"), # Assuming HIFI outputs B1 map
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
    Wrapper for `qmri_fit despot2`.
    """
    out_d = ensure_dir(out_dir)
    
    cmd_parts = _get_qmri_fit_command("despot2", legacy_binary="qmri_fit_despot2") + [
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
        "t2": _resolve_output_path(out_d, out_base, "T2"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
        "f0": _resolve_output_path(out_d, out_base, "F0"), # If estimated or passed through
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
    Wrapper for `qmri_fit despot2fm`.

    DESPOT2FM outputs the fitted single-component DESPOT2FM maps from the
    binary, not the mcDESPOT multicomponent parameter set.
    """
    out_d = ensure_dir(out_dir)
    
    cmd_parts = _get_qmri_fit_command("despot2fm", legacy_binary="qmri_fit_despot2fm") + [
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
    _normalize_legacy_outputs(
        out_d,
        out_base,
        ["T2", "M0", "F0"],
    )
    
    outputs = {
        "t2": _resolve_output_path(out_d, out_base, "T2"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
        "f0": _resolve_output_path(out_d, out_base, "F0"),
    }
    return outputs


def fit_mcdespot(
    spgr_file: ImageLike | Path,
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
    Wrapper for `qmri_fit mcdespot`.

    This uses the unified CLI contract: SPGR and SSFP stacks plus params, B1,
    optional F0/mask, and standard output/threading flags.
    """
    out_d = ensure_dir(out_dir)

    cmd_parts = (
        [_get_binary("qmri_fit_mcdespot_cuda")]
        if cuda
        else _get_qmri_fit_command("mcdespot", legacy_binary="qmri_fit_mcdespot")
    ) + [
        f"--spgr={extract_image_path(spgr_file)}",
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
    _normalize_legacy_outputs(
        out_d,
        out_base,
        ["MWF", "T1_fast", "T1_slow", "T2_fast", "T2_slow", "Tau", "T2", "M0", "F0"],
    )

    outputs = {
        "mwf": _resolve_output_path(out_d, out_base, "MWF"),
        "t1_fast": _resolve_output_path(out_d, out_base, "T1_fast"),
        "t1_slow": _resolve_output_path(out_d, out_base, "T1_slow"),
        "t2_fast": _resolve_output_path(out_d, out_base, "T2_fast"),
        "t2_slow": _resolve_output_path(out_d, out_base, "T2_slow"),
        "tau": _resolve_output_path(out_d, out_base, "Tau"),
    }
    return outputs
