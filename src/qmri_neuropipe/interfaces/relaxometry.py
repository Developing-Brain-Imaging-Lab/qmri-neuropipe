
from pathlib import Path
from typing import Optional, List, Dict, Any
import shlex
import shutil
import subprocess

from ..core.run import run_cmd
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike

_UNIFIED_FITTERS = ("qmri-fit", "qmri_fit")
_V4_CUDA_FITTERS = ("qmri_cuda", "qmr_cuda")


def _get_qmri_fit_command(
    subcommand: str, legacy_binary: Optional[str] = None, *, cuda: bool = False
) -> list[str]:
    """Resolve a fitter and its model entry point from PATH."""
    if cuda:
        for name in _V4_CUDA_FITTERS:
            if binary := shutil.which(name):
                return [binary, subcommand, "--device=gpu"]
        for name in ("qmri_fit_cuda", *_UNIFIED_FITTERS):
            if binary := shutil.which(name):
                help_result = subprocess.run(
                    [binary, "--help"], capture_output=True, text=True, errors="replace"
                )
                if help_result.returncode == 0 and "mcdespot_cuda" in help_result.stdout.split():
                    return [binary, "mcdespot_cuda"]
        if binary := shutil.which("qmri_fit_mcdespot_cuda"):
            return [binary]
        raise RuntimeError(
            "CUDA mcDESPOT requires qmri_cuda (or qmr_cuda), a CUDA-enabled "
            "qmri_fit with the mcdespot_cuda subcommand, or qmri_fit_mcdespot_cuda on PATH."
        )

    for name in ("qmri", *_UNIFIED_FITTERS):
        if binary := shutil.which(name):
            return [binary, subcommand]
    if legacy_binary:
        if binary := shutil.which(legacy_binary):
            return [binary]
    candidates = ", ".join(name for name in ("qmri", *_UNIFIED_FITTERS, legacy_binary) if name)
    raise RuntimeError(f"No fitter for {subcommand} found on PATH. Expected one of {candidates}.")


def _is_v4(command: list[str]) -> bool:
    return Path(command[0]).name in ("qmri", *_V4_CUDA_FITTERS)


def _append_cli_options(cmd_parts: list[str], options: Optional[Dict[str, Any]] = None) -> None:
    if not options:
        return
    for key, value in options.items():
        if value is None:
            continue
        if _is_v4(cmd_parts) and key == "t1f-despot-bounds":
            if str(value).lower() != "false":
                raise ValueError("t1f-despot-bounds requires qmri_fit; qmri v4 does not implement it.")
            continue
        if "--device=gpu" in cmd_parts and key == "device":
            if str(value).lower() != "gpu":
                raise ValueError("mcdespot.cuda: true requires device: gpu.")
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            # These options take an explicit boolean value; omitting False is
            # not equivalent for all qmri-fit builds (notably mcDESPOT's
            # t1f-despot-bounds parser).
            if key in {"t1f-despot-bounds", "scale-to-mean"}:
                cmd_parts.append(f"{flag}={'true' if value else 'false'}")
            elif value:
                cmd_parts.append(flag)
            continue
        cmd_parts.append(f"{flag}={value}")


def _run_fit(
    cmd_parts: list[str], label: str, algo: str, verbose: bool,
    extra_options: Optional[Dict[str, Any]],
) -> None:
    binary = Path(cmd_parts[0]).name
    # The workflow's historical LSQ spelling means nonlinear least squares.
    if algo.lower() == "lsq" and binary in ("qmri", *_UNIFIED_FITTERS):
        algo = "CERES"
    cmd_parts.append(f"--algo={algo}")
    if verbose and not _is_v4(cmd_parts):
        cmd_parts.append("--verbose")
    _append_cli_options(cmd_parts, extra_options)
    if binary in _UNIFIED_FITTERS and cmd_parts[1] in {"despot1", "despot1_hifi"}:
        # These docopt entry points read omitted paths as strings.
        if not any(arg.startswith("--mask=") for arg in cmd_parts):
            cmd_parts.append("--mask=")
        if cmd_parts[1] == "despot1_hifi" and not any(arg.startswith("--b1=") for arg in cmd_parts):
            cmd_parts.append("--b1=")
    run_cmd(shlex.join(cmd_parts), label=label)


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
    
    command = _get_qmri_fit_command("despot1", legacy_binary="qmri_fit_despot1")
    cmd_parts = command + [
        f"--spgr={spgr_p}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base + '_' if _is_v4(command) else out_base}",
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

    _run_fit(cmd_parts, "despot1_fit", algo, verbose, extra_options)
    
    # Expected outputs
    outputs = {
        "t1": _resolve_output_path(out_d, out_base, "T1"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
    }
    if (extra_options or {}).get("save-rates", False):
        outputs["r1"] = _resolve_output_path(out_d, out_base, "R1")
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
    b1_smoothing_fwhm: float = 0.0,
    work_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Wrapper for `qmri_fit despot1_hifi`.
    """
    import math
    from contextlib import ExitStack
    from tempfile import TemporaryDirectory

    fwhm = float(b1_smoothing_fwhm)
    if not math.isfinite(fwhm) or fwhm < 0:
        raise ValueError("b1_smoothing_fwhm must be finite and nonnegative (mm)")
    with ExitStack() as stack:
        out_d = ensure_dir(out_dir)
        fit_dir = out_d
        final_base = out_base
        if fwhm > 0:
            out_base = f"{out_base}_desc-unsmoothed"
            fit_dir = ensure_dir(work_dir) if work_dir is not None else Path(
                stack.enter_context(TemporaryDirectory(prefix="hifi-unsmoothed-"))
            )
            if fit_dir.resolve() == out_d.resolve():
                raise ValueError("HiFi work_dir must differ from the final output directory")
    
        command = _get_qmri_fit_command("despot1_hifi", legacy_binary="qmri_fit_despot1_hifi")
        cmd_parts = command + [
            f"--spgr={extract_image_path(spgr_file)}",
            f"--irspgr={extract_image_path(irspgr_file)}",
            f"--params={params_file}",
            f"--out_dir={fit_dir}",
            f"--out_base={out_base + '_' if _is_v4(command) else out_base}",
            f"--nthreads={nthreads}"
        ]
    
        if mask_file:
            cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

        _run_fit(cmd_parts, "despot1_hifi_fit", algo, verbose, extra_options)
    
        outputs = {
            "t1": _resolve_output_path(fit_dir, out_base, "T1"),
            "m0": _resolve_output_path(fit_dir, out_base, "M0"),
            "b1": _resolve_output_path(fit_dir, out_base, "B1"), # Assuming HIFI outputs B1 map
        }
        if (extra_options or {}).get("save-rates", False):
            outputs["r1"] = _resolve_output_path(fit_dir, out_base, "R1")
        if fwhm > 0:
            smoothed_b1 = out_d / f"{final_base}_B1.nii.gz"
            _smooth_hifi_b1(outputs["b1"], smoothed_b1, fwhm, mask_file)
            # Fit SPGR alone, fixing B1 to the smoothed HiFi estimate. Keep the
            # original joint-fit maps in the work directory only.
            outputs = fit_despot1(
                spgr_file=spgr_file, params_file=params_file, out_dir=out_d,
                b1_file=smoothed_b1, mask_file=mask_file, out_base=final_base,
                algo=algo, nthreads=nthreads, verbose=verbose,
                extra_options=extra_options,
            )
            outputs["b1"] = smoothed_b1
        import json
        (out_d / f"{final_base}_B1.json").write_text(json.dumps({
            "B1SmoothingFWHM": fwhm,
            "T1RefitWithFixedB1": fwhm > 0,
        }, indent=2) + "\n")
        return outputs


def _smooth_hifi_b1(source: Path, output: Path, fwhm: float, mask_file=None) -> None:
    """Smooth valid B1 samples using normalized Gaussian weights in mm."""
    import nibabel as nib
    import numpy as np
    from scipy.ndimage import gaussian_filter

    image = nib.load(str(source))
    data = image.get_fdata()
    if data.ndim != 3:
        raise ValueError("HiFi B1 smoothing requires a 3D map")
    valid = np.isfinite(data) & (data > 0)
    if mask_file is not None:
        mask = nib.load(str(extract_image_path(mask_file)))
        if mask.shape != image.shape or not np.allclose(mask.affine, image.affine):
            raise ValueError("HiFi B1 mask must match the B1 voxel grid")
        valid &= np.isfinite(mask.get_fdata()) & (mask.get_fdata() > 0)
    if not np.any(valid):
        raise ValueError("HiFi B1 map has no valid positive samples to smooth")
    sigma = fwhm / np.sqrt(8 * np.log(2)) / np.asarray(image.header.get_zooms()[:3])
    weights = gaussian_filter(valid.astype(float), sigma, mode="constant")
    weighted = gaussian_filter(np.where(valid, data, 0), sigma, mode="constant")
    smoothed = np.zeros_like(data)
    np.divide(weighted, weights, out=smoothed, where=valid & (weights > 0))
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(smoothed.astype(np.float32), image.affine, header), str(output))

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
    
    command = _get_qmri_fit_command("despot2", legacy_binary="qmri_fit_despot2")
    cmd_parts = command + [
        f"--ssfp={extract_image_path(ssfp_file)}",
        f"--t1={extract_image_path(t1_file)}",
        f"--b1={extract_image_path(b1_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base + '_' if _is_v4(command) else out_base}",
        f"--nthreads={nthreads}"
    ]
    
    if f0_file:
        cmd_parts.append(f"--f0={extract_image_path(f0_file)}")
        
    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

    _run_fit(cmd_parts, "despot2_fit", algo, verbose, extra_options)
    
    outputs = {
        "t2": _resolve_output_path(out_d, out_base, "T2"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
    }
    f0_path = _resolve_output_path(out_d, out_base, "F0")
    if not _is_v4(command) or f0_path.exists():
        outputs["f0"] = f0_path
    if (extra_options or {}).get("save-rates", False):
        outputs["r2"] = _resolve_output_path(out_d, out_base, "R2")
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
    
    command = _get_qmri_fit_command("despot2fm", legacy_binary="qmri_fit_despot2fm")
    cmd_parts = command + [
        f"--ssfp={extract_image_path(ssfp_file)}",
        f"--t1={extract_image_path(t1_file)}",
        f"--b1={extract_image_path(b1_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={out_base + '_' if _is_v4(command) else out_base}",
        f"--nthreads={nthreads}"
    ]
    
    if f0_file:
        cmd_parts.append(f"--f0={extract_image_path(f0_file)}")
        
    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

    _run_fit(cmd_parts, "despot2fm_fit", algo, verbose, extra_options)
    output_metrics = ["T2", "M0", "F0"]
    if (extra_options or {}).get("save-rates", False):
        output_metrics.append("R2")
    _normalize_legacy_outputs(
        out_d,
        out_base,
        output_metrics,
    )
    
    outputs = {
        "t2": _resolve_output_path(out_d, out_base, "T2"),
        "m0": _resolve_output_path(out_d, out_base, "M0"),
        "f0": _resolve_output_path(out_d, out_base, "F0"),
    }
    if (extra_options or {}).get("save-rates", False):
        outputs["r2"] = _resolve_output_path(out_d, out_base, "R2")
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
    optional F0/mask, and standard output/threading flags. ``t1_file`` is kept
    in the Python API for compatibility with existing workflow callers, but
    current CPU and CUDA mcDESPOT CLIs derive any requested DESPOT1 bounds from
    the SPGR data and do not accept a ``--t1`` option.
    """
    out_d = ensure_dir(out_dir)
    # qmri_fit concatenates mcDESPOT parameter names directly onto out_base
    # (for example, ``out_base + "VFm.nii.gz"``). Keep the public API tolerant
    # of either spelling while always giving the executable one trailing
    # underscore, yielding BIDS-like ``..._model-mcDESPOT_VFm.nii.gz`` names.
    canonical_out_base = out_base.rstrip("_")
    cli_out_base = f"{canonical_out_base}_"

    command = _get_qmri_fit_command(
        "mcdespot", legacy_binary="qmri_fit_mcdespot", cuda=cuda
    )
    cmd_parts = command + [
        f"--spgr={extract_image_path(spgr_file)}",
        f"--ssfp={extract_image_path(ssfp_file)}",
        f"--b1={extract_image_path(b1_file)}",
        f"--params={params_file}",
        f"--out_dir={out_d}",
        f"--out_base={cli_out_base}",
        f"--nthreads={nthreads}"
    ]

    # Pre-unified standalone CPU builds required an external DESPOT1 map. The
    # unified and CUDA interfaces no longer expose --t1, so only preserve it
    # for the one-element legacy command returned by the resolver.
    if not cuda and len(command) == 1:
        cmd_parts.append(f"--t1={extract_image_path(t1_file)}")

    if f0_file:
        cmd_parts.append(f"--f0={extract_image_path(f0_file)}")

    if mask_file:
        cmd_parts.append(f"--mask={extract_image_path(mask_file)}")

    _run_fit(cmd_parts, "mcdespot_fit", algo, verbose, extra_options)
    model = str((extra_options or {}).get("model", "2pool")).lower()
    metrics = ["VFm", "T1m", "T2m", "T1f", "T2f", "Tau", "F0"]
    if model == "3pool":
        metrics.extend(["VFcsf", "T1csf", "T2csf"])
    _normalize_legacy_outputs(
        out_d,
        canonical_out_base,
        metrics,
    )
    legacy_metadata = out_d / f"{canonical_out_base}run_metadata.json"
    canonical_metadata = out_d / f"{canonical_out_base}_run_metadata.json"
    if legacy_metadata.exists() and not canonical_metadata.exists():
        legacy_metadata.rename(canonical_metadata)

    output_keys = {
        "vfm": "VFm",
        "t1m": "T1m",
        "t2m": "T2m",
        "t1f": "T1f",
        "t2f": "T2f",
        "tau": "Tau",
        "f0": "F0",
    }
    if model == "3pool":
        output_keys.update(
            {"vfcsf": "VFcsf", "t1csf": "T1csf", "t2csf": "T2csf"}
        )
    outputs = {
        key: _resolve_output_path(out_d, canonical_out_base, metric)
        for key, metric in output_keys.items()
    }
    if _is_v4(command):
        # v4 omits parameters that were fixed or disabled (e.g. F0 or Tau).
        for key in ("f0", "tau"):
            if not outputs[key].exists():
                del outputs[key]
    return outputs
