from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
import json
import logging
import os
import subprocess

import nibabel as nib
import numpy as np

from ..core.run import run_cmd
from ..core.types import DWIFile, ImageLike
from ..io.dmri.bids import build_acqp_index
from ..core.utils import ensure_path, ensure_dir, extract_image_path

logger = logging.getLogger(__name__)

def _format_extra_opts(extra_opts: Optional[Dict[str, Any]], prefix: str = "--") -> list[str]:
    opts: list[str] = []
    if not extra_opts:
        return opts
    for key, val in extra_opts.items():
        if isinstance(val, bool):
            if val:
                opts.append(f"{prefix}{key}")
        else:
            # For header args like -basescale 1, FSL usually uses space not = for legacy tools
            # But eddy uses =.
            # Let's handle based on prefix?
            if prefix == "--":
                 opts.append(f"{prefix}{key}={val}")
            else:
                 opts.append(f"{prefix}{key} {val}")
    return opts

def _ensure_acqp_index(in_dwi: DWIFile, support_dir: Optional[Path] = None) -> tuple[Path | None, Path | None]:
    """
    Create acqp/index files if possible from BIDS metadata.
    """
    return build_acqp_index(
        in_dwi.json,
        in_dwi.img,
        entities=getattr(in_dwi, "entities", None),
        support_dir=support_dir,
    )


def _read_numeric_rows(path: Path) -> list[list[str]]:
    try:
        return [line.split() for line in path.read_text().splitlines() if line.split()]
    except OSError:
        return []


def _count_bval_dirs(path: Optional[Path]) -> Optional[int]:
    if not path:
        return None
    rows = _read_numeric_rows(Path(path))
    if not rows:
        return None
    return sum(len(row) for row in rows)


def _write_bvec_rows(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\n".join(" ".join(row) for row in rows) + "\n")


def _normalise_fsl_bvec(path: Path, expected_dirs: Optional[int] = None) -> Optional[int]:
    rows = _read_numeric_rows(path)
    if not rows:
        return None

    if len(rows) == 3:
        widths = {len(row) for row in rows}
        if len(widths) == 1:
            return widths.pop()

    if expected_dirs and len(rows) == expected_dirs and all(len(row) == 3 for row in rows):
        _write_bvec_rows(path, [list(col) for col in zip(*rows)])
        return expected_dirs

    flat = [value for row in rows for value in row]
    if expected_dirs and len(flat) == expected_dirs * 3:
        vectors = [flat[i:i + 3] for i in range(0, len(flat), 3)]
        _write_bvec_rows(path, [list(col) for col in zip(*vectors)])
        return expected_dirs

    return len(rows[0]) if len(rows) == 3 and rows[0] else None


def _validate_or_reuse_bvec(out_bvec: Path, fallback_bvec: Optional[Path], bval: Optional[Path], label: str) -> Path:
    expected_dirs = _count_bval_dirs(bval)
    actual_dirs = _normalise_fsl_bvec(out_bvec, expected_dirs) if out_bvec.exists() else None
    if expected_dirs and actual_dirs != expected_dirs and fallback_bvec and Path(fallback_bvec).exists():
        logger.warning(
            "%s produced an invalid bvec table (%s directions; expected %s). "
            "Reusing the incoming bvec table.",
            label,
            actual_dirs,
            expected_dirs,
        )
        shutil.copy2(fallback_bvec, out_bvec)
        _normalise_fsl_bvec(out_bvec, expected_dirs)
    return out_bvec


def bet(in_file: ImageLike | Path, out_file: Path, frac: float = 0.5, mask: bool = True, robust: bool = True) -> tuple[Path, Optional[Path]]:
    """
    Wrapper for FSL BET. Accepts ImageLike or Path.
    """
    img_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)

    robust_arg = "-R" if robust else ""
    bet_cmd = f"bet {img_p} {out_p} -f {frac} {robust_arg} " + ("-m" if mask else "")
    if not out_p.exists():
        run_cmd(bet_cmd, label="bet")

    if mask:
        # Standard BET mask naming: <output>_mask.nii.gz 
        # (assuming output has .nii.gz extension)
        # If out_p is "foo.nii.gz", mask is "foo_mask.nii.gz"
        mask_suffix = "_mask" + "".join(out_p.suffixes)
        # Actually fsl simply inserts _mask before the FIRST extension? 
        # Or before .nii.gz?
        # bet "foo.nii.gz" "bar.nii.gz" -m  -> bar_mask.nii.gz
        
        # Robust handling using string replacement on name if standard extension
        name = out_p.name
        if name.endswith(".nii.gz"):
            mask_name = name.replace(".nii.gz", "_mask.nii.gz")
        elif name.endswith(".nii"):
            mask_name = name.replace(".nii", "_mask.nii")
        else:
            mask_name = name + "_mask"
            
        mask_p = out_p.with_name(mask_name)
        return out_p, mask_p
        
    return out_p, None

def flirt(in_file: ImageLike | Path, ref_file: ImageLike | Path, out_file: Path, omat: Path = None, dof: int = 6, cost: str = "normmi", extra_args: str = "", extra_opts: Optional[Dict[str, Any]] = None):
    """
    Wrapper for FSL FLIRT.
    """
    logger.debug(f"Entering fsl.flirt. Outfile: {out_file}")
    in_p = extract_image_path(in_file)
    ref_p = extract_image_path(ref_file)
    out_p = ensure_dir(out_file)
    
    omat_cmd = ""
    if omat:
        omat_p = Path(omat)
        omat_cmd = f"-omat {omat_p}"
        # Skip only if BOTH exist
        if out_p.exists() and omat_p.exists():
             return out_p, omat_p
             
    elif out_p.exists():
         # If no omat requested and output exists, skip
         return out_p, None
    
    # Format extra options
    extra_flags = " ".join(_format_extra_opts(extra_opts, prefix="-"))
    
    cmd = f"flirt -in {in_p} -ref {ref_p} -out {out_p} {omat_cmd} -dof {dof} -cost {cost} {extra_args} {extra_flags}"
    stdout = run_cmd(cmd, label="flirt")
    
    if not out_p.exists():
        parent_dir = out_p.parent
        dir_listing = "Directory not found"
        if parent_dir.exists():
            dir_listing = "\n".join([str(p.name) for p in parent_dir.iterdir()])
            
        raise RuntimeError(f"FLIRT finished with exit code 0 but output file was not created: {out_p}\nSTDOUT:\n{stdout}\n\nContents of {parent_dir}:\n{dir_listing}\n\nCOMMAND:\n{cmd}")
        
    return out_p, Path(omat) if omat else None


def apply_xfm_4d(in_file: Path, ref_file: Path, out_file: Path, mat: Path, interp: str = "trilinear"):
    """
    Apply a 3D linear transform (FLIRT .mat) to a 4D series using applywarp.
    FSL's flirt only applies to the first volume of a 4D series.
    'applywarp --premat' is 4D-aware and handles this correctly even for identity/affine warps.
    """
    return applywarp(in_file, ref_file, out_file, premat=mat, interp=interp, force=True)


def merge(in_files: list[ImageLike | Path], out_file: Path, dimension: str = "t") -> Path:
    """
    Wrapper for fslmerge.
    
    Args:
        in_files: List of input images.
        out_file: Output merged image path.
        dimension: Dimension to merge along ('t', 'x', 'y', 'z', 'a'). Default 't'.
    """
    if not in_files:
        raise ValueError("No input files for fslmerge.")
        
    out_p = ensure_dir(out_file)
    
    # Convert all inputs to paths
    in_paths = [str(extract_image_path(f)) for f in in_files]
    in_str = " ".join(in_paths)
    
    cmd = f"fslmerge -{dimension} {out_p} {in_str}"
    
    if not out_p.exists():
         run_cmd(cmd, label="fslmerge")
         

    return out_p



def mcflirt(
    in_file: ImageLike | Path,
    out_file: Path,
    ref_vol: int = 0,
    cost: str = "normcorr",
    bins: int = 256,
    dof: int = 6,
    scaling: float = 6.0,
    smooth: float = 1.0,
    rotation: int = 0,
    verbose: int = 0,
    stages: int = 3,
    extra_args: str = ""
) -> Path:
    """
    Wrapper for FSL mcflirt.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    # mcflirt -in <input> -out <output> -refvol <vol> ...
    
    if out_p.exists():
        return out_p
        
    cmd = f"mcflirt -in {in_p} -out {out_p} -refvol {ref_vol} -cost {cost} -bins {bins} -dof {dof} -scaling {scaling} -smooth {smooth} -rotation {rotation} -stages {stages} {extra_args}"
    
    run_cmd(cmd, label="mcflirt")
    return out_p


def split(in_file: ImageLike | Path, out_basename: Path, dimension: str = "t") -> List[Path]:
    """
    Wrapper for fslsplit.
    Splits a 4D file into 3D volumes.
    
    Args:
        in_file: Input 4D file.
        out_basename: Basename for output files (e.g. /path/to/vol_).
        dimension: Dimension to split along. Default 't'.
        
    Returns:
        List of generated file paths.
    """
    in_p = extract_image_path(in_file)
    out_base = ensure_path(out_basename) # Just ensuring parent? no, out_basename is a prefix usually?
    # fslsplit <input> <output_basename> -t
    
    # Ensure parent exists
    if out_base.parent:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        
    
    # Use parent to glob
    parent = out_base.parent
    prefix = out_base.name

    # Glob pattern: prefix + 4 digits + .nii.gz (or .nii)
    # FSL usually output .nii.gz if FSLOUTPUTTYPE is NIFTI_GZ
    
    # OUTPUT CLEANUP:
    # Since fslsplit globs matches to return, existing stale files (e.g. from a prior run with more volumes)
    # can cause total file count to exceed expectation.
    # We must clean potential outputs before running.
    stale_files = list(parent.glob(f"{prefix}*"))
    for sf in stale_files:
         try:
             sf.unlink()
         except Exception:
             pass
             
    cmd = f"fslsplit {in_p} {out_base} -{dimension}"
    run_cmd(cmd, label="fslsplit")
    
    # Identify outputs
    files = sorted(list(parent.glob(f"{prefix}*")))
    return files


def applywarp(in_file: ImageLike | Path, ref_file: ImageLike | Path, out_file: Path, warp: Path = None, premat: Path = None, interp: str = "spline", extra_args: str = "", force: bool = False):
    """
    Wrapper for FSL applywarp.
    """
    in_p = extract_image_path(in_file)
    ref_p = extract_image_path(ref_file)
    out_p = ensure_dir(out_file)
    
    if out_p.exists() and not force:
        return out_p

    warp_cmd = f"-w {warp}" if warp else ""
    premat_cmd = f"--premat={premat}" if premat else ""
    
    cmd = f"applywarp -i {in_p} -r {ref_p} -o {out_p} {warp_cmd} {premat_cmd} --interp={interp} {extra_args}"
    run_cmd(cmd, label="applywarp")
    
    return out_p


def topup(
    in_dwis: Iterable[DWIFile],
    out_base: Path,
    *,
    config: Optional[Path] = None,
    field_output: bool = False,
    acqp: Optional[Path] = None,
    index: Optional[Path] = None,
    nthreads: int = 1,
) -> Path:
    """
    Run FSL topup using the mean B0 images from a group of reversed PE DWIs.
    """
    out_base = ensure_dir(out_base)

    b0_vols = []
    aff = None
    hdr = None
    acqp_lines: list[str] = []

    for dwi in in_dwis:
        acqp_path, index_path = (acqp, index) if acqp and index else _ensure_acqp_index(
            dwi,
            support_dir=out_base.parent / "topup_support",
        )
        if not acqp_path or not index_path:
            continue

        try:
            bvals = np.loadtxt(dwi.bval) if dwi.bval else None
        except Exception:
            bvals = None
        if bvals is None or not np.any(bvals == 0):
            continue

        try:
            data = nib.load(str(dwi.img))
        except Exception:
            continue

        idx = np.where(np.atleast_1d(bvals) == 0)[0]
        
        raw_data = data.get_fdata()
        extracted_vols = []
        
        if raw_data.ndim == 3:
            # Single volume 3D file. Assume it matches the single index.
            extracted_vols = [raw_data]
        elif raw_data.ndim == 4:
            # 4D file. Extract specific volumes.
            # Handle if idx is empty? (Covered by check above)
            # Slicing with array index preserves dimension at end: (X,Y,Z, N)
            vols_4d = raw_data[..., idx]
            # Unpack into list of 3D arrays
            for i in range(vols_4d.shape[-1]):
                extracted_vols.append(vols_4d[..., i])
        
        if aff is None:
            aff = data.affine
            hdr = data.header

        try:
            acqp_entries = acqp_path.read_text().strip().splitlines()
            index_entries = [int(v) for v in index_path.read_text().split()]
        except Exception:
            continue

        # Default to first acqp line if lengths mismatch
        for i, vol_i in enumerate(idx):
            if i < len(extracted_vols): # Safety check
                b0_vols.append(extracted_vols[i])
                acqp_idx = index_entries[vol_i] - 1 if vol_i < len(index_entries) else 0
                acqp_lines.append(acqp_entries[acqp_idx] if acqp_entries else "")

    


    if len(b0_vols) < 1 or not acqp_lines:
        raise RuntimeError("No B0 volumes found for topup input.")

    # Check if outputs exist
    movpar = out_base.with_name(out_base.name + "_movpar.txt")
    fieldcoef = out_base.with_name(out_base.name + "_fieldcoef.nii.gz")
    
    if movpar.exists() and fieldcoef.exists():
        return out_base

    imain = out_base.with_name(out_base.name + "_topup_imain.nii.gz")
    datain = out_base.with_name(out_base.name + "_topup_datain.txt")

    topup_data = np.stack(b0_vols, axis=-1)
    nib.save(nib.Nifti1Image(topup_data, aff, hdr), str(imain))
    datain.write_text("\n".join(acqp_lines) + "\n", encoding="utf-8")

    # Apply smoothing to input (user requested -s 1.15)
    imain_smoothed = out_base.with_name(out_base.name + "_topup_imain_smoothed.nii.gz")
    run_cmd(f"fslmaths {imain} -s 1.15 {imain_smoothed}", label="topup_smooth")

    cmd_parts = [
        "topup",
        f"--imain={imain_smoothed}",
        f"--datain={datain}",
        f"--out={out_base}",
    ]
    if config:
        cmd_parts.append(f"--config={config}")
    if field_output:
        cmd_parts.append(f"--fout={out_base}_field.nii.gz")
    
    if nthreads > 1:
        cmd_parts.append(f"--nthr={nthreads}")

    run_cmd(" ".join(cmd_parts), label="topup")
    return out_base


def eddy_correct(in_file: DWIFile, out_file: Path) -> DWIFile:
    """
    Legacy eddy_correct wrapper.
    """
    out_p = ensure_dir(out_file)
    out_base = out_p.with_suffix("").with_suffix("")
    out_bvec = out_base.with_suffix(".bvec")
    ecclog = out_base.with_suffix(".ecclog")

    ecc_cmd = f"eddy_correct {in_file.img} {out_p} 0"
    fdt_cmd = f"fdt_rotate_bvecs {in_file.bvec} {out_bvec} {ecclog}"

    if not out_p.exists():
        run_cmd(ecc_cmd, label="eddy_correct")

    if in_file.bvec and not out_bvec.exists():
        run_cmd(fdt_cmd, label="fdt_rotate_bvecs")
    if in_file.bvec:
        _validate_or_reuse_bvec(out_bvec, in_file.bvec, in_file.bval, "fdt_rotate_bvecs")

    return DWIFile(
        entities=in_file.entities,
        img=out_p,
        json=in_file.json,
        bval=in_file.bval,
        bvec=out_bvec,
        Delta=getattr(in_file, "Delta", None),
        delta=getattr(in_file, "delta", None),
    )


import subprocess
import shutil
import os
import re

def _which(cmd: str) -> Optional[str]:
    """Return path to command if found."""
    return shutil.which(cmd)

def _find_eddy_cuda() -> Optional[str]:
    """Find the best available eddy_cuda executable."""
    # 1. Check for generic symlink/binary first
    if shutil.which("eddy_cuda"):
        return "eddy_cuda"
        
    # 2. Search for versioned binaries (eddy_cuda8.0, eddy_cuda9.1, etc.)
    candidates = []
    search_paths = os.environ.get("PATH", "").split(os.pathsep)
    fsldir = os.environ.get("FSLDIR")
    if fsldir:
        search_paths.insert(0, os.path.join(fsldir, "bin"))
        
    seen = set()
    for path in search_paths:
        if not path or not os.path.isdir(path): continue
        try:
            for f in os.listdir(path):
                if f.startswith("eddy_cuda") and f not in seen:
                    full_path = os.path.join(path, f)
                    if os.access(full_path, os.X_OK):
                        candidates.append(f)
                        seen.add(f)
        except OSError:
            pass
            
    if not candidates:
        return None
        
    # Sort by version (highest first)
    def version_key(name):
        m = re.search(r"eddy_cuda(\d+\.?\d*)", name)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return 0.0
        return 0.0
        
    candidates.sort(key=version_key, reverse=True)
    return candidates[0]

# Cache for capability checks
_EDDY_HAS_NTHR: Dict[str, bool] = {}

def _check_eddy_supports_nthr(eddy_bin: str) -> bool:
    """Check if eddy binary supports --nthr flag."""
    if eddy_bin in _EDDY_HAS_NTHR:
        return _EDDY_HAS_NTHR[eddy_bin]
    
    try:
        # FSL tools often return non-zero on help, so we ignore check=True
        result = subprocess.run([eddy_bin, "--help"], capture_output=True, text=True)
        # Check both stdout and stderr (FSL is inconsistent)
        output = result.stdout + result.stderr
        supports = "--nthr" in output or "Number of threads" in output
        _EDDY_HAS_NTHR[eddy_bin] = supports
        return supports
    except Exception:
        # If we can't run it (e.g. not found), assume False or let the main call fail later
        return False

def check_gpu_availability() -> bool:
    """Check if a functional NVIDIA GPU is available."""
    if shutil.which("nvidia-smi"):
        try:
            # Run nvidia-smi to check if drivers are working
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False
    return False

def eddy(
    in_file: DWIFile,
    out_file: Path,
    *,
    mask: Optional[Path] = None,
    topup_base: Optional[str] = None,
    external_field: Optional[Path] = None,
    cuda: bool = False,
    cuda_device: int = 0,
    nthreads: int = 1,
    acqp: Optional[Path] = None,
    index: Optional[Path] = None,
    json_file: Optional[Path] = None,
    extra_opts: Optional[Dict[str, Any]] = None,
    force: bool = False,
) -> DWIFile:
    """
    Run FSL eddy or eddy_cuda with optional topup output.
    """
    out_p = ensure_dir(out_file)
    out_base = out_p.with_suffix("").with_suffix("")  # remove .nii.gz or .nii
    out_bvec = out_base.with_suffix(".bvec")
    
    if out_p.exists() and out_bvec.exists() and not force:
        return DWIFile(
            entities=in_file.entities,
            img=out_p,
            json=in_file.json,
            bval=in_file.bval,
            bvec=out_bvec,
            Delta=getattr(in_file, "Delta", None),
            delta=getattr(in_file, "delta", None),
        )

    # We must ensure we have a valid input path
    # in_dwi.img is a Path object or similar
    
    if not mask:
        mask = out_p.with_name("tmp_mask.nii.gz")
        bet(in_file=in_file, out_file=mask)
        # Apply dilation to temporary mask
        if _which("fslmaths"):
             maths(mask, mask, args="-dilM -dilM -dilM -bin")

    acqp_path, index_path = (acqp, index) if acqp and index else _ensure_acqp_index(
        in_file,
        support_dir=out_p.parent / "eddy_support",
    )
    if not acqp_path or not index_path:
        raise RuntimeError("acqparams/index files are required for eddy.")

    if not in_file.bvec or not in_file.bval:
        raise RuntimeError("bvec and bval files are required for eddy.")

    # Detect appropriate binary
    if cuda:
        if not check_gpu_availability():
            logger.warning("CUDA enabled but no functional NVIDIA GPU/driver detected via nvidia-smi. Falling back to CPU version of eddy.")
            cuda = False
        else:
            eddy_bin = _find_eddy_cuda()
            if not eddy_bin:
                 logger.warning("CUDA enabled but no 'eddy_cuda*' executable found. Falling back to CPU version of eddy.")
                 cuda = False

    if not cuda:
        # Prefer OpenMP version if available
        eddy_bin = "eddy_openmp" if _which("eddy_openmp") else "eddy"
        # If neither found, checking just "eddy" will likely fail in run_cmd but that's expected.

    # Only set CUDA_VISIBLE_DEVICES if explicitly requested via argument AND not running in pre-isolated env
    # In parallel mode, os.environ["CUDA_VISIBLE_DEVICES"] is already set to the specific GPU.
    # Prepending 'env CUDA_VISIBLE_DEVICES=...' overrides the isolation (resetting to physical GPU X).

    env_parts = ["env"]
    if cuda:
        # If env var is NOT set, we set it. If it IS set, we assume isolation is managed externally.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
             env_parts.append(f"CUDA_VISIBLE_DEVICES={cuda_device}")
    else:
        env_parts.append(f"OMP_NUM_THREADS={nthreads}")

    cmd_parts = [
        *env_parts,
        eddy_bin,
        f"--imain={in_file.img}",
        f"--mask={mask}",
        f"--index={index_path}",
        f"--acqp={acqp_path}",
        f"--bvecs={in_file.bvec}",
        f"--bvals={in_file.bval}",
        f"--out={out_base}",
    ]

    if not cuda:
        # Only pass --nthr if supported
        if _check_eddy_supports_nthr(eddy_bin):
            cmd_parts.append(f"--nthr={nthreads}")
        # Else we rely on OMP_NUM_THREADS set above
    if topup_base:
        cmd_parts.append(f"--topup={topup_base}")
    if external_field:
        cmd_parts.append(f"--field={external_field}")

    # Configurable JSON output
    # Only pass --json if explicitly requested in extra_opts (via 'json': True)
    enable_json = False
    if extra_opts is not None and 'json' in extra_opts:
        enable_json = bool(extra_opts['json'])
        # Start fresh or copy to avoid side effects if reused? extra_opts comes from kwargs or config dict
        # We delete logic outside or just pop here.
        # But wait, extra_opts is passed by reference potentially?
        # Safe to delete since we consumed it.
        del extra_opts['json']

    # Pass JSON sidecar if available AND requested
    json_path = json_file or in_file.json
    if enable_json and json_path and json_path.exists():
         cmd_parts.append(f"--json={json_path}")

    cmd_parts.extend(_format_extra_opts(extra_opts))

    try:
        run_cmd(" ".join(cmd_parts), label="eddy")
    except RuntimeError as exc:
        gpu_thread_limited = "compiled for GPU can only use 1 CPU thread" in str(exc)
        if not gpu_thread_limited or int(nthreads) == 1:
            raise

        retry_parts: list[str] = []
        has_omp = False
        has_nthr = False
        for part in cmd_parts:
            if part.startswith("OMP_NUM_THREADS="):
                retry_parts.append("OMP_NUM_THREADS=1")
                has_omp = True
            elif part.startswith("--nthr="):
                retry_parts.append("--nthr=1")
                has_nthr = True
            else:
                retry_parts.append(part)

        if not has_omp:
            if retry_parts and retry_parts[0] == "env":
                retry_parts.insert(1, "OMP_NUM_THREADS=1")
            else:
                retry_parts = ["env", "OMP_NUM_THREADS=1", *retry_parts]
        if not has_nthr and _check_eddy_supports_nthr(eddy_bin):
            retry_parts.append("--nthr=1")

        logger.warning(
            "Eddy binary is GPU-thread-limited; retrying with OMP_NUM_THREADS=1 and --nthr=1."
        )
        run_cmd(" ".join(retry_parts), label="eddy")

    rotated_bvec = out_base.with_suffix(".eddy_rotated_bvecs")
    if rotated_bvec.exists():
        shutil.move(rotated_bvec, out_bvec)
    if in_file.bvec:
        _validate_or_reuse_bvec(out_bvec, in_file.bvec, in_file.bval, "eddy")

    return DWIFile(
        entities=in_file.entities,
        img=out_p,
        json=in_file.json,
        bval=in_file.bval,
        bvec=out_bvec,
        Delta=getattr(in_file, "Delta", None),
        delta=getattr(in_file, "delta", None),
    )


def convert_xfm(in_file: Path, out_file: Path, inverse: bool = False, concat_mat: Path = None):
    """
    Wrapper for FSL convert_xfm.
    """
    out_p = ensure_dir(out_file)
    
    if out_p.exists():
        return out_p

    cmd_parts = ["convert_xfm", f"-omat {out_p}"]
    if inverse:
        cmd_parts.append(f"-inverse {in_file}")
    elif concat_mat:
        cmd_parts.append(f"-concat {concat_mat} {in_file}")
    else:
        cmd_parts.append(str(in_file))
        
    run_cmd(" ".join(cmd_parts), label="convert_xfm")
    return out_p

def reorient2std(in_file: ImageLike | Path, out_file: Path):
    """
    Wrapper for FSL fslreorient2std.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    if out_p.exists():
        return out_p
        
    cmd = f"fslreorient2std {in_p} {out_p}"
    run_cmd(cmd, label="fslreorient2std")
    return out_p

def maths(in_file: ImageLike | Path, out_file: Path, args: str, nthreads: int = 1):
    """
    Wrapper for fslmaths.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    cmd = f"fslmaths {in_p} {args} {out_p}"
    run_cmd(cmd, label="fslmaths")
    return out_p

def eddy_quad(
    eddy_base: Path,
    idx: Path,
    par: Path,
    mask: Path,
    bvals: Path,
    bvecs: Path,
    output_dir: Path,
    field: Optional[Path] = None,
    slspec: Optional[Path] = None,
    verbose: bool = False
):
    """
    Wrapper for FSL eddy_quad (QC for eddy).
    
    Args:
        eddy_base: Basename of eddy output (including path, without extensions)
        idx: Path to index file
        par: Path to acqparams file
        mask: Path to brain mask
        bvals: Path to bvals
        bvecs: Path to bvecs
        output_dir: Output directory
    """
    out_dir = Path(output_dir) # ensure_dir creates it, which eddy_quad dislikes
    
    # eddy_quad <eddyBase> -idx <eddyIndex> -par <eddyAcqParams> -m <nodifMask> -b <bvals> -g <bvecs> -o <outputDir>
    
    cmd_parts = [
        "eddy_quad",
        str(eddy_base),
        "-idx", str(idx),
        "-par", str(par),
        "-m", str(mask),
        "-b", str(bvals),
        "-g", str(bvecs),
        "-o", str(out_dir)
    ]
    
    if field:
        cmd_parts.extend(["-f", str(field)])
        
    if slspec:
         cmd_parts.extend(["-s", str(slspec)])
         
    if verbose:
        cmd_parts.append("-v")
        
    run_cmd(" ".join(cmd_parts), label="eddy_quad")
    return out_dir

def rotate_bvecs(bvecs: Path, mat: Path, out_bvecs: Path):
    """
    Rotate b-vectors using a rigid/affine matrix.
    Uses 'fdt_rotate_bvecs' if available, or manual numpy calculation.
    """
    # Robust Implementation using Numpy:
    # 1. Load bvecs (3xN)
    # 2. Load transform (FLIRT matrix 4x4)
    # 3. Apply rotation (upper 3x3) to bvecs: v' = R * v
    # 4. Save
    
    try:
        # Load bvecs
        bv = np.loadtxt(bvecs)
        # Handle shape (3, N) or (N, 3). FSL is usually (3, N)
        transposed = False
        if bv.shape[0] != 3 and bv.shape[1] == 3:
            bv = bv.T
            transposed = True
            
        # Load matrix
        aff = np.loadtxt(mat) # 4x4
        rot = aff[:3, :3]
        
        # Apply rotation
        new_bv = np.dot(rot, bv)
        
        # Renormalize
        norms = np.linalg.norm(new_bv, axis=0)
        # Avoid divide by zero
        norms[norms == 0] = 1
        new_bv = new_bv / norms
        
        if transposed:
            new_bv = new_bv.T
            
        # Use explicit delimiter and padding for readability
        np.savetxt(out_bvecs, new_bv, fmt='% .8f', delimiter=' ')
        
    except Exception as e:
        # Fallback to fsl command if numpy fails?
        # Ideally numpy is safer than relying on shell script
        raise RuntimeError(f"Failed to rotate bvecs: {e}")


def fit_dti(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    mask_file: Path,
    bval_file: Optional[Path] = None,
    bvec_file: Optional[Path] = None,
    save_tensor: bool = False,
    grad_nonlin: Optional[Path] = None,
    metrics: Optional[list[str]] = None,
) -> Dict[str, Path]:
    """
    Fit DTI using FSL dtifit.
    """
    import subprocess
    from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path
    
    out_dir = ensure_dir(out_dir)
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
    
    # prefix
    # Use build_bids_name with implicit suffix? No, dtifit adds suffixes.
    # We want base to be ..._model-DT
    # build_bids_name requires suffix? 
    # If we pass suffix, we get ..._model-DT_suffix.nii.gz
    # We want ..._model-DT as the prefix string.
    # We can assume build_bids_name works if we strip the extension.
    # But build_bids_name raises if no suffix.
    # So pass a dummy suffix like 'dti' and strip it?
    # Or construct manually? 
    # Better: use suffix='dwi' (which is probably present) but we want the prefix for dtifit which appends _FA etc.
    # If prefix is "sub-01_model-DT", dtifit makes "sub-01_model-DT_FA.nii.gz".
    # This matches BIDS if suffix is FA.
    
    # So we need to construct "sub-01_model-DT".
    # build_bids_name(..., suffix='param') -> sub-01_model-DT_param.nii.gz
    # strip _param.nii.gz?
    
    temp_name = build_bids_name({**ent_base, 'suffix': 'placeholder'})
    prefix_name = temp_name.replace('_placeholder.nii.gz', '').replace('_placeholder.nii', '')
    prefix_path = out_dir / prefix_name
    
    cmd = [
        'dtifit',
        f'--data={in_path}',
        f'--out={prefix_path}',
        f'--mask={mask_file}',
        f'--bvecs={bvec_file}',
        f'--bvals={bval_file}'
    ]
    
    if save_tensor:
        cmd.append('--save_tensor')
        
    if grad_nonlin:
        cmd.append(f'--gradnonlin={grad_nonlin}')
        
    run_cmd(" ".join(cmd), label="dtifit")
    
    metrics_norm = {metric.strip().lower() for metric in (metrics or ['fa', 'md', 'ad', 'rd', 'color_fa', 'tensor'])}

    def _write_sidecar(path: Path, output_metric: str, extras: Optional[dict] = None) -> None:
        payload = {
            "ModelName": "Diffusion Tensor Imaging",
            "FittingSoftware": "FSL dtifit",
            "InputData": in_path.name,
            "OutputMetric": output_metric,
        }
        if extras:
            payload.update(extras)
        with open(str(path).replace('.nii.gz', '.json'), 'w') as f:
            json.dump(payload, f, indent=4)

    prefix_str = str(prefix_path)
    output_files: Dict[str, Path] = {}
    produced: Dict[str, Path] = {
        'fa': Path(f"{prefix_str}_FA.nii.gz"),
        'md': Path(f"{prefix_str}_MD.nii.gz"),
        'l1': Path(f"{prefix_str}_L1.nii.gz"),
        'l2': Path(f"{prefix_str}_L2.nii.gz"),
        'l3': Path(f"{prefix_str}_L3.nii.gz"),
        'v1': Path(f"{prefix_str}_V1.nii.gz"),
        'v2': Path(f"{prefix_str}_V2.nii.gz"),
        'v3': Path(f"{prefix_str}_V3.nii.gz"),
        'tensor': Path(f"{prefix_str}_tensor.nii.gz"),
    }

    for metric_name, path in produced.items():
        if path.exists():
            extras = None
            if metric_name in {'v1', 'v2', 'v3'}:
                extras = {"VectorConvention": "voxel"}
            elif metric_name == 'tensor':
                extras = {"TensorConvention": "FSL", "TensorBasis": "voxel"}
            _write_sidecar(path, metric_name.upper() if len(metric_name) <= 3 else metric_name, extras=extras)

    if 'rd' in metrics_norm and produced['l2'].exists() and produced['l3'].exists():
        l2_img = nib.load(str(produced['l2']))
        l3_img = nib.load(str(produced['l3']))
        rd_path = out_dir / build_bids_name({**ent_base, 'suffix': 'RD'})
        rd_data = 0.5 * (l2_img.get_fdata(dtype=np.float32) + l3_img.get_fdata(dtype=np.float32))
        nib.save(nib.Nifti1Image(rd_data.astype(np.float32), l2_img.affine, l2_img.header), str(rd_path))
        produced['rd'] = rd_path
        _write_sidecar(rd_path, "RD")

    if 'color_fa' in metrics_norm and produced['fa'].exists() and produced['v1'].exists():
        fa_img = nib.load(str(produced['fa']))
        v1_img = nib.load(str(produced['v1']))
        fa_data = fa_img.get_fdata(dtype=np.float32)
        v1_data = v1_img.get_fdata(dtype=np.float32)
        decfa_path = out_dir / build_bids_name({**ent_base, 'suffix': 'DECFA'})
        decfa = np.abs(v1_data) * fa_data[..., np.newaxis]
        nib.save(nib.Nifti1Image(decfa.astype(np.float32), fa_img.affine, fa_img.header), str(decfa_path))
        produced['color_fa'] = decfa_path
        _write_sidecar(decfa_path, "DECFA", extras={"VectorConvention": "voxel"})

    if 'tensor_fsl' in metrics_norm and produced['tensor'].exists():
        tensor_fsl = out_dir / build_bids_name({**ent_base, 'suffix': 'tensorFSL'})
        shutil.copyfile(produced['tensor'], tensor_fsl)
        produced['tensor_fsl'] = tensor_fsl
        _write_sidecar(tensor_fsl, "tensorFSL", extras={"TensorConvention": "FSL", "TensorBasis": "voxel"})

    metric_aliases = {'ad': 'l1'}
    for metric_name in metrics_norm:
        source_name = metric_aliases.get(metric_name, metric_name)
        if source_name in produced and produced[source_name].exists():
            output_files[metric_name] = produced[source_name]

    return output_files


def resample_to_image(source_file: ImageLike | Path, reference_file: ImageLike | Path, out_file: Path, interpolator: str = "trilinear") -> Path:
    """
    Resample source_file to reference_file grid using FLIRT.
    Uses -applyxfm -usesqform to align based on header information (identity transform assumption).
    """
    src_p = extract_image_path(source_file)
    ref_p = extract_image_path(reference_file)
    out_p = ensure_dir(out_file)
    
    if out_p.exists():
        return out_p
        
    # flirt -in <src> -ref <ref> -applyxfm -usesqform -out <out> -interp <interp>
    cmd_parts = [
        "flirt",
        f"-in {src_p}",
        f"-ref {ref_p}",
        "-applyxfm",
        "-usesqform",
        f"-out {out_p}",
        f"-interp {interpolator}"
    ]
    
    run_cmd(" ".join(cmd_parts), label="flirt_resample")
    return out_p



    




def fast(
    in_files: Union[Path, ImageLike, list[Union[Path, ImageLike]]],
    out_base: Path,
    img_type: int = 1, # 1=T1, 2=T2, 3=PD
    num_classes: int = 3,
    hyper: float = 0.1,
    flags: str = ""
):
    """
    Wrapper for FSL FAST (FMRIB's Automated Segmentation Tool).
    
    Args:
        in_files: Input image(s) or list of images (multi-channel).
        out_base: Output basename (mask/pve/seg will be appended).
        img_type: Image type (1=T1, 2=T2, 3=PD).
        num_classes: Number of tissue classes (default 3).
        hyper: Hyper-parameter H (segmentation spatial smoothness) (default 0.1).
        flags: Additional flags string.
    """
    out_p = ensure_dir(out_base) # Usually out_base is a prefix path, but ensure_dir ensures PARENT exists
    
    # Check if outputs exist. FAST outputs many files.
    # We check for the main segmentation file: <out_base>_seg.nii.gz
    seg_file = Path(f"{out_base}_seg.nii.gz")
    if seg_file.exists():
        return
        
    # Prepare input args
    if not isinstance(in_files, list):
        in_files = [in_files]
        
    in_paths = [str(extract_image_path(f)) for f in in_files]
    
    cmd = ["fast"]
    cmd.append(f"-t {img_type}")
    cmd.append(f"-n {num_classes}")
    cmd.append(f"-H {hyper}")
    
    if flags:
        cmd.extend(flags.split())
        
    cmd.append(f"-o {out_base}")
    cmd.extend(in_paths)
    
    run_cmd(" ".join(cmd), label="fast")


def first(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    structures: str = "all",
    method: str = "auto",
    affine_mat: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Path]:
    """
    Wrapper for FSL FIRST (subcortical structure segmentation).
    
    Args:
        in_file: Input T1-weighted image.
        out_dir: Output directory.
        structures: Structures to segment. Options:
                   'all' - all structures
                   'L_Hipp,R_Hipp' - comma-separated list
                   See run_first_all --help for available structures.
        method: Registration method ('auto', 'fast', 'none').
        affine_mat: Pre-computed affine matrix to MNI (optional).
        verbose: Enable verbose output.
        
    Returns:
        Dictionary mapping structure names to segmentation file paths.
    """
    in_p = extract_image_path(in_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Output basename
    stem = in_p.name.replace('.nii.gz', '').replace('.nii', '')
    out_base = out_dir / stem
    
    # Check if already completed (FIRST creates *_all_fast_firstseg.nii.gz)
    first_seg = out_dir / f"{stem}_all_fast_firstseg.nii.gz"
    if first_seg.exists():
        return _parse_first_outputs(out_dir, stem)
    
    cmd_parts = ["run_first_all"]
    cmd_parts.append(f"-i {in_p}")
    cmd_parts.append(f"-o {out_base}")
    
    if structures != "all":
        cmd_parts.append(f"-s {structures}")
    
    if method != "auto":
        cmd_parts.append(f"-m {method}")
        
    if affine_mat:
        cmd_parts.append(f"-a {affine_mat}")
        
    if verbose:
        cmd_parts.append("-v")
        
    run_cmd(" ".join(cmd_parts), label="run_first_all")
    
    return _parse_first_outputs(out_dir, stem)


def _parse_first_outputs(out_dir: Path, stem: str) -> Dict[str, Path]:
    """Parse FIRST output files and return mapping of structure -> file path."""
    outputs = {}
    
    # Main combined segmentation
    combined_seg = out_dir / f"{stem}_all_fast_firstseg.nii.gz"
    if combined_seg.exists():
        outputs['combined'] = combined_seg
    
    # Individual structure segmentations (vtk meshes and masks)
    # FIRST creates: <stem>-<Structure>_first.nii.gz for individual masks
    # and <stem>-<Structure>_first.vtk for meshes
    
    structure_names = [
        'L_Accu', 'R_Accu',       # Accumbens
        'L_Amyg', 'R_Amyg',       # Amygdala
        'L_Caud', 'R_Caud',       # Caudate
        'L_Hipp', 'R_Hipp',       # Hippocampus
        'L_Pall', 'R_Pall',       # Pallidum/Globus Pallidus
        'L_Puta', 'R_Puta',       # Putamen
        'L_Thal', 'R_Thal',       # Thalamus
        'BrStem'                   # Brainstem
    ]
    
    for struct in structure_names:
        # Individual mask
        mask_file = out_dir / f"{stem}-{struct}_first.nii.gz"
        if mask_file.exists():
            outputs[struct] = mask_file
            
    return outputs


def fsl_anat(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    img_type: str = "T1",
    nobias: bool = False,
    nosubcortseg: bool = False,
    noseg: bool = False,
    noreg: bool = False,
    nocleanup: bool = False,
    strongbias: bool = False,
    weakbias: bool = False,
    noreorient: bool = False,
    nocrop: bool = False,
    clobber: bool = False,
) -> Dict[str, Path]:
    """
    Wrapper for FSL fsl_anat (comprehensive anatomical processing).
    
    Args:
        in_file: Input anatomical image.
        out_dir: Output directory (will be named <out_dir>.anat).
        img_type: Image type ('T1' or 'T2').
        nobias: Skip bias field correction.
        nosubcortseg: Skip subcortical segmentation (FIRST).
        noseg: Skip tissue segmentation (FAST).
        noreg: Skip registration to MNI.
        nocleanup: Keep intermediate files.
        strongbias: Use strongbias for FAST.
        weakbias: Use -weakbias for FAST.
        noreorient: Skip reorientation to standard.
        nocrop: Skip cropping.
        clobber: Overwrite existing output.
        
    Returns:
        Dictionary mapping output type to file path.
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_dir)
    
    # fsl_anat creates <out_dir>.anat/ directory
    anat_dir = out_p.with_suffix('.anat') if not str(out_p).endswith('.anat') else out_p
    
    # Check if already completed
    t1_brain = anat_dir / "T1_biascorr_brain.nii.gz"
    if t1_brain.exists() and not clobber:
        return _parse_fsl_anat_outputs(anat_dir)
    
    # Ensure parent exists
    anat_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cmd_parts = ["fsl_anat"]
    cmd_parts.append(f"-i {in_p}")
    cmd_parts.append(f"-o {out_p}")
    cmd_parts.append(f"-t {img_type}")
    
    if nobias: cmd_parts.append("--nobias")
    if nosubcortseg: cmd_parts.append("--nosubcortseg")
    if noseg: cmd_parts.append("--noseg")
    if noreg: cmd_parts.append("--noreg")
    if nocleanup: cmd_parts.append("--nocleanup")
    if strongbias: cmd_parts.append("--strongbias")
    if weakbias: cmd_parts.append("--weakbias")
    if noreorient: cmd_parts.append("--noreorient")
    if nocrop: cmd_parts.append("--nocrop")
    if clobber: cmd_parts.append("--clobber")
    
    run_cmd(" ".join(cmd_parts), label="fsl_anat")
    
    return _parse_fsl_anat_outputs(anat_dir)


def _parse_fsl_anat_outputs(anat_dir: Path) -> Dict[str, Path]:
    """Parse fsl_anat output directory and return mapping of output type -> file path."""
    outputs = {}
    
    # Standard outputs from fsl_anat
    output_map = {
        'T1': 'T1.nii.gz',
        'T1_orig': 'T1_orig.nii.gz',
        'T1_biascorr': 'T1_biascorr.nii.gz',
        'T1_biascorr_brain': 'T1_biascorr_brain.nii.gz',
        'T1_biascorr_brain_mask': 'T1_biascorr_brain_mask.nii.gz',
        'T1_fast_pve_0': 'T1_fast_pve_0.nii.gz',  # CSF
        'T1_fast_pve_1': 'T1_fast_pve_1.nii.gz',  # GM
        'T1_fast_pve_2': 'T1_fast_pve_2.nii.gz',  # WM
        'T1_fast_seg': 'T1_fast_seg.nii.gz',
        'T1_subcort_seg': 'T1_subcort_seg.nii.gz',  # FIRST output
        'T1_to_MNI_lin': 'T1_to_MNI_lin.nii.gz',
        'T1_to_MNI_nonlin': 'T1_to_MNI_nonlin.nii.gz',
        'T1_to_MNI_nonlin_field': 'T1_to_MNI_nonlin_field.nii.gz',
        'MNI_to_T1_nonlin_field': 'MNI_to_T1_nonlin_field.nii.gz',
    }
    
    for key, filename in output_map.items():
        filepath = anat_dir / filename
        if filepath.exists():
            outputs[key] = filepath
            
    # Also check for first_results directory
    first_dir = anat_dir / "first_results"
    if first_dir.exists():
        outputs['first_dir'] = first_dir
        
    return outputs


# ============================================================================
# Volume Extraction Utilities
# ============================================================================

def extract_fast_volumes(
    pve_files: Dict[str, Path],
    affine: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract tissue volumes from FSL FAST partial volume estimates.
    
    Args:
        pve_files: Dictionary with keys 'pve_0' (CSF), 'pve_1' (GM), 'pve_2' (WM)
                   pointing to the PVE files.
        affine: Affine matrix for voxel-to-mm conversion. If None, will read from file.
        
    Returns:
        Dictionary with tissue volumes in mm³:
        - CSF_Volume_mm3
        - GM_Volume_mm3
        - WM_Volume_mm3
        - TIV_mm3 (Total Intracranial Volume = CSF + GM + WM)
    """
    volumes = {}
    tissue_map = {'pve_0': 'CSF', 'pve_1': 'GM', 'pve_2': 'WM'}
    total = 0
    
    for pve_key, tissue_name in tissue_map.items():
        if pve_key not in pve_files or not pve_files[pve_key].exists():
            continue
            
        img = nib.load(str(pve_files[pve_key]))
        data = img.get_fdata()
        
        # Get voxel volume in mm³
        voxel_dims = img.header.get_zooms()[:3]
        voxel_vol = np.prod(voxel_dims)
        
        # Sum PVE values (partial volumes) * voxel volume
        vol_mm3 = float(np.sum(data) * voxel_vol)
        volumes[f'{tissue_name}_Volume_mm3'] = vol_mm3
        total += vol_mm3
        
    volumes['TIV_mm3'] = total
    
    return volumes


def extract_first_volumes(
    first_seg: Path,
    structure_labels: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Extract subcortical structure volumes from FSL FIRST segmentation.
    
    Args:
        first_seg: Path to FIRST combined segmentation (*_all_fast_firstseg.nii.gz).
        structure_labels: Optional override of structure name -> label mapping.
        
    Returns:
        Dictionary with structure volumes in mm³.
    """
    if not first_seg.exists():
        return {}
        
    # Default FSL FIRST label mapping
    # These labels are from the FSL FIRST documentation
    default_labels = {
        'Left_Thalamus': 10,
        'Left_Caudate': 11,
        'Left_Putamen': 12,
        'Left_Pallidum': 13,
        'BrainStem': 16,
        'Left_Hippocampus': 17,
        'Left_Amygdala': 18,
        'Left_Accumbens': 26,
        'Right_Thalamus': 49,
        'Right_Caudate': 50,
        'Right_Putamen': 51,
        'Right_Pallidum': 52,
        'Right_Hippocampus': 53,
        'Right_Amygdala': 54,
        'Right_Accumbens': 58,
    }
    
    labels = structure_labels or default_labels
    
    img = nib.load(str(first_seg))
    data = img.get_fdata()
    
    # Get voxel volume in mm³
    voxel_dims = img.header.get_zooms()[:3]
    voxel_vol = np.prod(voxel_dims)
    
    volumes = {}
    for struct_name, label_val in labels.items():
        mask = (data == label_val)
        vol_mm3 = float(np.sum(mask) * voxel_vol)
        volumes[f'{struct_name}_Volume_mm3'] = vol_mm3
        
    return volumes


def extract_freesurfer_volumes(
    subjects_dir: Path,
    subject_id: str,
) -> Dict[str, float]:
    """
    Extract volumes from FreeSurfer aseg.stats file.
    
    Args:
        subjects_dir: Path to FreeSurfer SUBJECTS_DIR.
        subject_id: Subject ID.
        
    Returns:
        Dictionary with structure volumes in mm³.
    """
    stats_file = subjects_dir / subject_id / "stats" / "aseg.stats"
    if not stats_file.exists():
        logger.warning(f"aseg.stats not found: {stats_file}")
        return {}
        
    volumes = {}
    
    try:
        with open(stats_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments
                if line.startswith('#'):
                    # Parse header measures like ICV
                    if 'Measure' in line and 'IntraCranialVol' in line:
                        parts = line.split(',')
                        if len(parts) >= 4:
                            try:
                                volumes['EstimatedTIV_mm3'] = float(parts[3].strip())
                            except ValueError:
                                pass
                    continue
                    
                if not line:
                    continue
                    
                # Parse data lines
                # Format: Index SegId NVoxels Volume_mm3 StructName ...
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        struct_name = parts[4]
                        vol_mm3 = float(parts[3])
                        volumes[f'{struct_name}_Volume_mm3'] = vol_mm3
                    except (ValueError, IndexError):
                        continue
                        
    except Exception as e:
        logger.warning(f"Failed to parse aseg.stats: {e}")
        
    return volumes


def save_volumes_to_file(
    volumes: Dict[str, float],
    out_path: Path,
    subject_id: str = "",
    session: str = "",
    format: str = "tsv",
) -> Path:
    """
    Save volume dictionary to CSV/TSV/XLSX file.
    
    Args:
        volumes: Dictionary of structure_name -> volume in mm³.
        out_path: Output file path.
        subject_id: Subject ID for the output.
        session: Session ID for the output.
        format: Output format ('tsv', 'csv', 'xlsx').
        
    Returns:
        Path to saved file.
    """
    import pandas as pd
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to tidy format
    rows = []
    for name, value in volumes.items():
        # Parse structure name and metric
        if name.endswith('_mm3'):
            struct = name.replace('_Volume_mm3', '').replace('_mm3', '')
            metric = 'Volume_mm3'
        else:
            struct = name
            metric = 'Value'
            
        rows.append({
            'Subject_ID': subject_id,
            'Session': session,
            'Structure': struct,
            'Metric': metric,
            'Value': value,
        })
        
    df = pd.DataFrame(rows)
    
    if format == 'tsv':
        df.to_csv(out_path, sep='\t', index=False)
    elif format == 'csv':
        df.to_csv(out_path, index=False)
    elif format == 'xlsx':
        df.to_excel(out_path, index=False, engine='openpyxl')
    else:
        raise ValueError(f"Unknown format: {format}")
        
    return out_path
