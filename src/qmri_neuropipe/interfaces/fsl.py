from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union
import json

import nibabel as nib
import numpy as np

from ..core.run import run_cmd
from ..core.types import DWIFile, ImageLike
from ..io.dmri.bids import build_acqp_index
from ..core.utils import ensure_path, ensure_dir, extract_image_path

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

def _ensure_acqp_index(in_dwi: DWIFile) -> tuple[Path | None, Path | None]:
    """
    Create acqp/index files if possible from BIDS metadata.
    """
    return build_acqp_index(in_dwi.json, in_dwi.img)


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
    run_cmd(cmd, label="flirt")
    return out_p, Path(omat) if omat else None


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
        
    cmd = f"fslsplit {in_p} {out_base} -{dimension}"
    run_cmd(cmd, label="fslsplit")
    
    # Identify outputs
    # fslsplit produces vol0000.nii.gz, vol0001.nii.gz ...
    # We need to glob them to return sorted list
    # The output basename is treated as a prefix.
    # If out_base is "vol", outputs are "vol0000.nii.gz"
    
    # Use parent to glob
    parent = out_base.parent
    prefix = out_base.name
    
    # Glob pattern: prefix + 4 digits + .nii.gz (or .nii)
    # FSL usually output .nii.gz if FSLOUTPUTTYPE is NIFTI_GZ
    # We can just glob prefix* and sort
    files = sorted(list(parent.glob(f"{prefix}*")))
    return files


def applywarp(in_file: ImageLike | Path, ref_file: ImageLike | Path, out_file: Path, warp: Path = None, premat: Path = None, interp: str = "spline", extra_args: str = ""):
    """
    Wrapper for FSL applywarp.
    """
    in_p = extract_image_path(in_file)
    ref_p = extract_image_path(ref_file)
    out_p = ensure_dir(out_file)
    
    if out_p.exists():
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
        acqp_path, index_path = (acqp, index) if acqp and index else _ensure_acqp_index(dwi)
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
        vols = data.get_fdata()[..., idx]
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
            b0_vols.append(vols[..., i])
            acqp_idx = index_entries[vol_i] - 1 if vol_i < len(index_entries) else 0
            acqp_lines.append(acqp_entries[acqp_idx] if acqp_entries else "")

    if not b0_vols or not acqp_lines:
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

    return DWIFile(
        entities=in_file.entities,
        img=out_p,
        json=in_file.json,
        bval=in_file.bval,
        bvec=out_bvec,
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
        )

    # We must ensure we have a valid input path
    # in_dwi.img is a Path object or similar
    
    if not mask:
        mask = out_p.with_name("tmp_mask.nii.gz")
        bet(in_file=in_file, out_file=mask)

    acqp_path, index_path = (acqp, index) if acqp and index else _ensure_acqp_index(in_file)
    if not acqp_path or not index_path:
        raise RuntimeError("acqparams/index files are required for eddy.")

    if not in_file.bvec or not in_file.bval:
        raise RuntimeError("bvec and bval files are required for eddy.")

    # Detect appropriate binary
    if cuda:
        eddy_bin = _find_eddy_cuda()
        if not eddy_bin:
             raise RuntimeError("CUDA enabled but no 'eddy_cuda*' executable found in PATH or FSLDIR/bin.")
    else:
        # Prefer OpenMP version if available
        eddy_bin = "eddy_openmp" if _which("eddy_openmp") else "eddy"
        # If neither found, checking just "eddy" will likely fail in run_cmd but that's expected.

    # Fix: Ensure shutil is imported if used later (it is used at line 335)
    import shutil
    
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

    cmd_parts.extend(_format_extra_opts(extra_opts))

    run_cmd(" ".join(cmd_parts), label="eddy")

    rotated_bvec = out_base.with_suffix(".eddy_rotated_bvecs")
    if rotated_bvec.exists():
        shutil.move(rotated_bvec, out_bvec)

    return DWIFile(
        entities=in_file.entities,
        img=out_p,
        json=in_file.json,
        bval=in_file.bval,
        bvec=out_bvec,
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
    grad_nonlin: Optional[Path] = None
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
    
    # Map outputs
    prefix_str = str(prefix_path)
    output_files = {}
    output_files['fa'] = Path(f"{prefix_str}_FA.nii.gz")
    output_files['md'] = Path(f"{prefix_str}_MD.nii.gz")
    output_files['ad'] = Path(f"{prefix_str}_L1.nii.gz") # L1 is AD
    
    # Note: RD is not directly output by dtifit (L2, L3 are separate)
    
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


def flirt(
    in_file: Union[str, Path, ImageLike],
    ref_file: Union[str, Path, ImageLike],
    out_file: Union[str, Path],
    omat: Optional[Union[str, Path]] = None,
    dof: int = 12,
    cost: str = "corratio",
    bins: int = 256,
    searchcost: str = "corratio",
    usesqform: bool = True,
    interp: str = "trilinear",
    extra_args: str = "",
    **kwargs
):
    """
    Run FSL FLIRT.
    """
    ensure_dir(Path(out_file).parent)
    
    cmd = [
        "flirt",
        f"-in {in_file}",
        f"-ref {ref_file}",
        f"-out {out_file}",
        f"-dof {dof}",
        f"-cost {cost}", 
        f"-searchcost {searchcost}",
        f"-bins {bins}",
        f"-interp {interp}"
    ]
    
    if omat:
        cmd.append(f"-omat {omat}")
        
    if usesqform:
        cmd.append("-usesqform")

    if extra_args:
        cmd.append(extra_args)

    # Handle kwargs
    for k, v in kwargs.items():
        if len(k) == 1: cmd.append(f"-{k} {v}")
        else: cmd.append(f"--{k}={v}") 

    run_cmd(" ".join(cmd), label=f"flirt ({dof} dof)")
    return out_file



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
