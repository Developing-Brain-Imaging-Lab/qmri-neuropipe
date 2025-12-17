from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import nibabel as nib
import numpy as np

from ..core.run import run_cmd
from ..core.types import DWIFile, ImageLike
from ..io.dmri.bids import build_acqp_index
from ..core.utils import ensure_path, ensure_dir, extract_image_path

def _format_extra_opts(extra_opts: Optional[Dict[str, Any]]) -> list[str]:
    opts: list[str] = []
    if not extra_opts:
        return opts
    for key, val in extra_opts.items():
        if isinstance(val, bool):
            if val:
                opts.append(f"--{key}")
        else:
            opts.append(f"--{key}={val}")
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

def flirt(in_file: ImageLike | Path, ref_file: ImageLike | Path, out_file: Path, omat: Path = None, dof: int = 6, cost: str = "normmi", extra_args: str = ""):
    """
    Wrapper for FSL FLIRT.
    """
    in_p = extract_image_path(in_file)
    ref_p = extract_image_path(ref_file)
    out_p = ensure_dir(out_file)
    
    omat_cmd = ""
    if omat:
        omat_p = Path(omat)
        if out_p.exists() and omat_p.exists():
            return out_p, omat_p
        omat_cmd = f"-omat {omat_p}"
    elif out_p.exists():
         return out_p, None
        
    cmd = f"flirt -in {in_p} -ref {ref_p} -out {out_p} {omat_cmd} -dof {dof} -cost {cost} {extra_args}"
    
    if out_p.exists() and (not omat or Path(omat).exists()):
        return out_p, omat if omat else None

    run_cmd(cmd, label="flirt")
    return out_p, omat if omat else None


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
) -> DWIFile:
    """
    Run FSL eddy or eddy_cuda with optional topup output.
    """
    out_p = ensure_dir(out_file)
    out_base = out_p.with_suffix("").with_suffix("")  # remove .nii.gz or .nii
    out_bvec = out_base.with_suffix(".bvec")
    
    if out_p.exists() and out_bvec.exists():
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

    eddy_bin = "eddy_cuda" if cuda else "eddy"
    env_parts = ["env", f"CUDA_VISIBLE_DEVICES={cuda_device}"] if cuda else ["env", f"OMP_NUM_THREADS={nthreads}"]

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
        cmd_parts.append(f"--nthr={nthreads}")
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
    # Prefer FSL tool if widely available (part of FSL >= 6.0?)
    # Usually fdt_rotate_bvecs needs 'bvecs' 'rotated_bvecs' 'mat_file'
    # But FSL's tool is often not in path or named differently depending on version.
    
    # Robust Implementation using Numpy:
    # 1. Load bvecs (3xN)
    # 2. Load transform (FLIRT matrix 4x4)
    # 3. Apply rotation (upper 3x3) to bvecs
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
        
        # Apply rotation: v' = R * v
        # Ensure normalization?
        
        # But wait, FLIRT matrix applies to coordinates.
        # For gradients, we usually need the INVERSE transpose of the affine?
        # For rotation matrix R, inv(R).T == R. So R is fine.
        # However, if there's shear/scale... FLIRT matrices are usually rigid/affine.
        
        # NOTE: bvecs should be unit vectors.
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
