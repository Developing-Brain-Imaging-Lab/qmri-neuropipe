from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from ...core import ProcessingError
from ...core.types import ImageLike
from ...core.utils import extract_image_path
from ...interfaces import ants
from ...interfaces.mrtrix import dwiextract, mrmath


LOGGER = logging.getLogger(__name__)
R0_MM = 250.0
RGE_MM = 10.0
LPS2RAS_3 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

try:
    import numba as nb
except Exception:  # pragma: no cover - optional dependency gate
    nb = None


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _get_gnl_metadata(image: ImageLike) -> dict:
    sidecar = getattr(image, "json", None)
    img_path = getattr(image, "img", None)
    LOGGER.info(f"Native GE GNL metadata source image: {img_path}")
    LOGGER.info(f"Native GE GNL metadata source sidecar: {sidecar}")
    if not sidecar or not Path(sidecar).exists():
        raise ProcessingError("Native GE GNL requires a JSON sidecar with import-time metadata.")
    payload = _load_json(Path(sidecar))
    gnl = payload.get("GradientNonlinearityCorrection")
    if not isinstance(gnl, dict):
        LOGGER.error(
            "GradientNonlinearityCorrection block missing in sidecar: %s. Available top-level keys: %s",
            sidecar,
            sorted(payload.keys()),
        )
        raise ProcessingError("Missing GradientNonlinearityCorrection block in sidecar.")
    offset = gnl.get("IsocenterOffsetScannerRASmm")
    if not isinstance(offset, list) or len(offset) != 3:
        raise ProcessingError("Missing IsocenterOffsetScannerRASmm in sidecar.")
    return gnl


def _read_ge_coefficients(coeffs_path: Path, r0: float = R0_MM, rge: float = RGE_MM):
    xynorm = np.ones(10, float)
    znorm = np.ones(10, float)
    xynorm[1] = 1.0 / np.sqrt(3.0)
    xynorm[2] = np.sqrt(8.0 / 3.0)
    xynorm[3] = np.sqrt(8.0 / 5.0)
    xynorm[4] = np.sqrt(64.0 / 15.0)
    znorm[1] = 2.0
    znorm[2] = 2.0
    znorm[3] = 8.0
    znorm[4] = 8.0
    for i in range(10):
        temp = (r0 / rge) ** i
        xynorm[i] *= temp
        znorm[i] *= temp

    xkeys = []
    ykeys = []
    zkeys = []
    xcoef = []
    ycoef = []
    zcoef = []
    rx = re.compile(r"^(SCALE[XYZ])\s*([0-9]+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$")
    with coeffs_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = rx.match(line)
            if not match:
                continue
            axis, tempind, temp = match.group(1), int(match.group(2)), float(match.group(3))
            if temp == 0.0:
                if axis == "SCALEX" and len(xcoef) == 0:
                    temp += 1.0
                elif axis == "SCALEY" and len(ycoef) == 0:
                    temp += 1.0
                elif axis == "SCALEZ" and len(zcoef) == 0:
                    temp += 1.0
                else:
                    continue
            idx = tempind - 1
            if idx < 0 or idx >= 10:
                continue
            if axis == "SCALEX":
                xcoef.append(temp * xynorm[idx])
                xkeys += [tempind, 1]
            elif axis == "SCALEY":
                ycoef.append(temp * xynorm[idx])
                ykeys += [tempind, -1]
            else:
                zcoef.append(temp * znorm[idx])
                zkeys += [tempind, 0]

    return (
        np.array(xkeys, np.int32),
        np.array(xcoef, np.float64),
        np.array(ykeys, np.int32),
        np.array(ycoef, np.float64),
        np.array(zkeys, np.int32),
        np.array(zcoef, np.float64),
    )


def _make_factorials(nmax: int) -> np.ndarray:
    fact = np.ones(nmax + 1, float)
    for i in range(2, nmax + 1):
        fact[i] = fact[i - 1] * i
    return fact


def _prep_itk_lps_geometry(img: nib.Nifti1Image):
    aff_ras = img.affine.astype(np.float64)
    ras2lps_4 = np.diag([-1.0, -1.0, 1.0, 1.0])
    aff_lps = ras2lps_4 @ aff_ras

    a = aff_lps[:3, :3]
    spacing = np.sqrt((a * a).sum(axis=0))
    d_itk = a @ np.diag(1.0 / spacing)

    shape = img.shape[:3]
    indv = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    new_origv = -(d_itk @ (spacing * indv))
    aff_grad_lps = aff_lps.copy()
    aff_grad_lps[2, 3] = new_origv[2]

    return aff_lps, d_itk, spacing, aff_grad_lps


if nb is not None:
    @nb.njit(cache=True)
    def _plgndr(ll, mm, x):
        pmm = 1.0
        if mm > 0:
            somx2 = np.sqrt(1.0 - x * x)
            fact = 1.0
            for _ in range(1, mm + 1):
                pmm = pmm * (-fact * somx2)
                fact += 2.0
        if ll == mm:
            return pmm
        pmmp1 = x * (2.0 * mm + 1.0) * pmm
        if ll == mm + 1:
            return pmmp1
        pll = 0.0
        for lm in range(mm + 2, ll + 1):
            pll = (x * (2.0 * lm - 1.0) * pmmp1 - (lm + mm - 1.0) * pmm) / (lm - mm)
            pmm = pmmp1
            pmmp1 = pll
        return pll

    @nb.njit(cache=True)
    def _sphericalz(ll, mm, phi, zz, fact):
        mma = mm if mm >= 0 else -mm
        if mm == 0:
            return _plgndr(ll, 0, zz)
        f1 = fact[ll - mma]
        f2 = fact[ll + mma]
        pl = _plgndr(ll, mma, zz)
        result = np.sqrt(2.0 * f1 / f2) * pl
        sign = -1.0 if (mm & 1) else 1.0
        if mm < 0:
            return sign * result * np.sin(mma * phi)
        return sign * result * np.cos(mma * phi)

    @nb.njit(cache=True)
    def _dshdx_x(ll, mml, phi, zz, fact):
        if mml >= 0:
            mm = mml
            if mm == 0:
                if ll == 1:
                    return 0.0
                return -_sphericalz(ll - 1, 1, phi, zz, fact) * np.sqrt(ll * (ll - 1) / 2.0)
            result = np.sqrt((ll + mm - 1) * (ll + mm)) * _sphericalz(ll - 1, mm - 1, phi, zz, fact) / 2.0
            if mm == 1:
                result *= np.sqrt(2.0)
            if mm <= ll - 2:
                result -= _sphericalz(ll - 1, mm + 1, phi, zz, fact) * np.sqrt((ll - mm - 1) * (ll - mm)) / 2.0
            return result
        mm = -mml
        if mm == 1:
            result = 0.0
        else:
            result = np.sqrt((ll + mm - 1) * (ll + mm)) * _sphericalz(ll - 1, -(mm - 1), phi, zz, fact) / 2.0
        if mm <= ll - 2:
            result -= _sphericalz(ll - 1, -(mm + 1), phi, zz, fact) * np.sqrt((ll - mm - 1) * (ll - mm)) / 2.0
        return result

    @nb.njit(cache=True)
    def _dshdx_y(ll, mml, phi, zz, fact):
        if mml >= 0:
            mm = mml
            if mm == 0:
                if ll == 1:
                    return 0.0
                return -_sphericalz(ll - 1, -1, phi, zz, fact) * np.sqrt(ll * (ll - 1) / 2.0)
            if mm == 1:
                result = 0.0
            else:
                result = -np.sqrt((ll + mm - 1) * (ll + mm)) * _sphericalz(ll - 1, -(mm - 1), phi, zz, fact) / 2.0
            if mm <= ll - 2:
                result -= _sphericalz(ll - 1, -(mm + 1), phi, zz, fact) * np.sqrt((ll - mm - 1) * (ll - mm)) / 2.0
            return result
        mm = -mml
        result = np.sqrt((ll + mm - 1) * (ll + mm)) * _sphericalz(ll - 1, mm - 1, phi, zz, fact) / 2.0
        if mm == 1:
            result *= np.sqrt(2.0)
        if mm <= ll - 2:
            result += _sphericalz(ll - 1, mm + 1, phi, zz, fact) * np.sqrt((ll - mm - 1) * (ll - mm)) / 2.0
        return result

    @nb.njit(cache=True)
    def _dshdx_z(ll, mml, phi, zz, fact):
        mm = mml if mml >= 0 else -mml
        if ll == mm:
            return 0.0
        return _sphericalz(ll - 1, mml, phi, zz, fact) * np.sqrt((ll - mm) * (ll + mm))

    @nb.njit(cache=True)
    def _dshdx(grad, ll, mml, rr, phi, zz, fact):
        if grad == 0:
            base = _dshdx_x(ll, mml, phi, zz, fact)
        elif grad == 1:
            base = _dshdx_y(ll, mml, phi, zz, fact)
        else:
            base = _dshdx_z(ll, mml, phi, zz, fact)
        return base * (rr ** (ll - 1))

    @nb.njit(cache=True)
    def _pixel_bmatrix(point_mm, r0, xkeys, xcoef, ykeys, ycoef, zkeys, zcoef, fact):
        x1 = point_mm[0] / r0
        y1 = point_mm[1] / r0
        z1 = point_mm[2] / r0
        rr = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        if rr == 0.0:
            return np.eye(3)
        phi = np.arctan2(y1, x1)
        zz = z1 / rr
        axx = ayy = azz = 0.0
        axy = azy = ayx = azx = axz = ayz = 0.0
        for kk in range(xcoef.size):
            ll = int(xkeys[2 * kk])
            mm = int(xkeys[2 * kk + 1])
            axx += xcoef[kk] * _dshdx(0, ll, mm, rr, phi, zz, fact)
            ayx += xcoef[kk] * _dshdx(1, ll, mm, rr, phi, zz, fact)
            azx += xcoef[kk] * _dshdx(2, ll, mm, rr, phi, zz, fact)
        for kk in range(ycoef.size):
            ll = int(ykeys[2 * kk])
            mm = int(ykeys[2 * kk + 1])
            axy += ycoef[kk] * _dshdx(0, ll, mm, rr, phi, zz, fact)
            ayy += ycoef[kk] * _dshdx(1, ll, mm, rr, phi, zz, fact)
            azy += ycoef[kk] * _dshdx(2, ll, mm, rr, phi, zz, fact)
        for kk in range(zcoef.size):
            ll = int(zkeys[2 * kk])
            mm = int(zkeys[2 * kk + 1])
            axz += zcoef[kk] * _dshdx(0, ll, mm, rr, phi, zz, fact)
            ayz += zcoef[kk] * _dshdx(1, ll, mm, rr, phi, zz, fact)
            azz += zcoef[kk] * _dshdx(2, ll, mm, rr, phi, zz, fact)
        out = np.empty((3, 3))
        out[0, 0] = axx
        out[0, 1] = axy
        out[0, 2] = axz
        out[1, 0] = ayx
        out[1, 1] = ayy
        out[1, 2] = ayz
        out[2, 0] = azx
        out[2, 1] = azy
        out[2, 2] = azz
        return out.T

    @nb.njit(parallel=True, cache=False)
    def _compute_graddev(out, aff_eval_lps, iso_scanner_ras, d_itk, r0, xkeys, xcoef, ykeys, ycoef, zkeys, zcoef, fact, lps2ras_3):
        nx, ny, nz, _ = out.shape
        for k in nb.prange(nz):
            for j in range(ny):
                for i in range(nx):
                    pt_lps0 = aff_eval_lps[0, 0] * i + aff_eval_lps[0, 1] * j + aff_eval_lps[0, 2] * k + aff_eval_lps[0, 3]
                    pt_lps1 = aff_eval_lps[1, 0] * i + aff_eval_lps[1, 1] * j + aff_eval_lps[1, 2] * k + aff_eval_lps[1, 3]
                    pt_lps2 = aff_eval_lps[2, 0] * i + aff_eval_lps[2, 1] * j + aff_eval_lps[2, 2] * k + aff_eval_lps[2, 3]
                    pt_ras0 = lps2ras_3[0, 0] * pt_lps0 + lps2ras_3[0, 1] * pt_lps1 + lps2ras_3[0, 2] * pt_lps2 - iso_scanner_ras[0]
                    pt_ras1 = lps2ras_3[1, 0] * pt_lps0 + lps2ras_3[1, 1] * pt_lps1 + lps2ras_3[1, 2] * pt_lps2 - iso_scanner_ras[1]
                    pt_ras2 = lps2ras_3[2, 0] * pt_lps0 + lps2ras_3[2, 1] * pt_lps1 + lps2ras_3[2, 2] * pt_lps2 - iso_scanner_ras[2]
                    b = _pixel_bmatrix(np.array([pt_ras0, pt_ras1, pt_ras2], dtype=np.float64), r0, xkeys, xcoef, ykeys, ycoef, zkeys, zcoef, fact)
                    # Ensure contiguous layout for Numba matmul optimization (avoid A/C-order warnings).
                    b = b.copy()
                    b = lps2ras_3 @ b @ lps2ras_3
                    b = b.T
                    b = d_itk.T @ b @ d_itk
                    b = b.T
                    out[i, j, k, 0] = b[0, 0]
                    out[i, j, k, 1] = b[0, 1]
                    out[i, j, k, 2] = b[0, 2]
                    out[i, j, k, 3] = b[1, 0]
                    out[i, j, k, 4] = b[1, 1]
                    out[i, j, k, 5] = b[1, 2]
                    out[i, j, k, 6] = b[2, 0]
                    out[i, j, k, 7] = b[2, 1]
                    out[i, j, k, 8] = b[2, 2]


def _extract_mean_b0(image: ImageLike, out_path: Path, force: bool = False):
    if out_path.exists() and not force:
        return out_path
    img_path = extract_image_path(image)
    if len(nib.load(str(img_path)).shape) == 3:
        nib.save(nib.load(str(img_path)), str(out_path))
        return out_path
    tmp_b0 = out_path.parent / f"{out_path.stem}_b0s.mif"
    dwiextract(img_path, tmp_b0, bzero=True, in_bvec=getattr(image, "bvec", None), in_bval=getattr(image, "bval", None), force=True)
    mrmath(tmp_b0, "mean", out_path, axis=3, force=True)
    tmp_b0.unlink(missing_ok=True)
    return out_path


def _same_grid(a: Path, b: Path) -> bool:
    img_a = nib.load(str(a))
    img_b = nib.load(str(b))
    return img_a.shape[:3] == img_b.shape[:3] and np.allclose(img_a.affine, img_b.affine, atol=1e-4)


def _rotation_from_fsl_affine(affine_path: Path, moving_reference: Path, fixed_reference: Path) -> np.ndarray:
    mat = np.loadtxt(str(affine_path))
    if mat.shape != (4, 4):
        raise ProcessingError(f"Expected 4x4 FSL affine at {affine_path}")
    moving_aff = nib.load(str(moving_reference)).affine
    fixed_aff = nib.load(str(fixed_reference)).affine
    world = fixed_aff @ mat @ np.linalg.inv(moving_aff)
    linear = world[:3, :3]
    u, _, vt = np.linalg.svd(linear)
    # Keep the signed rotation that matches the provided transform convention.
    # Forcing det>0 here can silently invert a reflected axis and appears as
    # spurious flips in downstream tensor reorientation.
    rot = u @ vt
    if np.isclose(np.linalg.det(rot), 0.0, atol=1e-8):
        raise ProcessingError(f"Degenerate FSL rotation extracted from {affine_path}")
    return rot


def _reorient_tensor_components(tensor_path: Path, rotation: np.ndarray) -> None:
    img = nib.load(str(tensor_path))
    data = img.get_fdata(dtype=np.float32)
    if data.shape[-1] != 9:
        raise ProcessingError(f"Expected 9-component tensor image, got shape {data.shape}")
    mats = data.reshape((-1, 3, 3))
    rotated = np.empty_like(mats)
    for idx in range(mats.shape[0]):
        rotated[idx] = rotation @ mats[idx] @ rotation.T
    out = rotated.reshape(data.shape)
    nib.save(nib.Nifti1Image(out.astype(np.float32), img.affine, img.header), str(tensor_path))


def create_native_ge_gnl_map(
    input_image: ImageLike,
    output_path: Path,
    grad_coeffs: Path,
    native_reference: Optional[ImageLike] = None,
    spatial_transform: Optional[dict] = None,
    nthreads: int = 1,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    logger = logger or LOGGER
    if nb is None:
        raise ProcessingError("Native GE GNL backend requires numba.")

    native_reference = native_reference or input_image
    gnl_meta = _get_gnl_metadata(native_reference)
    iso_scanner_ras = np.array(gnl_meta["IsocenterOffsetScannerRASmm"], dtype=np.float64)

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    native_b0 = output_dir / "native_gnl_b0_mean.nii.gz"
    final_b0 = output_dir / "final_gnl_b0_mean.nii.gz"
    native_tensor = output_dir / ".gnl_native_tensor_tmp.nii.gz"

    _extract_mean_b0(native_reference, native_b0, force=force)
    _extract_mean_b0(input_image, final_b0, force=force)

    if not native_tensor.exists() or force:
        logger.info("Calculating native GE gradient nonlinearity tensor map")
        img = nib.load(str(native_b0))
        _, d_itk, _, aff_grad_lps = _prep_itk_lps_geometry(img)
        xkeys, xcoef, ykeys, ycoef, zkeys, zcoef = _read_ge_coefficients(grad_coeffs)
        max_ll = int(max(xkeys[0::2].max(initial=1), ykeys[0::2].max(initial=1), zkeys[0::2].max(initial=1)))
        fact = _make_factorials(max(64, max_ll + 16))
        out = np.zeros((*img.shape[:3], 9), np.float32)
        _compute_graddev(
            out,
            aff_grad_lps.astype(np.float64),
            iso_scanner_ras,
            d_itk.astype(np.float64),
            R0_MM,
            xkeys,
            xcoef,
            ykeys,
            ycoef,
            zkeys,
            zcoef,
            fact,
            LPS2RAS_3,
        )
        nib.save(nib.Nifti1Image(out, img.affine, img.header), str(native_tensor))

    if _same_grid(native_b0, final_b0):
        if native_tensor != output_path:
            nib.save(nib.load(str(native_tensor)), str(output_path))
        native_tensor.unlink(missing_ok=True)
        return output_path

    logger.info("Rigidly mapping native GE GNL tensor into processed space")
    try:
        reg_prefix = output_dir / "native_to_final_gnl_"
        _, transforms = ants.registration(
            fixed_file=final_b0,
            moving_file=native_b0,
            out_prefix=reg_prefix,
            transform_type="Rigid",
            interpolator="linear",
            nthreads=nthreads,
        )
        ants.apply_transforms(
            fixed_file=final_b0,
            moving_file=native_tensor,
            out_file=output_path,
            transforms=transforms,
            interpolator="linear",
            imagetype=3,
            nthreads=nthreads,
        )
    finally:
        native_tensor.unlink(missing_ok=True)
    return output_path
