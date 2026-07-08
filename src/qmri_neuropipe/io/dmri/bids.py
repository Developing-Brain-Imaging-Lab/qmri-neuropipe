from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Iterable, Any
import json
import tempfile
import nibabel as nib
import numpy as np

from qmri_neuropipe.core.types import ImageFile, DWIFile
from qmri_neuropipe.core.utils import get_nifti_stem
from ..bids import build_bids_name, bids_find, _load_json_field, _sidecar  # already in your skeleton

_DIR_ENTITY_TO_PED = {
    "RL": "i",
    "LR": "i-",
    "AP": "j",
    "PA": "j-",
    "SI": "k",
    "IS": "k-",
}

_PED_TO_VECTOR = {
    "i": np.array([1.0, 0.0, 0.0]),
    "i-": np.array([-1.0, 0.0, 0.0]),
    "j": np.array([0.0, 1.0, 0.0]),
    "j-": np.array([0.0, -1.0, 0.0]),
    "k": np.array([0.0, 0.0, 1.0]),
    "k-": np.array([0.0, 0.0, -1.0]),
}

_FSL_TOPUP_SUPPORTED_PED = {"i", "i-", "j", "j-"}

# def load_dwi_from_bids(sub_dir: Path) -> Dict[Path, Path, Path, Path, Optional[Path], Optional[Path]]:
#     """
#     Very small helper: picks the first *_dwi.nii.gz and mates .bval/.bvec/.json.
#     Extend as needed (multiple runs, AP/PA handling, etc.).
#     """
#     dwi = sorted((sub_dir / "dwi").glob("*_dwi.nii.gz"))[0]
#     bval = Path(str(dwi).replace("_dwi.nii.gz", "_dwi.bval"))
#     bvec = Path(str(dwi).replace("_dwi.nii.gz", "_dwi.bvec"))
#     js   = Path(str(dwi).replace(".nii.gz", ".json")) if (Path(str(dwi).replace(".nii.gz", ".json"))).exists() else None

#     acqp, index = build_acqp_index(js, dwi)

#     return Dict(img=dwi, bval=bval, bvec=bvec, json=js, acqp=acqp, index=index)


def bids_find_dwi(root, derive_timing_sidecars: bool = True) -> list[DWIFile]:
    """
    Find DWI NIfTI images and their associated JSON/BVAL/BVEC/timing files.

    Uses the existing `bids_find(root, suffix='dwi', extension='.nii.gz')`.
    """
    dwi_ents = bids_find(root, suffix="dwi", extension=".nii.gz")
    results: list[DWIFile] = []

    for ent in dwi_ents:
        img = ent["path"]
        json_path = _sidecar(img, ".json")
        bval_path = _sidecar(img, ".bval")
        bvec_path = _sidecar(img, ".bvec")
        Delta_path = _sidecar(img, ".bigdelta")
        if not Delta_path.exists():
            legacy_Delta_path = _sidecar(img, ".Delta")
            if legacy_Delta_path.exists():
                Delta_path = legacy_Delta_path
        delta_path = _sidecar(img, ".delta")

        dwi = DWIFile(
            entities=ent,
            img=img,
            json=json_path if json_path.exists() else None,
            bval=bval_path if bval_path.exists() else None,
            bvec=bvec_path if bvec_path.exists() else None,
            Delta=Delta_path if Delta_path.exists() else None,
            delta=delta_path if delta_path.exists() else None,
        )
        if derive_timing_sidecars:
            ensure_dwi_timing_sidecars(dwi)
        results.append(dwi)

    return results


_BIG_DELTA_JSON_KEYS = (
    "Delta",
    "BigDelta",
    "big_delta",
    "DiffusionGradientSeparation",
    "DiffusionGradientSeparationTime",
    "GradientSeparation",
)

_SMALL_DELTA_JSON_KEYS = (
    "delta",
    "SmallDelta",
    "small_delta",
    "DiffusionGradientDuration",
    "DiffusionGradientDurationTime",
    "GradientDuration",
)


def _load_json_payload(json_path: Path | None) -> dict[str, Any]:
    if not json_path or not Path(json_path).exists():
        return {}
    try:
        with Path(json_path).open() as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            return payload[key]
    return None


def _dwi_volume_count(dwi: DWIFile) -> Optional[int]:
    if dwi.bval and Path(dwi.bval).exists():
        values = np.loadtxt(dwi.bval)
        return int(np.atleast_1d(values).size)
    if dwi.img and Path(dwi.img).exists():
        shape = nib.load(str(dwi.img)).shape
        return int(shape[3]) if len(shape) > 3 else 1
    return None


def _coerce_timing_vector(value: Any, n_volumes: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if arr.size == 1:
        arr = np.repeat(arr, n_volumes)
    elif arr.size != n_volumes:
        return None

    # Timing metadata should be seconds. Values like 40 or 10 usually indicate
    # milliseconds in vendor-derived JSON, so normalize them for model fitting.
    if np.nanmean(np.abs(arr)) > 1.0:
        arr = arr / 1000.0
    return arr


def ensure_dwi_timing_sidecars(dwi: DWIFile, overwrite: bool = False, logger=None) -> DWIFile:
    """Create .bigdelta/.delta sidecars from DWI JSON timing metadata when available."""
    if not isinstance(dwi, DWIFile):
        return dwi
    if dwi.Delta and dwi.delta and Path(dwi.Delta).exists() and Path(dwi.delta).exists() and not overwrite:
        return dwi

    payload = _load_json_payload(dwi.json)
    if not payload:
        return dwi

    try:
        n_volumes = _dwi_volume_count(dwi)
    except Exception as exc:
        if logger:
            logger.warning(f"Could not determine DWI volume count for timing sidecars: {exc}")
        return dwi
    if not n_volumes:
        return dwi

    big_delta = _coerce_timing_vector(_first_present(payload, _BIG_DELTA_JSON_KEYS), n_volumes)
    small_delta = _coerce_timing_vector(_first_present(payload, _SMALL_DELTA_JSON_KEYS), n_volumes)
    if big_delta is None or small_delta is None:
        return dwi

    Delta_path = _sidecar(dwi.img, ".bigdelta")
    delta_path = _sidecar(dwi.img, ".delta")
    if overwrite or not Delta_path.exists():
        np.savetxt(Delta_path, big_delta, fmt="%.9g")
    if overwrite or not delta_path.exists():
        np.savetxt(delta_path, small_delta, fmt="%.9g")

    dwi.Delta = Delta_path if Delta_path.exists() else dwi.Delta
    dwi.delta = delta_path if delta_path.exists() else dwi.delta
    if logger:
        logger.info(f"Created diffusion timing sidecars for {Path(dwi.img).name}: {Delta_path.name}, {delta_path.name}")
    return dwi
    
def build_dwi_filename(sub_id, session=None, entities=None):
    
    if entities:
        entities["sub"] = sub_id
    else:
        entities = {"sub": sub_id}
    
    if session:
        entities["ses"] = session
    # suffix is dwi, extension .nii.gz
    return build_bids_name(entities, suffix="dwi", extension=".nii.gz")

def build_dwi_path(root, sub_id, session=None, entities=None):
    """
    Build the full BIDS path for a DWI NIfTI file.
    """
    fname = build_dwi_filename(sub_id=sub_id, 
                               session=session, 
                               entities=entities)

    root = Path(root)
    parts = [root, f"sub-{sub_id}"]
    if session:
        parts.append(f"ses-{session}")
    parts.append("dwi")  # DWI lives in 'dwi' folder

    return Path(*parts) / fname

def infer_phase_encoding_direction(dwi: DWIFile | None = None, json_path: Path | None = None, entities: Optional[dict] = None):
    """
    Resolve PhaseEncodingDirection from BIDS JSON first, then common dir-* entity labels.
    """
    ped = _load_json_field(json_path or getattr(dwi, "json", None), "PhaseEncodingDirection")
    if ped:
        return ped

    source_entities = entities or getattr(dwi, "entities", {}) or {}
    dir_label = source_entities.get("dir")
    if dir_label:
        return _DIR_ENTITY_TO_PED.get(str(dir_label).strip().upper())

    return None


def phase_encoding_direction_to_vector(direction: str) -> np.ndarray:
    """Convert a BIDS phase-encoding direction to a voxel-axis vector."""
    try:
        return _PED_TO_VECTOR[str(direction)].copy()
    except KeyError as exc:
        raise ValueError(f"Unsupported PhaseEncodingDirection: {direction!r}") from exc


def infer_fsl_phase_encoding_direction(
    dwi: DWIFile | None = None,
    json_path: Path | None = None,
    entities: Optional[dict] = None,
) -> str | None:
    """
    Resolve a phase-encoding direction usable by FSL topup/eddy acqparams.

    BIDS allows ``k``/``k-`` for through-plane phase encoding, but FSL topup
    rejects acqparams rows with a nonzero third vector component. If JSON has a
    through-plane direction but the BIDS ``dir`` entity gives a conventional
    in-plane AP/PA/RL/LR label, prefer the entity for FSL.
    """
    json_ped = _load_json_field(json_path or getattr(dwi, "json", None), "PhaseEncodingDirection")
    source_entities = entities or getattr(dwi, "entities", {}) or {}
    dir_label = source_entities.get("dir")
    dir_ped = _DIR_ENTITY_TO_PED.get(str(dir_label).strip().upper()) if dir_label else None

    if json_ped in _FSL_TOPUP_SUPPORTED_PED:
        return json_ped
    if json_ped in {"k", "k-"} and dir_ped in _FSL_TOPUP_SUPPORTED_PED:
        return dir_ped
    return json_ped or dir_ped


def fsl_phase_encoding_direction_to_vector(direction: str) -> np.ndarray:
    """Convert a phase-encoding direction to an FSL topup/eddy acqparams vector."""
    vector = phase_encoding_direction_to_vector(direction)
    if not np.isclose(vector[2], 0.0):
        raise ValueError(
            "FSL topup/eddy acqparams require the third phase-encoding vector "
            f"component to be zero, but PhaseEncodingDirection={direction!r} "
            "maps to through-plane encoding. Fix the DWI sidecar or filename "
            "dir entity to an in-plane direction such as AP/PA/RL/LR, or disable topup."
        )
    return vector


def phase_encoding_vector_to_direction(vector: Iterable[float], atol: float = 1e-5) -> str:
    """Convert a cardinal voxel-axis vector to a BIDS phase-encoding direction."""
    vec = np.asarray(tuple(vector), dtype=float)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        raise ValueError(f"Invalid phase-encoding vector: {vector!r}")

    nonzero = np.flatnonzero(np.abs(vec) > atol)
    if len(nonzero) != 1:
        raise ValueError(f"Phase-encoding vector is not cardinal: {vec.tolist()}")

    axis = int(nonzero[0])
    label = "ijk"[axis]
    return f"{label}-" if vec[axis] < 0 else label


def phase_encoding_transform_matrix(
    source_affine: np.ndarray,
    target_affine: np.ndarray,
    atol: float = 1e-4,
) -> np.ndarray:
    """Return the signed permutation mapping source voxel vectors to target voxels."""
    source_linear = np.asarray(source_affine, dtype=float)[:3, :3]
    target_linear = np.asarray(target_affine, dtype=float)[:3, :3]
    transform = np.linalg.solve(target_linear, source_linear)
    rounded = np.rint(transform)

    is_signed_permutation = (
        np.allclose(transform, rounded, atol=atol)
        and np.all(np.sum(np.abs(rounded), axis=0) == 1)
        and np.all(np.sum(np.abs(rounded), axis=1) == 1)
    )
    if not is_signed_permutation:
        raise ValueError(
            "Image change is not a pure axis permutation/flip; cannot safely "
            f"transform phase encoding. Matrix: {transform.tolist()}"
        )
    return rounded.astype(int)


def transform_phase_encoding_direction(
    direction: str,
    source_affine: np.ndarray,
    target_affine: np.ndarray,
) -> str:
    """Express a BIDS phase-encoding direction in a reoriented image grid."""
    transform = phase_encoding_transform_matrix(source_affine, target_affine)
    vector = transform @ phase_encoding_direction_to_vector(direction)
    return phase_encoding_vector_to_direction(vector)


def transform_acqparams_file(
    source: Path,
    destination: Path,
    transform: np.ndarray,
) -> Path:
    """Transform the direction columns of an FSL acquisition-parameters file."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 phase-encoding transform, got {matrix.shape}")

    output_lines = []
    for line_number, raw_line in enumerate(Path(source).read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 4:
            raise ValueError(f"Invalid acqparams row {line_number} in {source}: {raw_line!r}")
        try:
            direction = np.asarray([float(value) for value in fields[:3]])
        except ValueError as exc:
            raise ValueError(
                f"Invalid direction in acqparams row {line_number} of {source}: {raw_line!r}"
            ) from exc
        transformed = matrix @ direction
        transformed[np.isclose(transformed, 0.0, atol=1e-8)] = 0.0
        direction_fields = [f"{value:g}" for value in transformed]
        output_lines.append(" ".join([*direction_fields, *fields[3:]]))

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return destination


def build_acqp_index(
    json_path: Path | None,
    dwi_path: Path,
    entities: Optional[dict] = None,
    support_dir: Path | None = None,
) -> Tuple[Path | None, Path | None]:
    """
    Build minimal FSL acqp.txt and index.txt from BIDS JSON.
    This is a stub that writes a single PE dir/ro time line if JSON is present.
    Expand for multi-PE or TOPUP use.
    """
    if not json_path or not json_path.exists():
        if not entities:
            return None, None
        meta = {}
    else:
        try:
            meta = json.loads(json_path.read_text())
        except Exception:
            return None, None

    ped = infer_fsl_phase_encoding_direction(json_path=json_path, entities=entities)
    if ped is None:
        return None, None

    trt = meta.get("TotalReadoutTime", 0.05)  # seconds

    acqp_dir = Path(support_dir) if support_dir else Path(tempfile.gettempdir()) / "qmri-neuropipe" / "eddy_support"
    acqp_dir.mkdir(parents=True, exist_ok=True)
    stem = get_nifti_stem(dwi_path)
    acqp = acqp_dir / f"{stem}_acqp.txt"
    index = acqp_dir / f"{stem}_index.txt"

    # Map BIDS PE to FSL acqp (i/j/k with +/-). Using j as default example:
    # Columns are: dx dy dz readout_time
    try:
        vector = fsl_phase_encoding_direction_to_vector(ped).astype(int)
    except ValueError:
        if ped in {"k", "k-"}:
            raise
        vector = phase_encoding_direction_to_vector("j").astype(int)
    line = " ".join(str(value) for value in vector)

    acqp.write_text(f"{line} {trt:.6f}\n", encoding="utf-8")

    # Determine number of volumes either from bval or by loading the image
    nvols: Optional[int] = None
    bval_candidates = list(Path(dwi_path.parent).glob("*_dwi.bval"))
    if bval_candidates:
        try:
            nvols = len(bval_candidates[0].read_text().split())
        except Exception:
            nvols = None
    if nvols is None:
        try:
            nvols = nib.load(str(dwi_path)).shape[-1]
        except Exception:
            nvols = None

    if nvols:
        index.write_text(("1\n" * nvols), encoding="utf-8")
    else:
        # Cannot construct a valid index file
        acqp.unlink(missing_ok=True)
        return None, None

    return acqp, index

def find_reversed_phase_groups(dwi_files: list[DWIFile], group_by: tuple[str, ...] = ("sub", "ses"),):
    """
    Given a list of DWIFile objects, find groups that can be treated as
    reversed phase-encoding sets for distortion correction.

    Returns
    -------
    list[list[DWIFile]]
        Each inner list is a group of DWIFile objects that share:
        - same subject/session (and other group_by entities)
        - same phase-encoding axis (i/j/k)
        - contain at least one '+' and one '-' direction
    """
    # key -> {+1: [files], -1: [files]}
    groups = defaultdict(lambda: {+1: [], -1: []})

    for dwi in dwi_files:
        ents = dwi.entities
        ped_raw = infer_phase_encoding_direction(dwi)
        if not ped_raw:
            # Could not determine PE direction; ignore for grouping
            continue

        # Normalize PhaseEncodingDirection: 'j-' -> axis='j', sign=-1
        axis = ped_raw[0]      # i / j / k
        sign = -1 if ped_raw.endswith("-") else +1

        # Build grouping key (sub, ses, axis, optionally acq/dir)
        key_parts = [ents.get(k, "") for k in group_by]
        key_parts.append(axis)
        key = tuple(key_parts)  # e.g., ('01', '01', 'j')

        groups[key][sign].append(dwi)

    combined_groups: list[list[DWIFile]] = []

    for key, sign_dict in groups.items():
        pos = sign_dict[+1]
        neg = sign_dict[-1]
        if pos and neg:
            # Simple strategy: treat all files in this key as a single group
            # (you can refine to pair by run, acq, etc.)
            combined = pos + neg
            combined_groups.append(combined)

    return combined_groups
