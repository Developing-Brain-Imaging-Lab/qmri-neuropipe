from __future__ import annotations

import gzip
import json
import logging
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import nibabel as nib
import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.tag import Tag

from ...core import BaseProcessingStep, ProcessingError


LOGGER = logging.getLogger(__name__)
PDB_TAG = Tag(0x0025, 0x101B)


@dataclass
class GeDicomSeriesMetadata:
    dicom_path: Path
    series_instance_uid: Optional[str]
    series_number: Optional[int]
    series_description: Optional[str]
    protocol_name: Optional[str]
    manufacturer: Optional[str]
    pdb_center_scanner_ras_rel_iso_mm: list[float]
    derivation_keys: list[str]
    pdb_center_parsing: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class _PdbLocationValue:
    raw: str
    value: float
    axis: Optional[int]
    letter: Optional[str]

    @property
    def has_axis(self) -> bool:
        return self.axis is not None


_AXIS_INDEX = {
    "R": 0,
    "L": 0,
    "A": 1,
    "P": 1,
    "S": 2,
    "I": 2,
}
_SIGNED_AXIS_VALUE = {
    "R": +1.0,
    "L": -1.0,
    "A": +1.0,
    "P": -1.0,
    "S": +1.0,
    "I": -1.0,
}
_AXIS_NAMES = {
    0: "R/L",
    1: "A/P",
    2: "S/I",
}


def _extract_pdb_text(ds: pydicom.dataset.FileDataset) -> str:
    if PDB_TAG not in ds:
        raise ValueError(f"GE PDB private tag missing: {PDB_TAG}")

    pdb = ds[PDB_TAG].value
    if isinstance(pdb, str):
        return pdb
    if not isinstance(pdb, (bytes, bytearray)):
        raise ValueError("Unsupported GE PDB payload type")

    raw = bytes(pdb)
    idx = raw.find(b"\x1f\x8b")
    if idx < 0:
        text = raw.decode("latin1", errors="replace").replace("\x00", "").strip()
        if _kv_from_pdb(text):
            return text
        raise ValueError("GE PDB payload does not contain a gzip stream or recognizable text")

    try:
        return gzip.decompress(raw[idx:]).decode("latin1", errors="replace")
    except Exception as exc:
        raise ValueError(f"Failed to decompress GE PDB gzip payload: {exc}") from exc


def _kv_from_pdb(txt: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for match in re.finditer(r'([A-Z0-9_]+)\s*(?:=|:)?\s*"([^"]*)"', txt):
        kv[match.group(1)] = match.group(2)
    return kv


def _parse_position_token(token: str) -> tuple[Optional[str], float]:
    token = token.strip().strip('"').strip()
    token = token.replace("\x00", "")
    token = re.sub(r"\s+", " ", token)
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    patterns = [
        rf'^([LRAPSI])\s*({number})\s*(?:MM)?$',
        rf'^({number})\s*([LRAPSI])\s*(?:MM)?$',
        rf'^({number})\s*(?:MM)?$',
    ]
    for idx, pattern in enumerate(patterns):
        match = re.match(pattern, token, flags=re.IGNORECASE)
        if not match:
            continue
        if idx == 0:
            return match.group(1).upper(), float(match.group(2))
        if idx == 1:
            return match.group(2).upper(), float(match.group(1))
        return None, float(match.group(1))

    raise ValueError(f"Unsupported GE position token: {token!r}")


def _parse_pdb_location_value(token: str) -> _PdbLocationValue:
    letter, value = _parse_position_token(token)
    raw = str(token).strip().strip('"').replace("\x00", "")
    if letter is None:
        return _PdbLocationValue(raw=raw, value=float(value), axis=None, letter=None)
    letter = letter.upper()
    return _PdbLocationValue(
        raw=raw,
        value=float(_SIGNED_AXIS_VALUE[letter] * value),
        axis=_AXIS_INDEX[letter],
        letter=letter,
    )


def _assign_axis_value(
    center: np.ndarray,
    used_axes: set[int],
    parsed: _PdbLocationValue,
    field_name: str,
) -> None:
    if parsed.axis is None:
        raise ValueError(f"Internal error: {field_name} has no anatomical axis.")
    if parsed.axis in used_axes:
        raise ValueError(
            f"Axis conflict while parsing {field_name}={parsed.raw!r}; "
            f"axis {_AXIS_NAMES[parsed.axis]} was already assigned."
        )
    center[parsed.axis] = parsed.value
    used_axes.add(parsed.axis)


def _infer_numeric_only_axis(used_axes: set[int], field_name: str) -> int:
    remaining_axes = [axis for axis in (0, 1, 2) if axis not in used_axes]
    if len(remaining_axes) != 1:
        assigned = ", ".join(_AXIS_NAMES[a] for a in sorted(used_axes)) or "none"
        remaining = ", ".join(_AXIS_NAMES[a] for a in remaining_axes) or "none"
        raise ValueError(
            f"Cannot infer anatomical axis for numeric-only {field_name}. "
            f"Assigned axes: {assigned}. Remaining axes: {remaining}."
        )
    return remaining_axes[0]


def _pdb_center_ras_rel_iso_with_details(kv: dict[str, str]) -> tuple[list[float], dict[str, Any]]:
    required = ["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"]
    missing = [key for key in required if key not in kv]
    if missing:
        raise ValueError(f"Missing GE PDB keys: {', '.join(missing)}")

    sloc = _parse_pdb_location_value(kv["SLOC1"])
    eloc = _parse_pdb_location_value(kv["ELOC1"])
    fov_values = [
        ("FOVCNT1", _parse_pdb_location_value(kv["FOVCNT1"])),
        ("FOVCNT2", _parse_pdb_location_value(kv["FOVCNT2"])),
    ]

    if sloc.axis is None or eloc.axis is None:
        raise ValueError(
            "SLOC1 and ELOC1 must include anatomical letters so the slice-select axis can be identified. "
            f"Got SLOC1={kv['SLOC1']!r}, ELOC1={kv['ELOC1']!r}."
        )
    if sloc.axis != eloc.axis:
        raise ValueError(
            "SLOC1 and ELOC1 appear to describe different anatomical axes: "
            f"SLOC1={kv['SLOC1']!r} -> {_AXIS_NAMES[sloc.axis]}, "
            f"ELOC1={kv['ELOC1']!r} -> {_AXIS_NAMES[eloc.axis]}."
        )

    center = np.full(3, np.nan, dtype=float)
    used_axes: set[int] = set()
    inferred_axes: dict[str, int] = {}
    center[sloc.axis] = 0.5 * (sloc.value + eloc.value)
    used_axes.add(sloc.axis)

    numeric_only_fov_values: list[tuple[str, _PdbLocationValue]] = []
    for field_name, parsed in fov_values:
        if parsed.has_axis:
            _assign_axis_value(center, used_axes, parsed, field_name)
        else:
            numeric_only_fov_values.append((field_name, parsed))

    for field_name, parsed in numeric_only_fov_values:
        axis = _infer_numeric_only_axis(used_axes, field_name)
        center[axis] = parsed.value
        used_axes.add(axis)
        inferred_axes[field_name] = axis

    if np.any(~np.isfinite(center)):
        raise ValueError(
            "Could not fully determine GE PDB prescription center. "
            f"Computed center={center.tolist()}, used_axes={sorted(used_axes)}."
        )

    center_list = [float(value) for value in center]
    parsed_fields: dict[str, dict[str, Any]] = {
        "SLOC1": {"axis": _AXIS_NAMES[sloc.axis], "value": sloc.value},
        "ELOC1": {"axis": _AXIS_NAMES[eloc.axis], "value": eloc.value},
    }
    for field_name, parsed in fov_values:
        axis = parsed.axis if parsed.axis is not None else inferred_axes[field_name]
        parsed_fields[field_name] = {
            "axis": _AXIS_NAMES[axis],
            "value": parsed.value,
            "axis_inferred": parsed.axis is None,
        }

    details = {
        "Method": "prescription_aware_anatomical_letter_axis_mapping",
        "AxisConvention": "scanner RAS relative to isocenter; R/A/S positive",
        "RawFields": {key: kv[key] for key in required},
        "ParsedFields": parsed_fields,
    }

    return center_list, details


def _pdb_center_ras_rel_iso(kv: dict[str, str]) -> list[float]:
    center, _details = _pdb_center_ras_rel_iso_with_details(kv)
    return center


def extract_ge_series_metadata(dicom_path: Path) -> GeDicomSeriesMetadata:
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
    manufacturer = str(getattr(ds, "Manufacturer", "") or "").strip()
    if "GE" not in manufacturer.upper():
        raise ValueError(f"Not a GE DICOM: {manufacturer or 'unknown manufacturer'}")

    kv = _kv_from_pdb(_extract_pdb_text(ds))
    offset, parsing = _pdb_center_ras_rel_iso_with_details(kv)

    return GeDicomSeriesMetadata(
        dicom_path=dicom_path,
        series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "") or "") or None,
        series_number=int(getattr(ds, "SeriesNumber")) if getattr(ds, "SeriesNumber", None) is not None else None,
        series_description=str(getattr(ds, "SeriesDescription", "") or "") or None,
        protocol_name=str(getattr(ds, "ProtocolName", "") or "") or None,
        manufacturer=manufacturer or None,
        pdb_center_scanner_ras_rel_iso_mm=offset,
        derivation_keys=["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"],
        pdb_center_parsing=parsing,
    )


def _iter_dicom_files(dicom_dir: Path) -> list[Path]:
    return sorted(path for path in dicom_dir.rglob("*") if path.is_file())


def _classify_ge_metadata_error(exc: Exception) -> str:
    msg = str(exc)
    if msg.startswith("Not a GE DICOM:"):
        return "not_ge"
    if msg.startswith("GE PDB private tag missing:"):
        return "missing_pdb_tag"
    if msg.startswith("GE PDB payload does not contain a gzip stream"):
        return "unrecognized_pdb_payload"
    if msg.startswith("Failed to decompress GE PDB gzip payload:"):
        return "corrupt_pdb_gzip"
    if msg.startswith("Missing GE PDB keys:"):
        return "missing_pdb_keys"
    if msg.startswith("Unsupported GE position token:"):
        return "unsupported_position_token"
    if msg.startswith("Unsupported GE PDB payload type"):
        return "unsupported_pdb_payload_type"
    return f"other:{msg}"


def _format_ge_metadata_scan_summary(
    *,
    dicom_dir: Path,
    scanned_files: int,
    failure_counts: Counter[str],
    failure_examples: dict[str, Path],
) -> str:
    if scanned_files == 0:
        return f"No files were found under {dicom_dir}."

    labels = {
        "not_ge": "non-GE files",
        "missing_pdb_tag": f"GE files missing private tag {PDB_TAG}",
        "unrecognized_pdb_payload": "GE files with unreadable PDB payloads",
        "corrupt_pdb_gzip": "GE files with corrupt PDB gzip payloads",
        "missing_pdb_keys": "GE files missing required PDB keys",
        "unsupported_position_token": "GE files with unsupported position tokens",
        "unsupported_pdb_payload_type": "GE files with unsupported PDB payload types",
    }

    parts = [f"Scanned {scanned_files} file(s)."]
    if not failure_counts:
        parts.append("No DICOM candidates could be parsed.")
        return " ".join(parts)

    ranked = failure_counts.most_common(3)
    formatted: list[str] = []
    for key, count in ranked:
        label = labels.get(key, key.removeprefix("other:"))
        example = failure_examples.get(key)
        if example:
            formatted.append(f"{label}: {count} (example: {example})")
        else:
            formatted.append(f"{label}: {count}")
    parts.append("Top rejection reasons: " + "; ".join(formatted) + ".")
    return " ".join(parts)


def collect_ge_series_metadata(
    dicom_dir: Path,
    logger: Optional[logging.Logger] = None,
    *,
    return_diagnostics: bool = False,
) -> list[GeDicomSeriesMetadata] | tuple[list[GeDicomSeriesMetadata], dict[str, Any]]:
    logger = logger or LOGGER
    series_map: dict[str, GeDicomSeriesMetadata] = {}
    failure_counts: Counter[str] = Counter()
    failure_examples: dict[str, Path] = {}
    scanned_files = 0

    for path in _iter_dicom_files(dicom_dir):
        scanned_files += 1
        try:
            meta = extract_ge_series_metadata(path)
        except (InvalidDicomError, IsADirectoryError):
            continue
        except Exception as exc:
            key = _classify_ge_metadata_error(exc)
            failure_counts[key] += 1
            failure_examples.setdefault(key, path)
            logger.debug(f"Skipping DICOM candidate {path}: {exc}")
            continue

        key = meta.series_instance_uid or f"series-number:{meta.series_number}"
        if key not in series_map:
            series_map[key] = meta

    series = list(series_map.values())
    if return_diagnostics:
        diagnostics = {
            "scanned_files": scanned_files,
            "failure_counts": dict(failure_counts),
            "failure_examples": {key: str(path) for key, path in failure_examples.items()},
            "summary": _format_ge_metadata_scan_summary(
                dicom_dir=dicom_dir,
                scanned_files=scanned_files,
                failure_counts=failure_counts,
                failure_examples=failure_examples,
            ),
        }
        return series, diagnostics
    return series


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _rounded_list(values: Sequence[float], ndigits: int = 12) -> list[float]:
    rounded_values: list[float] = []
    for value in values:
        rounded = round(float(value), ndigits)
        if rounded == -0.0:
            rounded = 0.0
        rounded_values.append(rounded)
    return rounded_values


def _get_nested(payload: dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _ensure_nested_dict(payload: dict[str, Any], path: Sequence[str]) -> dict[str, Any]:
    current = payload
    for key in path:
        value = current.get(key)
        if not isinstance(value, dict):
            value = {}
            current[key] = value
        current = value
    return current


def _infer_dir_entity(payload: dict[str, Any], json_path: Path) -> Optional[str]:
    candidates: list[str] = []

    entity_dir = _get_nested(payload, ["GradientTableOverride", "MatchingRule", "entities", "dir"])
    if isinstance(entity_dir, str):
        candidates.append(entity_dir)

    candidates.append(json_path.name)

    bids_guess = payload.get("BidsGuess")
    if isinstance(bids_guess, list):
        candidates.extend(str(item) for item in bids_guess)
    elif isinstance(bids_guess, str):
        candidates.append(bids_guess)

    for field in [
        ["GradientNonlinearityCorrection", "Derivation", "RepresentativeNifti"],
        ["GradientNonlinearityCorrection", "Derivation", "RepresentativeDicom"],
        ["SeriesDescription"],
        ["ProtocolName"],
    ]:
        value = _get_nested(payload, field)
        if isinstance(value, str):
            candidates.append(value)

    text = "\n".join(candidates)
    if re.search(r"(?:^|[_/.-])dir-AP(?:[_/.-]|$)", text, flags=re.IGNORECASE):
        return "AP"
    if re.search(r"(?:^|[_/.-])dir-PA(?:[_/.-]|$)", text, flags=re.IGNORECASE):
        return "PA"
    if re.search(r"pe1", text, flags=re.IGNORECASE):
        return "AP"
    if re.search(r"pe0", text, flags=re.IGNORECASE):
        return "PA"
    return None


def _phase_from_existing(payload: dict[str, Any]) -> Optional[str]:
    existing = payload.get("PhaseEncodingDirection")
    if not isinstance(existing, str) or not existing:
        return None
    if existing[0] not in {"i", "j", "k"}:
        return None
    return "j-" if existing.endswith("-") else "j"


def _phase_from_dir(dir_entity: Optional[str]) -> Optional[str]:
    if dir_entity is None:
        return None
    normalized = dir_entity.upper()
    if normalized == "AP":
        return "j"
    if normalized == "PA":
        return "j-"
    return None


def _phase_from_polarity(payload: dict[str, Any]) -> Optional[str]:
    polarity = payload.get("PhaseEncodingPolarityGE")
    if not isinstance(polarity, str):
        return None
    normalized = polarity.lower()
    if normalized == "unflipped":
        return "j"
    if normalized == "flipped":
        return "j-"
    return None


def _choose_phase_encoding_direction(payload: dict[str, Any], json_path: Path, mode: str = "dir") -> Optional[str]:
    dir_entity = _infer_dir_entity(payload, json_path)
    if mode == "dir":
        candidates = [_phase_from_dir(dir_entity), _phase_from_existing(payload), _phase_from_polarity(payload)]
    elif mode == "existing":
        candidates = [_phase_from_existing(payload), _phase_from_dir(dir_entity), _phase_from_polarity(payload)]
    elif mode == "polarity":
        candidates = [_phase_from_polarity(payload), _phase_from_dir(dir_entity), _phase_from_existing(payload)]
    else:
        raise ValueError(f"Unknown phase_from mode: {mode}")

    for candidate in candidates:
        if candidate in {"j", "j-"}:
            return candidate
    return None


def _apply_phase_encoding_fix(
    payload: dict[str, Any],
    json_path: Path,
    *,
    phase_from: str = "dir",
    strict: bool = False,
) -> bool:
    phase = _choose_phase_encoding_direction(payload, json_path, phase_from)
    if phase is None:
        if strict:
            raise ValueError(
                f"{json_path}: could not infer AP/PA PhaseEncodingDirection from dir entity, "
                "existing PhaseEncodingDirection, or PhaseEncodingPolarityGE."
            )
        return False
    if payload.get("PhaseEncodingDirection") == phase:
        return False
    payload["PhaseEncodingDirection"] = phase
    return True


def _resolve_sidecar_nifti(json_path: Path) -> Optional[Path]:
    base = str(json_path)
    if not base.endswith(".json"):
        return None
    nii_gz = Path(base[:-5] + ".nii.gz")
    if nii_gz.exists():
        return nii_gz
    nii = Path(base[:-5] + ".nii")
    if nii.exists():
        return nii
    return None


def _prep_itk_lps_geometry(img: nib.Nifti1Image) -> np.ndarray:
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
    return aff_grad_lps


def _derive_isocenter_offset_scanner_ras_mm(
    nifti_path: Path,
    pdb_center_scanner_ras_rel_iso_mm: list[float],
) -> list[float]:
    img = nib.load(str(nifti_path))
    aff_grad_lps = _prep_itk_lps_geometry(img)
    shape = img.shape[:3]
    cijk = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
    pt_lps_center = (aff_grad_lps @ np.array([cijk[0], cijk[1], cijk[2], 1.0]))[:3]
    pt_ras_center = np.array([-pt_lps_center[0], -pt_lps_center[1], pt_lps_center[2]], dtype=np.float64)
    offset = pt_ras_center - np.array(pdb_center_scanner_ras_rel_iso_mm, dtype=np.float64)
    return [float(v) for v in offset]


def _extract_ge_pdb_center_from_dicom(dicom_path: Path) -> tuple[list[float], dict[str, Any]]:
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
    kv = _kv_from_pdb(_extract_pdb_text(ds))
    return _pdb_center_ras_rel_iso_with_details(kv)


def _get_series_identifiers(sidecar: dict[str, Any]) -> dict[str, Any]:
    return {
        "SeriesInstanceUID": sidecar.get("SeriesInstanceUID"),
        "SeriesNumber": sidecar.get("SeriesNumber"),
        "SeriesDescription": sidecar.get("SeriesDescription"),
        "ProtocolName": sidecar.get("ProtocolName"),
    }


def _match_series(sidecar: dict[str, Any], series_meta: list[GeDicomSeriesMetadata]) -> Optional[GeDicomSeriesMetadata]:
    ids = _get_series_identifiers(sidecar)

    uid = ids["SeriesInstanceUID"]
    if uid:
        matches = [meta for meta in series_meta if meta.series_instance_uid == uid]
        if len(matches) == 1:
            return matches[0]

    number = ids["SeriesNumber"]
    if number is not None:
        try:
            number = int(number)
        except Exception:
            number = None
    if number is not None:
        matches = [meta for meta in series_meta if meta.series_number == number]
        if len(matches) == 1:
            return matches[0]

    text_fields = [
        ("SeriesDescription", "series_description"),
        ("ProtocolName", "protocol_name"),
    ]
    for sidecar_key, meta_key in text_fields:
        text = ids[sidecar_key]
        if not text:
            continue
        matches = [meta for meta in series_meta if getattr(meta, meta_key) == text]
        if len(matches) == 1:
            return matches[0]

    return None


def enrich_dwi_sidecar_with_ge_gnl(
    json_path: Path,
    series_meta: GeDicomSeriesMetadata,
    logger: Optional[logging.Logger] = None,
    *,
    fix_phase_encoding: bool = False,
    phase_from: str = "dir",
) -> bool:
    logger = logger or LOGGER
    payload = _load_json(json_path)
    gnl_meta = payload.get("GradientNonlinearityCorrection", {})
    nifti_path = _resolve_sidecar_nifti(json_path)
    if nifti_path is None:
        logger.warning(
            "Could not resolve NIfTI for %s; falling back to raw PDB center values for GNL metadata.",
            json_path.name,
        )
        isocenter_offset_scanner_ras_mm = series_meta.pdb_center_scanner_ras_rel_iso_mm
    else:
        isocenter_offset_scanner_ras_mm = _derive_isocenter_offset_scanner_ras_mm(
            nifti_path,
            series_meta.pdb_center_scanner_ras_rel_iso_mm,
        )

    new_block = {
        "Manufacturer": series_meta.manufacturer or "GE",
        "Method": "native_ge",
        "Source": "dicom_import",
        "IsocenterOffsetScannerRASmm": isocenter_offset_scanner_ras_mm,
        "Derivation": {
            "PDBKeys": series_meta.derivation_keys,
            "PDBCenterScannerRASRelativeToIsocenterMm": _rounded_list(series_meta.pdb_center_scanner_ras_rel_iso_mm),
            "PDBCenterParsing": series_meta.pdb_center_parsing,
            "RepresentativeDicom": str(series_meta.dicom_path),
            "RepresentativeNifti": str(nifti_path) if nifti_path else None,
            "NativeGeometryConvention": "make-L_ge_eval_frame",
            "SeriesInstanceUID": series_meta.series_instance_uid,
            "SeriesNumber": series_meta.series_number,
            "SeriesDescription": series_meta.series_description,
            "ProtocolName": series_meta.protocol_name,
        },
    }
    new_block["IsocenterOffsetScannerRASmm"] = _rounded_list(new_block["IsocenterOffsetScannerRASmm"])

    payload["GradientNonlinearityCorrection"] = new_block
    phase_changed = False
    if fix_phase_encoding:
        phase_changed = _apply_phase_encoding_fix(payload, json_path, phase_from=phase_from, strict=False)

    if gnl_meta == new_block and not phase_changed:
        logger.debug(f"GNL metadata already present in {json_path.name}")
        return False

    _save_json(json_path, payload)
    return True


def _target_dwi_sidecars(search_root: Path, context: dict[str, Any]) -> list[Path]:
    if "imported_dwi_sidecars" in context:
        imported = context.get("imported_dwi_sidecars") or []
        sidecars = [Path(p) for p in imported if Path(p).exists()]
        return sorted(sidecars)

    subject = context.get("subject")
    session = context.get("session")

    if subject:
        sub_dir = search_root / f"sub-{subject}"
        if session:
            sub_dir = sub_dir / f"ses-{session}"
        if sub_dir.exists():
            return sorted(sub_dir.rglob("*_dwi.json"))

    return sorted(search_root.rglob("*_dwi.json"))


def _normalize_target_sidecars(
    search_root: Optional[Path] = None,
    json_paths: Optional[Sequence[Path]] = None,
) -> list[Path]:
    targets: list[Path] = []

    if json_paths:
        for path in json_paths:
            path = Path(path)
            if path.is_dir():
                targets.extend(sorted(path.rglob("*_dwi.json")))
            else:
                targets.append(path)
    elif search_root:
        targets = _target_dwi_sidecars(Path(search_root), {})

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in targets:
        resolved = Path(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def enrich_existing_dwi_sidecars_with_ge_gnl(
    dicom_dir: Path,
    *,
    search_root: Optional[Path] = None,
    json_paths: Optional[Sequence[Path]] = None,
    logger: Optional[logging.Logger] = None,
    strict: bool = False,
    fix_phase_encoding: bool = False,
    phase_from: str = "dir",
) -> dict[str, Any]:
    """
    Enrich existing DWI JSON sidecars with GE gradient nonlinearity metadata.

    Parameters
    ----------
    dicom_dir : Path
        Directory containing source GE DICOMs.
    search_root : Path, optional
        Root directory under which to find ``*_dwi.json`` sidecars.
    json_paths : sequence of Path, optional
        Explicit JSON sidecars or directories containing them.
    logger : logging.Logger, optional
        Logger to use for status messages.
    strict : bool
        If True, raise when any sidecar cannot be matched to a DICOM series.
    fix_phase_encoding : bool
        If True, rewrite AP/PA DWI PhaseEncodingDirection to BIDS ``j``/``j-``.
    phase_from : {"dir", "existing", "polarity"}
        Preferred source for choosing ``j`` vs ``j-`` when fixing phase encoding.
    """
    logger = logger or LOGGER
    dicom_dir = Path(dicom_dir)
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    sidecars = _normalize_target_sidecars(search_root=search_root, json_paths=json_paths)
    if not sidecars:
        raise FileNotFoundError("No DWI JSON sidecars found to enrich.")

    missing = [path for path in sidecars if not path.exists()]
    if missing:
        raise FileNotFoundError(f"JSON sidecar not found: {missing[0]}")

    series_meta, diagnostics = collect_ge_series_metadata(
        dicom_dir,
        logger=logger,
        return_diagnostics=True,
    )
    if not series_meta:
        raise ProcessingError(
            f"No GE DICOM series with usable PDB metadata found under {dicom_dir}. "
            f"{diagnostics['summary']}"
        )

    updated = 0
    unchanged = 0
    unmatched: list[str] = []

    for json_path in sidecars:
        sidecar = _load_json(json_path)
        match = _match_series(sidecar, series_meta)

        if match is None and len(series_meta) == 1:
            match = series_meta[0]

        if match is None:
            unmatched.append(str(json_path))
            continue

        if enrich_dwi_sidecar_with_ge_gnl(
            json_path,
            match,
            logger=logger,
            fix_phase_encoding=fix_phase_encoding,
            phase_from=phase_from,
        ):
            updated += 1
        else:
            unchanged += 1

    if strict and unmatched:
        raise ProcessingError(
            "GNL metadata could not be matched for sidecars: " + ", ".join(unmatched)
        )

    if unmatched:
        logger.warning(
            "GNL metadata could not be matched for sidecars: " + ", ".join(unmatched)
        )

    logger.info(
        f"GE GNL metadata enrichment complete: updated={updated}, unchanged={unchanged}, unmatched={len(unmatched)}"
    )
    return {
        "updated": updated,
        "unchanged": unchanged,
        "unmatched": unmatched,
        "series_found": len(series_meta),
        "sidecars_considered": len(sidecars),
    }


def patch_existing_dwi_sidecars_from_representatives(
    *,
    search_root: Optional[Path] = None,
    json_paths: Optional[Sequence[Path]] = None,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
    backup: bool = True,
    fix_phase_encoding: bool = True,
    phase_from: str = "dir",
    update_isocenter_offset: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Patch existing DWI JSON sidecars using stored representative DICOM/NIfTI paths.

    This is intended for sidecars that already contain
    ``GradientNonlinearityCorrection.Derivation.RepresentativeDicom`` and usually
    ``RepresentativeNifti``. It repairs GE PDB center metadata, optionally
    recomputes ``IsocenterOffsetScannerRASmm``, and fixes AP/PA phase encoding
    to BIDS ``j``/``j-``.
    """
    logger = logger or LOGGER
    sidecars = _normalize_target_sidecars(search_root=search_root, json_paths=json_paths)
    if not sidecars:
        raise FileNotFoundError("No DWI JSON sidecars found to patch.")

    updated = 0
    unchanged = 0
    failed: list[dict[str, str]] = []
    details: list[dict[str, Any]] = []

    for json_path in sidecars:
        try:
            payload = _load_json(json_path)
            original = json.dumps(payload, sort_keys=True)
            derivation = _ensure_nested_dict(payload, ["GradientNonlinearityCorrection", "Derivation"])

            representative_dicom_raw = derivation.get("RepresentativeDicom")
            if not isinstance(representative_dicom_raw, str) or not representative_dicom_raw:
                raise ValueError(
                    "Missing GradientNonlinearityCorrection.Derivation.RepresentativeDicom"
                )
            representative_dicom = Path(representative_dicom_raw).expanduser()
            if not representative_dicom.exists():
                raise FileNotFoundError(f"RepresentativeDicom not found: {representative_dicom}")

            center, parsing = _extract_ge_pdb_center_from_dicom(representative_dicom)
            derivation["PDBKeys"] = ["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"]
            derivation["PDBCenterScannerRASRelativeToIsocenterMm"] = _rounded_list(center)
            derivation["PDBCenterParsing"] = parsing

            gnc = _ensure_nested_dict(payload, ["GradientNonlinearityCorrection"])
            gnc.setdefault("Manufacturer", "GE MEDICAL SYSTEMS")
            gnc.setdefault("Method", "native_ge")
            gnc.setdefault("Source", "dicom_import")

            representative_nifti: Optional[Path] = None
            representative_nifti_raw = derivation.get("RepresentativeNifti")
            if isinstance(representative_nifti_raw, str) and representative_nifti_raw:
                representative_nifti = Path(representative_nifti_raw).expanduser()
            if representative_nifti is None:
                representative_nifti = _resolve_sidecar_nifti(json_path)

            if update_isocenter_offset:
                if representative_nifti is None:
                    raise ValueError(
                        "RepresentativeNifti missing and sidecar NIfTI could not be resolved"
                    )
                if not representative_nifti.exists():
                    raise FileNotFoundError(f"RepresentativeNifti not found: {representative_nifti}")
                gnc["IsocenterOffsetScannerRASmm"] = _rounded_list(
                    _derive_isocenter_offset_scanner_ras_mm(representative_nifti, center)
                )
                derivation["RepresentativeNifti"] = str(representative_nifti)
                derivation["NativeGeometryConvention"] = "make-L_ge_eval_frame"

            phase_changed = False
            if fix_phase_encoding:
                phase_changed = _apply_phase_encoding_fix(
                    payload,
                    json_path,
                    phase_from=phase_from,
                    strict=strict,
                )

            changed = json.dumps(payload, sort_keys=True) != original
            if changed:
                updated += 1
                if not dry_run:
                    if backup:
                        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
                        if not backup_path.exists():
                            shutil.copy2(json_path, backup_path)
                    _save_json(json_path, payload)
            else:
                unchanged += 1

            details.append(
                {
                    "json": str(json_path),
                    "changed": changed,
                    "phase_changed": phase_changed,
                    "phase": payload.get("PhaseEncodingDirection"),
                    "pdb_center": derivation.get("PDBCenterScannerRASRelativeToIsocenterMm"),
                    "isocenter_offset": gnc.get("IsocenterOffsetScannerRASmm"),
                }
            )
        except Exception as exc:
            failed.append({"json": str(json_path), "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("Failed to patch %s: %s", json_path, exc)
            if strict:
                raise

    return {
        "updated": updated,
        "unchanged": unchanged,
        "failed": failed,
        "sidecars_considered": len(sidecars),
        "dry_run": dry_run,
        "details": details,
    }


class GEGnlMetadataEnrichmentStep(BaseProcessingStep):
    """
    Post-conversion sidecar enrichment for GE gradient nonlinearity metadata.
    """

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def run(self, first_arg, output_dir: Path, **kwargs):
        context = first_arg if isinstance(first_arg, dict) else {}

        import_cfg = self.config.get("import", {})
        gnl_cfg = import_cfg.get("gnl_metadata", {})
        if not gnl_cfg.get("enabled", False):
            return context

        manufacturer = str(gnl_cfg.get("manufacturer", "GE") or "GE").upper()
        if manufacturer != "GE":
            self.logger.info(f"Skipping GNL sidecar enrichment for manufacturer={manufacturer}")
            return context

        dicom_dir = kwargs.get("dicom_dir") or context.get("dicom_dir")
        if not dicom_dir:
            self.logger.warning("GNL metadata enrichment skipped: no dicom_dir available")
            return context

        dicom_dir = Path(dicom_dir)
        if not dicom_dir.exists():
            self.logger.warning(f"GNL metadata enrichment skipped: dicom_dir not found: {dicom_dir}")
            return context

        search_root = Path(output_dir)
        if not search_root.exists():
            self.logger.warning(f"GNL metadata enrichment skipped: output_dir not found: {search_root}")
            return context

        dwi_sidecars = _target_dwi_sidecars(search_root, context)
        if not dwi_sidecars:
            self.logger.info("No DWI sidecars found for GNL metadata enrichment")
            return context

        result = enrich_existing_dwi_sidecars_with_ge_gnl(
            dicom_dir=dicom_dir,
            json_paths=dwi_sidecars,
            logger=self.logger,
            strict=False,
            fix_phase_encoding=bool(gnl_cfg.get("fix_phase_encoding", False)),
            phase_from=str(gnl_cfg.get("phase_from", "dir") or "dir"),
        )
        context["gnl_metadata_sidecars_updated"] = result["updated"]
        return context
