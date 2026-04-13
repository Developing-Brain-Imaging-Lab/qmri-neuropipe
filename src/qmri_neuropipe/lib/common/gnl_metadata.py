from __future__ import annotations

import gzip
import json
import logging
import re
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


def _parse_letter_value(token: str) -> float:
    token = token.strip().strip('"')
    match = re.match(r'^([LRAPSI])\s*([-+]?\d+(?:\.\d+)?)$', token)
    if not match:
        raise ValueError(f"Unsupported GE position token: {token!r}")
    letter, value = match.group(1), float(match.group(2))
    return {
        "R": +value,
        "L": -value,
        "A": +value,
        "P": -value,
        "S": +value,
        "I": -value,
    }[letter]


def _pdb_center_ras_rel_iso(kv: dict[str, str]) -> list[float]:
    required = ["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"]
    missing = [key for key in required if key not in kv]
    if missing:
        raise ValueError(f"Missing GE PDB keys: {', '.join(missing)}")

    x0 = _parse_letter_value(kv["SLOC1"])
    x1 = _parse_letter_value(kv["ELOC1"])
    xc = 0.5 * (x0 + x1)
    y = _parse_letter_value(kv["FOVCNT1"])
    z = _parse_letter_value(kv["FOVCNT2"])
    return [float(xc), float(y), float(z)]


def extract_ge_series_metadata(dicom_path: Path) -> GeDicomSeriesMetadata:
    ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
    manufacturer = str(getattr(ds, "Manufacturer", "") or "").strip()
    if "GE" not in manufacturer.upper():
        raise ValueError(f"Not a GE DICOM: {manufacturer or 'unknown manufacturer'}")

    kv = _kv_from_pdb(_extract_pdb_text(ds))
    offset = _pdb_center_ras_rel_iso(kv)

    return GeDicomSeriesMetadata(
        dicom_path=dicom_path,
        series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "") or "") or None,
        series_number=int(getattr(ds, "SeriesNumber")) if getattr(ds, "SeriesNumber", None) is not None else None,
        series_description=str(getattr(ds, "SeriesDescription", "") or "") or None,
        protocol_name=str(getattr(ds, "ProtocolName", "") or "") or None,
        manufacturer=manufacturer or None,
        pdb_center_scanner_ras_rel_iso_mm=offset,
        derivation_keys=["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"],
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


def enrich_dwi_sidecar_with_ge_gnl(json_path: Path, series_meta: GeDicomSeriesMetadata, logger: Optional[logging.Logger] = None) -> bool:
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
            "PDBCenterScannerRASRelativeToIsocenterMm": series_meta.pdb_center_scanner_ras_rel_iso_mm,
            "RepresentativeDicom": str(series_meta.dicom_path),
            "RepresentativeNifti": str(nifti_path) if nifti_path else None,
            "NativeGeometryConvention": "make-L_ge_eval_frame",
            "SeriesInstanceUID": series_meta.series_instance_uid,
            "SeriesNumber": series_meta.series_number,
            "SeriesDescription": series_meta.series_description,
            "ProtocolName": series_meta.protocol_name,
        },
    }

    if gnl_meta == new_block:
        logger.debug(f"GNL metadata already present in {json_path.name}")
        return False

    payload["GradientNonlinearityCorrection"] = new_block
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

        if enrich_dwi_sidecar_with_ge_gnl(json_path, match, logger=logger):
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
        )
        context["gnl_metadata_sidecars_updated"] = result["updated"]
        return context
