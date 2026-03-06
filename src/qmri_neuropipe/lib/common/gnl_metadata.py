from __future__ import annotations

import gzip
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    isocenter_offset_scanner_ras_mm: list[float]
    derivation_keys: list[str]


def _extract_pdb_text(ds: pydicom.dataset.FileDataset) -> str:
    if PDB_TAG not in ds:
        raise ValueError(f"GE PDB private tag missing: {PDB_TAG}")

    pdb = ds[PDB_TAG].value
    if isinstance(pdb, str):
        return pdb
    if not isinstance(pdb, (bytes, bytearray)):
        raise ValueError("Unsupported GE PDB payload type")

    idx = bytes(pdb).find(b"\x1f\x8b")
    if idx < 0:
        raise ValueError("GE PDB payload does not contain a gzip stream")
    return gzip.decompress(bytes(pdb)[idx:]).decode("latin1", errors="replace")


def _kv_from_pdb(txt: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in txt.splitlines():
        line = line.strip()
        match = re.match(r'^([A-Z0-9_]+)\s+"(.*)"\s*$', line)
        if match:
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
        isocenter_offset_scanner_ras_mm=offset,
        derivation_keys=["SLOC1", "ELOC1", "FOVCNT1", "FOVCNT2"],
    )


def _iter_dicom_files(dicom_dir: Path) -> list[Path]:
    return sorted(path for path in dicom_dir.rglob("*") if path.is_file())


def collect_ge_series_metadata(dicom_dir: Path, logger: Optional[logging.Logger] = None) -> list[GeDicomSeriesMetadata]:
    logger = logger or LOGGER
    series_map: dict[str, GeDicomSeriesMetadata] = {}

    for path in _iter_dicom_files(dicom_dir):
        try:
            meta = extract_ge_series_metadata(path)
        except (InvalidDicomError, IsADirectoryError):
            continue
        except Exception as exc:
            logger.debug(f"Skipping DICOM candidate {path}: {exc}")
            continue

        key = meta.series_instance_uid or f"series-number:{meta.series_number}"
        if key not in series_map:
            series_map[key] = meta

    return list(series_map.values())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


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

    new_block = {
        "Manufacturer": series_meta.manufacturer or "GE",
        "Method": "native_ge",
        "Source": "dicom_import",
        "IsocenterOffsetScannerRASmm": series_meta.isocenter_offset_scanner_ras_mm,
        "Derivation": {
            "PDBKeys": series_meta.derivation_keys,
            "RepresentativeDicom": str(series_meta.dicom_path),
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

        dwi_sidecars = sorted(search_root.rglob("*_dwi.json"))
        if not dwi_sidecars:
            self.logger.info("No DWI sidecars found for GNL metadata enrichment")
            return context

        series_meta = collect_ge_series_metadata(dicom_dir, logger=self.logger)
        if not series_meta:
            self.logger.warning(f"No GE DICOM series with usable PDB metadata found under {dicom_dir}")
            return context

        updated = 0
        unmatched: list[str] = []

        for json_path in dwi_sidecars:
            sidecar = _load_json(json_path)
            match = _match_series(sidecar, series_meta)

            if match is None and len(series_meta) == 1:
                match = series_meta[0]

            if match is None:
                unmatched.append(json_path.name)
                continue

            try:
                if enrich_dwi_sidecar_with_ge_gnl(json_path, match, logger=self.logger):
                    updated += 1
            except Exception as exc:
                raise ProcessingError(f"Failed to update GNL metadata in {json_path}: {exc}") from exc

        if unmatched:
            self.logger.warning(
                "GNL metadata could not be matched for sidecars: "
                + ", ".join(unmatched)
            )

        self.logger.info(f"Updated GE GNL metadata in {updated} DWI sidecar(s)")
        context["gnl_metadata_sidecars_updated"] = updated
        return context
