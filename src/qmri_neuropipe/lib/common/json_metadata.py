from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .spatial_transforms import normalize_transform_chain


CRITICAL_METADATA_KEYS = [
    "GradientNonlinearityCorrection",
    "SpatialTransformChain",
]


def sanitize_metadata_payload(payload: dict) -> dict:
    sanitized = dict(payload)
    if "SpatialTransformChain" in sanitized:
        sanitized["SpatialTransformChain"] = normalize_transform_chain(
            sanitized.get("SpatialTransformChain")
        )
    return sanitized


def write_sanitized_json_copy(src_json: Optional[Path], dst_json: Path) -> Optional[Path]:
    if not src_json or not Path(src_json).exists():
        return None

    src_payload = sanitize_metadata_payload(json.loads(Path(src_json).read_text()))
    with Path(dst_json).open("w") as f:
        json.dump(src_payload, f, indent=2)
        f.write("\n")
    return dst_json


def copy_json_with_metadata(src_json: Optional[Path], dst_json: Path) -> Optional[Path]:
    if not src_json or not Path(src_json).exists():
        return None

    src_json = Path(src_json)
    src_payload = sanitize_metadata_payload(json.loads(src_json.read_text()))

    if dst_json.exists():
        try:
            dst_payload = sanitize_metadata_payload(json.loads(dst_json.read_text()))
        except Exception:
            dst_payload = {}
    else:
        dst_payload = {}

    merged = dst_payload.copy()
    for key, value in src_payload.items():
        if key not in merged:
            merged[key] = value

    for key in CRITICAL_METADATA_KEYS:
        if key in src_payload:
            merged[key] = src_payload[key]

    with dst_json.open("w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    return dst_json
