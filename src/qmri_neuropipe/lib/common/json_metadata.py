from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


CRITICAL_METADATA_KEYS = [
    "GradientNonlinearityCorrection",
    "SpatialTransformChain",
]


def copy_json_with_metadata(src_json: Optional[Path], dst_json: Path) -> Optional[Path]:
    if not src_json or not Path(src_json).exists():
        return None

    src_json = Path(src_json)
    src_payload = json.loads(src_json.read_text())

    if dst_json.exists():
        try:
            dst_payload = json.loads(dst_json.read_text())
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
