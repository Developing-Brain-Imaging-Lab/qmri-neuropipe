from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional


def normalize_transform_chain(transform: Optional[dict | list[dict]]) -> list[dict]:
    if transform is None:
        return []
    if isinstance(transform, list):
        return [deepcopy(item) for item in transform if isinstance(item, dict)]
    if isinstance(transform, dict):
        return [deepcopy(transform)]
    return []


def append_transform(chain: Optional[dict | list[dict]], transform: Optional[dict]) -> list[dict]:
    out = normalize_transform_chain(chain)
    if isinstance(transform, dict):
        out.append(deepcopy(transform))
    return out


def latest_usable_linear_transform(chain: Optional[dict | list[dict]]) -> Optional[dict]:
    for item in reversed(normalize_transform_chain(chain)):
        if item.get("usable_for_gnl_mapping") and item.get("type") == "linear":
            return item
    return None


def write_transform_chain_to_sidecar(json_path: Optional[Path], chain: Optional[dict | list[dict]]) -> None:
    if not json_path:
        return
    json_path = Path(json_path)
    if not json_path.exists():
        return
    payload = json.loads(json_path.read_text())
    payload["SpatialTransformChain"] = normalize_transform_chain(chain)
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
