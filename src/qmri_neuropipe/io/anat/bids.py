from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Iterable
import json
import logging

from qmri_neuropipe.core.types import ImageFile
from ..bids import build_bids_name, bids_find, _load_json_field, _sidecar  # already in your skeleton


def _normalized_selector_value(value):
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Path):
        return str(value).strip().lower()
    if isinstance(value, list):
        return [_normalized_selector_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalized_selector_value(item) for item in value)
    return value


def _selector_values_match(found, expected) -> bool:
    return _normalized_selector_value(found) == _normalized_selector_value(expected)


def _load_sidecar_payload(path: Optional[Path]) -> dict:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open() as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _anat_match(image: ImageFile, selector: dict) -> bool:
    if not isinstance(selector, dict) or not selector:
        return True

    bids_name = selector.get("bids_name")
    if bids_name and not _selector_values_match(image.img.stem.replace(".nii", ""), bids_name):
        return False

    entities = selector.get("entities")
    if isinstance(entities, dict):
        for key, expected in entities.items():
            if not _selector_values_match(image.entities.get(key), expected):
                return False

    json_fields = selector.get("json_fields")
    if isinstance(json_fields, dict):
        payload = _load_sidecar_payload(image.json)
        for key, expected in json_fields.items():
            if not _selector_values_match(payload.get(key), expected):
                return False

    return True


def select_anatomical_candidates(
    candidates: list[ImageFile],
    selector: Optional[dict],
    modality: str,
    logger: Optional[logging.Logger] = None,
) -> list[ImageFile]:
    if not selector:
        return candidates

    matches = [candidate for candidate in candidates if _anat_match(candidate, selector)]
    if len(matches) == 1:
        if logger:
            logger.info("Selected %s using configured anatomical selector: %s", modality, matches[0].img.name)
        return matches

    if not matches:
        raise RuntimeError(
            f"Anatomical selector for {modality} did not match any files. "
            f"Selector={selector} Candidates={[c.img.name for c in candidates]}"
        )

    raise RuntimeError(
        f"Anatomical selector for {modality} matched multiple files. "
        f"Selector={selector} Matches={[c.img.name for c in matches]}"
    )


def bids_find_t1w(root) -> list[ImageFile]:
    """
    Find anatomical T1w-like NIfTI images and their associated JSON files.

    BIDS MP2RAGE ``UNIT1`` images are valid T1-weighted anatomical inputs. They
    are normalized to the internal ``T1w`` suffix here because the downstream
    workflow uses that suffix to select T1-specific processing and derivative
    naming. The original filename and JSON sidecar remain unchanged.
    """
    t1w_ents = bids_find(root, suffix="T1w", extension=".nii.gz")
    unit1_ents = bids_find(root, suffix="UNIT1", extension=".nii.gz")
    results: list[ImageFile] = []

    for ent in sorted(t1w_ents + unit1_ents, key=lambda item: str(item["path"])):
        ent = dict(ent)
        img = ent["path"]
        json_path = _sidecar(img, ".json")
        ent["suffix"] = "T1w"

        t1w = ImageFile(entities=ent,
                        img=img,
                        json=json_path if json_path.exists() else None,
        )
        results.append(t1w)

    return results
    
def bids_find_t2w(root) -> list[ImageFile]:
    """
    Find Anatomical T1w NIfTI images and their associated JSON files.

    Uses the existing `bids_find(root, suffix='t1w', extension='.nii.gz')`.
    """
    t2w_ents = bids_find(root, suffix="T2w", extension=".nii.gz")
    results: list[ImageFile] = []

    for ent in t2w_ents:
        img = ent["path"]
        json_path = _sidecar(img, ".json")

        t2w = ImageFile(entities=ent,
                        img=img,
                        json=json_path if json_path.exists() else None,
        )
        results.append(t2w)

    return results


def bids_find_other_anat(root) -> list[ImageFile]:
    """Find raw anatomical NIfTIs other than images already treated as T1w/T2w."""
    excluded_suffixes = {
        "T1w", "UNIT1", "T2w",
        "mask", "dseg", "probseg", "seg", "xfm",
    }
    results: list[ImageFile] = []
    for ent in bids_find(root, extension=(".nii", ".nii.gz")):
        if ent.get("suffix") in excluded_suffixes:
            continue
        img = ent["path"]
        json_path = _sidecar(img, ".json")
        results.append(
            ImageFile(
                entities=ent,
                img=img,
                json=json_path if json_path.exists() else None,
            )
        )
    return sorted(results, key=lambda image: str(image.img))
    
