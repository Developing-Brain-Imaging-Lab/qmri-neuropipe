from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Iterable
import json

from qmri_neuropipe.core.types import ImageFile
from ..bids import build_bids_name, bids_find, _load_json_field, _sidecar  # already in your skeleton


def bids_find_t1w(root) -> list[ImageFile]:
    """
    Find Anatomical T1w NIfTI images and their associated JSON files.

    Uses the existing `bids_find(root, suffix='t1w', extension='.nii.gz')`.
    """
    t1w_ents = bids_find(root, suffix="T1w", extension=".nii.gz")
    results: list[ImageFile] = []

    for ent in t1w_ents:
        img = ent["path"]
        json_path = _sidecar(img, ".json")

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
    