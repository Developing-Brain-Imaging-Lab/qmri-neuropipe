from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Iterable
import json

from qmri_neuropipe.core.types import ImageFile
from ..bids import build_bids_name, bids_find, _load_json_field, _sidecar  # already in your skeleton


def bids_find_fmap(root) -> list[ImageFile]:
    """
    Find Anatomical T1w NIfTI images and their associated JSON files.

    Uses the existing `bids_find(root, suffix='t1w', extension='.nii.gz')`.
    """
    fmap_ents = bids_find(root, suffix="fmap", extension=".nii.gz")
    results: list[ImageFile] = []

    for ent in fmap_ents:
        img = ent["path"]
        json_path = _sidecar(img, ".json")

        fmap = ImageFile(entities=ent,
                         img=img,
                         json=json_path if json_path.exists() else None,
        )
        results.append(fmap)

    return results
    