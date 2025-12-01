from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Iterable
import json

from qmri_neuropipe.core.types import ImageFile, DWIFile
from ..bids import build_bids_name, bids_find, _load_json_field, _sidecar  # already in your skeleton

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


def bids_find_dwi(root) -> list[DWIFile]:
    """
    Find DWI NIfTI images and their associated JSON/BVAL/BVEC files.

    Uses the existing `bids_find(root, suffix='dwi', extension='.nii.gz')`.
    """
    dwi_ents = bids_find(root, suffix="dwi", extension=".nii.gz")
    results: list[DWIFile] = []

    for ent in dwi_ents:
        img = ent["path"]
        json_path = _sidecar(img, ".json")
        bval_path = _sidecar(img, ".bval")
        bvec_path = _sidecar(img, ".bvec")

        dwi = DWIFile(
            entities=ent,
            img=img,
            json=json_path if json_path.exists() else None,
            bval=bval_path if bval_path.exists() else None,
            bvec=bvec_path if bvec_path.exists() else None,
        )
        results.append(dwi)

    return results
    
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

def build_acqp_index(json_path: Path | None, dwi_path: Path) -> Tuple[Path | None, Path | None]:
    """
    Build minimal FSL acqp.txt and index.txt from BIDS JSON.
    This is a stub that writes a single PE dir/ro time line if JSON is present.
    Expand for multi-PE or TOPUP use.
    """
    if not json_path or not json_path.exists():
        return None, None

    meta = json.loads(json_path.read_text())
    pe = meta.get("PhaseEncodingDirection", "j")   # fallback
    trt = meta.get("TotalReadoutTime", 0.05)       # seconds

    acqp_dir = dwi_path.parent / "eddy"
    acqp_dir.mkdir(parents=True, exist_ok=True)
    acqp = acqp_dir / "acqp.txt"
    index = acqp_dir / "index.txt"

    # Map BIDS PE to FSL acqp (i/j/k with +/-). Using j as default example:
    # Columns are: dx dy dz readout_time
    line = {
        "i":  "1 0 0",
        "i-": "-1 0 0",
        "j":  "0 1 0",
        "j-": "0 -1 0",
        "k":  "0 0 1",
        "k-": "0 0 -1",
    }.get(pe, "0 1 0")

    acqp.write_text(f"{line} {trt:.6f}\n", encoding="utf-8")
    # Single line index for all volumes (adjust for multi-acqp scenarios)
    # Count volumes by reading bvals:
    nvols = len([x for x in Path(dwi_path.parent).glob("*_dwi.bval")][0].read_text().split())
    index.write_text(("1\n" * nvols), encoding="utf-8")

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
        ped_raw = _load_json_field(dwi.json, "PhaseEncodingDirection")
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


    """
    Group DWI files that appear to be reversed phase encoding pairs for TOPUP/EDDY.

    Strategy:
    - Group by "matching" BIDS entities except for 'dir' and 'run'.
    - Within each group, pair files that have opposite PhaseEncodingDirection.
    - Return a list of groups; each group is a list of DWIFile instances.
    """
    dwi_files = list(dwi_files)
    buckets: Dict[tuple, list[DWIFile]] = defaultdict(list)

    # Define which entities we *ignore* for pairing
    ignore_keys = {"dir", "run"}

    for d in dwi_files:
        # Build a key from entities that should match across phase-encoding pairs
        key_items = []
        for k, v in sorted(d.entities.items()):
            if k in ignore_keys:
                continue
            key_items.append((k, v))
        key = tuple(key_items)
        buckets[key].append(d)

    groups: list[list[DWIFile]] = []

    for key, files in buckets.items():
        used = set()
        n = len(files)

        for i in range(n):
            if i in used:
                continue

            pe_i = _load_json_field(files[i].json, "PhaseEncodingDirection")
            if pe_i is None:
                # No PE info; treat as its own group
                groups.append([files[i]])
                used.add(i)
                continue

            # Try to find a reversed partner
            pair = [files[i]]
            for j in range(i + 1, n):
                if j in used:
                    continue
                pe_j = _load_json_field(files[j].json, "PhaseEncodingDirection")
                if _is_pe_reversed(pe_i, pe_j):
                    pair.append(files[j])
                    used.add(j)

            used.add(i)
            groups.append(pair)

    return groups