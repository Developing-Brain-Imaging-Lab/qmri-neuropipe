
from pathlib import Path
from typing import List, Tuple, Optional
import re, json
import pandas as pd
from functools import lru_cache


BIDS_ENTITY_REGEX = re.compile(r'([a-zA-Z0-9]+)-([^\_]+)')


def _sidecar(path: Path, new_ext: str) -> Path:
    """
    Replace .nii or .nii.gz with a different extension (e.g. .json, .bval, .bvec).
    """
    from ..core.utils import get_nifti_stem
    base = get_nifti_stem(path)
    return path.with_name(base + new_ext)

def _load_json_field(json_path: Path | None, field: str):
    if json_path is None or not json_path.exists():
        return None
    try:
        with json_path.open() as f:
            data = json.load(f)
        return data.get(field)
    except Exception:
        return None

def select_participants_sessions(bids_dir: Path, participants: List[str]|None, sessions: List[str]|None, skip_validation: bool=False) -> List[Tuple[str, str|None]]:
    out=[]
    for p in sorted(Path(bids_dir).glob("sub-*")):
        if not p.is_dir(): continue
        sub=p.name.split("-",1)[1]
        if participants and sub not in participants: continue
        ses_dirs=sorted(p.glob("ses-*"))
        if ses_dirs:
            for sd in ses_dirs:
                ses=sd.name.split("-",1)[1]
                if sessions and ses not in sessions: continue
                out.append((sub,ses))
        else:
            out.append((sub,None))

    return out

def build_bids_name(entities, suffix=None, extension=".nii.gz"):
    """
    Build a BIDS-like filename from entities dict.
    
    Parameters
    ----------
    entities : dict
        E.g. {"sub": "001", "ses": "01", "run": "1"}.
    suffix : str, optional
        If None, will use entities["suffix"] if present.
    extension : str, optional
        File extension, default '.nii.gz'.
    
    Returns
    -------
    str
        Filename like 'sub-001_ses-01_run-1_dwi.nii.gz'.
    """
    # If suffix isn’t passed, fall back to entities
    if suffix is None:
        suffix = entities.get("suffix")
        if suffix is None:
            raise ValueError("suffix must be provided or present in entities['suffix'].")

    parts = []

    # Standard BIDS order (you can adjust as needed)
    # Standard BIDS order (you can adjust as needed)
    for key in ("sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "space", "model", "desc", "echo", "flip", "inv", "mt", "chunk"):
        value = entities.get(key)
        if value:
            parts.append(f"{key}-{value}")

    parts.append(suffix)

    return "_".join(parts) + extension

def build_from_parsed(parsed, root=None, override=None):
    """
    Rebuild a BIDS filename/path from a parsed BIDS dict.
    
    Parameters
    ----------
    parsed : dict
        Output from bids_find() / parse_bids_filename()
    root : str or Path, optional
        New base directory for output paths
    override : dict, optional
        Any entities you want to overwrite (e.g., run='2', session='03')

    Returns
    -------
    Path or str
    """
    # Separate entities from metadata
    entities = {k: parsed.get(k) for k in [
        "sub","ses","task","acq","ce","rec","dir",
        "echo","flip","inv","mt","run","chunk"
    ] if parsed.get(k) is not None}

    # apply overrides
    if override:
        entities.update(override)

    return build_bids_name(
        root=root,
        suffix=parsed["suffix"],
        extension=parsed["extension"],
        **entities
    )

@lru_cache(maxsize=1024)
def parse_bids_filename(path):
    """
    Extract BIDS entities from a filename, returning a dict.
    
    Cached version for better performance when parsing the same files repeatedly.
    """
    path = Path(path)
    name = path.name
    
    entities = {k: None for k in [
        "sub", "ses", "task", "acq", "ce", "rec", "dir", "run",
        "echo", "flip", "inv", "mt", "chunk"
    ]}
    
    matches = BIDS_ENTITY_REGEX.findall(name)
    
    for key, value in matches:
        if key in entities:
            entities[key] = value
    
    # Add suffix and extension
    # e.g. sub-01_ses-02_acq-mb4_dwi.nii.gz → suffix=dwi, ext=.nii.gz
    suffix = name.split("_")[-1].split(".")[0]
    extension = "".join(path.suffixes)
    
    entities["suffix"] = suffix
    entities["extension"] = extension
    entities["path"] = path
    
    return entities

extract_bids_entities = parse_bids_filename
get_entities_from_path = parse_bids_filename

def bids_find(root, suffix=None, extension=None):
    """
    Recursively find BIDS-like files under root.
    
    Optimized version with targeted glob patterns for better performance.

    Returns
    -------
    list of dict
        Parsed BIDS entities + 'path' and 'extension' for each file.
    """
    root = Path(root)
    results = []
    
    # OPTIMIZATION: Use targeted patterns to avoid walking entire tree
    # Build glob pattern based on extension filter
    if extension:
        # Convert to list if single extension
        exts = [extension] if isinstance(extension, str) else extension
        patterns = []
        
        # Create specific patterns for each extension
        for ext in exts:
            if ext in ('.nii', '.nii.gz'):
                patterns.extend(['**/*.nii', '**/*.nii.gz'])
            elif ext == '.json':
                patterns.append('**/*.json')
            elif ext in ('.bval', '.bvec'):
                patterns.extend(['**/*.bval', '**/*.bvec'])
            else:
                patterns.append(f'**/*{ext}')
        
        # Remove duplicates
        patterns = list(set(patterns))
    else:
        # Default: search common BIDS extensions only (much faster than "*")
        patterns = ['**/*.nii', '**/*.nii.gz', '**/*.json', '**/*.bval', '**/*.bvec']
    
    # Collect all matching paths
    paths_to_process = set()
    for pattern in patterns:
        paths_to_process.update(root.rglob(pattern))
    
    # Process paths
    for path in paths_to_process:
        if not path.is_file():
            continue

        # Get full extension
        full_ext = "".join(path.suffixes)
        
        # Extension filter (already mostly filtered by glob, but double-check)
        if extension and full_ext not in (extension if isinstance(extension, (list, tuple)) else [extension]):
            continue

        ent = parse_bids_filename(path)

        # Suffix filter
        if suffix and ent.get("suffix") != suffix:
            continue

        # Make sure we always have these fields:
        ent.setdefault("path", path)
        ent.setdefault("extension", full_ext)

        results.append(ent)

    return results

def bids_find_sorted(root, suffix=None, extension=None):
    """
    Wrapper around bids_find that returns a sorted list of entities.
    Sort order: sub, ses, run, suffix, path.
    """
    files = bids_find(root, suffix=suffix, extension=extension)

    def sort_key(ent):
        return (
            ent.get("sub", ""),
            ent.get("ses", ""),
            ent.get("run", ""),
            ent.get("suffix", ""),
            str(ent.get("path", "")),
        )

    return sorted(files, key=sort_key)

def bids_to_dataframe(entries):
    """
    Convert parsed BIDS entries into a DataFrame for easy inspection.
    """
    return pd.DataFrame(entries)

def bids_collect_series(root, suffix=None):
    """
    Collect BIDS series (NIfTI + JSON + optional bval/bvec) under root.
    
    Optimized version with better filtering and grouping logic.

    Parameters
    ----------
    root : str or Path
        BIDS root.
    suffix : str, optional
        Filter by suffix (e.g. 'dwi', 'T1w', 'bold'). If None, all.

    Returns
    -------
    list of dict
        Each dict has:
            - 'entities': dict of BIDS entities (sub, ses, run, etc.)
            - 'nii'     : Path or None
            - 'json'    : Path or None
            - 'bval'    : Path or None
            - 'bvec'    : Path or None
    """
    # OPTIMIZATION: Only search for relevant extensions upfront
    # This is much faster than searching for everything
    if suffix:
        # If suffix specified, we can be more targeted
        # Look for NIfTI files first, then find sidecars
        nii_files = bids_find(root, suffix=suffix, extension=['.nii', '.nii.gz'])
        
        # Group by canonical key and pre-populate with NIfTI
        groups = {}
        for ent in nii_files:
            key = (
                ent.get("sub"),
                ent.get("ses"),
                ent.get("task"),
                ent.get("acq"),
                ent.get("dir"),
                ent.get("run"),
                ent.get("suffix"),
                ent.get("space"),
                ent.get("desc"),
            )
            
            # Create entities dict (exclude path/extension)
            entities = {
                k: v for k, v in ent.items()
                if k not in ("path", "extension")
            }
            
            nii_path = ent["path"]
            
            groups[key] = {
                "entities": entities,
                "nii": nii_path,
                "json": None,
                "bval": None,
                "bvec": None,
            }
            
            # Check for sidecars using _sidecar helper
            json_path = _sidecar(nii_path, '.json')
            if json_path.exists():
                groups[key]["json"] = json_path
            
            # For DWI, check for bval/bvec
            if ent.get("suffix") == "dwi":
                bval_path = _sidecar(nii_path, '.bval')
                bvec_path = _sidecar(nii_path, '.bvec')
                if bval_path.exists():
                    groups[key]["bval"] = bval_path
                if bvec_path.exists():
                    groups[key]["bvec"] = bvec_path
    else:
        # No suffix filter - use old logic but optimized
        files = bids_find(root, suffix=None, extension=['.nii', '.nii.gz', '.json', '.bval', '.bvec'])
        
        groups = {}
        
        for ent in files:
            key = (
                ent.get("sub"),
                ent.get("ses"),
                ent.get("task"),
                ent.get("acq"),
                ent.get("dir"),
                ent.get("run"),
                ent.get("suffix"),
                ent.get("space"),
                ent.get("desc"),
            )

            rec = groups.setdefault(
                key,
                {
                    "entities": {
                        k: v
                        for k, v in ent.items()
                        if k not in ("path", "extension")
                    },
                    "nii": None,
                    "json": None,
                    "bval": None,
                    "bvec": None,
                },
            )

            ext = ent["extension"]
            path = ent["path"]

            if ext in (".nii", ".nii.gz"):
                rec["nii"] = path
            elif ext == ".json":
                rec["json"] = path
            elif ext == ".bval":
                rec["bval"] = path
            elif ext == ".bvec":
                rec["bvec"] = path

    # Turn into a sorted list
    def sort_key(item):
        ents = item["entities"]
        return (
            ents.get("sub", ""),
            ents.get("ses", ""),
            ents.get("run", ""),
            ents.get("suffix", ""),
        )

    return sorted(groups.values(), key=sort_key)
