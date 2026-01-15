
from pathlib import Path
from typing import List
from ...core.types import ImageFile
from ..bids import BIDS_ENTITY_REGEX, _load_json_field, _sidecar, build_from_parsed

def bids_find_relax(search_dir: Path) -> List[ImageFile]:
    """
    Find Relaxometry (VFA/DESPOT) files in valid BIDS subdirectories.
    Looks for SPGR, SSFP, IR-SPGR, B1 maps in 'anat' (and 'fmap').
    """
    files = []
    
    # Files to look for:
    # Entities: acq-spgr, acq-ssfp, acq-afi, acq-irspgr
    # Suffixes: _VFA, _T1w, _MP2RAGE?
    # User specified: suffixes `_VFA.nii.gz` and tags.
    
    candidates = []
    # Search anat and fmap
    for dtype in ['anat', 'fmap']:
        dpath = search_dir / dtype
        if dpath.exists():
            candidates.extend(dpath.glob("*.nii.gz"))
            
    for f in candidates:
        name = f.name
        # Parse entities
        # Use simple regex or assume standard BIDS name structure
        # We can reuse ..io.bids.parse_bids_filename if available, or just regex
        
        # Simple entity parsing
        entities = {}
        for match in BIDS_ENTITY_REGEX.finditer(name):
             entities[match.group(1)] = match.group(2)
             
        # Determine suffix
        # ending after last _?
        if "_" in name:
             suffix_part = name.split("_")[-1]
             entities['suffix'] = suffix_part.replace(".nii.gz", "").replace(".nii", "")
        
        # Filter for Relaxometry relevance
        # 1. explicit acq tags
        acq = entities.get('acq', '').lower()
        desc = entities.get('desc', '').lower()
        suffix = entities.get('suffix', '').lower()
        
        is_relax = False
        if 'spgr' in acq or 'ssfp' in acq or 'irspgr' in acq or 'afi' in acq:
             is_relax = True
        elif 'spgr' in desc or 'ssfp' in desc: # Non-standard but possible
             is_relax = True
        elif 'vfa' in suffix or 'vfa' in name.lower():
             is_relax = True
        elif 'b1' in suffix or 'b1' in acq:
             is_relax = True
             
        if is_relax:
             # Load Sidecar
             json_file = _sidecar(f, ".json")
             json_data = _load_json_field(json_file, None) if json_file.exists() else {} # Just load whole dict? 
             # wait _load_json_field loads a field.
             # We need whole dict for ImageFile usually? ImageFile.json is Dict.
             
             import json
             jdata = {}
             if json_file.exists():
                  try:
                       with open(json_file) as jf: jdata = json.load(jf)
                  except: pass
             
             files.append(ImageFile(img=f, entities=entities, json=jdata, bids_dir=search_dir.parent))
             
    return files
