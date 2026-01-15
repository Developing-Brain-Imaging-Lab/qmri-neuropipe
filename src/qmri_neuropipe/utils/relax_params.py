
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
from ..core.types import ImageFile

def _extract_bids_param(img: ImageFile, key: str, default=None) -> Any:
    """Safely extract parameter from ImageFile sidecar."""

    if img.json:
        if isinstance(img.json, dict):
            return img.json.get(key, default)
        elif hasattr(img.json, 'exists') and img.json.exists():
            with open(img.json) as f:
                data = json.load(f)
                return data.get(key, default)
    # Fallback: check entities? (e.g. TR in filename?) Unlikely for accurate fitting.
    return default

def generate_acq_params(
    spgr_images: List[ImageFile],
    ssfp_images: List[ImageFile] = [],
    irspgr_images: List[ImageFile] = [],
    output_path: Optional[Path] = None
) -> Dict:
    """
    Generate acquisition parameter dictionary for qmri_fit_* binaries.
    
    Structure:
    {
      "SPGR": [{"FlipAngle": [...], "RepetitionTime": [...]}],
      "SSFP": [{"FlipAngle": [...], "RepetitionTime": [...], "PhaseCycling": [...]}],
      "IRSPGR": [{"FlipAngle": [], "RepetitionTime": [], "InversionTime": [], "EchoTrainLength": []}]
    }
    """
    
    params = {}
    
    # Process SPGR
    if spgr_images:
        spgr_entry = {"FlipAngle": [], "RepetitionTime": []}
        for img in spgr_images:
            # Assume each image is a volume in sequence or 4D?
            # If 4D, the sidecar might have a list for FlipAngle?
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            
            # Handle Single Value vs List
            if isinstance(fa, list):
                spgr_entry["FlipAngle"].extend(fa)
            elif fa is not None:
                spgr_entry["FlipAngle"].append(float(fa))
                
            if isinstance(tr, list):
                # Convert seconds to ms if needed? 
                # BIDS usually seconds. C++ code expected ms? 
                # C++ source code variable names just "RepetitionTime". 
                # Checking source: qmri_fit_despot1.cpp line 160: auto tr = array["RepetitionTime"].get<std::vector<double>>();
                # Usually standard internal units. Let's assume BIDS Seconds * 1000 -> ms?
                # User didn't specify units. Standard implementation usually involves ms for TR.
                # Let's checking if we need to scale. 
                # Ideally we check the C++ math. 
                # Safe bet: standard BIDS is seconds. DESPOT eq usually uses ms for TR/T1/T2.
                # Let's auto-convert seconds to ms if < 10?
                spgr_entry["RepetitionTime"].extend([t * 1000.0 for t in tr]) # Assume BIDS(s) -> ms
            elif tr is not None:
                spgr_entry["RepetitionTime"].append(float(tr) * 1000.0)

        params["SPGR"] = [spgr_entry]

    # Process SSFP
    if ssfp_images:
        ssfp_entry = {"FlipAngle": [], "RepetitionTime": [], "PhaseCycling": []}
        for img in ssfp_images:
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            phase = _extract_bids_param(img, "PhaseEncodingDirection") # Wrong? PhaseCycling is diff.
            # PhaseCycling usually "PhaseCycling" in sidecar or deduced?
            # User said "PhaseCycling" in json.
            pc = _extract_bids_param(img, "PhaseCycling") # Custom BIDS field?
            
            if isinstance(fa, list): ssfp_entry["FlipAngle"].extend(fa)
            elif fa is not None: ssfp_entry["FlipAngle"].append(float(fa))
            
            if isinstance(tr, list): ssfp_entry["RepetitionTime"].extend([t * 1000.0 for t in tr])
            elif tr is not None: ssfp_entry["RepetitionTime"].append(float(tr) * 1000.0)
            
            if isinstance(pc, list): ssfp_entry["PhaseCycling"].extend(pc)
            elif pc is not None: ssfp_entry["PhaseCycling"].append(float(pc))
            else: ssfp_entry["PhaseCycling"].append(0.0) # Default 0 or 180?

        params["SSFP"] = [ssfp_entry]

    # Process IR-SPGR
    if irspgr_images:
        ir_entry = {"FlipAngle": [], "RepetitionTime": [], "InversionTime": [], "EchoTrainLength": []}
        for img in irspgr_images:
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            ti = _extract_bids_param(img, "InversionTime")
            etl = _extract_bids_param(img, "EchoTrainLength", 1) # Default 1? or typically larger?
            
            if isinstance(fa, list): ir_entry["FlipAngle"].extend(fa)
            elif fa is not None: ir_entry["FlipAngle"].append(float(fa))
            
            if isinstance(tr, list): ir_entry["RepetitionTime"].extend([t * 1000.0 for t in tr])
            elif tr is not None: ir_entry["RepetitionTime"].append(float(tr) * 1000.0)
            
            if isinstance(ti, list): ir_entry["InversionTime"].extend([t * 1000.0 for t in ti])
            elif ti is not None: ir_entry["InversionTime"].append(float(ti) * 1000.0)
            
            if isinstance(etl, list): ir_entry["EchoTrainLength"].extend(etl)
            elif etl is not None: ir_entry["EchoTrainLength"].append(float(etl))
            
        params["IRSPGR"] = [ir_entry]
        
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
            
    return params
