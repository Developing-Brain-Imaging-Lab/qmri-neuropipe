
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
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
      "SPGR": [{"FlipAngle": [...], "RepetitionTime": [...], "EchoTime": [...]}],
      "SSFP": [{"FlipAngle": [...], "RepetitionTime": [...], "EchoTime": [...], "PhaseCycling": [...]}],
      "IRSPGR": [{"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "InversionTime": [], "EchoTrainLength": []}]
    }
    """
    
    params = {}

    def _append_time(entry: dict, key: str, value: Any) -> None:
        if isinstance(value, list):
            entry[key].extend([float(v) for v in value])
        elif value is not None:
            entry[key].append(float(value))
    
    # Process SPGR
    if spgr_images:
        spgr_entry = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": []}
        for img in spgr_images:
            # Assume each image is a volume in sequence or 4D?
            # If 4D, the sidecar might have a list for FlipAngle?
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            te = _extract_bids_param(img, "EchoTime")
            
            # Handle Single Value vs List
            if isinstance(fa, list):
                spgr_entry["FlipAngle"].extend(fa)
            elif fa is not None:
                spgr_entry["FlipAngle"].append(float(fa))
                
            _append_time(spgr_entry, "RepetitionTime", tr)
            _append_time(spgr_entry, "EchoTime", te)

        params["SPGR"] = [spgr_entry]

    # Process SSFP
    if ssfp_images:
        ssfp_entry = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "PhaseCycling": []}
        for img in ssfp_images:
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            te = _extract_bids_param(img, "EchoTime")
            # PhaseCycling usually "PhaseCycling" in sidecar or deduced?
            # User said "PhaseCycling" in json.
            pc = _extract_bids_param(img, "PhaseCycling") # Custom BIDS field?
            
            if isinstance(fa, list): ssfp_entry["FlipAngle"].extend(fa)
            elif fa is not None: ssfp_entry["FlipAngle"].append(float(fa))
            
            _append_time(ssfp_entry, "RepetitionTime", tr)
            _append_time(ssfp_entry, "EchoTime", te)
            
            if isinstance(pc, list): ssfp_entry["PhaseCycling"].extend(pc)
            elif pc is not None: ssfp_entry["PhaseCycling"].append(float(pc))
            else: ssfp_entry["PhaseCycling"].append(0.0) # Default 0 or 180?

        params["SSFP"] = [ssfp_entry]

    # Process IR-SPGR
    if irspgr_images:
        ir_entry = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": [], "InversionTime": [], "EchoTrainLength": []}
        for img in irspgr_images:
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            te = _extract_bids_param(img, "EchoTime")
            ti = _extract_bids_param(img, "InversionTime")
            etl = _extract_bids_param(img, "EchoTrainLength", 1) # Default 1? or typically larger?
            
            if isinstance(fa, list): ir_entry["FlipAngle"].extend(fa)
            elif fa is not None: ir_entry["FlipAngle"].append(float(fa))
            
            _append_time(ir_entry, "RepetitionTime", tr)
            _append_time(ir_entry, "EchoTime", te)
            
            _append_time(ir_entry, "InversionTime", ti)
            
            if isinstance(etl, list): ir_entry["EchoTrainLength"].extend(etl)
            elif etl is not None: ir_entry["EchoTrainLength"].append(float(etl))
            
        params["IRSPGR"] = [ir_entry]
        
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
            
    return params
