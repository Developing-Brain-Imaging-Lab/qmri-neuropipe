
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import nibabel as nib
from ..core.types import ImageFile


NUMERIC_METADATA_KEYS = {
    "FlipAngle",
    "RepetitionTime",
    "EchoTime",
    "InversionTime",
    "EchoTrainLength",
    "PhaseCycling",
    "TRRatio",
}


def _coerce_numeric_metadata_value(value: Any) -> Any:
    """
    Coerce numeric sidecar metadata values into Python numbers.

    dcm2bids configs and hand-edited JSON sometimes encode numeric fields as
    strings, e.g. ``"55.0"``. Relaxometry fitting expects actual numbers for
    arithmetic and ordering, so normalize scalar strings and lists recursively.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return _coerce_numeric_metadata_value(parsed)
        try:
            return float(text)
        except ValueError:
            return value

    if isinstance(value, list):
        return [_coerce_numeric_metadata_value(item) for item in value]

    return value


def _extract_bids_param(img: ImageFile, key: str, default=None) -> Any:
    """Safely extract parameter from ImageFile sidecar."""

    value = default
    if img.json:
        if isinstance(img.json, dict):
            value = img.json.get(key, default)
        elif hasattr(img.json, 'exists') and img.json.exists():
            with open(img.json) as f:
                data = json.load(f)
                value = data.get(key, default)
        if key in NUMERIC_METADATA_KEYS:
            return _coerce_numeric_metadata_value(value)
        return value

    # Fallback: check entities? (e.g. TR in filename?) Unlikely for accurate fitting.
    if key in NUMERIC_METADATA_KEYS:
        return _coerce_numeric_metadata_value(default)
    return default


def _nvols(img: ImageFile) -> int:
    try:
        shape = nib.load(str(img.img)).shape
        return int(shape[3]) if len(shape) >= 4 else 1
    except Exception:
        return 1


def _append_param_series(entry: dict, key: str, value: Any, n_expected: int) -> None:
    if isinstance(value, list):
        entry[key].extend([float(v) for v in value])
    elif value is not None:
        entry[key].extend([float(value)] * max(int(n_expected), 1))


def _collapse_constant_scalar_series(entry: dict, keys: List[str]) -> None:
    for key in keys:
        values = entry.get(key)
        if not isinstance(values, list) or not values:
            continue
        first = float(values[0])
        if all(float(v) == first for v in values):
            entry[key] = first

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

    # Process SPGR
    if spgr_images:
        spgr_entry = {"FlipAngle": [], "RepetitionTime": [], "EchoTime": []}
        for img in spgr_images:
            # Assume each image is a volume in sequence or 4D?
            # If 4D, the sidecar might have a list for FlipAngle?
            fa = _extract_bids_param(img, "FlipAngle")
            tr = _extract_bids_param(img, "RepetitionTime")
            te = _extract_bids_param(img, "EchoTime")
            vol_count = _nvols(img)
            fa_count = len(fa) if isinstance(fa, list) else 1
            expected_count = max(vol_count, fa_count)
            
            # Handle Single Value vs List
            if isinstance(fa, list):
                spgr_entry["FlipAngle"].extend([float(v) for v in fa])
            elif fa is not None:
                spgr_entry["FlipAngle"].extend([float(fa)] * expected_count)
                
            _append_param_series(spgr_entry, "RepetitionTime", tr, expected_count)
            _append_param_series(spgr_entry, "EchoTime", te, expected_count)

        _collapse_constant_scalar_series(spgr_entry, ["RepetitionTime", "EchoTime"])
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
            vol_count = _nvols(img)
            fa_count = len(fa) if isinstance(fa, list) else 1
            pc_count = len(pc) if isinstance(pc, list) else 1
            expected_count = max(vol_count, fa_count, pc_count)
            
            if isinstance(fa, list): ssfp_entry["FlipAngle"].extend([float(v) for v in fa])
            elif fa is not None: ssfp_entry["FlipAngle"].extend([float(fa)] * expected_count)
            
            _append_param_series(ssfp_entry, "RepetitionTime", tr, expected_count)
            _append_param_series(ssfp_entry, "EchoTime", te, expected_count)
            
            if isinstance(pc, list): ssfp_entry["PhaseCycling"].extend([float(v) for v in pc])
            elif pc is not None: ssfp_entry["PhaseCycling"].extend([float(pc)] * expected_count)
            else: ssfp_entry["PhaseCycling"].extend([0.0] * expected_count) # Default 0 or 180?

        _collapse_constant_scalar_series(ssfp_entry, ["RepetitionTime", "EchoTime"])
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
            vol_count = _nvols(img)
            fa_count = len(fa) if isinstance(fa, list) else 1
            ti_count = len(ti) if isinstance(ti, list) else 1
            etl_count = len(etl) if isinstance(etl, list) else 1
            expected_count = max(vol_count, fa_count, ti_count, etl_count)
            
            if isinstance(fa, list): ir_entry["FlipAngle"].extend([float(v) for v in fa])
            elif fa is not None: ir_entry["FlipAngle"].extend([float(fa)] * expected_count)
            
            _append_param_series(ir_entry, "RepetitionTime", tr, expected_count)
            _append_param_series(ir_entry, "EchoTime", te, expected_count)
            
            _append_param_series(ir_entry, "InversionTime", ti, expected_count)
            
            if isinstance(etl, list): ir_entry["EchoTrainLength"].extend([float(v) for v in etl])
            elif etl is not None: ir_entry["EchoTrainLength"].extend([float(etl)] * expected_count)
            
        _collapse_constant_scalar_series(
            ir_entry,
            ["RepetitionTime", "EchoTime", "InversionTime", "EchoTrainLength"],
        )
        params["IRSPGR"] = [ir_entry]
        
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
            
    return params
