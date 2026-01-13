import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Any

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Recursively flatten a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_qc_metrics_csv(qc_data_list: List[Dict[str, Any]], output_file: Path):
    """
    Save a list of QC metrics (dictionaries) to a CSV/TSV file.
    
    Args:
        qc_data_list: List of dictionaries containing QC metrics for each subject.
        output_file: Path to save the CSV file.
    """
    if not qc_data_list:
        return

    # Flatten each entry
    flattened_data = []
    for entry in qc_data_list:
        # Separate identifying info if needed, but usually flattened is fine.
        # Ensure we flatten nested structures like 'motion': {'abs_mean': ...}
        flat = flatten_dict(entry)
        flattened_data.append(flat)

    if not flattened_data:
        return
        
    df = pd.DataFrame(flattened_data)
    
    # Sort columns?
    # Ensure 'subject' and 'session' are first if present
    cols = df.columns.tolist()
    first_cols = [c for c in ['subject', 'session', 'file_name'] if c in cols]
    other_cols = [c for c in cols if c not in first_cols]
    df = df[first_cols + other_cols]

    # Save
    ensure_dir_exists(output_file)
    sep = '\t' if output_file.suffix == '.tsv' else ','
    df.to_csv(output_file, sep=sep, index=False)

def ensure_dir_exists(path: Path):
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
