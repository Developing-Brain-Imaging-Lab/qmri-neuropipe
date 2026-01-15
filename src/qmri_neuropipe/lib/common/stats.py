
from pathlib import Path
from typing import Dict, List, Optional
import nibabel as nib
import numpy as np
import pandas as pd

from ...core import BaseProcessingStep
from ...core.types import ImageFile
from ...core.utils import ensure_dir

class ROIStatsStep(BaseProcessingStep):
    """
    Extract statistics (Mean, Std) from quantitative maps using a segmentation mask.
    """
    
    def run(self, 
            map_image: ImageFile, 
            segmentation: ImageFile, 
            output_dir: Path, 
            force: bool = False,
            label_map: Optional[Dict[int, str]] = None
           ) -> Path:
           
        output_dir = ensure_dir(output_dir)
        out_csv = output_dir / f"{map_image.img.stem.replace('.nii', '').replace('.gz', '')}_roi_stats.tsv"
        
        if out_csv.exists() and not force:
             self.logger.info(f"Skipping ROI Extraction (Exists): {out_csv.name}")
             return out_csv
             
        self.logger.info(f"Extracting ROI Stats for {map_image.img.name}")
        
        # Load Data
        img_nii = nib.load(map_image.img)
        seg_nii = nib.load(segmentation.img)
        
        data = img_nii.get_fdata()
        seg = seg_nii.get_fdata()
        
        # Ensure dimensions match
        if data.shape != seg.shape:
             # Basic check. If 4D map?
             if data.ndim == 4 and seg.ndim == 3:
                 # Loop over 4th dim?
                 pass
             elif data.shape != seg.shape:
                 self.logger.warning(f"Shape mismatch: Map {data.shape} vs Seg {seg.shape}. Resampling needed? skipping.")
                 return out_csv
         
        labels = np.unique(seg)
        labels = labels[labels > 0] # Skip background
        
        stats = []
        for lab in labels:
             mask = (seg == lab)
             vals = data[mask]
             
             mean_val = np.mean(vals)
             std_val = np.std(vals)
             
             name = str(int(lab))
             if label_map and int(lab) in label_map:
                 name = label_map[int(lab)]
                 
             stats.append({
                 "LabelID": int(lab),
                 "LabelName": name,
                 "Mean": mean_val,
                 "Std": std_val,
                 "Count": len(vals)
             })
             
        df = pd.DataFrame(stats)
        df.to_csv(out_csv, sep='\t', index=False)
        return out_csv
