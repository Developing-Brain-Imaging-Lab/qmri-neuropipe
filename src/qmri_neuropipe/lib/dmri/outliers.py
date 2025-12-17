"""
dMRI Outlier Removal Step
"""
from pathlib import Path
from typing import Optional, List, Any
import numpy as np
import nibabel as nib
import logging

from ...core.base import BaseProcessingStep
from ...core.types import DWIFile
from ...io.dmri.bids import build_bids_name

class OutlierRemovalStep(BaseProcessingStep):
    """
    Identifies and removes outlier volumes from DWI data.
    """
    
    def __init__(self, config, logger, provenance, method: str = "manual", threshold: float = 0.05, manual_indices: Optional[List[int]] = None):
        super().__init__(config, logger, provenance)
        self.method = method
        self.threshold = threshold
        self.manual_indices = manual_indices
        
    def run(self, context: dict, output_dir: Path, **kwargs) -> dict:
        """
        Run outlier removal.
        """
        dwi_files: List[DWIFile] = context.get("dwi_files", [])
        if not dwi_files:
             return context
             
        # Assumes working on 'current_image' or modifying the list?
        # Usually checking the processed data so far.
        # But 'dwi_files' in context might be the list of all inputs.
        # PreprocessingWorkflow handles context['current_image'].
        
        current_img: DWIFile = context.get("current_image")
        if not current_img:
             return context
        
        bad_indices = []
        
        # --- Check if output already exists ---
        fname = current_img.img.name.replace(".nii", "_clean.nii")
        # Handle .nii.gz case properly if needed, but the replace works for .nii.gz too
        # If input: desc-eddy_dwi.nii.gz -> desc-eddy_dwi_clean.nii.gz
        
        out_path = output_dir / fname
        out_bvec_path = output_dir / (current_img.bvec.stem + "_clean.bvec") if current_img.bvec else None
        
        # If output exists, we assume outlier removal was performed successfully previously.
        # Note: If previous run found NO outliers, out_path wouldn't exist, and we'd correctly fall through to check again.
        if out_path.exists():
             # Check timestamps
             in_mtime = current_img.img.stat().st_mtime
             out_mtime = out_path.stat().st_mtime
             
             if in_mtime > out_mtime:
                 self.logger.info(f"Outlier input ({current_img.img.name}) is newer than output. Re-running.")
                 self.logger.debug(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, Diff={in_mtime-out_mtime:.2f}s")
             else:
                 self.logger.info(f"Skipping Outlier Removal (Output exists and up-to-date: {out_path.name})")
             
                 # Construct output object
                 new_bval_path = output_dir / (current_img.bval.stem + "_clean.bval") if current_img.bval else None
                 # Re-verify existence of sidecars
                 if new_bval_path and not new_bval_path.exists(): new_bval_path = current_img.bval
                 if out_bvec_path and not out_bvec_path.exists(): out_bvec_path = current_img.bvec
                 
                 new_dwi_file = DWIFile(
                      entities=current_img.entities,
                      img=out_path,
                      json=current_img.json,
                      bval=new_bval_path,
                      bvec=out_bvec_path
                 )
                 
                 context["current_image"] = new_dwi_file
                 
                 # Attempt to load stats if possible? 
                 # For now, just return context to ensure downstream steps get the CLEAN image.
                 return context

        # 1. Identify Bad Indices
        if self.method == "manual":
             if self.manual_indices:
                  bad_indices = self.manual_indices
             else:
                  # Check for sidecar file?
                  # Or config
                  pass
                  
        elif self.method == "eddy_qc" or self.method == "threshold":
             # Look for eddy outlier report associated with the current image
             # The current image is likely the output of eddy (e.g. ..._desc-eddycorrected_dwi.nii.gz)
             # The outlier map should be ..._desc-eddycorrected_dwi.eddy_outlier_map
             
             # Expected path
             img_path = context.get('current_image').img
             # Remove extensions (.nii or .nii.gz)
             base_path = img_path.with_suffix("").with_suffix("") if img_path.name.endswith(".gz") else img_path.with_suffix("")
             
             outlier_map_path = base_path.with_suffix(".eddy_outlier_map")
             
             # Fallback: check if we just stripped only one suffix if it wasn't .gz?
             # Safe way:
             if not outlier_map_path.exists():
                  # Try appending to pure stem?
                  # If file is sub-01_desc-eddy_dwi.nii.gz, base is sub-01_desc-eddy_dwi
                  # map is sub-01_desc-eddy_dwi.eddy_outlier_map
                  pass
             
             if not outlier_map_path.exists():
                 self.logger.warning(f"Eddy outlier map not found at {outlier_map_path}. Skipping outlier removal.")
             else:
                 self.logger.info(f"Reading eddy outlier map: {outlier_map_path}")
                 try:
                     # Load map: rows=volumes, cols=slices
                     # 1 = outlier, 0 = valid
                     # Read as integer matrix
                     # Skip header row as it contains text
                     map_data = np.loadtxt(outlier_map_path, dtype=int, skiprows=1)
                     
                     # Check dimensions
                     # If only 1 volume, map_data might be 1D array of slices?
                     if map_data.ndim == 1:
                         # Single volume case
                         map_data = map_data.reshape(1, -1)
                     
                     n_vols, n_slices = map_data.shape
                     
                     for i in range(n_vols):
                         n_outliers = np.sum(map_data[i, :])
                         percent = n_outliers / n_slices
                         
                         if percent > self.threshold:
                             bad_indices.append(i)
                             self.logger.info(f"Volume {i}: {n_outliers}/{n_slices} slices are outliers ({percent:.2%}). Marked for removal.")
                             
                 except Exception as e:
                     self.logger.warning(f"Failed to parse eddy outlier map: {e}")
             
        elif self.method == "deep_learning":
             self.logger.warning("Deep learning outlier detection not implemented.")
        
        if not bad_indices:
             self.logger.info("No outliers identified to remove.")
             return context
             
        # 2. Remove Volumes
        self.logger.info(f"Removing {len(bad_indices)} outlier volumes: {bad_indices}")
        
        img = nib.load(current_img.img)
        data = img.get_fdata()
        
        # Determine keep indices
        n_vols = data.shape[-1]
        keep_indices = [i for i in range(n_vols) if i not in bad_indices]
        
        if len(keep_indices) == 0:
             raise ValueError("All volumes identified as outliers! Cannot proceed.")
             
        # Create cleaned image
        new_data = data[..., keep_indices]
        new_img = nib.Nifti1Image(new_data, img.affine, img.header)
        
        # Save new image
        fname = current_img.img.name.replace(".nii", "_clean.nii")
        # Ensure .gz if original was .gz and replacement didn't catch it (replace affects only first match? No, all. But .nii inside .nii.gz?)
        # If foo.nii.gz -> replace .nii -> foo_clean.nii.gz. Correct.
        # If foo.nii -> foo_clean.nii. Correct.
        
        out_path = output_dir / fname
        nib.save(new_img, out_path)
        self.logger.info(f"Saved outlier-removed DWI to: {out_path} (dims: {new_data.shape})")

        # Handle bvals/bvecs
        new_bval_path = None
        new_bvec_path = None
        
        # Bvals
        if current_img.bval and current_img.bval.exists():
             bvals = np.loadtxt(current_img.bval)
             # bvals might be 1D or 2D row
             if bvals.ndim == 1:
                  new_bvals = bvals[keep_indices]
             else:
                  new_bvals = bvals[:, keep_indices] # Usually (N,)
             
             new_bval_path = output_dir / (current_img.bval.stem + "_clean.bval")
             np.savetxt(new_bval_path, new_bvals, fmt='%d', newline=' ') # FSL style usually single line
             
        # Bvecs
        if current_img.bvec and current_img.bvec.exists():
             bvecs = np.loadtxt(current_img.bvec)
             # bvecs is usually (3, N)
             if bvecs.shape[0] == 3:
                  new_bvecs = bvecs[:, keep_indices]
             else:
                  new_bvecs = bvecs[keep_indices, :].T # Reshape to 3xN logic?
             
             new_bvec_path = output_dir / (current_img.bvec.stem + "_clean.bvec")
             np.savetxt(new_bvec_path, new_bvecs, fmt='%.6f')

        # --- Calculate Statistics ---
        total_vols = n_vols
        num_removed = len(bad_indices)
        percent_removed = (num_removed / total_vols) * 100
        
        stats = {
            "total_volumes": total_vols,
            "removed_volumes": num_removed,
            "percent_removed": percent_removed,
            "bvalue_stats": []
        }
        
        # Detailed stats (reuse loaded bvals if possible, or reload)
        current_bvals = None
        if current_img.bval and current_img.bval.exists():
             try:
                 bvals = np.loadtxt(current_img.bval)
                 if bvals.ndim == 1:
                     current_bvals = bvals
                 else:
                     current_bvals = bvals.ravel()
                     
                 rounded_bvals = np.round(current_bvals, -2) # Round to nearest 100
                 unique_b = np.unique(rounded_bvals)
                 
                 for b in unique_b:
                     b_indices = np.where(rounded_bvals == b)[0]
                     total_b = len(b_indices)
                     removed_b = len(set(b_indices).intersection(set(bad_indices)))
                     pct_b = (removed_b / total_b * 100) if total_b > 0 else 0
                     
                     stats["bvalue_stats"].append({
                         "b_value": int(b),
                         "total": total_b,
                         "removed": removed_b,
                         "percent": pct_b
                     })
             except Exception as e:
                 self.logger.warning(f"Could not calculate b-value stats: {e}")
        
        # Update context
        new_dwi_file = DWIFile(
             entities=current_img.entities,
             img=out_path,
             json=current_img.json,
             bval=new_bval_path,
             bvec=new_bvec_path
        )
        
        context["current_image"] = new_dwi_file
        context["outlier_stats"] = stats
        return context
