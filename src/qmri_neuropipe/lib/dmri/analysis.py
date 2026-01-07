"""
Diffusion Analysis Module.

This module provides processing steps for registering atlases and extracting
ROI/Tract statistics from diffusion models.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import logging
import csv
import numpy as np
import nibabel as nib
import shutil

from ...core import BaseProcessingStep, ProcessingError
from ...interfaces import ants, fsl # Assuming ANTs or FSL for registration
from ...io.bids import build_bids_name

class AtlasRegistrationStep(BaseProcessingStep):
    """
    Step to register standard atlases to the subject's diffusion space.
    
    Atlases:
    - JHU (ICBM-DTI-81, JHU-WhiteMatter-labels)
    - MNI standard atlases
    - IIT (if provided)
    
    This step typically:
    1. Registers MNI152 template to Subject FA (or b0) using ANTs/FSL.
    2. Applies the inverse transform to the Atlas labels (nearest neighbor).
    """
    def __init__(self, config, logger, provenance, method='ants', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method 
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        atlas_out = output_dir / "Atlases"
        atlas_out.mkdir(parents=True, exist_ok=True)
        
        # 1. Identify Target Image (Subject Space)
        # Best target for atlas registration is usually FA.
        # Check context for FA map.
        dti_results = context.get('modeling_results', {}).get('DTI', {})
        target_img = dti_results.get('FA')
        
        if not target_img:
            # Fallback to b0 or mean_b0 (which might be the "current_image" if preproc was done)
            # Or compute FA on the fly? Pipeline usually runs Fitting first.
            self.logger.warning("No FA map found for registration target. Trying DWI/B0...")
            target_img = dwi.img
            # Ideally we compute FA quickly or use b0? b0 has different contrast from T1 template.
        
        # 2. Identify Templates & Atlases
        # These paths should be provided in config or found in standard locations.
        # We assume standard FSL/MNI paths if not provided.
        
        # TODO: Parameterize this via config or environment
        # For now, we will look for FSLDIR.
        # Ideally, user config 'normalization.template' provides the template.
        
        template_img = self.config.get("normalization", {}).get("template")
        if not template_img:
             # Default to standard MNI if not specified? 
             # Or skip if we can't find one.
             self.logger.warning("No template specified in config['normalization']['template']. Skipping Atlas step.")
             return context
             
        # Where are the atlas labels?
        # Usually in same dir as template or specified.
        # Implementation Detail: We need a mapping of "Atlas Name" -> "Label File".
        # Let's assume a simplified config structure or standard FSL atlases.
        
        atlases_to_reg = self.config.get("analysis", {}).get("atlases", {})
        # Example config:
        # analysis:
        #   atlases:
        #      JHU: /path/to/JHU-ICBM-labels.nii.gz
        
        if not atlases_to_reg:
             self.logger.info("No atlases configured for registration.")
             return context
             
        # 3. Registration (Template -> Subject)
        # We need to warp Template -> Subject to move Labels -> Subject
        # Or Subject -> Template and use Inverse.
        # Normalization step might have already run Subject -> Template. 
        # Check context for 'normalization_warp'?
        
        # If we have warp from NormalizationStep (Subject -> Template), we can use the Inverse (Template -> Subject).
        
        if self.method == 'ants':
             pass # Use existing ANTswrap if available?
             
        # For simplicity and robustness, let's run a quick registration or re-use.
        # If we re-use, we need to know where the warp is.
        # Let's assume we do a fresh registration or use inverse of normalization if stored.
        
        # If we don't have existing warp, run registration: Template (Moving) -> Subject (Fixed).
        # Note: Usually we run Subject -> Template for spatial normalization.
        # Here we want Labels (in Template) -> Subject.
        # So we want Transform(Template -> Subject).
        
        # Let's run Rigid+Affine+SyN: Template -> Subject FA.
        
        warp_out = atlas_out / "warp_template_to_subject"
        
        # Check existing
        # ...
        
        self.logger.info(f"Registering Template to Subject for Atlas propagation...")
        
        # Using ants.registration
        reg_res = ants.registration(
            fixed=target_img,
            moving=template_img,
            out_prefix=str(warp_out), # ants wrapper usually handles full path prefixes
            type_of_transform='SyN', # or Affine if quick
            nthreads=self.nthreads
        )
        
        # 4. Apply Transforms to Atlases
        registered_atlases = {}
        
        for name, label_path in atlases_to_reg.items():
             label_path = Path(label_path)
             if not label_path.exists():
                 self.logger.warning(f"Atlas {name} label file not found: {label_path}")
                 continue
                 
             self.logger.info(f"Warping {name} atlas to subject space...")
             out_label = atlas_out / f"{name}_in_subject.nii.gz"
             
             # Apply transform
             # NearestNeighbor interpolation for labels!
             ants.apply_transforms(
                 fixed=target_img,
                 moving=label_path,
                 transformlist=reg_res['fwdtransforms'], # Template -> Subject
                 out_file=out_label,
                 interpolator='nearestNeighbor',
                 nthreads=self.nthreads
             )
             
             registered_atlases[name] = out_label
             
        # Populate context
        context.setdefault('segmentations', {})['Atlases'] = registered_atlases
        
        return context


class StatsExtractionStep(BaseProcessingStep):
    """
    Step to extract mean/median/std statistics of diffusion metrics within ROIs.
    
    Iterates over:
    - Modeling Results (DTI, NODDI, etc.) -> Metric Maps (FA, MD, ODI...)
    - Segmentations (TractSeg bundles, PyAFQ bundles, Atlas labels)
    
    Outputs CSV files.
    """
    def __init__(self, config, logger, provenance, nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        out_stats = output_dir / "Statistics"
        out_stats.mkdir(parents=True, exist_ok=True)
        
        dwi = context.get('current_image') if isinstance(context, dict) else None
        subject_id = dwi.entities.get('sub', 'unknown') if dwi else 'unknown'
        
        # 1. Gather Metrics
        # structure: context['modeling_results'][ModelName][MetricName] = Path
        all_models = context.get('modeling_results', {})
        
        if not all_models:
             self.logger.warning("No modeling results found. Skipping statistics extraction.")
             return context
             
        # 2. Gather Segmentations
        # structure: context['segmentations'][MethodName] = Path (dir or file) or Dict
        all_segs = context.get('segmentations', {})
        
        if not all_segs:
             self.logger.warning("No segmentations found. Skipping statistics extraction.")
             return context
             
        # 3. Iterate and Compute
        
        # We process each Segmentation Source separately (different CSV per source usually cleaner)
        
        for seg_source, seg_data in all_segs.items():
            self.logger.info(f"Extracting statistics for segmentation source: {seg_source}")
            
            # Identify ROIs
            # TractSeg: seg_data is directory with binary masks
            # PyAFQ: seg_data is directory?
            # Atlases: seg_data is Dict[AtlasName: Path]
            
            rois = {} # name -> path
            
            if seg_source == 'TractSeg':
                 # Directory of masks
                 seg_dir = Path(seg_data)
                 if seg_dir.is_dir():
                     for p in seg_dir.glob("*.nii.gz"):
                         # Name is filename without ext
                         name = p.name.replace('.nii.gz', '')
                         rois[name] = p
            elif seg_source == 'Atlases':
                 # Dict of AtlasName -> LabelMap
                 # Note: LabelMap has multiple integer labels. We need to split them?
                 # Or compute stats per label ID.
                 # For simplicity, let's treat each Atlas as a container of ROIs.
                 # We might handle multi-label maps differently.
                 pass 
                 
            elif seg_source == 'PyAFQ':
                 # Attempt to find NIfTI bundle masks in the output directory
                 # Standard structure might contain a 'bundles' subfolder
                 afq_dir = Path(seg_data)
                 # Recursively search for bundle masks provided they are nifti
                 # Common naming: 'CST_L.nii.gz' inside 'bundles'
                 potential_masks = list(afq_dir.glob("**/bundles/*.nii.gz"))
                 if not potential_masks:
                     potential_masks = list(afq_dir.glob("**/*_mask.nii.gz"))
                     
                 for p in potential_masks:
                     name = p.name.replace('.nii.gz', '').replace('_mask', '')
                     rois[name] = p
            
            if not rois and seg_source != 'Atlases':
                 self.logger.warning(f"No ROI masks found for {seg_source}")
                 continue
                 
            # --- Processing Loop ---
            
            # Initialize Data Structure for CSV
            # Columns: Subject, Segmentation, Model, Metric, ROI, Mean, Median, StdDev
            
            csv_rows = []
            
            # A. Single-File Masks (TractSeg)
            if rois:
                for roi_name, roi_path in rois.items():
                    # Load mask once
                    try:
                        mask_img = nib.load(str(roi_path))
                        mask_data = mask_img.get_fdata() > 0.5 # Binary
                    except Exception as e:
                        self.logger.error(f"Failed to load ROI {roi_name}: {e}")
                        continue
                        
                    if np.sum(mask_data) == 0:
                        continue
                        
                    for model_name, metrics in all_models.items():
                        for metric_name, metric_path in metrics.items():
                             # Load Metric
                             try:
                                 met_img = nib.load(str(metric_path))
                                 met_data = met_img.get_fdata()
                             except Exception as e:
                                 self.logger.error(f"Failed to load metric {metric_path}: {e}")
                                 continue
                             
                             # Extract values
                             # Ensure dimensions match
                             if met_data.shape != mask_data.shape:
                                  # Resample? Or Skip? 
                                  # If in same space, should match. TractSeg outputs in DWI space.
                                  self.logger.warning(f"Shape mismatch: {roi_name} {mask_data.shape} vs {metric_name} {met_data.shape}")
                                  continue
                                  
                             values = met_data[mask_data]
                             # Remove NaNs/Infs
                             values = values[np.isfinite(values)]
                             
                             if values.size == 0:
                                  continue
                                  
                             row = {
                                 "Subject": subject_id,
                                 "SegmentationSource": seg_source,
                                 "ROI": roi_name,
                                 "Model": model_name,
                                 "Metric": metric_name,
                                 "Mean": np.mean(values),
                                 "Median": np.median(values),
                                 "StdDev": np.std(values),
                                 "Min": np.min(values),
                                 "Max": np.max(values),
                                 "VoxelCount": values.size
                             }
                             csv_rows.append(row)

            # B. Multi-Label Atlases
            if seg_source == 'Atlases':
                 for atlas_name, label_path in seg_data.items():
                     try:
                         label_img = nib.load(str(label_path))
                         label_vol = label_img.get_fdata().astype(int)
                     except Exception as e:
                         self.logger.error(f"Failed to load Atlas {atlas_name}: {e}")
                         continue
                         
                     # Unique labels
                     unique_labels = np.unique(label_vol)
                     unique_labels = unique_labels[unique_labels > 0] # Skip background
                     
                     for lbl in unique_labels:
                         mask_data = (label_vol == lbl)
                         roi_name = f"{atlas_name}_Label_{lbl}"
                         
                         # Iterate Models/Metrics
                         for model_name, metrics in all_models.items():
                            for metric_name, metric_path in metrics.items():
                                 try:
                                     met_img = nib.load(str(metric_path))
                                     met_data = met_img.get_fdata()
                                 except Exception: continue
                                 
                                 if met_data.shape != mask_data.shape: continue
                                 
                                 values = met_data[mask_data]
                                 values = values[np.isfinite(values)]
                                 
                                 if values.size == 0: continue
                                 
                                 row = {
                                     "Subject": subject_id,
                                     "SegmentationSource": seg_source,
                                     "ROI": roi_name,
                                     "Model": model_name,
                                     "Metric": metric_name,
                                     "Mean": np.mean(values),
                                     "Median": np.median(values),
                                     "StdDev": np.std(values),
                                     "Min": np.min(values),
                                     "Max": np.max(values),
                                     "VoxelCount": values.size
                                 }
                                 csv_rows.append(row)
            
            # Save CSV for this Source
            if csv_rows:
                csv_path = out_stats / f"sub-{subject_id}_{seg_source}_stats.csv"
                keys = csv_rows[0].keys()
                try:
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(csv_rows)
                    self.logger.info(f"Saved stats to {csv_path}")
                    
                    # Store path in context
                    context.setdefault('statistics', {})[seg_source] = csv_path
                except Exception as e:
                    self.logger.error(f"Failed to write CSV: {e}")
                    
        return context

