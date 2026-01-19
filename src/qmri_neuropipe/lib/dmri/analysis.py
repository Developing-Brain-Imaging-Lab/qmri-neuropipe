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
        # 1. Identify Target Image (Subject Space)
        # Search all models for FA map (e.g. DTI, DKI)
        target_img = None
        modeling_results = context.get('modeling_results', {})
        
        for model_name, metrics in modeling_results.items():
             # Check exact match
             if 'FA' in metrics:
                 target_img = metrics['FA']
                 break
             # Check case-insensitive
             for k, v in metrics.items():
                 if k.upper() == 'FA':
                     target_img = v
                     break
             if target_img: break
        
        if not target_img:
            # Fallback to b0 logic...
            self.logger.warning("No FA map found for registration target. Trying DWI/B0 (First Volume)...")
            
            # Ensure we use a 3D volume.
            # If dwi.img is 4D, we must extract B0.
            # However, the registration interface expects a file path.
            # We can try to assume the interface handles splitting or we create a temp B0.
            # For now, let's try to find an explicit B0 if available in context?
            # Or just use the dwi path and hope the user provided a 3D image?
            # Creating a temp file requires more logic. 
            # Let's trust the improved lookup first.
            target_img = dwi.img
            
            # Check dimensions if possible
            try:
                hdr = nib.load(target_img).header
                if hdr.get_data_shape()[3] > 1:
                     # It's 4D. 
                     pass # We can't fix it easily without writing a new file.
                     # But we can warn.
                     self.logger.warning(f"Target image {target_img.name} is 4D. Registration may fail if fixed/moving dimensions mismatch.")
            except:
                pass
            # Ideally we compute FA quickly or use b0? b0 has different contrast from T1 template.
        
        # 2. Identify Templates & Atlases
        # These paths should be provided in config or found in standard locations.
        # We assume standard FSL/MNI paths if not provided.
        
        # TODO: Parameterize this via config or environment
        # For now, we will look for FSLDIR.
        # Ideally, user config 'normalization.template' provides the template.
        
        norm_cfg = self.config.get("normalization") or self.config.get("dmri", {}).get("normalization", {})
        template_img = norm_cfg.get("template")
        if not template_img:
             # Default to standard MNI if not specified? 
             # Or skip if we can't find one.
             self.logger.warning("No template specified in config['normalization']['template']. Skipping Atlas step.")
             return context
             
        # Where are the atlas labels?
        # Usually in same dir as template or specified.
        # Implementation Detail: We need a mapping of "Atlas Name" -> "Label File".
        # Let's assume a simplified config structure or standard FSL atlases.
        
        analysis_cfg = self.config.get("analysis") or self.config.get("dmri", {}).get("analysis", {})
        atlases_to_reg = analysis_cfg.get("atlases", {})
        # Example config:
        # analysis:
        #   atlases:
        #      JHU: /path/to/JHU-ICBM-labels.nii.gz
        
        if not atlases_to_reg:
             self.logger.info("No atlases configured for registration.")
             return context
             
        # 2. Parse Atlases and Group by Template
        # Structure: template_groups[template_path] = list of (atlas_name, label_path)
        template_groups = {}
        
        default_template = norm_cfg.get("template")
        
        for name, cfg in atlases_to_reg.items():
             label_p = None
             tpl_p = default_template
             
             if isinstance(cfg, (str, Path)):
                 label_p = cfg
             elif isinstance(cfg, dict):
                 label_p = cfg.get('labels') or cfg.get('file') # support 'labels' or 'file'
                 if 'template' in cfg:
                      tpl_p = cfg['template']
             
             if not label_p:
                 self.logger.warning(f"Atlas {name} missing label path. Skipping.")
                 continue
                 
             if not tpl_p:
                  self.logger.warning(f"Atlas {name} has no template defined and no global default. Skipping.")
                  continue
                  
             tpl_p = str(tpl_p) # Ensure string key
             if tpl_p not in template_groups:
                 template_groups[tpl_p] = []
             template_groups[tpl_p].append((name, label_p))
             
        # 3. Registration (Template -> Subject)
        
        # Warp Output Directory
        transforms_dir = atlas_out / "transforms"
        transforms_dir.mkdir(exist_ok=True)
        
        registered_atlases = {}
        
        # Iterate unique templates
        for tpl_str, atlases in template_groups.items():
             tpl_path = Path(tpl_str)
             if not tpl_path.exists():
                 self.logger.warning(f"Template not found: {tpl_path}. Skipping associated atlases: {[a[0] for a in atlases]}")
                 continue
                 
             self.logger.info(f"Processing Template group: {tpl_path.name} ({len(atlases)} atlases)")
             
             # Define Warp Prefix based on Template Name
             # We try to be unique: sub-X_ses-Y_from-TemplateName_to-Subject_mode-image_xfm
             t_name = tpl_path.name.replace('.nii.gz', '').replace('.nii', '')
             
             ents = dwi.entities.copy()
             ents['suffix'] = 'xfm'
             ents['from'] = 'Template' # Generic 'Template' or specific?
             ents['to'] = 'Subject'
             ents['mode'] = 'image'
             if 'desc' in ents: del ents['desc']
             
             # Add desc to distinguish templates if needed, or rely on 'from'? 
             # BIDS entity 'from' is custom-ish here unless we use 'desc'. 
             # Let's use desc for template name if possible, or just filename uniqueness.
             prefix_ents = ents.copy()
             prefix_ents['desc'] = t_name 
             
             warp_prefix_name = build_bids_name(prefix_ents).replace('.nii.gz', '') + "_"
             warp_out = transforms_dir / warp_prefix_name
             
             # Run Registration (Template -> Subject)
             # Check if we should skip? 
             # For now, just run. Wrapper usually overwrites or logic handles it.
             self.logger.info(f"Registering {tpl_path.name} to Subject FA...")
             
             _, tx_forward = ants.registration(
                fixed_file=target_img,
                moving_file=tpl_path,
                out_prefix=warp_out, 
                transform_type='SyN',
                nthreads=self.nthreads
             )
             
             # Apply to all atlases in this group
             for name, label_path in atlases:
                  label_path = Path(label_path)
                  if not label_path.exists():
                      self.logger.warning(f"Atlas {name} label file not found: {label_path}")
                      continue
                      
                  self.logger.info(f"Warping {name} atlas to subject space...")
                  
                  # BIDS Naming: sub-XX_ses-XX_desc-<atlas-name>_dseg.nii.gz
                  lbl_ents = dwi.entities.copy()
                  lbl_ents['desc'] = name
                  lbl_ents['suffix'] = 'dseg'
                  
                  out_label_name = build_bids_name(lbl_ents)
                  if not out_label_name.endswith('.nii.gz'): out_label_name += '.nii.gz'
                  
                  out_label = atlas_out / out_label_name
                  
                  ants.apply_transforms(
                     fixed_file=target_img,
                     moving_file=label_path,
                     transforms=tx_forward,
                     out_file=out_label,
                     interpolator='nearestNeighbor',
                     nthreads=self.nthreads
                  )
                  
                  registered_atlases[name] = out_label
                  
        # Populate context with ALL registered atlases
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

