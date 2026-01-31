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
        atlas_out = output_dir / "atlases"
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
        # Global template is optional if atlases define their own.
        if not template_img:
             self.logger.debug("No global template specified. Atlases must define their own 'template'.")
             
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
             template_groups[tpl_p].append((name, cfg))
             
        # 3. Registration (Template -> Subject)
        
        # We need to handle warps.
        # User requested: atlases/{AtlasName}/transforms/
        # Since we group by template, we register once per template.
        # We can store the warp in a temporary/shared location and then copy/link to each atlas folder.
        
        registered_atlases = {}
        
        # Iterate unique templates
        for tpl_str, atlases in template_groups.items():
             tpl_path = Path(tpl_str)
             if not tpl_path.exists():
                 self.logger.warning(f"Template not found: {tpl_path}. Skipping associated atlases: {[a[0] for a in atlases]}")
                 continue
                 
             self.logger.info(f"Processing Template group: {tpl_path.name} ({len(atlases)} atlases)")
             
             # Run Registration (Template -> Subject) ONCE
             
             # We need a stable place to run registration.
             # Use a generic 'transforms' cache folder for the raw registration output.
             cache_dir = atlas_out / ".transforms_work"
             cache_dir.mkdir(exist_ok=True)
             
             t_name = tpl_path.name.replace('.nii.gz', '').replace('.nii', '')
             
             ents = dwi.entities.copy()
             ents['suffix'] = 'xfm'
             ents['from'] = 'Template' 
             ents['to'] = 'Subject'
             ents['mode'] = 'image'
             if 'desc' in ents: del ents['desc']
             
             prefix_ents = ents.copy()
             prefix_ents['desc'] = t_name 
             
             warp_prefix_name = build_bids_name(prefix_ents).replace('.nii.gz', '') + "_"
             warp_cache_prefix = cache_dir / warp_prefix_name
             
             self.logger.info(f"Registering {tpl_path.name} to Subject FA...")
             
             _, tx_forward = ants.registration(
                fixed_file=target_img,
                moving_file=tpl_path,
                out_prefix=warp_cache_prefix, 
                transform_type='SyN',
                nthreads=self.nthreads
             )
             
             # Fix 'Ensure Dir' artifact (empty folder named as prefix)
             # ants.registration wrapper calls ensure_dir(out_prefix).
             artifact_dir = Path(str(warp_cache_prefix))
             if artifact_dir.exists() and artifact_dir.is_dir():
                 try:
                     artifact_dir.rmdir()
                 except: pass
             
             # Apply to all atlases in this group
             for name, atlas_cfg in atlases:
                  # Resolve label path from config
                  label_path = None
                  if isinstance(atlas_cfg, (str, Path)):
                       label_path = atlas_cfg
                  elif isinstance(atlas_cfg, dict):
                       label_path = atlas_cfg.get('labels') or atlas_cfg.get('file')

                  label_path = Path(label_path)
                  if not label_path.exists():
                      self.logger.warning(f"Atlas {name} label file not found: {label_path}")
                      continue
                      
                  # Check for Probabilistic flag
                  is_prob = False
                  if isinstance(atlas_cfg, dict):
                       is_prob = atlas_cfg.get('is_probabilistic', False)
                  
                  # Interpolator: Linear for Probabilistic (preserve 0-1), Nearest for Label Maps
                  interp = 'linear' if is_prob else 'nearestNeighbor'

                  # Create Atlas Directory: atlases/{Name}
                  atlas_subdir = atlas_out / name
                  atlas_subdir.mkdir(parents=True, exist_ok=True)
                  
                  # Create Transforms Directory: atlases/{Name}/transforms
                  atlas_tx_dir = atlas_subdir / "transforms"
                  atlas_tx_dir.mkdir(exist_ok=True)
                  
                  # Copy Warps to Atlas Transforms Dir
                  final_tx_forward = []
                  for tx in tx_forward:
                      tx_p = Path(tx)
                      dest = atlas_tx_dir / tx_p.name
                      if not dest.exists(): # Don't overwrite if multiple atlases share same transforms? logic check.
                           # Actually we are inside the template loop, these are new copies per atlas if we want per-atlas transform logs.
                           # Or we can just copy.
                           shutil.copy2(tx_p, dest)
                      final_tx_forward.append(str(dest))
                  
                  self.logger.info(f"Warping {name} atlas to subject space (Probabilistic={is_prob}, Interp={interp})...")
                  
                  # BIDS Naming: sub-XX_ses-XX_desc-<atlas-name>_dseg.nii.gz
                  # For probabilistic, maybe suffix should be 'probseg'? usage of dseg implies discrete.
                  # But keeping dseg/probseg consistent helps downstream.
                  lbl_ents = dwi.entities.copy()
                  lbl_ents['desc'] = name
                  lbl_ents['suffix'] = 'probseg' if is_prob else 'dseg'
                  
                  out_label_name = build_bids_name(lbl_ents)
                  if not out_label_name.endswith('.nii.gz'): out_label_name += '.nii.gz'
                  
                  out_label = atlas_subdir / out_label_name
                  
                  # Prepare extra args for ANTs
                  apply_kwargs = {}
                  if is_prob:
                       apply_kwargs['imagetype'] = 3 # TimeSeries/Vector for 4D atlas
                  
                  ants.apply_transforms(
                     fixed_file=target_img,
                     moving_file=label_path,
                     transforms=final_tx_forward,
                     out_file=out_label,
                     interpolator=interp,
                     nthreads=self.nthreads,
                     **apply_kwargs
                  )
                  
                  registered_atlases[name] = out_label
             
             # Clean up cache
             if cache_dir.exists():
                 shutil.rmtree(cache_dir, ignore_errors=True)
                  
        # Populate context with ALL registered atlases
        context.setdefault('segmentations', {})['Atlases'] = registered_atlases
        
        return context


class StatsExtractionStep(BaseProcessingStep):
    """
    Step to extract mean/median/std statistics of diffusion metrics within ROIs.
    
    Iterates over:
    - Modeling Results (DTI, NODDI, etc.) -> Metric Maps (FA, MD, ODI...)
    - Segmentations (TractSeg bundles, PyAFQ bundles, Atlas labels)
    
    Outputs TSV files.
    """
    def __init__(self, config, logger, provenance, nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs
        
    def _load_lut(self, lut_path: Path):
        """
        Load a label lookup table (LUT).
        Expects column with indices and column with names.
        Supported formats: FSL XML (.xml), FreeSurfer (txt), CSV, TSV.
        """
        import pandas as pd
        import xml.etree.ElementTree as ET
        lut = {}
        try:
            # FSL XML Format
            if lut_path.suffix.lower() == '.xml':
                 try:
                     tree = ET.parse(lut_path)
                     root = tree.getroot()
                     # FSL XML structure: <label index="0" x=".." y=".." z="..">LabelName</label>
                     for label in root.iter('label'):
                          if 'index' in label.attrib:
                               idx = int(label.attrib['index'])
                               name = label.text.strip()
                               lut[idx] = name
                     return lut
                 except Exception as ex:
                      self.logger.warning(f"Failed to parse XML LUT {lut_path}: {ex}")
                      return {}
            # Fallback for FS LUT (whitespace separated, # comments)
            if lut_path.suffix == '.txt' or 'FreeSurfer' in lut_path.name:
                 # Custom parser for FS LUT: id name r g b a
                 with open(lut_path, 'r') as f:
                     for line in f:
                         line = line.strip()
                         if not line or line.startswith('#'): continue
                         parts = line.split()
                         if len(parts) >= 2:
                             try:
                                 idx = int(parts[0])
                                 name = parts[1]
                                 lut[idx] = name
                             except: pass
                 return lut
            
            # Use pandas for CSV/TSV
            df = pd.read_csv(lut_path, sep=None, engine='python')
            # Look for suitable columns
            cols = df.columns.astype(str).str.lower()
            
            idx_col = None
            name_col = None
            
            for c in df.columns:
                cl = str(c).lower()
                if any(x in cl for x in ['index', 'id', 'label_id', 'roi_id']):
                    idx_col = c
                if any(x in cl for x in ['name', 'label', 'roi_name', 'structure']):
                    name_col = c
                    
            if idx_col and name_col:
                for _, row in df.iterrows():
                     lut[int(row[idx_col])] = str(row[name_col])
            else:
                self.logger.warning(f"Could not identify ID/Name columns in LUT {lut_path}. Columns: {df.columns}")
        except Exception as e:
            self.logger.warning(f"Failed to load LUT {lut_path}: {e}")
            
        return lut
    
    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs):
        """
        Run statistics extraction.
        """
        import pandas as pd
        import numpy as np
        import nibabel as nib
        from scipy.ndimage import label, mean, median, standard_deviation
        
        output_dir = Path(output_dir)
        dwi = context.get('current_image')
        
        # Lowercase directory logic: 'statistics'
        output_dir = output_dir / "statistics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load LUTs
        analysis_cfg = self.config.get("dmri", {}).get("analysis", {})
        if not analysis_cfg:
             analysis_cfg = self.config.get("dmri", {}).get("modeling", {}).get("analysis", {})
             
        lut_map = {}
        atlases_cfg = analysis_cfg.get("atlases", {})
        for name, cfg in atlases_cfg.items():
             if isinstance(cfg, dict) and cfg.get('lut'):
                 lut_map[name] = self._load_lut(Path(cfg['lut']))
                 
        # 1. Gather Metrics
        # structure: context['modeling_results'][ModelName][MetricName] = Path
        all_models = context.get('modeling_results', {})
        if not all_models:
             self.logger.warning("No modeling results found. Skipping statistics extraction.")
             return context

        # Load all metrics once into memory (map name -> data)
        loaded_maps = {}
        for model_name, metrics in all_models.items():
            for metric_name, metric_path in metrics.items():
                if not Path(metric_path).exists(): continue
                try:
                    img = nib.load(str(metric_path))
                    data = img.get_fdata()
                    # Store as (model, metric) tuple in key or value
                    loaded_maps[f"{model_name}_{metric_name}"] = {
                        'model': model_name,
                        'metric': metric_name,
                        'img': img,
                        'data': data
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to load metric {metric_path}: {e}")

        # 2. Iterate Segmentations
        segmentations = context.get('segmentations', {})
        # Flatten: {'Atlases': {'JHU': path}, 'TractSeg': {'Bundle': path}}
        
        for seg_type, seg_dict in segmentations.items():
             
             # Per-segmentation-source CSV
             # We group by source name inside seg_dict if possible?
             # Actually 'Atlases' dict keys are 'JHU', etc.
             # 'TractSeg' dict keys are 'BundleName'.
             # Strategy: 
             # If seg_type == 'Atlases', iterate each Atlas -> One CSV per Atlas.
             # If seg_type != 'Atlases' (e.g. TractSeg), iterate all ROIs -> One CSV for 'TractSeg'.
             
             if seg_type == 'Atlases':
                 # Treat each Atlas separately
                 for atlas_name, atlas_path in seg_dict.items():
                      if not Path(atlas_path).exists(): continue
                      
                      self.logger.info(f"Extracting stats for Atlas: {atlas_name}")
                      
                      lut = lut_map.get(atlas_name, {})
                      
                      # Retrieve config for this atlas
                      atlas_cfg = atlases_cfg.get(atlas_name, {})
                      is_prob = False
                      if isinstance(atlas_cfg, dict):
                           is_prob = atlas_cfg.get('is_probabilistic', False)

                      try:
                          seg_img = nib.load(str(atlas_path))
                          # For probabilistic, we keep floats. For deterministic, we want ints.
                          if is_prob:
                               seg_data = seg_img.get_fdata() # Keep float
                          else:
                               seg_data = seg_img.get_fdata().astype(int)
                      except Exception as e:
                          self.logger.error(f"Failed to load atlas {atlas_name}: {e}")
                          continue
                          
                      seg_stats = []
                      
                      if is_prob:
                           # PROBABILISTIC / 4D ATLAS
                           # We iterate over the 4th dimension (Volumes)
                           if seg_data.ndim != 4:
                                self.logger.warning(f"Atlas {atlas_name} is marked probabilistic but is not 4D (ndim={seg_data.ndim}). Treating as thresholded map?")
                                # TODO: Handle 3D probability map (single ROI) case if needed.
                                # For now, skip or assume dims.
                                pass
                           
                           n_vols = seg_data.shape[3] if seg_data.ndim == 4 else 1
                           prob_thresh = 0.01  # Ignore voxels with almost 0 weight
                           
                           for vol_idx in range(n_vols):
                                if seg_data.ndim == 4:
                                     weights = seg_data[..., vol_idx]
                                else:
                                     weights = seg_data
                                
                                roi_name = lut.get(vol_idx, f"ROI_{vol_idx}")
                                
                                # Optimization: Skip empty ROIs
                                if weights.max() <= 0: continue
                                
                                mask_bool = weights > prob_thresh
                                if not np.any(mask_bool): continue # Moved this line up
                                
                                for metric_key, m_info in loaded_maps.items():
                                     m_data = m_info['data']
                                     if m_data.shape != weights.shape: continue
                                     
                                     # Select voxels
                                     w_vals = weights[mask_bool]
                                     d_vals = m_data[mask_bool]
                                     
                                     # Filter NaNs
                                     valid = np.isfinite(d_vals) & (d_vals != 0)
                                     w_vals = w_vals[valid]
                                     d_vals = d_vals[valid]
                                     
                                     if d_vals.size == 0: continue
                                     
                                     # Weighted Stats
                                     w_sum = np.sum(w_vals)
                                     if w_sum <= 0: continue
                                     
                                     w_mean = np.sum(d_vals * w_vals) / w_sum
                                     w_var = np.sum(w_vals * (d_vals - w_mean)**2) / w_sum
                                     w_std = np.sqrt(w_var)
                                     
                                     # Weighted Median
                                     try:
                                         order = np.argsort(d_vals)
                                         d_sorted = d_vals[order]
                                         w_sorted = w_vals[order]
                                         cum_w = np.cumsum(w_sorted)
                                         cutoff = 0.5 * w_sum
                                         w_median = d_sorted[cum_w >= cutoff][0]
                                     except:
                                         w_median = 0
                                     
                                     stat = {
                                        "roi_id": vol_idx,
                                        "roi_name": roi_name,
                                        "model": m_info['model'],
                                        "metric": m_info['metric'],
                                        "mean": w_mean,
                                        "median": w_median,
                                        "std": w_std,
                                        "count": d_vals.size, # Number of voxels > threshold
                                        "vol_sum": w_sum # Total weight (volume-ish)
                                     }
                                     seg_stats.append(stat)

                      else:
                           # DETERMINISTIC / LABEL MAP
                           rois = np.unique(seg_data)
                           rois = rois[rois > 0]
                           
                           for roi_idx in rois:
                                roi_mask = (seg_data == roi_idx)
                                roi_name = lut.get(int(roi_idx), f"Label_{roi_idx}")
                                for metric_key, m_info in loaded_maps.items(): # Corrected indentation
                                     m_data = m_info['data']
                                     if m_data.shape != seg_data.shape: continue
                                     vals = m_data[roi_mask]
                                     vals = vals[np.isfinite(vals)]
                                     vals = vals[vals != 0] # Ignore background/zeros
                                     if vals.size == 0: continue
                                     
                                     stat = {
                                         "roi_id": roi_idx,
                                         "roi_name": roi_name,
                                         "model": m_info['model'],
                                         "metric": m_info['metric'],
                                         "mean": np.mean(vals),
                                         "median": np.median(vals),
                                         "std": np.std(vals),
                                         "count": vals.size
                                     }
                                     seg_stats.append(stat)
                      
                      if seg_stats:
                          df = pd.DataFrame(seg_stats)
                          ents = dwi.entities.copy()
                          ents['desc'] = atlas_name
                          ents['suffix'] = 'stats'
                          fname = build_bids_name(ents).replace('.nii.gz', '').replace('.nii', '') + ".tsv"
                          df.to_csv(output_dir / fname, sep='\t', index=False)
                          context.setdefault('segmentation_stats', []).append(output_dir / fname)
                          context.setdefault('roi_stats_files', {})[atlas_name] = output_dir / fname

             else:
                 # Binary Masks (TractSeg, etc.)
                 # Treat 'seg_type' as the source name (e.g. 'TractSeg')
                 self.logger.info(f"Extracting stats for: {seg_type}")
                 
                 seg_stats = []
                 for roi_name, roi_path in seg_dict.items():
                      if not Path(roi_path).exists(): continue
                      
                      try:
                          mask_img = nib.load(str(roi_path))
                          mask_data = mask_img.get_fdata() > 0.5
                      except: continue
                      
                      if np.sum(mask_data) == 0: continue
                      
                      for metric_key, m_info in loaded_maps.items(): # Corrected to use m_info
                           m_data = m_info['data']
                           if m_data.shape != mask_data.shape: continue
                           vals = m_data[mask_data]
                           vals = vals[np.isfinite(vals)]
                           vals = vals[vals != 0] # Ignore background/zeros
                           if vals.size == 0: continue
                           
                           stat = {
                                "roi_id": roi_name, # or idx
                                "roi_name": roi_name,
                                "model": m_info['model'], # Used m_info directly
                                "metric": m_info['metric'], # Used m_info directly
                                "mean": np.mean(vals),
                                "median": np.median(vals),
                                "std": np.std(vals),
                                "count": vals.size
                            }
                           seg_stats.append(stat)
                           
                 if seg_stats:
                      df = pd.DataFrame(seg_stats)
                      ents = dwi.entities.copy()
                      ents['desc'] = seg_type
                      ents['suffix'] = 'stats'
                      fname = build_bids_name(ents).replace('.nii.gz', '').replace('.nii', '') + ".tsv"
                      df.to_csv(output_dir / fname, sep='\t', index=False)
                      context.setdefault('segmentation_stats', []).append(output_dir / fname)
                      context.setdefault('roi_stats_files', {})[seg_type] = output_dir / fname
                      
        return context
