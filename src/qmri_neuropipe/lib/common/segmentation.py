from pathlib import Path
import logging
import numpy as np
import nibabel as nib
import pandas as pd
from typing import Dict, Optional, Union, List

from ...core import BaseProcessingStep
from ...core.utils import ensure_dir

class SegmentationStep(BaseProcessingStep):
    """
    Step to compute statistics within regions of interest (ROIs) defined by an atlas.
    
    Can be used for:
    - dMRI metrics (FA, MD, etc.)
    - Anatomical intensities (T1w, T2w) - *Requires coregistration to atlas space*
    
    Inputs:
        context: Pipeline context.
        output_dir: Directory to save stats.
        
    Configuration:
        atlas_file: Path to label image (NIfTI).
        atlas_labels: Path to lookup table (TSV/TXT).
        metrics: List of metric names/suffixes to include. If empty, tries to find all available.
    """
    
    def __init__(self, config, logger, provenance, atlas_file: str = None, atlas_labels: str = None, metrics: list = None, **kwargs):
        super().__init__(config, logger, provenance)
        self.atlas_file = Path(atlas_file) if atlas_file else None
        self.atlas_labels = Path(atlas_labels) if atlas_labels else None
        self.target_metrics = metrics or [] # List of suffixes or names to include
        self.kwargs = kwargs
        
    def run(self, context: dict, output_dir: Path, **kwargs) -> dict:
        """
        Run segmentation statistics.
        
        Args:
            context: Pipeline context.
            output_dir: Directory to save stats file.
        """
        # Determine output path early for skip check
        subject = context.get('subject', 'unknown')
        session = context.get('session', '')
        out_name = f"sub-{subject}"
        if session: out_name += f"_ses-{session}"
        out_name += "_desc-roi_stats.tsv"
        out_path = output_dir / out_name

        if out_path.exists() and not kwargs.get('force', False):
             self.logger.info(f"Skipping segmentation stats (exists): {out_path}")
             if 'segmentation_stats' not in context: context['segmentation_stats'] = []
             context['segmentation_stats'].append(out_path)
             return context

        if not self.atlas_file:
             # Try to get from kwargs (runtime)
             if 'atlas_file' in kwargs: self.atlas_file = Path(kwargs['atlas_file'])
             
        if not self.atlas_file or not self.atlas_file.exists():
            self.logger.warning(f"Segmentation skipped: Atlas file not found at {self.atlas_file}")
            return context
            
        # 1. Gather Metrics/Images to measure
        # We need a dictionary: { "MetricName": path_to_nifti }
        metrics_dict = {}
        
        # dMRI Context
        if 'normalized_results' in context and context['normalized_results']:
            # Results from NormalizationStep (usually dict of paths)
            normalized = context['normalized_results']
            if isinstance(normalized, dict):
                 metrics_dict.update(normalized)
                 
        elif 'modeling_results' in context:
            # Fallback to native space results if normalization didn't run
            for model, results in context['modeling_results'].items():
                if isinstance(results, dict):
                    metrics_dict.update(results)

        # Anat Context
        # Check for standard keys or explicit passed images
        if 'preprocessed_t1w' in context:
             img_obj = context['preprocessed_t1w']
             if hasattr(img_obj, 'img'): metrics_dict['T1w'] = img_obj.img
             
        if 'preprocessed_t2w' in context:
             img_obj = context['preprocessed_t2w']
             # Use coregistered if available? 
             # Usually we want the one in the same space as atlas.
             # If using NonlinearRegistration (MNI), keys might be updated or different.
             # Ideally validation checks geometry.
             if hasattr(img_obj, 'img'): metrics_dict['T2w'] = img_obj.img
             
        # Explicit override/addition from kwargs
        if 'images' in kwargs and isinstance(kwargs['images'], dict):
             metrics_dict.update(kwargs['images'])
        
        if not metrics_dict:
             self.logger.warning("Segmentation skipped: No input images found in context.")
             return context

        # Filter metrics if target_metrics specified
        if self.target_metrics:
            filtered = {}
            for k, v in metrics_dict.items():
                # Check if key matches target (case insensitive or exact)
                # target 'FA' should match 'FA', 'tensor_FA'
                for t in self.target_metrics:
                    if t.lower() == k.lower() or t.lower() in k.lower():
                        filtered[k] = v
                        # Don't break, might match multiple? No, k is unique.
            metrics_dict = filtered
            
        if not metrics_dict:
            self.logger.warning(f"Segmentation skipped: No images matched targets {self.target_metrics}")
            return context

        # 2. Load Atlas
        try:
            atlas_img = nib.load(str(self.atlas_file))
            atlas_data = atlas_img.get_fdata().astype(int)
            atlas_affine = atlas_img.affine
        except Exception as e:
            self.logger.error(f"Failed to load atlas {self.atlas_file}: {e}")
            return context
            
        # Load Labels (Optional)
        roi_map = {} # id -> name
        if self.atlas_labels and self.atlas_labels.exists():
            try:
                if self.atlas_labels.suffix in ['.csv', '.tsv', '.txt']:
                    with open(self.atlas_labels, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[0].isdigit():
                             roi_map[int(parts[0])] = parts[1]
            except Exception as e:
                self.logger.warning(f"Failed to parse labels {self.atlas_labels}: {e}")
        
        # Determine labels to process
        unique_labels = np.unique(atlas_data)
        unique_labels = unique_labels[unique_labels > 0] # Exclude background
        
        if not roi_map:
            for l in unique_labels:
                roi_map[l] = f"ROI_{l}"
        else:
            # Fill missing
            for l in unique_labels:
                if l not in roi_map:
                    roi_map[l] = f"ROI_{l}"

        # 3. Compute Statistics
        stats_rows = []
        subject = context.get('subject', 'unknown')
        session = context.get('session', '')
        
        ensure_dir(output_dir)
        import scipy.ndimage as nd
        
        # Pre-filter labels present in atlas
        labels_list = sorted([l for l in roi_map.keys() if l in unique_labels])
        
        if not labels_list:
             self.logger.warning("No ROIs found in atlas data.")
             return context

        for metric_name, metric_path in metrics_dict.items():
            try:
                # Resolve path if it's an ImageFile object
                p = metric_path
                if hasattr(p, 'img'): p = p.img
                p = Path(p)
                
                if not p.exists(): continue
                
                img = nib.load(str(p))
                data = img.get_fdata()
                
                # Squeeze extra dims if 4D with 1 vol
                if data.ndim == 4 and data.shape[3] == 1:
                    data = data.squeeze()
                
                # Check geometry
                # Simple shape check
                if data.shape != atlas_data.shape:
                    self.logger.warning(f"Dimension mismatch for {metric_name} ({data.shape}) vs Atlas ({atlas_data.shape}). Skipping.")
                    continue
                
                # Calculate
                means = nd.mean(data, labels=atlas_data, index=labels_list)
                stds = nd.standard_deviation(data, labels=atlas_data, index=labels_list)
                counts = nd.sum(np.ones_like(data), labels=atlas_data, index=labels_list)
                
                for idx, label_id in enumerate(labels_list):
                   roi_name = roi_map[label_id]
                   stats_rows.append({
                       "subject": subject,
                       "session": session,
                       "metric": metric_name,
                       "roi_id": label_id,
                       "roi_name": roi_name,
                       "mean": means[idx],
                       "std": stds[idx],
                       "count": counts[idx]
                   })
                   
            except Exception as e:
                self.logger.warning(f"Error processing metric {metric_name}: {e}")
        
        # 4. Save Results
        if stats_rows:
            df = pd.DataFrame(stats_rows)
            out_name = f"sub-{subject}"
            if session: out_name += f"_ses-{session}"
            out_name += "_desc-roi_stats.tsv"
            
            out_path = output_dir / out_name
            df.to_csv(out_path, sep='\t', index=False, float_format='%.6f')
            self.logger.info(f"Saved ROI statistics to {out_path}")
            
            if 'segmentation_stats' not in context: context['segmentation_stats'] = []
            context['segmentation_stats'].append(out_path)
            
        return context
