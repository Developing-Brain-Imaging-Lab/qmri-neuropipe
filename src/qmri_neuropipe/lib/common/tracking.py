from pathlib import Path
from typing import Optional, Any, Dict
import logging
import json
import numpy as np
import nibabel as nib

from ...core import BaseProcessingStep
from ...core.types import ImageFile
from .tracker import NeuroimagingTracker

class TrackingStep(BaseProcessingStep):
    """
    Pipeline step to update the study-wide tracker with participant data.
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None, provenance = None):
        super().__init__(config, logger, provenance)
        self.tracker_path = self.config.get('tracker_file')
        if not self.tracker_path:
            # Try to find it in the output_dir
            out_dir = Path(self.config.get('output_dir', '.'))
            self.tracker_path = out_dir / "study_tracker.xlsx"
        else:
            self.tracker_path = Path(self.tracker_path)

    def run(self, context: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Aggregate data from the current context and update the tracker.
        """
        # 1. Use existing tracker from config if available, otherwise create one
        tracker = self.config.tracker
        if not tracker:
            if not self.tracker_path:
                self.logger.warning("No tracker file specified and no active tracker in config. Skipping TrackingStep.")
                return context
            tracker = NeuroimagingTracker(self.tracker_path, logger=self.logger)

        subject = context.get('subject')
        session = context.get('session')
        study = context.get('study_name', self.config.get('study_name'))
        
        if not subject or not session:
            self.logger.error("Missing subject or session in context. Cannot update tracker.")
            return context

        self.logger.info(f"Updating study-wide tracker: {tracker.excel_path.name}")
        
        # 2. Update Module Statuses
        for key, val in context.items():
            if key.endswith('_status') and isinstance(val, str):
                module = key[:-7] # remove _status
                tracker.update_status(subject, session, module, val, study)

        # 3. Update QC Metrics
        qc_metrics = context.get('qc_metrics', {}).copy()
        
        # Recover QC from disk if missing in context
        if not qc_metrics:
             quad_json = output_dir / "qc" / "eddy_quad" / "qc.json"
             if quad_json.exists():
                  try:
                      with open(quad_json) as f:
                          m = json.load(f)
                          qc_metrics["QC_DWI_Motion_Abs_mm"] = m.get('qc_mot_abs', 0)
                          qc_metrics["QC_DWI_Motion_Rel_mm"] = m.get('qc_mot_rel', 0)
                          qc_metrics["QC_DWI_Motion_FD_Mean"] = m.get('qc_mot_rel', 0) # Mapping Rel Motion to FD Mean placeholder
                          if 'qc_s2s_b0_avg' in m: 
                               qc_metrics["QC_DWI_b0_SNR"] = m['qc_s2s_b0_avg']
                               qc_metrics["QC_DWI_SNR"] = m['qc_s2s_b0_avg']
                          qc_metrics["QC_DWI_Outliers_Total_Pct"] = m.get('qc_outliers_tot', 0)
                  except: pass

        # Add Scan Metadata
        current_img = context.get('current_image')
        if isinstance(current_img, ImageFile):
             try:
                 img_path = Path(current_img.img)
                 if img_path.exists():
                     img = nib.load(str(img_path))
                     # Directions/Volumes
                     if len(img.shape) > 3:
                          qc_metrics['QC_DWI_Directions'] = img.shape[3]
                     
                     # Resolution
                     pixdim = img.header.get_zooms()[:3]
                     qc_metrics['QC_DWI_Resolution'] = " x ".join([f"{p:.2f}" for p in pixdim])
                     
                     # B-values
                     if hasattr(current_img, 'bval') and current_img.bval and current_img.bval.exists():
                          bvals = np.loadtxt(current_img.bval)
                          unique_b = np.unique(np.round(bvals, -1)).astype(int)
                          qc_metrics['QC_DWI_Bvals'] = ", ".join(map(str, unique_b))
             except Exception as e:
                  self.logger.warning(f"Failed to extract scan metadata for tracker: {e}")

        # Add Outlier Stats
        outlier_stats = context.get('outlier_stats', {})
        if not outlier_stats:
             # Try to find outlier stats file
             outlier_files = list(output_dir.glob("*desc-outliers_stats.json"))
             if outlier_files:
                  try:
                      with open(outlier_files[0]) as f:
                          outlier_stats = json.load(f)
                  except: pass

        if outlier_stats:
             qc_metrics['QC_DWI_Outliers_Removed_Volumes'] = outlier_stats.get('removed_volumes', 0)
             qc_metrics['QC_DWI_Outliers_Removed_Pct'] = outlier_stats.get('percent_removed', 0)
             
        # Add Total Slices Flagged (from eddy)
        # Search for .eddy_outlier_map in output_dir
        outlier_maps = list(output_dir.glob("*.eddy_outlier_map"))
        if outlier_maps:
             try:
                  # Skip header, read as matrix, sum
                  out_data = np.loadtxt(outlier_maps[0], skiprows=1)
                  qc_metrics['QC_DWI_Total_Outlier_Slices'] = int(np.sum(out_data))
             except: pass

        if qc_metrics:
            tracker.add_metrics(subject, session, qc_metrics, study)

        # 4. Update ROI Stats
        roi_files = context.get('roi_stats_files', {}).copy()
        
        # Recover ROI stats from disk if missing
        stats_dir = output_dir / "statistics"
        if stats_dir.exists():
             for tsv in stats_dir.glob("*.tsv"):
                  # Try to infer atlas name from filename (desc-ATLAS)
                  name = tsv.name
                  atlas_name = "Unknown"
                  if 'desc-' in name:
                       atlas_name = name.split('desc-')[1].split('_')[0]
                  
                  if atlas_name not in roi_files:
                       roi_files[atlas_name] = tsv

        for atlas_name, tsv_path in roi_files.items():
            tracker.add_roi_stats(subject, session, Path(tsv_path), atlas_name, study)

        # 5. Save the tracker (Force save at the end of subject)
        tracker.save(force=True)
        
        return context
