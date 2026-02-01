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
        
        if not subject:
            self.logger.error("Missing subject in context. Cannot update tracker.")
            return context
        
        # Session is optional
        session_str = str(session) if session else "N/A"

        from datetime import datetime
        self.logger.info(f"Updating study-wide tracker: {tracker.excel_path.name}")
        
        # 1.5 Update baseline Metadata and Status
        metadata = {
            "Last_Seen_Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "BIDS_Dir": str(self.config.get('bids_dir', '')),
            "Output_Dir": str(self.config.get('output_dir', ''))
        }
        # Add any extra metadata from context if available (e.g. from participants.tsv)
        if 'subject_metadata' in context:
             metadata.update(context['subject_metadata'])
             
        tracker.update_metadata(subject, session, metadata, study)
        tracker.update_status(subject, session, "Overall_Pipeline", "Completed", study)

        # 2. Update Module Statuses from context
        for key, val in context.items():
            if key.endswith('_status') and isinstance(val, str):
                module = key[:-7] # remove _status
                # Convert lowercase module names to CamelCase or Title Case if needed?
                mod_name = module.replace("_", " ").title().replace(" ", "")
                tracker.update_status(subject, session, mod_name, val, study)

        # 3. Update QC Metrics
        qc_metrics = context.get('qc_metrics', {}).copy()
        
        # --- AGGREGATION LOGIC ---
        # If we have multiple images (runs), we aggregate their metrics
        qc_registry = context.get('qc_registry', {})
        total_removed = 0
        total_vols = 0
        all_aggregates = {} # Map metric name to list of values
        
        # Also check for outlier_stats at top level (for single image cases or specific pipelines)
        top_outliers = context.get('outlier_stats', {}) or context.get('outliers', {})
        if top_outliers:
            total_removed += top_outliers.get('removed_volumes', 0)
            total_vols += top_outliers.get('total_volumes', 0)

        # Iterate over registry to find all metrics
        bval_stats_agg = {} # Map b_value to {'total': X, 'removed': Y}
        
        for img_name, record in qc_registry.items():
            # Outlier stats from record (dmri.py uses "outliers", outliers.py uses "outlier_stats")
            img_outliers = record.get('outliers', record.get('outlier_stats', {}))
            if img_outliers:
                total_removed += img_outliers.get('removed_volumes', 0)
                total_vols += img_outliers.get('total_volumes', 0)
                
                # B-value breakdown
                for b_entry in img_outliers.get('bvalue_stats', []):
                    bv = b_entry.get('b_value', 0)
                    if bv not in bval_stats_agg: bval_stats_agg[bv] = {'total': 0, 'removed': 0}
                    bval_stats_agg[bv]['total'] += b_entry.get('total', 0)
                    bval_stats_agg[bv]['removed'] += b_entry.get('removed', 0)
            
            # QC Metrics (from eddy_quad or other steps)
            # Many are flat in the record now
            for k, v in record.items():
                if k.startswith('QC_DWI_') and isinstance(v, (int, float)):
                    if k not in all_aggregates: all_aggregates[k] = []
                    all_aggregates[k].append(v)
                
                # Handle nested CNR list if present (legacy or specific report format)
                if k == 'cnr' and isinstance(v, list):
                    for entry in v:
                        label = entry.get('Metric', 'Unknown')
                        val = entry.get('Value', 0)
                        key = f"QC_DWI_{label.replace(' ', '_')}_CNR"
                        if key not in all_aggregates: all_aggregates[key] = []
                        all_aggregates[key].append(val)

        # Merge aggregates into qc_metrics
        for k, vals in all_aggregates.items():
            if vals:
                # Average for SNR/CNR/Motion
                qc_metrics[k] = np.mean(vals)

        # Finalize Outlier aggregates
        if total_vols > 0:
            qc_metrics['QC_DWI_Outliers_Removed_Volumes'] = total_removed
            qc_metrics['QC_DWI_Outliers_Removed_Pct'] = (total_removed / total_vols) * 100
            
        for bv, counts in bval_stats_agg.items():
            qc_metrics[f'QC_DWI_Bval_{bv}_Total'] = counts['total']
            qc_metrics[f'QC_DWI_Bval_{bv}_Removed'] = counts['removed']
            if counts['total'] > 0:
                qc_metrics[f'QC_DWI_Bval_{bv}_Pct'] = (counts['removed'] / counts['total']) * 100
        
        # Recover QC from disk if STILL missing (fallback for sessions without registry)
        if not any(k.startswith('QC_DWI') for k in qc_metrics):
             quad_json = output_dir / "qc" / "eddy_quad" / "qc.json"
             if quad_json.exists():
                  try:
                      with open(quad_json) as f:
                          m = json.load(f)
                          qc_metrics["QC_DWI_Motion_Abs_mm"] = m.get('qc_mot_abs', 0)
                          qc_metrics["QC_DWI_Motion_Rel_mm"] = m.get('qc_mot_rel', 0)
                          qc_metrics["QC_DWI_Motion_FD_Mean"] = m.get('qc_mot_rel', 0)
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
