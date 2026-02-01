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
        tracker.update_status(subject, session, "Overall_Pipeline", "Complete", study)

        # Infer modality for more specific status tracking
        inferred_modality = None
        out_str = str(output_dir).lower()
        if 'dwi' in out_str or 'diffusion' in out_str:
            inferred_modality = 'Diffusion'
        elif 'anat' in out_str:
            inferred_modality = 'Anatomical'
        elif 'relax' in out_str:
            inferred_modality = 'Relaxometry'
            
        if inferred_modality:
            tracker.update_status(subject, session, "Preprocessing", "Complete", study, modality=inferred_modality)
            tracker.update_status(subject, session, "Analysis", "Complete", study, modality=inferred_modality)
            tracker.update_status(subject, session, "Overall", "Complete", study, modality=inferred_modality)

        # 2. Update Module Statuses from context
        for key, val in context.items():
            if key.endswith('_status') and isinstance(val, str):
                module = key[:-7] # remove _status
                # Convert lowercase module names to CamelCase or Title Case if needed?
                mod_name = module.replace("_", " ").title().replace(" ", "")
                tracker.update_status(subject, session, mod_name, val, study)

        # 3. Update QC Metrics
        qc_metrics = {}
        
        # --- ROBUST FILESYSTEM RECOVERY & AGGREGATION ---
        # Search for all outlier stats files
        outlier_files = list(output_dir.glob("*desc-outliers_stats.json"))
        total_removed = 0
        total_vols = 0
        bval_stats_agg = {}
        
        for out_json in outlier_files:
            try:
                with open(out_json) as f:
                    stats = json.load(f)
                    total_removed += stats.get('removed_volumes', 0)
                    total_vols += stats.get('total_volumes', 0)
                    
                    # B-value breakdown
                    for b_entry in stats.get('bvalue_stats', []):
                        bv = b_entry.get('b_value', 0)
                        if bv not in bval_stats_agg: bval_stats_agg[bv] = {'total': 0, 'removed': 0}
                        bval_stats_agg[bv]['total'] += b_entry.get('total', 0)
                        bval_stats_agg[bv]['removed'] += b_entry.get('removed', 0)
            except Exception as e:
                self.logger.warning(f"Failed to read outlier stats {out_json.name}: {e}")

        if total_vols > 0:
            qc_metrics['DWI_Outliers_Removed_Volumes'] = total_removed
            qc_metrics['DWI_Outliers_Removed_Pct'] = (total_removed / total_vols) * 100
            
        for bv, counts in bval_stats_agg.items():
            qc_metrics[f'DWI_Bval_{bv}_Total'] = counts['total']
            qc_metrics[f'DWI_Bval_{bv}_Removed'] = counts['removed']
            if counts['total'] > 0:
                qc_metrics[f'DWI_Bval_{bv}_Pct'] = (counts['removed'] / counts['total']) * 100

        # Search for all eddy_quad QC files
        # They might be in qc/eddy_quad_RUN/qc.json OR qc/eddy_quad_RUN/qc_summary.json
        quad_files = list(output_dir.glob("qc/eddy_quad*/qc.json")) + \
                     list(output_dir.glob("qc/eddy_quad*/qc_summary.json"))
        all_quad_metrics = {} # Map metric to list of values
        
        for q_json in quad_files:
            try:
                with open(q_json) as f:
                    m = json.load(f)
                    
                    # If it's our persistent summary, it already has the correct prefixes
                    if q_json.name == "qc_summary.json":
                        for k, v in m.items():
                             if k.startswith("DWI_"):
                                 if k not in all_quad_metrics: all_quad_metrics[k] = []
                                 try: all_quad_metrics[k].append(float(v))
                                 except: pass
                        continue

                    # Extract standard metrics
                    for src_key, target_key in [
                        ('qc_mot_abs', 'DWI_Motion_Abs_mm'),
                        ('qc_mot_rel', 'DWI_Motion_Rel_mm'),
                        ('qc_s2s_b0_avg', 'DWI_SNR'),
                        ('qc_outliers_tot', 'DWI_Outliers_Total_Pct')
                    ]:
                        if src_key in m:
                            if target_key not in all_quad_metrics: all_quad_metrics[target_key] = []
                            all_quad_metrics[target_key].append(m[src_key])
                    
                    # CNR per shell (list of dicts)
                    if 'cnr' in m and isinstance(m['cnr'], list):
                        for entry in m['cnr']:
                            label = entry.get('Metric', 'Unknown')
                            val = entry.get('Value', 0)
                            key = f"DWI_{label.replace(' ', '_')}_CNR"
                            if key not in all_quad_metrics: all_quad_metrics[key] = []
                            all_quad_metrics[key].append(val)
            except Exception as e:
                self.logger.warning(f"Failed to read QC metrics {q_json}: {e}")

        # Average quad metrics
        for k, vals in all_quad_metrics.items():
            if vals:
                qc_metrics[k] = np.mean(vals)
                if 'Motion_Rel' in k:
                    qc_metrics['DWI_Motion_FD_Mean'] = qc_metrics[k] # Alias Rel to FD Mean

        # --- FALLBACK: report_data.json recovery ---
        # If some metrics are still missing (e.g. intermediate files were deleted),
        # try to recover from the persistent report_data.json
        if not qc_metrics or 'DWI_Motion_Abs_mm' not in qc_metrics:
            report_data_file = output_dir.parent / "report_data.json"
            if report_data_file.exists():
                try:
                    self.logger.debug(f"Attempting to recover metrics from {report_data_file}")
                    with open(report_data_file) as f:
                        rd = json.load(f)
                    
                    # Look for "Quality Control" entries in dmri steps
                    # report_data.json -> dmri -> steps -> [ {name: "Quality Control", tables: [...]}, ... ]
                    for step in rd.get('dmri', {}).get('steps', []):
                        if step.get('name') == 'Quality Control':
                            for table in step.get('tables', []):
                                title = table.get('title', '')
                                rows = table.get('rows', [])
                                
                                if 'Motion' in title:
                                    for row in rows:
                                        m_name = row.get('Metric', '')
                                        val = row.get('Value')
                                        if val is not None:
                                            if 'Absolute' in m_name: qc_metrics['DWI_Motion_Abs_mm'] = float(val)
                                            elif 'Relative' in m_name:
                                                qc_metrics['DWI_Motion_Rel_mm'] = float(val)
                                                qc_metrics['DWI_Motion_FD_Mean'] = float(val)
                                
                                elif 'CNR' in title:
                                    for row in rows:
                                        # Shell row looks like {'Shell': 'Shell 1 (CNR)', 'Value': '14.96'}
                                        sh = row.get('Shell', row.get('Metric', ''))
                                        val = row.get('Value')
                                        if val is not None:
                                            # Normalize shell name to key
                                            key = f"DWI_{sh.replace(' ', '_').replace('(', '').replace(')', '')}"
                                            qc_metrics[key] = float(val)

                                elif 'Outliers Summary' in title:
                                    for row in rows:
                                        m_name = row.get('Metric', '')
                                        val = row.get('Value')
                                        if val is not None and 'Total Outliers (%)' in m_name:
                                            try:
                                                # Strip % if present
                                                val_f = float(str(val).replace('%', '').strip())
                                                qc_metrics['DWI_Outliers_Total_Pct'] = val_f
                                            except: pass
                except Exception as e:
                    self.logger.warning(f"Failed to recover metrics from report_data.json: {e}")

        # --- CONTEXT OVERRIDES ---
        # Allow registry entries to override if available (useful for non-persistent or custom steps)
        qc_registry = context.get('qc_registry', {})
        for img_name, record in qc_registry.items():
            for k, v in record.items():
                # Map old prefixes to new prefix
                if k.startswith('QC_DWI_'):
                    new_k = k.replace('QC_DWI_', 'DWI_')
                    qc_metrics[new_k] = v
                elif k.startswith('DWI_'):
                    qc_metrics[k] = v

        # Add Scan Metadata
        current_img = context.get('current_image')
        if isinstance(current_img, ImageFile):
             try:
                 img_path = Path(current_img.img)
                 if img_path.exists():
                     img = nib.load(str(img_path))
                     # Directions/Volumes
                     if len(img.shape) > 3:
                          qc_metrics['DWI_Directions'] = img.shape[3]
                     
                     # Resolution
                     pixdim = img.header.get_zooms()[:3]
                     qc_metrics['DWI_Resolution'] = " x ".join([f"{p:.2f}" for p in pixdim])
                     
                     # B-values
                     if hasattr(current_img, 'bval') and current_img.bval and current_img.bval.exists():
                          bvals = np.loadtxt(current_img.bval)
                          unique_b = np.unique(np.round(bvals, -1)).astype(int)
                          qc_metrics['DWI_Bvals'] = ", ".join(map(str, unique_b))
             except Exception as e:
                  self.logger.warning(f"Failed to extract scan metadata for tracker: {e}")
 
        # Add Total Slices Flagged (from eddy)
        # Search for .eddy_outlier_map in output_dir
        outlier_maps = list(output_dir.glob("*.eddy_outlier_map"))
        if outlier_maps:
             try:
                  # Skip header, read as matrix, sum
                  out_data = np.loadtxt(outlier_maps[0], skiprows=1)
                  qc_metrics['DWI_Total_Outlier_Slices'] = int(np.sum(out_data))
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
