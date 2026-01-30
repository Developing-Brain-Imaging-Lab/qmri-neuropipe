from pathlib import Path
from typing import Optional, Any, Dict
import logging

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
        if not self.tracker_path:
            self.logger.warning("No tracker file specified. Skipping TrackingStep.")
            return context

        subject = context.get('subject')
        session = context.get('session')
        study = context.get('study_name', self.config.get('study_name'))
        
        if not subject or not session:
            self.logger.error("Missing subject or session in context. Cannot update tracker.")
            return context

        tracker = NeuroimagingTracker(self.tracker_path, logger=self.logger)
        
        self.logger.info(f"Updating study-wide tracker: {self.tracker_path.name}")
        
        # 1. Update Module Statuses
        # Iterate through context to find step completion flags
        # Convention: context['step_name_status'] = 'completed'
        for key, val in context.items():
            if key.endswith('_status') and isinstance(val, str):
                module = key[:-7] # remove _status
                tracker.update_status(subject, session, module, val, study)

        # 2. Update QC Metrics
        # Convention: context['qc_metrics'] = {'SNR': 20, 'FD': 0.1, ...}
        qc_metrics = context.get('qc_metrics', {})
        if qc_metrics:
            tracker.add_metrics(subject, session, qc_metrics, study)

        # 3. Update ROI Stats
        # Convention: context['roi_stats_files'] = {'DTI': Path, 'NODDI': Path}
        roi_files = context.get('roi_stats_files', {})
        for sheet_suffix, tsv_path in roi_files.items():
            sheet_name = f"{sheet_suffix}_Metrics"
            tracker.add_roi_stats(subject, session, Path(tsv_path), sheet_name, study)

        # 4. Save the tracker
        tracker.save()
        
        return context
