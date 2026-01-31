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
        qc_metrics = context.get('qc_metrics', {})
        if qc_metrics:
            tracker.add_metrics(subject, session, qc_metrics, study)

        # 4. Update ROI Stats
        roi_files = context.get('roi_stats_files', {})
        for atlas_name, tsv_path in roi_files.items():
            tracker.add_roi_stats(subject, session, Path(tsv_path), atlas_name, study)

        # 5. Save the tracker (Force save at the end of subject)
        tracker.save(force=True)
        
        return context
