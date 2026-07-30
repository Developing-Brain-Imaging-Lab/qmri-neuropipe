"""
Top-level fMRI processing workflow pipeline.
Routes the BOLD data through BIDS App Containers like fmriprep,
or through native HCP pipelines.
"""

from typing import Any, Optional
import logging
from pathlib import Path

from ...core import BaseWorkflow, PipelineContext
from ...lib.fmri.bids_apps import FmriPrepStep
from ...lib.fmri.hcp import HCPfMRIStep

class FmriWorkflow(BaseWorkflow):
    """
    Workflow for functional MRI processing.
    Delegates to specific fMRI pipelines based on configuration.
    """
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None):
        self.pipeline_steps = []
        super().__init__(config, logger)
        
    def _initialize_steps(self):
        fmriprep_config = self.config.get("fmriprep", {})
        hcp_config = self.config.get("hcp", {})
        
        # If user has an fmriprep config dict, assume they want it
        use_fmriprep = fmriprep_config.get("enabled", False)
        
        if use_fmriprep:
            self.logger.info("fMRIWorkflow: Routing to fMRIPrep BIDS app.")
            self.pipeline_steps.append(
                FmriPrepStep(
                    config=self.config,
                    logger=self.logger,
                    provenance=self.provenance
                )
            )
        else:
            self.logger.info("fMRIWorkflow: fmriprep disabled. Routing to native HCP skeleton.")
            self.pipeline_steps.append(
                 HCPfMRIStep(
                     config=self.config,
                     logger=self.logger,
                     provenance=self.provenance
                 )
            )
            
    def run(self, subjects: Optional[list] = None, sessions: Optional[list] = None, pairs: Optional[list] = None, **kwargs) -> dict:
        stats = {'n_success': 0, 'n_failed': 0, 'n_skipped': 0}
        
        if pairs:
            active_subjects = [p[0] for p in pairs]
        elif subjects:
            active_subjects = subjects
        else:
            self.logger.warning("No subjects provided to FmriWorkflow.")
            return stats
            
        # fmriprep processes participants
        output_dir = getattr(self.config, 'output_dir', Path("./derivatives"))
            
        for sub in active_subjects:
            # Create a mock internal context image for the steps to extract subject label
            mock_img = type('obj', (object,), {'entities': {'sub': sub}})()
            context = PipelineContext(
                {"current_image": mock_img, "subject": sub}
            )
            
            try:
                for step in self.pipeline_steps:
                    self.logger.info(f"Running fMRI step: {step.__class__.__name__} for sub-{sub}")
                    context = step.run(context, output_dir=output_dir, **kwargs)
                stats['n_success'] += 1
            except Exception as e:
                self.logger.error(f"fMRI Workflow failed for sub-{sub}: {e}")
                stats['n_failed'] += 1
                if getattr(self.config, 'stop_on_error', False):
                    raise e
                    
        return stats
