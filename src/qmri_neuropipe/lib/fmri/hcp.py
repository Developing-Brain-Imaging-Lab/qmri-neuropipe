"""
Skeleton for HCP-style fMRI workflows.
Falls back to native execution if fMRIPrep containers are not utilized.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

from ...core.base import BaseProcessingStep

class HCPfMRIStep(BaseProcessingStep):
    """
    Placeholder for native HCP fMRI Volume and Surface pipelines.
    Currently a stub that warns the user that this native integration
    is pending full implementation.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, provenance=None):
        super().__init__(config, logger, provenance)
        self.workflow_type = self.config.get("hcp_workflow_type", "Volume")
        
    def run(self, context: dict | object, output_dir: Path, **kwargs) -> Any:
        input_image = context.get('current_image') if isinstance(context, dict) else context
        
        if not input_image:
            self.logger.warning("No input image provided to HCP fMRI step.")
            return context
            
        self.logger.warning(
            f"Native HCP fMRI Pipeline ({self.workflow_type}) is currently a stub. "
            f"Please use the FmriPrep container pipeline (`fmriprep: enabled: true`) via BIDS Apps for now."
        )
        
        return context
