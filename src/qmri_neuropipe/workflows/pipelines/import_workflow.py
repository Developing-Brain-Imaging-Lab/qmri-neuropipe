
from pathlib import Path
from typing import Optional
from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.lib.common.importing import Dcm2NiixStep, Dcm2BidsStep, ImportGnlMetadataStep

class ImportWorkflow(BaseWorkflow):
    """
    Workflow for converting DICOMs to BIDS/NIfTI.
    """
    def _initialize_steps(self):
        self.modality = "Import"
        self.steps = []
        
    def build_pipeline(self, context: dict):
        self.steps = []
        import_cfg = self.config.get('import', {})
        
        method = import_cfg.get('method', 'dcm2bids')
        
        if method == 'dcm2bids':
            self.add_step(Dcm2BidsStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))
        elif method == 'dcm2niix':
            self.add_step(Dcm2NiixStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

        if import_cfg.get('gnl_metadata', {}).get('enabled', False):
            self.add_step(ImportGnlMetadataStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))
            
    def run(self, dicom_dir: Path, output_dir: Path, context: dict) -> dict:
        self.logger.info("Starting Import Workflow")
        context = dict(context)
        context["dicom_dir"] = Path(dicom_dir)
        step_context = {k: v for k, v in context.items() if k != "dicom_dir"}
        
        for step in self.steps:
            if isinstance(step, (Dcm2BidsStep, Dcm2NiixStep)):
                step.run(dicom_dir, output_dir, **step_context)
            else:
                result = step(context, output_dir=output_dir, **context)
                if isinstance(result, dict):
                    context.update(result)
                
        return context
