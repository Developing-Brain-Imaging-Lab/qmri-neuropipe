"""
fMRI processing steps leveraging BIDS App containers or pipelines.
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

from ...core.base import BaseProcessingStep, ValidationError
from ...core.types import ImageLike
from ...interfaces.fmriprep import run_fmriprep

class FmriPrepStep(BaseProcessingStep):
    """
    Executes fMRIPrep on the entire BIDS directory.
    This step operates mostly at a dataset/subject level, taking over major
    preprocessing workloads for BOLD data.
    """
    
    def __init__(self, config: Any, logger: Optional[logging.Logger] = None, provenance=None):
        super().__init__(config, logger, provenance)
        
        fmriprep_cfg = self.config.get("fmriprep", {})
        
        self.container_path = fmriprep_cfg.get("container_path")
        self.docker_image = fmriprep_cfg.get("docker_image")
        self.fs_license = fmriprep_cfg.get("fs_license_file")
        self.custom_args = fmriprep_cfg.get("custom_args", [])
        
    def run(self, context: dict | object, output_dir: Path, **kwargs) -> Any:
        # Resolve inputs
        input_image = context.get('current_image') if isinstance(context, dict) else context
        
        if not input_image or not hasattr(input_image, 'entities'):
            raise ValidationError("FmriPrepStep requires a valid BIDS image to extract subject information.")
            
        subject_id = input_image.entities.get('sub')
        if not subject_id:
            raise ValidationError("Could not identify 'sub' from image entities to pass to fMRIPrep.")

        # Determine BIDS root
        bids_dir = getattr(self.config, 'bids_dir', None)
        if not bids_dir:
            raise ValidationError("bids_dir must be configured to run FmriPrepStep.")

        # Output resolution
        fmriprep_out = output_dir / "derivatives" / "fmriprep"
        fmriprep_out.mkdir(parents=True, exist_ok=True)
        
        work_dir = output_dir / "work" / "fmriprep"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        fs_license_path = Path(self.fs_license) if self.fs_license else None
        container_p = Path(self.container_path) if self.container_path else None
        
        # Check if already run (heuristically check for sub-XXX html report)
        report_html = fmriprep_out / f"sub-{subject_id}.html"
        force = kwargs.get('force', False) or self.config.get('force', False)
        
        if report_html.exists() and not force:
            self.logger.info(f"Skipping fMRIPrep, found existing output report: {report_html.name}")
            return context

        self.logger.info(f"Running fMRIPrep for sub-{subject_id}...")
        
        n_cpus = self.config.get("n_cpus", 1)
        omp_nthreads = self.config.get("omp_nthreads", 1)

        run_fmriprep(
            bids_dir=bids_dir,
            output_dir=output_dir / "derivatives", # fMRIPrep handles the 'fmriprep' folder creation
            participant_label=str(subject_id),
            container_path=container_p,
            docker_image=self.docker_image,
            fs_license_file=fs_license_path,
            work_dir=work_dir,
            nthreads=n_cpus,
            omp_nthreads=omp_nthreads,
            custom_args=self.custom_args,
            logger=self.logger
        )
        
        # We don't necessarily update "current_image" here, as fmriprep outputs 
        # a massive tree of derivatives. Just pass context along for any post-processing.
        return context
