# my_dmri_pipeline.py
from pathlib import Path
from qmri_neuropipe.core import (
    BasePipeline, BaseWorkflow, PipelineConfig,
)
from qmri_neuropipe.core.types import ImageFile, DWIFile
from qmri_neuropipe.io.bids import _load_json_field
from qmri_neuropipe.io.anat.bids import bids_find_t1w, bids_find_t2w
from qmri_neuropipe.io.dmri.bids import bids_find_dwi, find_reversed_phase_groups
from qmri_neuropipe.lib.common.denoise import DenoisingStep
from qmri_neuropipe.lib.common.gibbs import GibbsUnringingStep
from qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep


# Define a simple preprocessing workflow
class PreprocessingWorkflow(BaseWorkflow):
    """
    Preprocessing workflow for DWI:
    - Takes a context dict (DWI files, topup groups, etc.)
    - Applies configurable steps in sequence.
    """

    def _initialize_steps(self):
        self.steps = []

    def build_pipeline(self, context: dict):

        steps = []

        #Read in the configuration and user inputs to build the overall pipeline
        dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {})

        # Example: config-driven switches
        if dmri_cfg['denoising']['enabled']:

            denoise_cfg=dmri_cfg.get('denoising', {})
            method = denoise_cfg.get('method')
            if method is None:
                self.logger.warning(
                    "No denoising.method specified in config — using default 'mrtrix'"
                )
                method = "mrtrix"

            patch_radius=denoise_cfg.get('patch_radius')
            if patch_radius is None:
                self.logger.warning(
                    "No denoising.patch_radius specified in config — using default 2"
                )
                patch_radius=2

            block_radius=denoise_cfg.get('block_radius')
            if block_radius is None:
                self.logger.warning(
                    "No denoising.block_radius specified in config — using default 5"
                )
                block_radius=5

            
            #Get the options     
            steps.append(DenoisingStep(config=self.config, 
                                       logger=self.logger, 
                                       provenance=self.provenance,
                                       method=method,
                                       patch_radius=patch_radius,
                                       block_radius=block_radius))

        if dmri_cfg['degibbs']['enabled']:

            degibbs_cfg=dmri_cfg.get('degibbs', {})
            method = degibbs_cfg.get('method')
            if method is None:
                self.logger.warning(
                    "No gibbs_correction.method specified in config — using default 'mrtrix'"
                )
                method = "mrtrix"


            steps.append(GibbsUnringingStep(config=self.config, 
                                            logger=self.logger, 
                                            provenance=self.provenance,
                                            method=method))
            
        if dmri_cfg['eddy']['enabled']:
            eddy_cfg=dmri_cfg.get('eddy', {})
            method=eddy_cfg.get('method')
            if method is None:
                self.logger.warning(
                    "No eddy correction method specified in config — using default 'eddy'"
                )
                method = "eddy"

            steps.append(EddyCorrectionStep(config=self.config, 
                                            logger=self.logger, 
                                            provenance=self.provenance,
                                            method=method))



        # if self.config.get("do_topup_eddy", True):
        #     steps.append(
        #         TopupEddyStep(self.config, self.logger, self.provenance)
        #     )

        # if self.config.get("do_bias_correction", True):
        #     steps.append(
        #         BiasCorrectionStep(self.config, self.logger, self.provenance)
        #     )
        #

        self.steps = steps



    def run(self, context: dict, output_dir: Path):
        """
        Parameters
        ----------
        context : dict
            Arbitrary context passed through the steps, e.g.:
            {
                "dwi_files": list[DWIFile],
                "topup_groups": list[list[DWIFile]],
                "subject": str,
                "session": str | None,
            }
        output_dir : Path

        Returns
        -------
        dict
            Updated context after all preprocessing steps.
        """
        return self.execute_steps(context, output_dir=output_dir)
    

# Define complete pipeline
class DMRIPipeline(BasePipeline):
    """Diffusion MRI Processing pipeline."""
    
    @property
    def name(self):
        return 'dmri-pipeline'
    
    @property
    def version(self):
        return '1.0.0'
    
    def _initialize_pipeline(self):
        self.preprocessing = PreprocessingWorkflow(self.config, self.logger, self.provenance)
    
    def process_subject(self, subject: str, session: str | None = None):
        ses = f"ses-{session}" if session else ""
        subj_dir = (Path(self.config.get('bids_dir')) / f'sub-{subject}' / ses / 'dwi')
        output_dir = self._get_output_dir(subject, session)

        # 1. Build the Preprocessing Context 
        #   - Find the Structural Files for the subject
        #   - Find the DWI files for the subject (list[DWIFile])
        #   - Find other associated images needed for processing 

        # anat_files = bids_find_anat(subj_dir)
        t1w_files: list[ImageFile]  = bids_find_t1w(subj_dir)
        t2w_files: list[ImageFile]  = bids_find_t2w(subj_dir)
        dwi_files: list[DWIFile]    = bids_find_dwi(subj_dir)
        fmap_files: list[ImageFile] = bids_find_fmap(subj_dir)
        
        if not dwi_files:
            self.logger.warning(f"No DWI files found for sub-{subject} {ses}. Skipping.")
            return

        # 2. (MOVE TO WORKFLOW) Group by reversed phase encoding for TOPUP/EDDY
        topup_groups = find_reversed_phase_groups(dwi_files)
        self.logger.info(
            f"Found {len(dwi_files)} DWI files and {len(topup_groups)} topup group(s) "
            f"for sub-{subject} {ses}."
        )

        # 3. Build preprocessing context
        context = {
            "subject": subject,
            "session": session,
            "current_image": dwi_files[0],
            "topup_groups": topup_groups,
        }

        # 4. Build the preprocessing workflow/pipeline
        self.preprocessing.build_pipeline(context)

        # 4. Run preprocessing workflow
        preprocessed_context = self.preprocessing.run(context, output_dir)

        # 5. Extract and log final outputs
        preprocessed_dwis = preprocessed_context.get("preprocessed_dwis", dwi_files)

        for d in preprocessed_dwis:
            self.logger.info(f"Preprocessed DWI: {d.img}")

        self.logger.info(
            f"Preprocessing complete for sub-{subject} {ses}: "
            f"{len(preprocessed_dwis)} DWI file(s) ready for downstream steps."
        )

# Run the pipeline
if __name__ == '__main__':
    # Create configuration
    config = PipelineConfig(
        bids_dir='/data/my_study',
        output_dir='/data/derivatives/simple-pipeline',
        n_cpus=8,
        skip_existing=True
    )
    
    # Create and run pipeline
    pipeline = DMRIPipeline(config)
    pipeline.run()  # Processes all subjects automatically!
