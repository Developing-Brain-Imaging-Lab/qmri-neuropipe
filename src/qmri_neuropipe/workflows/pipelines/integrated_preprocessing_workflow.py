"""
Integrated Preprocessing Workflow with ExecutionEngine.

This module contains the PreprocessingWorkflow class that uses ExecutionEngine
for step execution, providing better separation of concerns and reusability.
"""

from pathlib import Path
from typing import Optional

from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.core.types import DWIFile

# Processing steps
from qmri_neuropipe.lib.common.denoise import DenoisingStep
from qmri_neuropipe.lib.common.gibbs import GibbsUnringingStep
from qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep
from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep
from qmri_neuropipe.lib.dmri.topup import TopupStep
from qmri_neuropipe.lib.common.bias import BiasCorrectionStep
from qmri_neuropipe.lib.dmri.grad_nonlin import (
    TortoiseGradNonlinCorrectStep,
    AlignFinalGNLTensorStep
)
from qmri_neuropipe.lib.common.registration import CoregistrationStep
from qmri_neuropipe.lib.dmri.grad_check import GradientCheckStep
from qmri_neuropipe.lib.dmri.reorient import DMRIReorientStep
from qmri_neuropipe.lib.dmri.merge import MergeStep
from qmri_neuropipe.lib.dmri.drbuddi import NativeDrbuddiStep
from qmri_neuropipe.lib.common.mask import BrainMaskingStep
from qmri_neuropipe.lib.common.resample import ResampleStep
from qmri_neuropipe.lib.dmri.outliers import OutlierRemovalStep
from qmri_neuropipe.lib.dmri.qc import EddyQuadStep
from qmri_neuropipe.lib.dmri.motion import NiiFreezeStep


class PreprocessingWorkflow(BaseWorkflow):
    def recover_intermediates(self, work_dir: Path, output_dir: Path):
        """
        Recover intermediate data from the final output directory back to the working directory.
        This allows the pipeline to skip steps that were previously computed and saved.
        """
        import shutil

        self.logger.info("Attempting to recover intermediate data...")

        intermediate_store = output_dir / "intermediate"
        if not intermediate_store.exists():
            self.logger.debug(f"No intermediate storage found at {intermediate_store}")
            return

        if (intermediate_store / "topup").exists():
            target_work = work_dir / "topup"
            if not target_work.exists():
                self.logger.info(f"Recovering Topup results from {intermediate_store / 'topup'}")
                shutil.copytree(intermediate_store / "topup", target_work, dirs_exist_ok=True)

        if (intermediate_store / "synb0").exists():
            target_work = work_dir / "synb0"
            if not target_work.exists():
                self.logger.info(f"Recovering Synb0 results from {intermediate_store / 'synb0'}")
                shutil.copytree(intermediate_store / "synb0", target_work, dirs_exist_ok=True)

        if (intermediate_store / "merge").exists():
            target_work = work_dir / "merge"
            if not target_work.exists():
                self.logger.info(f"Recovering Merge results from {intermediate_store / 'merge'}")
                shutil.copytree(intermediate_store / "merge", target_work, dirs_exist_ok=True)

        if (intermediate_store / "eddy").exists():
            target_work = work_dir / "eddy"
            if not target_work.exists():
                self.logger.info(f"Recovering Eddy results from {intermediate_store / 'eddy'}")
                shutil.copytree(intermediate_store / "eddy", target_work, dirs_exist_ok=True)

        if (intermediate_store / "nativedrbuddi").exists():
            target_work = work_dir / "nativedrbuddi"
            if not target_work.exists():
                self.logger.info(f"Recovering Native DRBUDDI results from {intermediate_store / 'nativedrbuddi'}")
                shutil.copytree(intermediate_store / "nativedrbuddi", target_work, dirs_exist_ok=True)

        if (intermediate_store / "masking").exists():
            target_work = work_dir / "masking"
            if not target_work.exists():
                self.logger.info(f"Recovering Brain Mask results from {intermediate_store / 'masking'}")
                shutil.copytree(intermediate_store / "masking", target_work, dirs_exist_ok=True)

    """
    Preprocessing workflow for DWI using ExecutionEngine.
    
    This workflow builds a pipeline of preprocessing steps based on
    configuration and uses ExecutionEngine for execution.
    """

    def _initialize_steps(self):
        self.modality = "Diffusion"
        self.steps = []

    def build_pipeline(self, context: dict):
        """Build the preprocessing pipeline based on configuration."""
        self.steps = []  # Reset steps
        dwi_files: list[DWIFile] = context.get("dwi_files", [])
        topup_groups = context.get("topup_groups", [])
        
        self.logger.info(
            f"Building preprocessing pipeline: "
            f"{len(dwi_files)} DWI files, {len(topup_groups)} topup groups"
        )

        dmri_cfg = (self.config.get('dmri') or {}).get('preprocessing', {})
        
        # Add steps based on configuration
        self._add_merge_step(dmri_cfg, context)
        self._add_reorientation_step(dmri_cfg)
        self._add_resample_step(dmri_cfg)
        self._add_distortion_correction_steps(dmri_cfg, context)
        self._add_gradient_nonlinearity_step(dmri_cfg)
        self._add_gradient_check_step(dmri_cfg)
        self._add_manual_outlier_removal_step(dmri_cfg)
        self._add_denoising_step(dmri_cfg)
        self._add_gibbs_step(dmri_cfg)
        self._add_motion_correction_step(dmri_cfg)
        self._add_post_eddy_distortion_refinement_step(dmri_cfg, context)
        self._add_automated_outlier_removal_step(dmri_cfg)
        self._add_bias_correction_step(dmri_cfg)
        self._add_coregistration_step(dmri_cfg, context)
        self._add_brain_masking_step(dmri_cfg)
        self._add_final_gnl_alignment_step(dmri_cfg)
        
        self.logger.info(f"Pipeline built with {len(self.steps)} steps")

    def _add_reorientation_step(self, dmri_cfg: dict):
        """Add reorientation step if enabled."""
        reor_cfg = dmri_cfg.get('reorient', {})
        if reor_cfg.get('enabled', False):
            self.logger.info("Adding DMRIReorientStep")
            self.add_step(DMRIReorientStep(
                self.config, self.logger, self.provenance
            ))

    def _add_resample_step(self, dmri_cfg: dict):
        """Add resampling step if enabled."""
        res_cfg = dmri_cfg.get('resample', {})
        if res_cfg.get('enabled', False):
            self.logger.info("Adding ResampleStep")
            self.add_step(ResampleStep(
                self.config,
                self.logger,
                self.provenance,
                resolution=res_cfg.get('resolution')
            ))

    def _add_gradient_check_step(self, dmri_cfg: dict):
        """Add gradient check step if enabled."""
        grad_check_cfg = dmri_cfg.get('grad_check', {})
        if grad_check_cfg.get('enabled', False):
            self.logger.info("Adding GradientCheckStep")
            self.add_step(GradientCheckStep(
                self.config, self.logger, self.provenance
            ))

    def _add_manual_outlier_removal_step(self, dmri_cfg: dict):
        """Add manual outlier removal step if enabled."""
        outlier_cfg = dmri_cfg.get('outliers', {})
        if outlier_cfg.get('enabled', False) and outlier_cfg.get('method') == 'manual':
            self.logger.info("Adding OutlierRemovalStep (manual)")
            self.add_step(OutlierRemovalStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method='manual',
                manual_indices=outlier_cfg.get('manual_indices'),
                volumes_file=outlier_cfg.get('volumes_file')
            ))

    def _add_distortion_correction_steps(self, dmri_cfg: dict, context: dict):
        """Add distortion correction steps based on configuration."""
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        dist_method = distcorr_cfg.get('method', 'none')
        fallback = distcorr_cfg.get('fallback', False)

        topup_groups = context.get("topup_groups", [])
        has_reverse_pe = len(topup_groups) > 0
        t1w_files = context.get("t1w_files", [])
        t2w_files = context.get("t2w_files", [])
        dwi_files = context.get("dwi_files", [])
        synb0_cfg = distcorr_cfg.get("synb0", {}) or {}
        synb0_t1w_source = str(synb0_cfg.get("t1w_source", "raw")).lower()
        synb0_supersynth_input = str(synb0_cfg.get("supersynth_input", "auto")).lower()
        uses_dwi_supersynth = (
            synb0_t1w_source in {"dwi_supersynth", "supersynth_dwi", "diffusion_supersynth", "b0_supersynth"}
            or (
                synb0_t1w_source in {"supersynth", "prefer_supersynth"}
                and synb0_supersynth_input in {"dwi", "b0", "diffusion", "mean_b0"}
            )
        )
        has_synb0_anat = bool(t1w_files) or (
            synb0_t1w_source in {"supersynth", "prefer_supersynth"} and bool(t2w_files)
        ) or (uses_dwi_supersynth and bool(dwi_files))

        if dist_method == 'synb0':
            if has_synb0_anat:
                self.logger.info("Adding Synb0EstimationStep + TopupStep")
                self.add_step(Synb0EstimationStep(
                    self.config, self.logger, self.provenance
                ))
                self.add_step(TopupStep(
                    self.config, self.logger, self.provenance
                ))
                context['do_topup'] = True
            else:
                self.logger.warning("Synb0 requested but no usable anatomical reference found")
                
        elif dist_method in {'topup', 'topup+drbuddi'}:
            if has_reverse_pe:
                self.logger.info("Adding TopupStep (native reverse-PE)")
                self.add_step(TopupStep(
                    self.config, self.logger, self.provenance
                ))
                context['do_topup'] = True
            elif fallback and has_synb0_anat:
                self.logger.info("Fallback: Adding Synb0EstimationStep + TopupStep")
                self.add_step(Synb0EstimationStep(
                    self.config, self.logger, self.provenance
                ))
                self.add_step(TopupStep(
                    self.config, self.logger, self.provenance
                ))
                context['do_topup'] = True
            else:
                self.logger.warning("Topup requested but no reverse-PE data")
        elif dist_method == 'drbuddi':
            context['do_drbuddi'] = True
                
        elif dist_method == 'none':
            self.logger.info("Distortion correction disabled")
        else:
            self.logger.warning(f"Unknown distcorr method '{dist_method}'")

        if dist_method == 'topup+drbuddi':
            context['do_drbuddi'] = True

    def _add_merge_step(self, dmri_cfg: dict, context: dict):
        """Add merge step if multiple DWI files need to be combined."""
        dwi_files = context.get("dwi_files", [])
        merge_cfg = dmri_cfg.get('merging', {})
        
        do_merge = False
        if len(dwi_files) > 1:
            if merge_cfg.get('enabled', True) or self._should_merge_for_distcorr(dmri_cfg, context):
                do_merge = True
            
        if do_merge:
            self.logger.info("Adding MergeStep")
            self.add_step(MergeStep(
                self.config, self.logger, self.provenance
            ))

    def _should_merge_for_distcorr(self, dmri_cfg: dict, context: dict) -> bool:
        """Infer whether distortion correction requires a merged series downstream."""
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        dist_method = distcorr_cfg.get('method', 'none')
        fallback = distcorr_cfg.get('fallback', False)
        has_topup_groups = bool(context.get("topup_groups"))
        has_t1w = bool(context.get("t1w_files"))

        if dist_method in {'topup', 'topup+drbuddi'}:
            if has_topup_groups:
                return True
            if fallback and has_t1w:
                return True
            return False
        if dist_method == 'drbuddi':
            return has_topup_groups
        if dist_method == 'synb0':
            return has_t1w
        return False

    def _add_denoising_step(self, dmri_cfg: dict):
        """Add denoising step if enabled."""
        denoise_cfg = dmri_cfg.get('denoising', {})
        if denoise_cfg.get('enabled', False):
            method = denoise_cfg.get('method', 'mrtrix')
            params = denoise_cfg.get('parameters', {})
            
            self.logger.info(f"Adding DenoisingStep (method={method})")
            self.add_step(DenoisingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                patch_radius=params.get('patch_radius', denoise_cfg.get('patch_radius', 2)),
                block_radius=params.get('block_radius', denoise_cfg.get('block_radius', 5)),
                mask_dilation=denoise_cfg.get('mask_dilation', 2),
                pca_method=params.get('pca_method', denoise_cfg.get('pca_method', 'eig'))
            ))

    def _add_gibbs_step(self, dmri_cfg: dict):
        """Add Gibbs unringing step if enabled."""
        degibbs_cfg = dmri_cfg.get('degibbs', {})
        if degibbs_cfg.get('enabled', False):
            method = degibbs_cfg.get('method', 'mrtrix')
            self.logger.info(f"Adding GibbsUnringingStep (method={method})")
            self.add_step(GibbsUnringingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method
            ))

    def _add_motion_correction_step(self, dmri_cfg: dict):
        """Add motion/eddy correction step if enabled."""
        motion_cfg = dmri_cfg.get('motion_correction', {})
        legacy_eddy_cfg = dmri_cfg.get('eddy', {})
        
        motion_method = motion_cfg.get('method')
        if not motion_method:
            if legacy_eddy_cfg.get('enabled', False):
                motion_method = 'eddy'
            else:
                motion_method = 'none'

        if motion_method == 'eddy':
            method = legacy_eddy_cfg.get('method', 'eddy')
            self.logger.info(f"Adding EddyCorrectionStep (method={method})")
            self.add_step(EddyCorrectionStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                mask_dilation=legacy_eddy_cfg.get('mask_dilation', 3)
            ))
            
            if method == 'eddy':
                self.logger.info("Adding EddyQuadStep")
                self.add_step(EddyQuadStep(
                    config=self.config,
                    logger=self.logger,
                    provenance=self.provenance
                ))
                  
        elif motion_method == 'niifreeze':
            self.logger.info("Adding NiiFreezeStep")
            self.add_step(NiiFreezeStep(config=self.config))
             
        elif motion_method == 'none':
            self.logger.info("Motion correction disabled")
        else:
            self.logger.warning(f"Unknown motion method '{motion_method}'")

    def _add_post_eddy_distortion_refinement_step(self, dmri_cfg: dict, context: dict):
        """Add native DRBUDDI-like refinement after Eddy when requested."""
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        dist_method = distcorr_cfg.get('method', 'none')
        if dist_method not in {'drbuddi', 'topup+drbuddi'}:
            return

        motion_cfg = dmri_cfg.get('motion_correction', {})
        legacy_eddy_cfg = dmri_cfg.get('eddy', {})
        motion_method = motion_cfg.get('method')
        if not motion_method:
            motion_method = 'eddy' if legacy_eddy_cfg.get('enabled', False) else 'none'

        if motion_method != 'eddy':
            self.logger.warning("Native DRBUDDI requested but Eddy is not enabled. Skipping native DRBUDDI step.")
            context['do_drbuddi'] = False
            return

        drbuddi_cfg = distcorr_cfg.get('drbuddi', {})
        self.logger.info("Adding NativeDrbuddiStep")
        self.add_step(NativeDrbuddiStep(
            self.config,
            self.logger,
            self.provenance,
            transform_type=drbuddi_cfg.get('transform_type', 'SyNOnly'),
            interpolator=drbuddi_cfg.get('interpolator', 'linear'),
            registration_options=drbuddi_cfg.get('registration_options', {}),
            symmetric_pairwise=drbuddi_cfg.get('symmetric_pairwise', True),
            pe_axis_constraint=drbuddi_cfg.get('pe_axis_constraint', 1.0),
        ))

    def _add_automated_outlier_removal_step(self, dmri_cfg: dict):
        """Add automated outlier removal step if enabled."""
        outlier_cfg = dmri_cfg.get('outliers', {})
        if outlier_cfg.get('enabled', False) and outlier_cfg.get('method') != 'manual':
            method = outlier_cfg.get('method', 'threshold')
            self.logger.info(f"Adding OutlierRemovalStep ({method})")
            self.add_step(OutlierRemovalStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                threshold=outlier_cfg.get('threshold', 0.05)
            ))

    def _add_bias_correction_step(self, dmri_cfg: dict):
        """Add bias field correction step if enabled."""
        bias_cfg = dmri_cfg.get('bias_correction', {})
        do_bias = bias_cfg.get('enabled')
        if do_bias is None:
            do_bias = self.config.get("do_bias_correction", False)
             
        if do_bias and bool(bias_cfg.get('enabled', True)):
            method = bias_cfg.get('method') or self.config.get("bias_method", "ants")
            self.logger.info(f"Adding BiasCorrectionStep (method={method})")
            self.add_step(BiasCorrectionStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method
            ))

    def _add_coregistration_step(self, dmri_cfg: dict, context: dict):
        """Add coregistration step if enabled."""
        coreg_cfg = dmri_cfg.get('coregistration', {})
        do_coreg = coreg_cfg.get('enabled')
        if do_coreg is None:
            do_coreg = self.config.get("do_coregistration", False)

        if do_coreg:
            method = coreg_cfg.get('method') or self.config.get("coreg_method", "ants")
            self.logger.info(f"Adding CoregistrationStep (method={method})")
            self.add_step(CoregistrationStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                options={k: v for k, v in coreg_cfg.items() if k not in {"enabled", "method"}},
            ))

    def _add_gradient_nonlinearity_step(self, dmri_cfg: dict):
        """Add gradient nonlinearity correction step if enabled."""
        gnl_cfg = dmri_cfg.get('grad_nonlin', {})
        if gnl_cfg.get('enabled', False):
            self.logger.info("Adding TortoiseGradNonlinCorrectStep")
            self.add_step(TortoiseGradNonlinCorrectStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                is_resampled=False
            ))

    def _add_brain_masking_step(self, dmri_cfg: dict):
        """Add brain masking step if enabled."""
        mask_cfg = dmri_cfg.get('brain_masking', {})
        do_masking = mask_cfg.get('enabled')
        if do_masking is None:
            do_masking = self.config.get("do_brain_masking", False)
             
        if do_masking:
            method = mask_cfg.get('method') or self.config.get("masking_method", "mrtrix")
            self.logger.info(f"Adding BrainMaskingStep (method={method})")
            self.add_step(BrainMaskingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                apply_mask=mask_cfg.get('apply_mask', True),
                mask_input=mask_cfg.get('mask_input', 'b0'),
                use_gpu=mask_cfg.get('use_gpu')
            ))

    def _add_final_gnl_alignment_step(self, dmri_cfg: dict):
        """Add final GNL alignment step so tensor is in final DWI space before modeling."""
        gnl_cfg = dmri_cfg.get('grad_nonlin', {})
        if gnl_cfg.get('enabled', False):
            self.logger.info("Adding AlignFinalGNLTensorStep")
            self.add_step(AlignFinalGNLTensorStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

    def run(
        self,
        output_dir: Path,
        context: dict,
        reporter=None
    ) -> dict:
        """
        Execute the preprocessing workflow using ExecutionEngine.
        
        Parameters
        ----------
        output_dir : Path
            Working directory for intermediate files
        context : dict
            Processing context with input data
        reporter : ReportGenerator, optional
            Reporter for generating reports
            
        Returns
        -------
        dict
            Updated context after preprocessing
        """
        from ...utils.execution_engine import ExecutionEngine
        from ...utils.data_io import DataIOManager
        
        self.logger.info("Starting PreprocessingWorkflow with ExecutionEngine")
        
        dwi_files = context.get("dwi_files", [])
        if not dwi_files:
            self.logger.warning("No DWI files to process")
            return context
        
        # Report inputs
        if reporter and dwi_files:
            reporter.set_dmri_input_summary(f"DWI Files: {len(dwi_files)}")
            
            # Plot first input
            dwi = dwi_files[0]
            fig_path = output_dir / "report_input_dwi_b0.png"
            try:
                from qmri_neuropipe.lib.reporting.viz import create_ortho_view
                create_ortho_view(dwi.img, fig_path, title="Input DWI (first volume)")
                reporter.add_dmri_input_figure(fig_path, caption=dwi.img.name)
            except Exception as e:
                self.logger.warning(f"Failed to plot input DWI: {e}")
        
        # Create execution engine
        engine = ExecutionEngine(self.config, self.logger)
        
        # Execute steps with progress tracking
        try:
            from qmri_neuropipe.core.ui import console
            from rich.progress import (
                Progress, SpinnerColumn, TextColumn,
                BarColumn, TimeRemainingColumn
            )
            has_rich = True
        except ImportError:
            has_rich = False
            console = None
        
        # Calculate total tasks
        GLOBAL_STEPS = (
            Synb0EstimationStep, TopupStep, GradientCheckStep,
            DMRIReorientStep, MergeStep
        )
        calc_total = sum(
            1 if isinstance(s, GLOBAL_STEPS) else len(dwi_files)
            for s in self.steps
        )
        
        # Execute with or without progress bar
        if has_rich and getattr(console, "is_terminal", True):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True,
                console=console
            ) as progress:
                task = progress.add_task(
                    "[cyan]Preprocessing...",
                    total=calc_total
                )
                
                def progress_callback(description, advance):
                    if description:
                        progress.update(task, description=description)
                    if advance:
                        progress.advance(task)
                
                context = engine.execute_steps(
                    steps=self.steps,
                    context=context,
                    output_dir=output_dir,
                    reporter=reporter,
                    progress_callback=progress_callback
                )
        else:
            context = engine.execute_steps(
                steps=self.steps,
                context=context,
                output_dir=output_dir,
                reporter=reporter
            )
        
        # Save QC metrics if available
        try:
            from qmri_neuropipe.lib.reporting.export import save_qc_metrics_csv
            
            all_qc = context.get("all_qc_metrics", [])
            if all_qc:
                qc_csv = self.config.output_dir / "qc" / "group_qc_metrics.csv"
                qc_csv.parent.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Exporting QC metrics to {qc_csv}")
                save_qc_metrics_csv(all_qc, qc_csv)
        except Exception as e:
            self.logger.warning(f"Failed to save QC CSV: {e}")

        context["processing_steps"] = [step.__class__.__name__ for step in self.steps]
        try:
            io_manager = DataIOManager(self.config, self.logger)
            skip_existing = self.config.get("skip_existing", True)
            io_manager.save_final_outputs(context, self.config.output_dir, skip_existing=skip_existing)
        except Exception as e:
            self.logger.warning(f"Failed to save final outputs: {e}")

        self.logger.info("PreprocessingWorkflow complete")
        return context
