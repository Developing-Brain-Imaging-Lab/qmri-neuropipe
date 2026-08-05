"""
Integrated Preprocessing Workflow with ExecutionEngine.

This module contains the PreprocessingWorkflow class that uses ExecutionEngine
for step execution, providing better separation of concerns and reusability.
"""

from pathlib import Path

from qmri_neuropipe.core import BaseWorkflow, PipelineContext
from qmri_neuropipe.core.step_control import get_rerun_from_step
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.utils.data_io import DataIOManager
from qmri_neuropipe.utils.execution_engine import ExecutionEngine

# Processing steps
from qmri_neuropipe.lib.common.denoise import DenoisingStep
from qmri_neuropipe.lib.common.gibbs import GibbsUnringingStep
from qmri_neuropipe.lib.dmri.eddy import EddyCorrectionStep
from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep
from qmri_neuropipe.lib.dmri.topup import TopupStep
from qmri_neuropipe.lib.dmri.apply_topup import ApplyTopupStep
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
from qmri_neuropipe.lib.dmri.tortoise_v4 import TortoiseV4CorrectionStep
from qmri_neuropipe.lib.dmri.ants_motion import AntsDiffusionMotionCorrectionStep


_TORTOISE_V4_METHODS = {'tortoise_v4', 'tortoise-v4', 'tortoise'}


def _resolve_tortoise_v4_config(dmri_cfg: dict) -> tuple[bool, dict]:
    """Resolve the new preprocessing stream and its legacy motion alias."""
    motion_cfg = dmri_cfg.get('motion_correction', {}) or {}
    legacy = motion_cfg.get('tortoise_v4', {}) or {}
    top_level_present = 'tortoise_v4' in dmri_cfg
    top_level = dmri_cfg.get('tortoise_v4', {}) or {}

    # The top-level preprocessing stream is authoritative when present.  Do
    # not inherit stale integrated Synb0/DRBUDDI flags from the legacy motion
    # alias, since that can silently move a requested post-TORTOISE correction
    # to the pre-TORTOISE DRBUDDI path.
    options = dict(top_level if top_level_present else legacy)
    if top_level_present:
        enabled = bool(options.get('enabled', True))
    else:
        enabled = motion_cfg.get('method') in _TORTOISE_V4_METHODS
    options.pop('enabled', None)

    # PyYAML follows YAML 1.1 and parses an unquoted ``off`` as False.  Accept
    # that spelling as the documented disabled EPI mode.
    if 'epi' in options and (options['epi'] is False or options['epi'] is None):
        options['epi'] = 'off'

    synb0_cfg = options.get('synb0')
    if isinstance(synb0_cfg, dict):
        synb0_enabled = bool(synb0_cfg.get('enabled', True))
        options.setdefault('use_synb0', synb0_enabled)
        if synb0_enabled:
            options.setdefault('epi', 'DRBUDDI')
    if str(options.get('epi', 'off')).lower() == 'drbuddi':
        options.setdefault('use_reverse_pe', not bool(options.get('use_synb0', False)))
    return enabled, options


def _tortoise_coregistration_enabled(options: dict) -> bool:
    nested = options.get('coregistration_to_anatomy') or {}
    if not isinstance(nested, dict):
        return bool(nested or options.get('coregister_to_anatomical', False))
    return bool(
        nested.get('enabled', False)
        or options.get('coregister_to_anatomical', False)
    )


def _post_tortoise_distcorr_requested(dmri_cfg: dict) -> bool:
    """Return whether distortion correction was explicitly placed after TORTOISE."""
    distcorr_cfg = dmri_cfg.get('distcorr', {}) or {}
    application = str(distcorr_cfg.get('application', '')).strip().lower()
    application = application.replace('-', '_')
    return application in {'post_tortoise', 'after_tortoise'} or bool(
        distcorr_cfg.get('after_tortoise', False)
    )


class PreprocessingWorkflow(BaseWorkflow):
    def recover_intermediates(self, work_dir: Path, output_dir: Path):
        """
        Recover intermediate data from the final output directory back to the working directory.
        This allows the pipeline to skip steps that were previously computed and saved.
        """
        self.logger.info("Attempting to recover intermediate data...")

        intermediate_store = output_dir / "intermediate"
        if not intermediate_store.exists():
            self.logger.debug(f"No intermediate storage found at {intermediate_store}")
            return

        recovered = self.recover_intermediate_tree(work_dir, intermediate_store)

        if recovered:
            self.logger.info(f"Recovered {len(recovered)} intermediate step directorie(s).")

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
        self._log_effective_distortion_plan(dmri_cfg)
        
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
        self._add_post_tortoise_distortion_correction_steps(dmri_cfg, context)
        self._add_post_eddy_distortion_refinement_step(dmri_cfg, context)
        self._add_automated_outlier_removal_step(dmri_cfg)
        self._add_bias_correction_step(dmri_cfg)
        self._add_coregistration_step(dmri_cfg, context)
        self._add_brain_masking_step(dmri_cfg)
        self._add_final_gnl_alignment_step(dmri_cfg)
        
        self.logger.info(f"Pipeline built with {len(self.steps)} steps")

    def _log_effective_distortion_plan(self, dmri_cfg: dict) -> None:
        """Report susceptibility routing before any workflow steps are added."""
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        distcorr_cfg = dmri_cfg.get('distcorr', {}) or {}
        method = str(distcorr_cfg.get('method', 'none')).lower()
        epi = str(tortoise_cfg.get('epi', 'off')).lower()

        if _post_tortoise_distcorr_requested(dmri_cfg):
            self.logger.info(
                f"Effective distortion plan: TORTOISEV4 (epi={epi}) -> Synb0 -> "
                "Topup -> ApplyTopup"
            )
        elif tortoise_enabled and epi != 'off':
            source = 'Synb0' if tortoise_cfg.get('use_synb0', False) else 'reverse-PE data'
            self.logger.info(
                f"Effective distortion plan: {source} -> TORTOISEV4 epi={epi.upper()}"
            )
        else:
            self.logger.info(
                f"Effective distortion plan: distcorr.method={method} before motion correction"
            )

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
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        coreg_cfg = tortoise_cfg.get('coregistration_to_anatomy') or {}
        if not isinstance(coreg_cfg, dict):
            coreg_cfg = {}
        if tortoise_enabled and (
            tortoise_cfg.get('output_res')
            or tortoise_cfg.get('output_voxels')
            or _tortoise_coregistration_enabled(tortoise_cfg)
            or str(coreg_cfg.get('output_resolution', '')).lower() in {'native', 'anatomical'}
        ):
            self.logger.info("TORTOISEV4 owns final resampling; skipping pipeline ResampleStep")
            return
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
        if _post_tortoise_distcorr_requested(dmri_cfg):
            self.logger.info("Deferring susceptibility correction until after TORTOISEV4")
            return
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        dist_method = distcorr_cfg.get('method', 'none')
        fallback = distcorr_cfg.get('fallback', False)
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        tortoise_owns_epi = (
            tortoise_enabled
            and str(tortoise_cfg.get('epi', 'off')).lower() != 'off'
        )

        if tortoise_owns_epi:
            context['tortoise_owns_distcorr'] = True
            if tortoise_cfg.get('use_synb0', False):
                if not bool(
                    context.get("t1w_files")
                    or context.get("t2w_files")
                    or context.get("anatomical_files")
                ):
                    raise ValueError("TORTOISEV4 use_synb0 requires an anatomical image")
                self.logger.info("Adding Synb0EstimationStep for TORTOISEV4 DRBUDDI input")
                self.add_step(Synb0EstimationStep(
                    self.config,
                    self.logger,
                    self.provenance,
                    synb0_config=tortoise_cfg.get('synb0'),
                ))
            else:
                self.logger.info("TORTOISEV4 owns susceptibility distortion correction")
            return

        topup_groups = context.get("topup_groups", [])
        has_reverse_pe = len(topup_groups) > 0
        t1w_files = context.get("t1w_files", [])
        # Synb0 requires an undistorted anatomical scan. An acquired T1w is
        # preferred; other anatomical contrasts are converted to T1w by
        # SuperSynth before model inference.
        has_synb0_anat = bool(t1w_files or context.get("anatomical_files"))

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
                self.logger.warning("Synb0 requested but no anatomical image was found")
                
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

    def _add_post_tortoise_distortion_correction_steps(self, dmri_cfg: dict, context: dict):
        """Add the opt-in TORTOISE -> Synb0 -> topup -> applytopup chain."""
        if not _post_tortoise_distcorr_requested(dmri_cfg):
            return

        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        distcorr_cfg = dmri_cfg.get('distcorr', {}) or {}
        method = str(distcorr_cfg.get('method', 'none')).lower()
        epi = str(tortoise_cfg.get('epi', 'off')).lower()

        if not tortoise_enabled:
            raise ValueError("distcorr.application=post_tortoise requires TORTOISEV4")
        if epi != 'off':
            raise ValueError(
                "Post-TORTOISE Topup requires tortoise_v4.epi: off so susceptibility "
                "distortion is not corrected twice"
            )
        if method != 'synb0':
            raise ValueError(
                "Post-TORTOISE distortion correction currently supports only "
                "distcorr.method: synb0; native reverse-PE support requires a future "
                "two-stream TORTOISE workflow"
            )
        if tortoise_cfg.get('use_synb0') or tortoise_cfg.get('use_reverse_pe'):
            raise ValueError(
                "Configure Synb0 under distcorr for the post-TORTOISE workflow; "
                "do not enable tortoise_v4.use_synb0 or use_reverse_pe"
            )
        if context.get('topup_groups'):
            raise ValueError(
                "Post-TORTOISE Synb0 currently supports a single acquired PE stream, "
                "not native reverse-PE topup groups"
            )
        if not bool(
            context.get("t1w_files")
            or context.get("t2w_files")
            or context.get("anatomical_files")
        ):
            raise ValueError("Post-TORTOISE Synb0 requires an anatomical image")

        context['do_topup'] = True
        self.logger.info("Adding post-TORTOISE Synb0 estimation, Topup, and ApplyTopup")
        synb0_cfg = dict(distcorr_cfg.get('synb0') or {})
        if isinstance(distcorr_cfg.get('skull_strip'), dict):
            synb0_cfg.setdefault('skull_strip', distcorr_cfg['skull_strip'])
        self.add_step(Synb0EstimationStep(
            self.config,
            self.logger,
            self.provenance,
            synb0_config=synb0_cfg,
        ))
        self.add_step(TopupStep(self.config, self.logger, self.provenance))
        self.add_step(ApplyTopupStep(
            self.config,
            self.logger,
            self.provenance,
            method=distcorr_cfg.get('apply_method', 'jac'),
        ))

    def _add_merge_step(self, dmri_cfg: dict, context: dict):
        """Add merge step if multiple DWI files need to be combined."""
        dwi_files = context.get("dwi_files", [])
        merge_cfg = dmri_cfg.get('merging', {})
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        if (
            tortoise_enabled
            and (tortoise_cfg.get('use_reverse_pe', False) or tortoise_cfg.get('use_synb0', False))
        ):
            self.logger.info("Keeping PE series separate for TORTOISEV4 up/down processing")
            return
        
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
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        if (
            tortoise_enabled
            and str(tortoise_cfg.get('denoising', 'off')).lower() != 'off'
        ):
            self.logger.info("TORTOISEV4 owns denoising; skipping pipeline DenoisingStep")
            return
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
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        if (
            tortoise_enabled
            and bool(tortoise_cfg.get('gibbs', False))
        ):
            self.logger.info("TORTOISEV4 owns Gibbs correction; skipping pipeline GibbsUnringingStep")
            return
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

        tortoise_enabled, tortoise_options = _resolve_tortoise_v4_config(dmri_cfg)
        if tortoise_enabled:
            tortoise_options.setdefault(
                'reference_selection', motion_cfg.get('reference_selection', {})
            )
            self.logger.info("Adding integrated TORTOISEV4 preprocessing stream")
            self.add_step(TortoiseV4CorrectionStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                **tortoise_options,
            ))
            return
        
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

        elif motion_method == 'ants':
            options = dict(motion_cfg.get('ants', {}))
            self.logger.info("Adding AntsDiffusionMotionCorrectionStep (rigid motion)")
            self.add_step(AntsDiffusionMotionCorrectionStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                mode='motion',
                slice_to_volume=options.get('slice_to_volume', False),
                reference_selection=motion_cfg.get('reference_selection', {}),
                transform_type=options.get('transform_type', 'Rigid'),
                interpolator=options.get('interpolator', 'linear'),
                registration_options=options.get('registration_options', {}),
            ))

        elif motion_method in {'native', 'ants_native', 'native_ants'}:
            options = dict(motion_cfg.get('native', {}))
            self.logger.info("Adding native ANTs motion/eddy correction")
            self.add_step(AntsDiffusionMotionCorrectionStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                mode='motion_eddy',
                slice_to_volume=options.get('slice_to_volume', True),
                reference_selection=motion_cfg.get('reference_selection', {}),
                transform_type=options.get('transform_type', 'Affine'),
                interpolator=options.get('interpolator', 'linear'),
                registration_options=options.get('registration_options', {}),
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
        if context.get('tortoise_owns_distcorr'):
            return
        distcorr_cfg = dmri_cfg.get('distcorr', {})
        dist_method = distcorr_cfg.get('method', 'none')
        if dist_method not in {'drbuddi', 'topup+drbuddi'}:
            return

        motion_cfg = dmri_cfg.get('motion_correction', {})
        legacy_eddy_cfg = dmri_cfg.get('eddy', {})
        motion_method = motion_cfg.get('method')
        if not motion_method:
            motion_method = 'eddy' if legacy_eddy_cfg.get('enabled', False) else 'none'

        if motion_method not in {'eddy', 'tortoise_v4', 'tortoise-v4', 'tortoise', 'native', 'ants_native', 'native_ants'}:
            self.logger.warning("Native DRBUDDI requested but no compatible motion/eddy method is enabled. Skipping native DRBUDDI step.")
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
        tortoise_enabled, tortoise_cfg = _resolve_tortoise_v4_config(dmri_cfg)
        if tortoise_enabled and _tortoise_coregistration_enabled(tortoise_cfg):
            self.logger.info("TORTOISEV4 owns anatomical alignment; skipping pipeline CoregistrationStep")
            return
        coreg_cfg = dmri_cfg.get('coregistration', {})
        do_coreg = coreg_cfg.get('enabled')
        if do_coreg is None:
            do_coreg = self.config.get("do_coregistration", False)

        if do_coreg:
            method = coreg_cfg.get('method') or self.config.get("coreg_method", "ants")
            coreg_options = dict(coreg_cfg)
            nested_options = coreg_options.pop("options", None)
            if isinstance(nested_options, dict):
                merged_options = dict(nested_options)
                merged_options.update(coreg_options)
                coreg_options = merged_options
            self.logger.info(f"Adding CoregistrationStep (method={method})")
            self.add_step(CoregistrationStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                options={k: v for k, v in coreg_options.items() if k not in {"enabled", "method"}},
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
    ) -> PipelineContext:
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
        context = PipelineContext.ensure(context)
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
            Synb0EstimationStep, TopupStep, ApplyTopupStep, GradientCheckStep,
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
            io_manager.normalize_context_derivative_entities(context)
            rerun_from_step = get_rerun_from_step(
                self.config,
                "dmri.preprocessing",
                "preprocessing",
            )
            skip_existing = (
                self.config.get("skip_existing", True)
                and not self.config.get("force", False)
                and not rerun_from_step
            )
            if rerun_from_step:
                self.logger.info(
                    "Publishing refreshed final outputs after rerun_from_step=%s.",
                    rerun_from_step,
                )
            io_manager.save_final_outputs(context, self.config.output_dir, skip_existing=skip_existing)
        except Exception as e:
            self.logger.warning(f"Failed to save final outputs: {e}")

        self.logger.info("PreprocessingWorkflow complete")
        return PipelineContext.ensure(context)
