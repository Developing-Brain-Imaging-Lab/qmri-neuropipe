"""
Anatomical Processing Pipeline (anat_proc).

1) Resample/resize
2) Reorient to standard
3) Denoise
4) Gibbs correction
5) Bias correction
6) Coregistration (T1w <-> T2w)
7) Brain Mask
8) FreeSurfer Recon-all (optional)
9) Nonlinear Registration (to template)
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging
import shutil
import json
import time
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

from ...core import BasePipeline, BaseWorkflow, PipelineConfig, ProcessingError
from ...core.step_control import get_rerun_from_step, step_force_active, step_matches, any_step_matches
from ...core.types import ImageFile
from ...io.anat.bids import bids_find_t1w, bids_find_t2w, select_anatomical_candidates
from ...io.bids import build_bids_name

# Steps
from ...lib.common.resample import ResampleStep
from ...lib.common.tracking import TrackingStep
from ...lib.common.reorient import ReorientStep
from ...lib.common.denoise import DenoisingStep
from ...lib.common.gibbs import GibbsUnringingStep
from ...lib.common.bias import BiasCorrectionStep
from ...lib.common.registration import CoregistrationStep, NonlinearRegistrationStep
from ...lib.common.mask import BrainMaskingStep
from ...lib.anat.recon import ReconAllStep, FreeSurferStatsStep
from ...lib.anat.super_synth import SuperSynthStep
from ...lib.common.sharpen import SharpeningStep
from ...lib.common.segmentation import SegmentationStep
from ...interfaces.mriqc import run_mriqc
from ...interfaces import ants, fsl  # For manual apply_transforms
from ...lib.reporting.report import ReportGenerator

# Rich and Viz imports moved to local scope below

@dataclass
class PreprocessingConfig:
    resample: Dict[str, Any] = field(default_factory=dict)
    reorient: Dict[str, Any] = field(default_factory=dict)
    denoising: Dict[str, Any] = field(default_factory=dict)
    degibbs: Dict[str, Any] = field(default_factory=dict)
    gibbs: Dict[str, Any] = field(default_factory=dict)
    bias_correction: Dict[str, Any] = field(default_factory=dict)
    sharpen: Dict[str, Any] = field(default_factory=dict)
    brain_masking: Dict[str, Any] = field(default_factory=dict)
    coregistration: Dict[str, Any] = field(default_factory=dict)
    normalization: Dict[str, Any] = field(default_factory=dict)
    recon_all: Dict[str, Any] = field(default_factory=dict)
    use_freesurfer: bool = False
    force_run: bool = False
    skull_stripped_outputs: bool = False

@dataclass
class NormalizationConfig:
    enabled: bool = False
    template: Optional[str] = None
    method: str = "ants"
    options: Dict[str, Any] = field(default_factory=dict)
    save_transforms: bool = True
    skull_stripped_outputs: bool = False

@dataclass
class SegmentationConfig:
    enabled: bool = False
    atlas_file: Optional[str] = None
    atlas_labels: Optional[Union[List[str], str]] = None
    metrics: Optional[List[str]] = None
    atlas_threshold: Optional[float] = None

@dataclass
class QCConfig:
    enabled: bool = False
    modalities: Optional[List[str]] = None

@dataclass
class FreeSurferConfig:
    enabled: bool = False


@dataclass
class SuperSynthConfig:
    enabled: bool = False
    mode: str = "invivo"
    sharpen_synths: bool = False
    device: Optional[str] = None
    compute_volumes: bool = False


@dataclass
class AnatomicalConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    freesurfer: FreeSurferConfig = field(default_factory=FreeSurferConfig)
    super_synth: SuperSynthConfig = field(default_factory=SuperSynthConfig)


class AnatPreprocessingWorkflow(BaseWorkflow):
    """
    Workflow for preprocessing T1w (and optionally T2w) images.
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger, provenance: Any):
        self.anat_config = AnatomicalConfig()
        super().__init__(config, logger, provenance)
        # Parse relevant anatomical config section(s) into AnatomicalConfig dataclass
        anat_raw_cfg = self.config.get("anat", {})
        preprocessing_cfg = anat_raw_cfg.get("preprocessing", {})
        normalization_cfg = anat_raw_cfg.get("preprocessing", {}).get("normalization", {})
        segmentation_cfg = anat_raw_cfg.get("segmentation", {})
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        freesurfer_use = preprocessing_cfg.get("use_freesurfer", False) or anat_raw_cfg.get("use_freesurfer", False)
        ss_cfg = anat_raw_cfg.get("super_synth", {})
        brain_masking_cfg = preprocessing_cfg.get("brain_masking", {})
        
        self.anat_config = AnatomicalConfig(
            preprocessing=PreprocessingConfig(
                resample=preprocessing_cfg.get("resample", {}),
                reorient=preprocessing_cfg.get("reorient", {}),
                denoising=preprocessing_cfg.get("denoising", {}),
                degibbs=preprocessing_cfg.get("degibbs", {}),
                gibbs=preprocessing_cfg.get("gibbs", {}),
                bias_correction=preprocessing_cfg.get("bias_correction", {}),
                sharpen=preprocessing_cfg.get("sharpen", {}),
                brain_masking=brain_masking_cfg,
                coregistration=preprocessing_cfg.get("coregistration", {}),
                normalization=normalization_cfg,
                recon_all=preprocessing_cfg.get("recon_all", {}),
                use_freesurfer=freesurfer_use,
                force_run=preprocessing_cfg.get("force_run", False),
                skull_stripped_outputs=preprocessing_cfg.get(
                    "skull_stripped_outputs",
                    brain_masking_cfg.get("output_skull_stripped", False),
                ),
            ),
            normalization=NormalizationConfig(
                enabled=normalization_cfg.get("enabled", False),
                template=normalization_cfg.get("template"),
                method=normalization_cfg.get("method", "ants"),
                options=normalization_cfg.get("options", {}),
                save_transforms=normalization_cfg.get(
                    "save_transforms",
                    normalization_cfg.get("save_transform", True),
                ),
                skull_stripped_outputs=normalization_cfg.get(
                    "skull_stripped_outputs",
                    preprocessing_cfg.get(
                        "skull_stripped_outputs",
                        brain_masking_cfg.get("output_skull_stripped", False),
                    ),
                ),
            ),
            segmentation=SegmentationConfig(
                enabled=segmentation_cfg.get("enabled", False),
                atlas_file=segmentation_cfg.get("atlas_file"),
                atlas_labels=segmentation_cfg.get("atlas_labels"),
                metrics=segmentation_cfg.get("metrics"),
                atlas_threshold=segmentation_cfg.get("atlas_threshold"),
            ),
            qc=QCConfig(
                enabled=qc_cfg.get("enabled", False),
                modalities=qc_cfg.get("modalities"),
            ),
            freesurfer=FreeSurferConfig(
                enabled=preprocessing_cfg.get("recon_all", {}).get("enabled", False) or freesurfer_use
            ),
            super_synth=SuperSynthConfig(
                enabled=ss_cfg.get("enabled", False),
                mode=ss_cfg.get("mode", "invivo"),
                sharpen_synths=ss_cfg.get("sharpen_synths", False),
                device=ss_cfg.get("device"),
                compute_volumes=ss_cfg.get("compute_volumes", False),
            ),
        )

        self.steps = []
        self._initialize_steps()

    def _add_anat_step(self, *args, **kwargs) -> None:
        figures = kwargs.pop("figures", None)
        extra_details = kwargs.pop("details", None)

        reporter = None
        step_name = None
        base_details = None

        if len(args) >= 3 and hasattr(args[0], "add_anat_step"):
            reporter, step_name, base_details = args[:3]
        elif len(args) >= 2:
            step_name, base_details = args[:2]
            reporter = kwargs.pop("reporter", None) or getattr(self, "reporter", None)
        else:
            return

        if not reporter:
            return

        details = dict(base_details)
        if isinstance(extra_details, dict):
            details.update(extra_details)

        reporter.add_anat_step(step_name, details, figures=figures)

    @staticmethod
    def _has_input_modality(context: dict, suffix: str) -> bool:
        return bool(context.get(f"{suffix.lower()}_files", []))

    def _initialize_steps(self):
        pre_cfg = self.anat_config.preprocessing

        # 1. Resample
        if pre_cfg.resample.get("enabled"):
            self.add_step(
                ResampleStep(
                    self.config,
                    self.logger,
                    self.provenance,
                    resolution=pre_cfg.resample.get("resolution"))
            )

        # 2. Reorient
        if pre_cfg.reorient.get("enabled", False):
            self.add_step(ReorientStep(self.config, self.logger, self.provenance))

        # 3. Denoise
        den_cfg = pre_cfg.denoising
        if den_cfg.get("enabled"):
            self.add_step(DenoisingStep(self.config, self.logger, self.provenance, method=den_cfg.get("method", "ants")))

        # 4. Gibbs
        gibbs_cfg = pre_cfg.degibbs or pre_cfg.gibbs
        if gibbs_cfg.get("enabled"):
            self.add_step(GibbsUnringingStep(self.config, self.logger, self.provenance, method=gibbs_cfg.get("method", "mrtrix")))

        # 5. Bias
        bias_cfg = pre_cfg.bias_correction
        if bias_cfg.get("enabled"):
            self.add_step(BiasCorrectionStep(self.config, self.logger, self.provenance, method=bias_cfg.get("method", "ants")))

        # 5b. Sharpening (Optional)
        sharp_cfg = pre_cfg.sharpen
        if sharp_cfg.get("enabled"):
            self.add_step(SharpeningStep(self.config, self.logger, self.provenance, method=sharp_cfg.get("method", "ants")))

        # 6. Brain Mask
        mask_cfg = pre_cfg.brain_masking
        use_fs = pre_cfg.use_freesurfer
        if mask_cfg.get("enabled", False) and not use_fs:
            self.add_step(BrainMaskingStep(
                self.config,
                self.logger,
                self.provenance,
                method=mask_cfg.get("method", "ants"),
                use_gpu=mask_cfg.get("use_gpu")
            ))

        # 7. Recon-all
        recon_cfg = pre_cfg.recon_all
        if recon_cfg.get("enabled") or use_fs:
            self.add_step(ReconAllStep(self.config, self.logger, self.provenance))
            # Always add stats extraction if FS is enabled
            self.add_step(FreeSurferStatsStep(self.config, self.logger, self.provenance))

        # 8. Nonlinear Registration
        norm_cfg = pre_cfg.normalization
        if norm_cfg.get("enabled"):
            self.add_step(
                NonlinearRegistrationStep(
                    self.config,
                    self.logger,
                    self.provenance,
                    template=norm_cfg.get("template"),
                    method=norm_cfg.get("method", "ants"),
                    options=norm_cfg.get("options", {}),
                    save_transforms=norm_cfg.get(
                        "save_transforms",
                        norm_cfg.get("save_transform", True),
                    ),
                    space_entity=norm_cfg.get("space_entity", norm_cfg.get("space_name", norm_cfg.get("space", "Standard"))),
                )
            )

        # 9. Segmentation
        seg_cfg = self.anat_config.segmentation
        if seg_cfg.enabled:
            self.add_step(SegmentationStep(
                self.config, self.logger, self.provenance,
                atlas_file=seg_cfg.atlas_file,
                atlas_labels=seg_cfg.atlas_labels,
                metrics=seg_cfg.metrics,
                atlas_threshold=seg_cfg.atlas_threshold
            ))

        # 10. SuperSynth segmentation + optional volume extraction
        if self.anat_config.super_synth.enabled:
            self.add_step(SuperSynthStep(self.config, self.logger, self.provenance))

    def _validate_outputs(self, context: dict, step_metrics: List[dict], reporter=None) -> None:
        """
        Validate the presence of expected outputs and record errors.

        Args:
            context: Processing context dictionary.
            step_metrics: List of step metrics dictionaries.
            reporter: Optional reporter to log validation errors.
        """
        errors = context.setdefault('errors', [])

        def _check_image_file(img_obj, label: str):
            if img_obj is None:
                msg = f"{label} missing in context."
                self.logger.warning(msg)
                errors.append(msg)
                if reporter:
                    self._add_anat_step(f"Validation_{label}", {"Status": "Missing"}, details={"message": msg})
                return False
            if not hasattr(img_obj, 'img') or not img_obj.img.exists():
                msg = f"{label} image file does not exist at {getattr(img_obj, 'img', 'Unknown')}."
                self.logger.warning(msg)
                errors.append(msg)
                if reporter:
                    self._add_anat_step(f"Validation_{label}", {"Status": "File Missing"}, details={"message": msg})
                return False
            return True

        # Validate preprocessed T1w only when T1w input was selected.
        if self._has_input_modality(context, "T1w"):
            _check_image_file(context.get("preprocessed_t1w"), "Preprocessed_T1w")

        # Validate preprocessed T2w only when T2w input was selected.
        if self._has_input_modality(context, "T2w"):
            pre_t2 = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
            _check_image_file(pre_t2, "Preprocessed_T2w")

        # Validate brain mask
        _check_image_file(context.get("brain_mask"), "Brain_Mask")

        # Validate segmentation if present
        if "segmentation" in context:
            seg = context["segmentation"]
            _check_image_file(seg, "Segmentation")

        # Validate 5tt if present
        if "5tt" in context:
            f5tt = context["5tt"]
            _check_image_file(f5tt, "5tt")

        # Report summary errors if any
        if errors and reporter:
            error_rows = [{"Error": err} for err in errors]
            reporter.add_anat_summary("Validation Errors", error_rows)

    def _run_mriqc_qc(self, images: List[ImageFile], output_dir: Path, context: dict, reporter=None):
        """
        Run MRIQC on given images if QC enabled and modality matches.

        Args:
            images: list of ImageFile objects.
            output_dir: directory to store QC outputs.
            context: processing context dict to update with QC results.
            reporter: optional reporter to log results.
        """
        qc_cfg = self.anat_config.qc
        if not qc_cfg.enabled:
            return

        qc_modalities = set(qc_cfg.modalities or [])
        imgs_to_qc = []
        for img in images:
            if img is None or not hasattr(img, 'entities'):
                continue
            # Determine modality from entities or suffix
            mod = None
            if 'suffix' in img.entities:
                mod = img.entities.get('suffix')
            elif 'modality' in img.entities:
                mod = img.entities.get('modality')
            if mod and mod.upper() in qc_modalities:
                imgs_to_qc.append(img.img)

        if not imgs_to_qc:
            self.logger.info("No images matched QC modalities, skipping MRIQC.")
            return

        # Prepare MRIQC output dir
        mriqc_out_dir = output_dir / "mriqc"
        mriqc_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            run_mriqc(
                bids_dir=self.config.bids_dir,
                output_dir=mriqc_out_dir,
                participant_label=context.get('subject'),
                session_id=context.get('session'),
                n_procs=self.config.n_cpus,
                modalities=list(qc_modalities),
                verbose_reports=self.config.verbose
            )

            # Attach QC outputs to context for reporting
            context.setdefault('qc_outputs', []).append(str(mriqc_out_dir))

            if reporter:
                self._add_anat_step("MRIQC", {"Status": "Completed", "QC_Dir": str(mriqc_out_dir)},
                                      details={"Modalities": list(qc_modalities)})
        except Exception as e:
            err_msg = f"MRIQC failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)
            if reporter:
                self._add_anat_step("MRIQC", {"Status": "Failed"}, details={"error": str(e)})

    def _preprocess_t1w(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, figures_dir: Path, step_metrics: Optional[List[dict]] = None) -> (dict, List[dict]):
        """
        Preprocess T1w images with the configured steps.

        Returns:
            Updated context and metrics list.
        """
        if step_metrics is None:
            step_metrics = []

        errors = context.setdefault('errors', [])

        t1w_files = context.get('t1w_files', [])
        if not t1w_files:
            msg = "No T1w image found."
            self.logger.error(msg)
            errors.append(msg)
            raise ValueError(msg)

        use_fs = self.anat_config.preprocessing.use_freesurfer
        save_inter = self.config.get("save_intermediate", False)
        skip_existing = self.config.get("skip_existing", False)
        force_run = self.anat_config.preprocessing.force_run
        if force_run:
            self.logger.info("Anatomical force_run enabled: Ignoring existing outputs.")
            skip_existing = False

        current_t1 = t1w_files[0]
        context["current_image"] = current_t1
        processed_t1 = current_t1
        rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
        force_from_step_active = bool(getattr(self, "_anat_force_from_step_active", False))

        for step in self.steps:
            if isinstance(step, (CoregistrationStep, BrainMaskingStep, NonlinearRegistrationStep, SegmentationStep, FreeSurferStatsStep)):
                continue
            force_from_step_active = step_force_active(force_from_step_active, step, rerun_from_step)
            self._anat_force_from_step_active = force_from_step_active

            # If using FS, skip standard T1w preproc steps
            if use_fs and isinstance(step, (ResampleStep, ReorientStep, DenoisingStep, GibbsUnringingStep, BiasCorrectionStep, SharpeningStep)):
                tracker = self.config.tracker
                if tracker and context.get('subject') and context.get('session'):
                    tracker_module = step.normalize_tracker_module(step.__class__.__name__)
                    study = context.get('study_name', self.config.get('study_name'))
                    tracker.update_status(context['subject'], context['session'], tracker_module,
                                          "completed (FreeSurfer)", study=study, modality="Anatomical")
                continue

            step_name = step.__class__.__name__
            self.logger.info(f"Running T1w step: {step_name}")

            # Determine descriptor for this step
            step_desc = None
            if isinstance(step, ResampleStep):
                step_desc = 'resample'
            elif isinstance(step, ReorientStep):
                step_desc = 'reorient'
            elif isinstance(step, DenoisingStep):
                step_desc = 'denoise'
            elif isinstance(step, GibbsUnringingStep):
                step_desc = 'gibbs'
            elif isinstance(step, BiasCorrectionStep):
                step_desc = 'bias'
            elif isinstance(step, SharpeningStep):
                step_desc = 'sharpen'
            elif isinstance(step, ReconAllStep):
                step_desc = None  # Saved differently
            elif isinstance(step, NonlinearRegistrationStep):
                step_desc = 'normalize'

            skipped = False
            if final_output_dir and step_desc and not force_from_step_active:
                ents = dict(processed_t1.entities)
                ents['desc'] = step_desc
                if 'suffix' not in ents:
                    ents['suffix'] = 'T1w'

                fname = build_bids_name(ents)
                expected_path = final_output_dir / fname

                # Check for existence (standard or final preproc fallback)
                if not expected_path.exists() and skip_existing and step == self.steps[-1]:
                    p_ents = dict(processed_t1.entities)
                    p_ents['desc'] = 'preproc'
                    if 'suffix' not in p_ents:
                        p_ents['suffix'] = 'T1w'
                    p_fname = build_bids_name(p_ents)
                    p_dest = final_output_dir / p_fname
                    if p_dest.exists():
                        expected_path = p_dest
                        fname = p_fname

                if expected_path.exists() and (skip_existing or save_inter):
                    self.logger.info(f"Skipping {step_name} (Found existing output: {fname})")
                    try:
                        # Wrap as ImageFile
                        img_obj = ImageFile(entities=ents, img=expected_path)
                        processed_t1 = img_obj
                        skipped = True

                        step_metrics.append({
                            "Step": f"T1w_{step_name}",
                            "Status": "Skipped (Found)",
                            "Duration": "0s"
                        })

                        tracker_module = step.normalize_tracker_module(step_name)
                        tracker = self.config.tracker
                        subject = context.get('subject')
                        session = context.get('session')
                        study = context.get('study_name', self.config.get('study_name'))

                        if tracker and subject and session:
                            tracker.update_status(subject, session, tracker_module, "completed (cached)",
                                                  study=study, modality="Anatomical")
                            tracker.save()
                    except Exception as e:
                        self.logger.warning(f"Failed to load existing intermediate {fname}: {e}. Re-running.")

            if not skipped:
                st = time.time()
                try:
                    step_force = force_run or force_from_step_active
                    if force_from_step_active:
                        self.logger.info(f"Forcing T1w {step_name} because rerun_from_step has been reached.")
                    if isinstance(step, ReconAllStep):
                        context["current_image"] = processed_t1
                        processed_t1 = step(context, output_dir=output_dir, force=step_force)
                    else:
                        processed_t1 = step(processed_t1, output_dir=output_dir, force=step_force)
                except Exception as e:
                    err_msg = f"Error during T1w {step_name}: {e}"
                    self.logger.error(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step(f"T1w_{step_name}", {"Status": "Failed"}, details={"error": str(e)})
                    continue

                dur = time.time() - st

                step_metrics.append({
                    "Step": f"T1w_{step_name}",
                    "Status": "Completed",
                    "Duration": f"{dur:.2f}s"
                })

                if isinstance(processed_t1, dict):
                    context.update(processed_t1)
                    processed_t1 = context.get("current_image")

                # Validate output file existence after step
                if hasattr(processed_t1, 'img') and not processed_t1.img.exists():
                    err_msg = f"Output file missing after T1w {step_name}: {processed_t1.img}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step(f"T1w_{step_name}", {"Status": "Output Missing"}, details={"path": str(processed_t1.img)})

            # Save intermediate results
            if save_inter and not skipped and final_output_dir and step_desc:
                try:
                    ents = dict(processed_t1.entities)
                    ents['desc'] = step_desc
                    if 'suffix' not in ents:
                        ents['suffix'] = 'T1w'
                    fname = build_bids_name(ents)
                    dest = final_output_dir / fname

                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(processed_t1.img, dest)
                        if processed_t1.json and processed_t1.json.exists():
                            shutil.copy(processed_t1.json, dest.with_suffix("").with_suffix(".json"))
                except Exception as e:
                    self.logger.warning(f"Failed to save intermediate T1w {step_name} output: {e}")

            # Reporting: T1w
            if reporter and not skipped:
                try:
                    from ...lib.reporting.viz import create_ortho_view

                    fig_list = []

                    if isinstance(step, DenoisingStep):
                        p = figures_dir / "t1w_denoised.png"
                        create_ortho_view(processed_t1.img, p, title="Denoised T1w")
                        fig_list.append({"path": str(p), "title": "Denoising", "caption": "Denoised T1w"})

                    elif isinstance(step, GibbsUnringingStep):
                        p = figures_dir / "t1w_gibbs.png"
                        create_ortho_view(processed_t1.img, p, title="Gibbs Corrected T1w")
                        fig_list.append({"path": str(p), "title": "Gibbs", "caption": "Gibbs Corrected T1w"})

                    details = {"Modality": "T1w"}
                    if hasattr(step, "method"):
                        details["Method"] = step.method
                    if hasattr(step, "patch_radius"):
                        details["Patch Radius"] = step.patch_radius

                    self._add_anat_step(step_name, details, figures=fig_list)

                except ImportError:
                    pass
                except Exception as e:
                    self.logger.warning(f"Failed to plot T1w report figure: {e}")

        context["preprocessed_t1w"] = processed_t1
        self.logger.info(f"T1w processing complete: {processed_t1.img}")

        # Run QC if enabled on T1w
        try:
            self._run_mriqc_qc([processed_t1], output_dir, context, reporter)
        except Exception as e:
            err_msg = f"T1w QC step failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)

        # Validate outputs after all steps
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _preprocess_t2w(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, figures_dir: Path, step_metrics: Optional[List[dict]] = None) -> (dict, List[dict]):
        """
        Preprocess T2w images with the configured steps.

        Returns:
            Updated context and metrics list.
        """
        if step_metrics is None:
            step_metrics = []

        errors = context.setdefault('errors', [])

        t2w_files = context.get('t2w_files', [])
        if not t2w_files:
            return context, step_metrics

        save_inter = self.config.get("save_intermediate", False)
        skip_existing = self.config.get("skip_existing", False)
        force_run = self.anat_config.preprocessing.force_run

        current_t2 = t2w_files[0]
        processed_t2 = current_t2
        rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
        force_from_step_active = bool(getattr(self, "_anat_force_from_step_active", False))

        for step in self.steps:
            if isinstance(step, (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep, CoregistrationStep, SegmentationStep, FreeSurferStatsStep)):
                continue
            force_from_step_active = step_force_active(force_from_step_active, step, rerun_from_step)
            self._anat_force_from_step_active = force_from_step_active

            step_name = step.__class__.__name__
            self.logger.info(f"Running T2w step: {step_name}")

            # Determine descriptor for this step
            step_desc = None
            if isinstance(step, ResampleStep):
                step_desc = 'resample'
            elif isinstance(step, ReorientStep):
                step_desc = 'reorient'
            elif isinstance(step, DenoisingStep):
                step_desc = 'denoise'
            elif isinstance(step, GibbsUnringingStep):
                step_desc = 'gibbs'
            elif isinstance(step, BiasCorrectionStep):
                step_desc = 'bias'
            elif isinstance(step, SharpeningStep):
                step_desc = 'sharpen'

            skipped = False
            if final_output_dir and step_desc and not force_from_step_active:
                ents = dict(processed_t2.entities)
                ents['desc'] = step_desc
                if 'suffix' not in ents:
                    ents['suffix'] = 'T2w'

                fname = build_bids_name(ents)
                expected_path = final_output_dir / fname

                if expected_path.exists() and (skip_existing or save_inter):
                    self.logger.info(f"Skipping {step_name} (Found existing output: {fname})")
                    try:
                        img_obj = ImageFile(expected_path, self.config.bids_dir)
                        processed_t2 = img_obj
                        skipped = True
                        step_metrics.append({"Step": f"T2w_{step_name}", "Status": "Skipped (Found)", "Duration": "0s"})

                        tracker_module = step.normalize_tracker_module(step_name)
                        tracker = self.config.tracker
                        subject = context.get('subject')
                        session = context.get('session')
                        study = context.get('study_name', self.config.get('study_name'))

                        if tracker and subject and session:
                            tracker.update_status(subject, session, tracker_module, "completed (cached)", study=study, modality="Anatomical")
                            tracker.save()
                    except Exception:
                        pass

            if not skipped:
                st = time.time()
                try:
                    step_force = force_run or force_from_step_active
                    if force_from_step_active:
                        self.logger.info(f"Forcing T2w {step_name} because rerun_from_step has been reached.")
                    processed_t2 = step(processed_t2, output_dir=output_dir, force=step_force)
                except Exception as e:
                    err_msg = f"Error during T2w {step_name}: {e}"
                    self.logger.error(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step(f"T2w_{step_name}", {"Status": "Failed"}, details={"error": str(e)})
                    continue
                dur = time.time() - st

                step_metrics.append({
                    "Step": f"T2w_{step_name}",
                    "Status": "Completed",
                    "Duration": f"{dur:.2f}s"
                })

                if isinstance(processed_t2, dict):
                    processed_t2 = processed_t2.get("current_image", processed_t2)

                # Validate output file existence after step
                if hasattr(processed_t2, 'img') and not processed_t2.img.exists():
                    err_msg = f"Output file missing after T2w {step_name}: {processed_t2.img}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step(f"T2w_{step_name}", {"Status": "Output Missing"}, details={"path": str(processed_t2.img)})

            # Save intermediate T2w
            if save_inter and not skipped and final_output_dir and step_desc:
                try:
                    ents = dict(processed_t2.entities)
                    ents['desc'] = step_desc
                    if 'suffix' not in ents:
                        ents['suffix'] = 'T2w'
                    fname = build_bids_name(ents)
                    dest = final_output_dir / fname

                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(processed_t2.img, dest)
                        if processed_t2.json and processed_t2.json.exists():
                            shutil.copy(processed_t2.json, dest.with_suffix("").with_suffix(".json"))
                except Exception as e:
                    self.logger.warning(f"Failed to save intermediate T2w {step_name} output: {e}")

            # Reporting: T2w
            if reporter:
                try:
                    from ...lib.reporting.viz import create_ortho_view

                    fig_list = []
                    if isinstance(step, DenoisingStep):
                        p = figures_dir / "t2w_denoised.png"
                        create_ortho_view(processed_t2.img, p, title="Denoised T2w")
                        fig_list.append({"path": str(p), "title": "Denoising", "caption": "Denoised T2w"})

                    details = {"Modality": "T2w"}
                    if hasattr(step, "method"):
                        details["Method"] = step.method

                    self._add_anat_step(step_name, details, figures=fig_list)
                except Exception as e:
                    self.logger.warning(f"Failed to plot T2w report figure: {e}")

        self.logger.info(f"T2w processing complete: {processed_t2.img}")

        context["preprocessed_t2w"] = processed_t2

        # Run QC if enabled on T2w
        try:
            self._run_mriqc_qc([processed_t2], output_dir, context, reporter)
        except Exception as e:
            err_msg = f"T2w QC step failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)

        # Validate outputs after all steps
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _run_coregistration(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, figures_dir: Path, step_metrics: List[dict]) -> (dict, List[dict]):
        """
        Run coregistration step between T1w and T2w if enabled.

        Returns:
            Updated context and metrics list.
        """
        t1w_files = context.get('t1w_files', [])
        t2w_files = context.get('t2w_files', [])
        coreg_cfg_run = self.anat_config.preprocessing.coregistration

        do_coreg = (t1w_files and t2w_files and coreg_cfg_run.get("enabled", False))

        if not do_coreg:
            self.logger.info("Skipping Coregistration (Not enabled or insufficient data).")
            return context, step_metrics

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.update(description="[cyan]Coregistration")
            except Exception:
                pass

        errors = context.setdefault('errors', [])

        def _prepare_t2w_for_freesurfer_coreg() -> ImageFile:
            use_fs = self.anat_config.preprocessing.use_freesurfer
            if not use_fs:
                return context["preprocessed_t2w"]

            mask_step = next((s for s in self.steps if isinstance(s, BrainMaskingStep)), None)
            if not mask_step:
                self.logger.warning(
                    "FreeSurfer anatomical coregistration requested, but no BrainMaskingStep is configured. "
                    "Proceeding with the unmasked T2w image."
                )
                return context["preprocessed_t2w"]

            self.logger.info(
                "FreeSurfer coregistration uses a skull-stripped T1w target. "
                "Generating a skull-stripped T2w moving image before T2w -> T1w registration."
            )
            masked_t2w, _ = mask_step(context["preprocessed_t2w"], output_dir=output_dir, return_mask=True)
            return masked_t2w

        try:
            # Flatten options logic (Top-level + Nested 'options')
            coreg_options = dict(coreg_cfg_run)
            if "options" in coreg_options:
                sub_opts = coreg_options.pop("options")
                if isinstance(sub_opts, dict):
                    coreg_options.update(sub_opts)

            coreg_step = CoregistrationStep(
                self.config,
                self.logger,
                self.provenance,
                method=coreg_cfg_run.get("method", "fsl"),
                options={k: v for k, v in coreg_options.items() if k not in {"enabled", "method"}},
            )
            coreg_step.modality = "Anatomical"
            rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
            force_coreg = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(coreg_step, rerun_from_step)
            self._anat_force_from_step_active = force_coreg
            if force_coreg:
                self.logger.info("Forcing Coregistration because rerun_from_step has been reached.")

            ref_img = coreg_cfg_run.get("reference_image", "t1w").lower()

            st = time.time()
            if ref_img in {"supersynth", "syntht1w", "synthetic_t1w", "supersynth_multivariate"}:
                from ...lib.anat.super_synth import ensure_supersynth_outputs_for_image, ensure_supersynth_t1w

                moving = context.get("preprocessed_t2w") or context.get("preprocessed_t1w")
                if moving is None:
                    raise ProcessingError("No moving anatomical image available for SuperSynth coregistration.")

                use_multivariate = (
                    ref_img == "supersynth_multivariate"
                    or str(coreg_options.get("supersynth_registration", "")).lower() == "multivariate"
                )

                if use_multivariate:
                    fixed_source = context.get("preprocessed_t1w")
                    if fixed_source is None:
                        raise ProcessingError("SuperSynth multivariate coregistration requires a preprocessed T1w fixed source.")

                    ss_root = output_dir / "coregistration" / "supersynth_multivariate"
                    fixed_outputs = ensure_supersynth_outputs_for_image(
                        fixed_source,
                        ss_root / "fixed",
                        self.config,
                        self.logger,
                        mode=coreg_options.get("supersynth_mode"),
                        device=coreg_options.get("supersynth_device"),
                        sharpen_synths=coreg_options.get("supersynth_sharpen_synths"),
                        force=bool(coreg_options.get("force", False)) or force_coreg,
                    )
                    moving_outputs = ensure_supersynth_outputs_for_image(
                        moving,
                        ss_root / "moving",
                        self.config,
                        self.logger,
                        mode=coreg_options.get("supersynth_mode"),
                        device=coreg_options.get("supersynth_device"),
                        sharpen_synths=coreg_options.get("supersynth_sharpen_synths"),
                        force=bool(coreg_options.get("force", False)) or force_coreg,
                    )
                    required = ("synth_t1w", "synth_t2w")
                    if not all(k in fixed_outputs and k in moving_outputs for k in required):
                        raise ProcessingError("SuperSynth multivariate coregistration requires synthetic T1w and T2w outputs for both images.")

                    synth_ref = ImageFile(
                        entities={**fixed_source.entities, "desc": "synthT1w"},
                        img=fixed_outputs["synth_t1w"],
                    )
                    coreg_options.update({
                        "registration_fixed": fixed_outputs["synth_t1w"],
                        "registration_moving": moving_outputs["synth_t1w"],
                        "registration_fixed_extras": [fixed_outputs["synth_t2w"]],
                        "registration_moving_extras": [moving_outputs["synth_t2w"]],
                        "transform_type": coreg_options.get("transform_type", "SyNOnly"),
                    })
                    coreg_step = CoregistrationStep(
                        self.config,
                        self.logger,
                        self.provenance,
                        method="ants",
                        options={k: v for k, v in coreg_options.items() if k not in {"enabled", "method"}},
                    )
                    self.logger.info(
                        "Coregistration: SuperSynth multivariate registration "
                        "(moving synth T1w/T2w -> fixed synth T1w/T2w), applying transform to original anatomical image."
                    )
                else:
                    synth_ref = ensure_supersynth_t1w(
                        context,
                        output_dir / "coregistration",
                        self.config,
                        self.logger,
                        input_preference=coreg_options.get("supersynth_input", "auto"),
                        mode=coreg_options.get("supersynth_mode"),
                        device=coreg_options.get("supersynth_device"),
                        sharpen_synths=coreg_options.get("supersynth_sharpen_synths"),
                        force=bool(coreg_options.get("force", False)) or force_coreg,
                        subdir="supersynth_reference",
                    )
                    if not synth_ref:
                        raise ProcessingError("SuperSynth anatomical coregistration target could not be generated.")
                    self.logger.info("Coregistration: Reference=SuperSynth T1w. Registering anatomical image -> synthetic T1w.")

                res_anat = coreg_step(moving, output_dir=output_dir, target=synth_ref.img, options=coreg_options, force=force_coreg)
                if isinstance(res_anat, dict):
                    res_anat = res_anat.get("current_image")
                context["preprocessed_anat_coreg"] = res_anat
                context["preprocessed_t1w_synth"] = synth_ref

                if moving is context.get("preprocessed_t2w"):
                    context["preprocessed_t2w_coreg"] = res_anat
                else:
                    context["preprocessed_t1w"] = res_anat

                if reporter:
                    try:
                        from ...lib.reporting.viz import plot_comparison
                        p = figures_dir / "coreg_anat_on_supersynth.png"
                        plot_comparison(synth_ref.img, res_anat.img, p, title="Coreg Anat -> SuperSynth T1w")
                        details = {"Method": coreg_step.method, "Reference": "SuperSynth T1w", "Moving": moving.entities.get("suffix", "Anat")}
                        fig_item = [{"path": str(p), "title": "Coregistration", "caption": "Anatomical image (overlay) on SuperSynth T1w"}]
                        self._add_anat_step("Coregistration", details, figures=fig_item)
                    except Exception as e:
                        self.logger.warning(f"Failed to plot Coregistration report: {e}")

            elif ref_img == 't2w':
                self.logger.info("Coregistration: Reference=T2w. Registering T1w -> T2w.")
                res_t1 = coreg_step(context["preprocessed_t1w"], output_dir=output_dir, target=context["preprocessed_t2w"].img, options=coreg_options, force=force_coreg)
                if isinstance(res_t1, dict):
                    res_t1 = res_t1.get("current_image")
                context["preprocessed_t1w"] = res_t1

                if reporter:
                    try:
                        from ...lib.reporting.viz import plot_comparison
                        p = figures_dir / "coreg_t1_on_t2.png"
                        plot_comparison(context["preprocessed_t2w"].img, res_t1.img, p, title="Coreg T1w -> T2w")
                        details = {"Method": coreg_step.method, "Reference": "T2w", "Moving": "T1w"}
                        fig_item = [{"path": str(p), "title": "Coregistration", "caption": "T1w (overlay) on T2w"}]
                        self._add_anat_step("Coregistration", details, figures=fig_item)
                    except Exception as e:
                        self.logger.warning(f"Failed to plot Coregistration report: {e}")

            else:
                self.logger.info("Coregistration: Reference=T1w. Registering T2w -> T1w.")
                moving_t2 = _prepare_t2w_for_freesurfer_coreg()
                res_t2 = coreg_step(moving_t2, output_dir=output_dir, target=context["preprocessed_t1w"].img, options=coreg_options, force=force_coreg)
                if isinstance(res_t2, dict):
                    res_t2 = res_t2.get("current_image")
                context["preprocessed_t2w_coreg"] = res_t2

                if reporter:
                    try:
                        from ...lib.reporting.viz import plot_comparison
                        p = figures_dir / "coreg_t2_on_t1.png"
                        plot_comparison(context["preprocessed_t1w"].img, res_t2.img, p, title="Coreg T2w -> T1w")
                        details = {"Method": coreg_step.method, "Reference": "T1w", "Moving": "T2w"}
                        fig_item = [{"path": str(p), "title": "Coregistration", "caption": "T2w (overlay) on T1w"}]
                        self._add_anat_step("Coregistration", details, figures=fig_item)
                    except Exception as e:
                        self.logger.warning(f"Failed to plot Coregistration report: {e}")

            dur = time.time() - st
            step_metrics.append({"Step": "Coregistration", "Status": "Completed", "Duration": f"{dur:.2f}s"})

            # Validate output file existence
            img_to_validate = context.get("preprocessed_t1w") if ref_img == 't2w' else context.get("preprocessed_t2w_coreg")
            if img_to_validate and hasattr(img_to_validate, 'img') and not img_to_validate.img.exists():
                err_msg = f"Coregistration output file missing: {img_to_validate.img}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Coregistration", {"Status": "Output Missing"}, details={"path": str(img_to_validate.img)})

        except Exception as e:
            err_msg = f"Coregistration failed: {e}"
            self.logger.error(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step("Coregistration", {"Status": "Failed"}, details={"error": str(e)})
            return context, step_metrics

        # Save intermediate coregistration result
        save_inter = self.config.get("save_intermediate", False)
        if save_inter and final_output_dir:
            try:
                img_to_save = context.get("preprocessed_t1w")
                if ref_img == 't1w':
                    img_to_save = context.get("preprocessed_t2w_coreg")

                if img_to_save:
                    ents = dict(img_to_save.entities)
                    ents['desc'] = 'coreg'
                    if 'suffix' not in ents:
                        ents['suffix'] = 'T1w' if ref_img == 't2w' else 'T2w'
                    fname = build_bids_name(ents)
                    dest = final_output_dir / fname
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(img_to_save.img, dest)
                        if img_to_save.json and img_to_save.json.exists():
                            shutil.copy(img_to_save.json, dest.with_suffix("").with_suffix(".json"))
            except Exception as e:
                err_msg = f"Failed to save coregistration intermediate: {e}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Coregistration", {"Status": "Failed to Save Intermediate"}, details={"error": str(e)})

        # Run QC if enabled on outputs
        try:
            outputs_for_qc = []
            if ref_img == 't2w':
                outputs_for_qc.append(context.get("preprocessed_t1w"))
                outputs_for_qc.append(context.get("preprocessed_t2w"))
            else:
                outputs_for_qc.append(context.get("preprocessed_t1w"))
                outputs_for_qc.append(context.get("preprocessed_t2w_coreg"))
            self._run_mriqc_qc(outputs_for_qc, output_dir, context, reporter)
        except Exception as e:
            err_msg = f"Coregistration QC step failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)

        # Validate outputs again after saving and QC
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _run_brain_masking(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, figures_dir: Path, step_metrics: List[dict]) -> (dict, List[dict]):
        """
        Generate brain mask using the configured step if enabled.

        Returns:
            Updated context and metrics list.
        """
        mask_step = next((s for s in self.steps if isinstance(s, BrainMaskingStep)), None)
        if not mask_step:
            return context, step_metrics

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.update(description="[cyan]Brain Masking")
            except Exception:
                pass

        pre_cfg = self.anat_config.preprocessing
        coreg_cfg_local = pre_cfg.coregistration
        ref_mode = coreg_cfg_local.get("reference_image", "t1w").lower()

        ref_img_key = "preprocessed_t1w"
        if ref_mode == 't2w':
            ref_img_key = "preprocessed_t2w"

        # Fallback if ref_img_key not present
        if ref_img_key == "preprocessed_t1w" and "preprocessed_t1w" not in context and "preprocessed_t2w" in context:
            ref_img_key = "preprocessed_t2w"
        elif ref_img_key == "preprocessed_t2w" and "preprocessed_t2w" not in context and "preprocessed_t1w" in context:
            ref_img_key = "preprocessed_t1w"

        target_img = context.get(ref_img_key)
        errors = context.setdefault('errors', [])

        if not target_img:
            msg = "No reference image found for brain masking."
            self.logger.warning(msg)
            errors.append(msg)
            if reporter:
                self._add_anat_step("BrainMasking", {"Status": "Failed"}, details={"message": msg})
            return context, step_metrics

        self.logger.info(f"Generating binary brain mask using reference: {ref_img_key} ({mask_step.method})")

        save_inter = self.config.get("save_intermediate", False)
        skip_existing = self.config.get("skip_existing", False)
        skull_stripped_outputs = bool(self.anat_config.preprocessing.skull_stripped_outputs)
        rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
        force_masking = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(mask_step, rerun_from_step)
        self._anat_force_from_step_active = force_masking
        if force_masking:
            self.logger.info("Forcing Brain Masking because rerun_from_step has been reached.")

        def _store_skull_stripped_preproc(img_key: str, img_obj: Optional[ImageFile], mask_obj: Optional[ImageFile]) -> None:
            if not skull_stripped_outputs:
                return
            if not img_obj or not mask_obj:
                return
            publish_dir = final_output_dir or output_dir
            brain_obj = self._write_masked_anat_derivative(
                img_obj,
                mask_obj,
                publish_dir,
                desc="preproc-brain",
                suffix=img_obj.entities.get("suffix", "T1w"),
                errors=errors,
                reporter=reporter,
                label=f"BrainMasking_{img_obj.entities.get('suffix', img_key)}",
                force=force_masking,
            )
            if brain_obj:
                context[f"{img_key}_brain"] = brain_obj

        skipped_mask = False
        if final_output_dir and skip_existing and not force_masking:
            m_ents = dict(target_img.entities)
            m_ents['desc'] = 'preproc'
            m_ents['suffix'] = 'mask'
            for k in ['space', 'res']:
                m_ents.pop(k, None)

            m_fname = build_bids_name(m_ents)
            m_dest = final_output_dir / m_fname

            if m_dest.exists():
                self.logger.info(f"Skipping Brain Masking (Found existing mask: {m_fname})")
                context["brain_mask"] = ImageFile(entities=m_ents, img=m_dest)
                _store_skull_stripped_preproc(ref_img_key, target_img, context["brain_mask"])
                skipped_mask = True
                step_metrics.append({"Step": "BrainMasking", "Status": "Skipped (Found)", "Duration": "0s"})

                tracker_module = mask_step.normalize_tracker_module(mask_step.__class__.__name__)
                tracker = self.config.tracker
                subject = context.get('subject')
                session = context.get('session')
                study = context.get('study_name', self.config.get('study_name'))

                if tracker and subject and session:
                    tracker.update_status(subject, session, tracker_module, "completed (cached)", study=study, modality="Anatomical")
                    tracker.save()

        if not skipped_mask:
            st = time.time()
            try:
                brain_masked, mask = mask_step(target_img, output_dir=output_dir, return_mask=True, force=force_masking)
                dur = time.time() - st
                step_metrics.append({"Step": "BrainMasking", "Status": "Completed", "Duration": f"{dur:.2f}s"})

                context["brain_mask"] = mask
                _store_skull_stripped_preproc(ref_img_key, brain_masked, mask)

                # Validate mask output
                if not (hasattr(mask, 'img') and mask.img.exists()):
                    err_msg = f"Brain mask output file missing: {getattr(mask, 'img', None)}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step("BrainMasking", {"Status": "Output Missing"}, details={"path": str(getattr(mask, 'img', 'Unknown'))})

                if reporter:
                    p = figures_dir / "brain_mask_check.png"
                    try:
                        from ...lib.reporting.viz import plot_comparison
                        plot_comparison(target_img.img, mask.img, p, title="Brain Mask Check", overlay_alpha=0.3, overlay_cmap="autumn")
                        details = {"Method": mask_step.method, "Reference Image": ref_img_key}
                        fig_item = [{"path": str(p), "title": "Brain Mask", "caption": "Brain Mask (overlay) on Reference"}]
                        self._add_anat_step("BrainMasking", details, figures=fig_item)
                    except Exception as e:
                        self.logger.warning(f"Failed to plot brain mask report: {e}")
            except Exception as e:
                err_msg = f"Brain Masking failed: {e}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                step_metrics.append({"Step": "BrainMasking", "Status": "Failed", "Duration": "0s"})
                if reporter:
                    self._add_anat_step("BrainMasking", {"Status": "Failed"}, details={"error": str(e)})

        # Save intermediate brain mask
        if save_inter and final_output_dir and "brain_mask" in context:
            try:
                mask_to_save = context["brain_mask"]
                ents = dict(mask_to_save.entities)
                ents['desc'] = 'brain'  # Raw brain mask from step
                ents['suffix'] = 'mask'
                fname = build_bids_name(ents)
                dest = final_output_dir / fname
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(mask_to_save.img, dest)
            except Exception as e:
                err_msg = f"Failed to save intermediate brain mask: {e}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("BrainMasking", {"Status": "Failed to Save Intermediate"}, details={"error": str(e)})

        if skull_stripped_outputs and "brain_mask" in context:
            _store_skull_stripped_preproc("preprocessed_t1w", context.get("preprocessed_t1w"), context.get("brain_mask"))
            _store_skull_stripped_preproc(
                "preprocessed_t2w",
                context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w"),
                context.get("brain_mask"),
            )

        # Run QC if enabled on brain mask
        try:
            self._run_mriqc_qc([context.get("brain_mask")], output_dir, context, reporter)
        except Exception as e:
            err_msg = f"Brain Masking QC step failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.advance()
            except Exception:
                pass

        # Validate outputs after brain masking
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _write_masked_anat_derivative(
        self,
        img_obj: ImageFile,
        mask_obj: ImageFile,
        output_dir: Path,
        *,
        desc: str,
        suffix: str,
        errors: List[str],
        reporter=None,
        label: str = "BrainMaskedImage",
        force: bool = False,
    ) -> Optional[ImageFile]:
        """Write an anatomical image with all voxels outside ``mask_obj`` set to zero."""
        if not (
            img_obj
            and hasattr(img_obj, "img")
            and img_obj.img
            and Path(img_obj.img).exists()
            and mask_obj
            and hasattr(mask_obj, "img")
            and mask_obj.img
            and Path(mask_obj.img).exists()
        ):
            return None

        ents = dict(getattr(img_obj, "entities", {}) or {})
        ents["desc"] = desc
        ents.setdefault("suffix", suffix)
        dest = output_dir / build_bids_name(ents)
        if dest.exists() and self.config.get("skip_existing", False) and not force:
            return ImageFile(entities=ents, img=dest, json=getattr(img_obj, "json", None))

        try:
            img_nii = nib.load(str(img_obj.img))
            mask_nii = nib.load(str(mask_obj.img))
            img_data = img_nii.get_fdata()
            mask_data = mask_nii.get_fdata() > 0.5

            if img_data.shape[:3] != mask_data.shape[:3]:
                msg = (
                    f"Cannot create {desc} derivative for {Path(img_obj.img).name}: "
                    f"image shape {img_data.shape[:3]} does not match mask shape {mask_data.shape[:3]}."
                )
                self.logger.warning(msg)
                errors.append(msg)
                return None

            if img_data.ndim == 4 and mask_data.ndim == 3:
                mask_data = mask_data[..., np.newaxis]

            dest.parent.mkdir(parents=True, exist_ok=True)
            masked_data = img_data * mask_data
            masked_nii = nib.Nifti1Image(masked_data.astype(img_nii.get_data_dtype()), img_nii.affine, img_nii.header)
            nib.save(masked_nii, str(dest))

            if img_obj.json and img_obj.json.exists():
                shutil.copy2(img_obj.json, dest.with_suffix("").with_suffix(".json"))

            return ImageFile(entities=ents, img=dest, json=getattr(img_obj, "json", None))
        except Exception as e:
            msg = f"Failed to create {desc} derivative for {Path(img_obj.img).name}: {e}"
            self.logger.warning(msg)
            errors.append(msg)
            if reporter:
                self._add_anat_step(label, {"Status": "Failed"}, details={"error": str(e), "desc": desc})
            return None

    def _run_normalization(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, figures_dir: Path, step_metrics: List[dict]) -> (dict, List[dict]):
        """
        Run nonlinear normalization if enabled.

        Returns:
            Updated context and metrics list.
        """
        norm_step = next((s for s in self.steps if isinstance(s, NonlinearRegistrationStep)), None)
        if not norm_step:
            return context, step_metrics
        norm_space = getattr(norm_step, "space_entity", None) or "Standard"

        save_inter = self.config.get("save_intermediate", False)
        skip_existing = self.config.get("skip_existing", False)
        errors = context.setdefault('errors', [])
        rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
        force_norm = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(norm_step, rerun_from_step)
        self._anat_force_from_step_active = force_norm
        if force_norm:
            self.logger.info("Forcing Normalization because rerun_from_step has been reached.")

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.update(description=f"[cyan]Normalization ({norm_step.template})")
            except Exception:
                pass

        primary_img = context.get("preprocessed_t1w")
        primary_key = "preprocessed_t1w"
        primary_suffix = "T1w"

        if primary_img is None:
            primary_img = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
            primary_key = "preprocessed_t2w"
            primary_suffix = "T2w"

        if primary_img:
            self.logger.info(f"Running Normalization on {primary_suffix}...")
            try:
                context["current_image"] = primary_img
                normalization_result = norm_step(
                    context,
                    output_dir=output_dir,
                    force=force_norm,
                )
                if isinstance(normalization_result, dict):
                    context = normalization_result
            except Exception as e:
                err_msg = f"Normalization step failed: {e}"
                self.logger.error(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Normalization", {"Status": "Failed"}, details={"error": str(e)})
                return context, step_metrics

            normalized_primary = context.get("current_image")
            context[primary_key] = normalized_primary

            if not (normalized_primary and hasattr(normalized_primary, 'img') and normalized_primary.img.exists()):
                err_msg = f"Normalization output {primary_suffix} image missing or invalid."
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Normalization", {"Status": "Output Missing"}, details={"path": str(getattr(normalized_primary, 'img', 'Unknown'))})

            # Apply to T2w
            transform = context.get("template_transform")
            t2_img = context.get("preprocessed_t2w")
            if "preprocessed_t2w_coreg" in context:
                t2_img = context["preprocessed_t2w_coreg"]

            transform_type = str(context.get("template_transform_type", "ants") or "ants").lower()
            if primary_suffix == "T1w" and t2_img and transform and transform.exists():
                self.logger.info("Applying Normalization Warp to T2w...")
                ents = dict(t2_img.entities)
                ents['space'] = norm_space
                ents['desc'] = 'norm'
                if 'suffix' not in ents:
                    ents['suffix'] = 'T2w'

                norm_t2_path = output_dir / build_bids_name(ents)

                try:
                    if not norm_t2_path.exists() or not skip_existing or force_norm:
                        template = norm_step.template or self.anat_config.preprocessing.normalization.get("template")

                        if template:
                            if transform_type == "fsl" or Path(transform).suffix == ".mat":
                                interp = (
                                    self.anat_config.preprocessing.normalization.get("options", {}).get("interpolation")
                                    or "trilinear"
                                )
                                fsl.applywarp(
                                    in_file=t2_img.img,
                                    ref_file=Path(template),
                                    out_file=norm_t2_path,
                                    premat=Path(transform),
                                    interp=interp,
                                    force=True,
                                )
                                norm_t2_obj = ImageFile(entities=ents, img=norm_t2_path)
                                context["preprocessed_t2w"] = norm_t2_obj

                                if not norm_t2_obj.img.exists():
                                    err_msg = f"T2w normalized output file missing after FLIRT apply: {norm_t2_obj.img}"
                                    self.logger.warning(err_msg)
                                    errors.append(err_msg)
                                    if reporter:
                                        self._add_anat_step("Normalization_T2w", {"Status": "Output Missing"}, details={"path": str(norm_t2_obj.img)})
                            else:
                                prefix = Path(transform)
                                warp = prefix.with_suffix("").parent / (prefix.name + "1Warp.nii.gz")
                                affine = prefix.with_suffix("").parent / (prefix.name + "0GenericAffine.mat")

                                if warp.exists() and affine.exists():
                                    ants.apply_transforms(
                                        fixed_file=Path(template),
                                        moving_file=t2_img.img,
                                        out_file=norm_t2_path,
                                        transforms=[warp, affine],
                                        interpolator='linear'
                                    )
                                    norm_t2_obj = ImageFile(entities=ents, img=norm_t2_path)
                                    context["preprocessed_t2w"] = norm_t2_obj

                                    if not norm_t2_obj.img.exists():
                                        err_msg = f"T2w normalized output file missing after apply_transforms: {norm_t2_obj.img}"
                                        self.logger.warning(err_msg)
                                        errors.append(err_msg)
                                        if reporter:
                                            self._add_anat_step("Normalization_T2w", {"Status": "Output Missing"}, details={"path": str(norm_t2_obj.img)})
                                else:
                                    warn_msg = "Could not find warp/affine files for T2 normalization."
                                    self.logger.warning(warn_msg)
                                    errors.append(warn_msg)
                                    if reporter:
                                        self._add_anat_step("Normalization_T2w", {"Status": "Warp Files Missing"}, details={"warp": str(warp), "affine": str(affine)})
                        else:
                            warn_msg = "No template reference found for T2 normalization apply."
                            self.logger.warning(warn_msg)
                            errors.append(warn_msg)
                            if reporter:
                                self._add_anat_step("Normalization_T2w", {"Status": "Template Missing"})
                    else:
                        self.logger.info(f"Skipping T2w Normalization (Exists): {norm_t2_path}")
                        context["preprocessed_t2w"] = ImageFile(entities=ents, img=norm_t2_path)

                        tracker = self.config.tracker
                        if tracker and context.get('subject') and context.get('session'):
                            study = context.get('study_name', self.config.get('study_name'))
                            tracker.update_status(context['subject'], context['session'], "Coregistration", "completed (cached)",
                                                  study=study, modality="Anatomical")
                            tracker.save()
                except Exception as e:
                    err_msg = f"Failed to apply normalization warp to T2w: {e}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step("Normalization_T2w", {"Status": "Failed"}, details={"error": str(e)})

        def _normalize_brain_mask():
            mask_obj = context.get("brain_mask")
            if not (mask_obj and hasattr(mask_obj, "img") and Path(mask_obj.img).exists()):
                warn_msg = (
                    "Normalized skull-stripped output was requested, but no brain mask is available. "
                    "Enable anat.preprocessing.brain_masking and verify that it completes successfully."
                )
                self.logger.warning(warn_msg)
                errors.append(warn_msg)
                return None
            transform = context.get("template_transform")
            template = norm_step.template or self.anat_config.preprocessing.normalization.get("template")
            if not (transform and template):
                warn_msg = (
                    "Normalized skull-stripped output was requested, but the normalization transform "
                    "or template is unavailable."
                )
                self.logger.warning(warn_msg)
                errors.append(warn_msg)
                return None

            ents = dict(mask_obj.entities)
            ents['space'] = norm_space
            ents['desc'] = 'norm'
            ents['suffix'] = 'mask'

            norm_mask_path = output_dir / build_bids_name(ents)
            if norm_mask_path.exists() and skip_existing and not force_norm:
                return ImageFile(entities=ents, img=norm_mask_path)

            try:
                transform_type = str(context.get("template_transform_type", "ants") or "ants").lower()
                if transform_type == "fsl" or Path(transform).suffix == ".mat":
                    if not Path(transform).exists():
                        warn_msg = f"Normalization transform does not exist: {transform}"
                        self.logger.warning(warn_msg)
                        errors.append(warn_msg)
                        return None
                    fsl.applywarp(
                        in_file=mask_obj.img,
                        ref_file=Path(template),
                        out_file=norm_mask_path,
                        premat=Path(transform),
                        interp="nn",
                        force=True,
                    )
                else:
                    prefix = Path(transform)
                    warp = prefix.with_suffix("").parent / (prefix.name + "1Warp.nii.gz")
                    affine = prefix.with_suffix("").parent / (prefix.name + "0GenericAffine.mat")
                    if not (warp.exists() and affine.exists()):
                        warn_msg = (
                            "Could not normalize the brain mask: "
                            f"missing {warp.name} or {affine.name}."
                        )
                        self.logger.warning(warn_msg)
                        errors.append(warn_msg)
                        return None
                    ants.apply_transforms(
                        fixed_file=Path(template),
                        moving_file=mask_obj.img,
                        out_file=norm_mask_path,
                        transforms=[warp, affine],
                        interpolator='nearestNeighbor'
                    )

                if not norm_mask_path.exists():
                    raise ProcessingError(f"Normalized brain mask was not created: {norm_mask_path}")
                return ImageFile(entities=ents, img=norm_mask_path)
            except Exception as e:
                err_msg = f"Failed to normalize the anatomical brain mask: {e}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step(
                        "Normalization_BrainMask",
                        {"Status": "Failed"},
                        details={"error": str(e)},
                    )
                return None

        if self.anat_config.normalization.skull_stripped_outputs:
            normalized_mask = _normalize_brain_mask()
            if normalized_mask:
                normalized_images = (
                    ("normalized_t1w_brain", context.get("preprocessed_t1w"), "T1w"),
                    ("normalized_t2w_brain", context.get("preprocessed_t2w"), "T2w"),
                )
                for context_key, normalized_img, suffix in normalized_images:
                    if not (
                        normalized_img
                        and hasattr(normalized_img, "entities")
                        and normalized_img.entities.get("space")
                    ):
                        continue
                    brain_obj = self._write_masked_anat_derivative(
                        normalized_img,
                        normalized_mask,
                        final_output_dir or output_dir,
                        desc="norm-brain",
                        suffix=suffix,
                        errors=errors,
                        reporter=reporter,
                        label=f"Normalization_{suffix}_Brain",
                        force=force_norm,
                    )
                    if brain_obj:
                        context[context_key] = brain_obj
                        self.logger.info(f"Saved normalized skull-stripped {suffix}: {brain_obj.img}")

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.advance()
            except Exception:
                pass

        # Publish normalized images and transforms to the final anat directory.
        if final_output_dir:
            def _publish_norm_img(img_obj, suffix: str):
                if not (img_obj and hasattr(img_obj, 'entities') and hasattr(img_obj, 'img') and img_obj.img.exists()):
                    return
                if not img_obj.entities.get('space'):
                    return

                ents = dict(img_obj.entities)
                ents['desc'] = 'norm'
                if 'suffix' not in ents:
                    ents['suffix'] = suffix
                fname = build_bids_name(ents)
                dest = final_output_dir / fname
                try:
                    if img_obj.img.resolve() != dest.resolve():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_obj.img, dest)
                        if img_obj.json and img_obj.json.exists():
                            shutil.copy2(img_obj.json, dest.with_suffix("").with_suffix(".json"))
                    img_obj.img = dest
                except Exception as e:
                    err_msg = f"Failed to publish normalized {suffix}: {e}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
                    if reporter:
                        self._add_anat_step(f"Normalization_{suffix}", {"Status": "Failed to Save"}, details={"error": str(e)})

            def _publish_norm_transforms():
                transform = context.get("template_transform")
                if not transform:
                    return

                transform = Path(transform)
                transform_type = str(context.get("template_transform_type", "ants") or "ants").lower()
                candidates = []

                if transform_type == "ants":
                    candidates.extend([
                        transform.parent / f"{transform.name}0GenericAffine.mat",
                        transform.parent / f"{transform.name}1Warp.nii.gz",
                        transform.parent / f"{transform.name}1InverseWarp.nii.gz",
                    ])
                    if transform.exists():
                        candidates.append(transform)
                else:
                    candidates.append(transform)

                for src in candidates:
                    if not src.exists():
                        continue
                    dest = final_output_dir / src.name
                    try:
                        if src.resolve() != dest.resolve():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dest)
                    except Exception as e:
                        err_msg = f"Failed to publish normalization transform {src.name}: {e}"
                        self.logger.warning(err_msg)
                        errors.append(err_msg)
                        if reporter:
                            self._add_anat_step("Normalization_Transform", {"Status": "Failed to Save"}, details={"error": str(e), "file": str(src)})

            _publish_norm_img(context.get("preprocessed_t1w"), 'T1w')
            _publish_norm_img(context.get("preprocessed_t2w"), 'T2w')
            _publish_norm_transforms()

        # Run QC if enabled on normalized outputs
        try:
            self._run_mriqc_qc([context.get("preprocessed_t1w"), context.get("preprocessed_t2w")], output_dir, context, reporter)
        except Exception as e:
            err_msg = f"Normalization QC step failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)

        # Validate outputs after normalization
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _run_freesurfer(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, step_metrics: List[dict]) -> (dict, List[dict]):
        """
        Run FreeSurfer stats extraction if enabled.

        Returns:
            Updated context and metrics list.
        """
        fs_stats_step = next((s for s in self.steps if isinstance(s, FreeSurferStatsStep)), None)
        if not fs_stats_step:
            return context, step_metrics

        if not context.get("preprocessed_t1w"):
            self.logger.info("Skipping FreeSurfer stats (no preprocessed T1w available).")
            return context, step_metrics

        errors = context.setdefault('errors', [])

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.update(description="[cyan]FreeSurfer Stats")
            except Exception:
                pass

        try:
            self.logger.info("Parsing FreeSurfer Stats...")
            rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
            force_fs_stats = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(fs_stats_step, rerun_from_step)
            self._anat_force_from_step_active = force_fs_stats
            if force_fs_stats:
                self.logger.info("Forcing FreeSurfer Stats because rerun_from_step has been reached.")
            fs_stats_step(context, output_dir=output_dir, force=force_fs_stats)
        except Exception as e:
            err_msg = f"FreeSurfer stats extraction failed: {e}"
            self.logger.error(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step("FreeSurferStats", {"Status": "Failed"}, details={"error": str(e)})

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.advance()
            except Exception:
                pass

        # Validate FreeSurfer outputs if possible (not explicit here)

        # Run QC if enabled on FreeSurfer outputs (if any)
        # No specific outputs defined, so skipping

        # Validate outputs after FreeSurfer
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _run_super_synth(
        self,
        output_dir: Path,
        context: dict,
        final_output_dir: Optional[Path],
        reporter,
        step_metrics: List[dict],
    ) -> tuple:
        """Run SuperSynth segmentation (and optional volume extraction) if enabled."""
        ss_step = next((s for s in self.steps if isinstance(s, SuperSynthStep)), None)
        if not ss_step:
            return context, step_metrics

        # Require a preprocessed T1w (or any current anatomical image).
        if not context.get("preprocessed_t1w") and not context.get("current_image"):
            self.logger.info("Skipping SuperSynth (no preprocessed T1w available).")
            return context, step_metrics

        errors = context.setdefault("errors", [])

        try:
            self.logger.info("Running SuperSynth segmentation...")
            rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
            force_super_synth = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(ss_step, rerun_from_step)
            self._anat_force_from_step_active = force_super_synth
            if force_super_synth:
                self.logger.info("Forcing SuperSynth because rerun_from_step has been reached.")
            ss_step(context, output_dir=output_dir, force=force_super_synth)

            # Log volumes to the study tracker when computed.
            volumes = context.get("super_synth_volumes")
            if volumes:
                tracker = self.config.tracker
                subject = context.get("subject")
                session = context.get("session")
                study = self.config.get("study_name")
                if tracker and subject and session:
                    tracker.log_volume_statistics(
                        subject, session, volumes,
                        method="supersynth", study=study,
                    )
                    tracker.save()

            if reporter:
                details = {"Status": "Complete", "Regions": len(context.get("super_synth_volumes", {}))}
                csv = context.get("super_synth_volumes_csv")
                if csv:
                    details["Volumes_CSV"] = str(csv)
                self._add_anat_step(reporter, "SuperSynth", details)

        except Exception as e:
            err_msg = f"SuperSynth failed: {e}"
            self.logger.error(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step(reporter, "SuperSynth", {"Status": "Failed", "error": str(e)})

        return context, step_metrics

    def _run_segmentation(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, step_metrics: List[dict]) -> (dict, List[dict]):
        """
        Run segmentation step if enabled.

        Returns:
            Updated context and metrics list.
        """
        seg_step = next((s for s in self.steps if isinstance(s, SegmentationStep)), None)
        if not seg_step:
            return context, step_metrics

        errors = context.setdefault('errors', [])

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.update(description="[cyan]Segmentation")
            except Exception:
                pass

        try:
            self.logger.info("Running Segmentation...")
            rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
            force_segmentation = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(seg_step, rerun_from_step)
            self._anat_force_from_step_active = force_segmentation
            if force_segmentation:
                self.logger.info("Forcing Segmentation because rerun_from_step has been reached.")
            seg_step(context, output_dir=output_dir, force=force_segmentation)
        except Exception as e:
            err_msg = f"Segmentation step failed: {e}"
            self.logger.error(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step("Segmentation", {"Status": "Failed"}, details={"error": str(e)})
            return context, step_metrics

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.advance()
            except Exception:
                pass

        # Save intermediates
        save_inter = self.config.get("save_intermediate", False)
        if save_inter and final_output_dir:
            try:
                if "segmentation" in context:
                    seg = context["segmentation"]
                    if hasattr(seg, 'img') and seg.img.exists():
                        ents = dict(seg.entities)
                        fname = build_bids_name(ents)
                        dest = final_output_dir / fname
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(seg.img, dest)
                            if seg.json and seg.json.exists():
                                shutil.copy(seg.json, dest.with_suffix("").with_suffix(".json"))
                if "5tt" in context:
                    f5tt = context["5tt"]
                    if hasattr(f5tt, 'img') and f5tt.img.exists():
                        ents = dict(f5tt.entities)
                        fname = build_bids_name(ents)
                        dest = final_output_dir / fname
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(f5tt.img, dest)
            except Exception as e:
                err_msg = f"Failed to save intermediate segmentation outputs: {e}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Segmentation", {"Status": "Failed to Save Intermediate"}, details={"error": str(e)})

        # Validate segmentation outputs
        if "segmentation" in context:
            seg = context["segmentation"]
            if not (hasattr(seg, 'img') and seg.img.exists()):
                err_msg = f"Segmentation output file missing: {getattr(seg, 'img', None)}"
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Segmentation", {"Status": "Output Missing"}, details={"path": str(getattr(seg, 'img', 'Unknown'))})

        # Run QC if enabled on segmentation outputs
        try:
            to_qc = []
            if "segmentation" in context:
                to_qc.append(context["segmentation"])
            if to_qc:
                self._run_mriqc_qc(to_qc, output_dir, context, reporter)
        except Exception as e:
            err_msg = f"Segmentation QC step failed: {e}"
            self.logger.warning(err_msg)
            context.setdefault('errors', []).append(err_msg)

        # Validate outputs after segmentation
        self._validate_outputs(context, step_metrics, reporter)

        return context, step_metrics

    def _handle_reporting_and_outputs(self, context: dict, final_output_dir: Optional[Path], reporter, step_metrics: List[dict]):
        """
        Handle final reporting, summaries and gathering output paths.

        Args:
            context: Processing context with results.
            final_output_dir: Directory to save final outputs.
            reporter: Reporting object for figures and summaries.
            step_metrics: List of dictionaries detailing step results.
        """
        if not final_output_dir:
            if reporter:
                reporter.add_anat_summary("Anatomical Execution Summary", step_metrics)
            return

        anat_outputs = {
            "Final Preprocessed Images": [],
            "Modeling Derivatives": [],
            "Normalized Derivatives": [],
            "Segmentation Outputs": []
        }

        # T1w Path
        pre_t1 = context.get("preprocessed_t1w")
        if pre_t1:
            ents = dict(pre_t1.entities)
            ents['desc'] = 'preproc'
            if 'suffix' not in ents:
                ents['suffix'] = 'T1w'
            t1_path = final_output_dir / build_bids_name(ents)
            anat_outputs["Final Preprocessed Images"].append({"key": "Preprocessed T1w", "path": str(t1_path)})

        pre_t1_brain = context.get("preprocessed_t1w_brain")
        if pre_t1_brain:
            anat_outputs["Final Preprocessed Images"].append({
                "key": "Skull-stripped Preprocessed T1w",
                "path": str(pre_t1_brain.img),
            })

        # T2w Path
        pre_t2 = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
        if pre_t2:
            ents = dict(pre_t2.entities)
            ents['desc'] = 'preproc'
            if 'suffix' not in ents:
                ents['suffix'] = 'T2w'
            t2_path = final_output_dir / build_bids_name(ents)
            anat_outputs["Final Preprocessed Images"].append({"key": "Preprocessed T2w", "path": str(t2_path)})

        pre_t2_brain = context.get("preprocessed_t2w_brain")
        if pre_t2_brain:
            anat_outputs["Final Preprocessed Images"].append({
                "key": "Skull-stripped Preprocessed T2w",
                "path": str(pre_t2_brain.img),
            })

        # Mask Path
        mask_img = context.get("brain_mask")
        if mask_img:
            ents = dict(mask_img.entities)
            ents['desc'] = 'preproc'
            ents['suffix'] = 'mask'
            mask_path = final_output_dir / build_bids_name(ents)
            anat_outputs["Segmentation Outputs"].append({"key": "Brain Mask", "path": str(mask_path)})

        # Scan for Intermediate Outputs (Normalized, Segmentation, Coreg)
        for f in final_output_dir.glob("*desc-norm*.nii.gz"):
            anat_outputs["Normalized Derivatives"].append({"key": f.name, "path": str(f)})

        for f in final_output_dir.glob("*dseg*.nii.gz"):
            anat_outputs["Segmentation Outputs"].append({"key": f.name, "path": str(f)})

        for f in final_output_dir.glob("*probseg*.nii.gz"):
            anat_outputs["Segmentation Outputs"].append({"key": f.name, "path": str(f)})

        for f in final_output_dir.glob("*5tt*.nii.gz"):
            anat_outputs["Segmentation Outputs"].append({"key": f.name, "path": str(f)})

        if reporter and anat_outputs:
            reporter.set_anat_outputs(anat_outputs)

        if reporter:
            reporter.add_anat_summary("Anatomical Execution Summary", step_metrics)

    def _load_preprocessed_from_output(self, context: dict, final_output_dir: Optional[Path]) -> Optional[dict]:
        """Load existing preprocessed anatomical outputs from the final output directory."""
        if not final_output_dir:
            return None

        t1w_files = context.get("t1w_files", [])
        t2w_files = context.get("t2w_files", [])

        if not t1w_files and not t2w_files:
            return None

        preprocessed_t1w = None
        preprocessed_t2w = None
        preprocessed_t1w_brain = None
        preprocessed_t2w_brain = None
        normalized_t1w_brain = None
        normalized_t2w_brain = None
        brain_mask = None

        if t1w_files:
            t1w = t1w_files[0]
            t1_entities = dict(getattr(t1w, "entities", {}) or {})
            t1_entities.setdefault("suffix", "T1w")
            t1_entities["desc"] = "preproc"
            t1_name = build_bids_name(t1_entities)
            t1_path = final_output_dir / t1_name

            if not t1_path.exists():
                return None

            preprocessed_t1w = ImageFile(entities=t1_entities, img=t1_path, json=None)

        if t2w_files:
            t2w = t2w_files[0]
            t2_entities = dict(getattr(t2w, "entities", {}) or {})
            t2_entities.setdefault("suffix", "T2w")
            t2_entities["desc"] = "preproc"
            t2_name = build_bids_name(t2_entities)
            t2_path = final_output_dir / t2_name
            if t2_path.exists():
                preprocessed_t2w = ImageFile(entities=t2_entities, img=t2_path, json=None)
            else:
                return None

        if preprocessed_t1w is None and preprocessed_t2w is None:
            return None

        mask_ref = preprocessed_t1w or preprocessed_t2w
        mask_entities = dict(getattr(mask_ref, "entities", {}) or {})
        mask_entities.pop("space", None)
        mask_entities["suffix"] = "mask"
        mask_name = build_bids_name(mask_entities)
        mask_path = final_output_dir / mask_name
        if mask_path.exists():
            brain_mask = ImageFile(entities=mask_entities, img=mask_path, json=None)

        if brain_mask and self.anat_config.preprocessing.skull_stripped_outputs:
            if preprocessed_t1w:
                t1_brain_entities = dict(preprocessed_t1w.entities)
                t1_brain_entities["desc"] = "preproc-brain"
                t1_brain_name = build_bids_name(t1_brain_entities)
                t1_brain_path = final_output_dir / t1_brain_name
                if not t1_brain_path.exists():
                    return None
                preprocessed_t1w_brain = ImageFile(entities=t1_brain_entities, img=t1_brain_path, json=None)

            if preprocessed_t2w:
                t2_brain_entities = dict(preprocessed_t2w.entities)
                t2_brain_entities["desc"] = "preproc-brain"
                t2_brain_name = build_bids_name(t2_brain_entities)
                t2_brain_path = final_output_dir / t2_brain_name
                if not t2_brain_path.exists():
                    return None
                preprocessed_t2w_brain = ImageFile(entities=t2_brain_entities, img=t2_brain_path, json=None)

        norm_cfg = self.anat_config.preprocessing.normalization or {}
        if self.anat_config.normalization.skull_stripped_outputs and norm_cfg.get("enabled"):
            norm_space = norm_cfg.get("space_entity", norm_cfg.get("space_name", norm_cfg.get("space", "Standard")))

            if preprocessed_t1w:
                norm_t1_entities = dict(preprocessed_t1w.entities)
                norm_t1_entities["space"] = norm_space
                norm_t1_entities["desc"] = "norm-brain"
                norm_t1_path = final_output_dir / build_bids_name(norm_t1_entities)
                if not norm_t1_path.exists():
                    return None
                normalized_t1w_brain = ImageFile(entities=norm_t1_entities, img=norm_t1_path, json=None)

            if preprocessed_t2w:
                norm_t2_entities = dict(preprocessed_t2w.entities)
                norm_t2_entities["space"] = norm_space
                norm_t2_entities["desc"] = "norm-brain"
                norm_t2_path = final_output_dir / build_bids_name(norm_t2_entities)
                if not norm_t2_path.exists():
                    return None
                normalized_t2w_brain = ImageFile(entities=norm_t2_entities, img=norm_t2_path, json=None)

        return {
            "preprocessed_t1w": preprocessed_t1w,
            "preprocessed_t2w": preprocessed_t2w,
            "preprocessed_t1w_brain": preprocessed_t1w_brain,
            "preprocessed_t2w_brain": preprocessed_t2w_brain,
            "normalized_t1w_brain": normalized_t1w_brain,
            "normalized_t2w_brain": normalized_t2w_brain,
            "brain_mask": brain_mask
        }

    def _recon_all_finished(self, context: dict) -> bool:
        pre_cfg = self.anat_config.preprocessing
        if not (pre_cfg.recon_all.get("enabled") or pre_cfg.use_freesurfer):
            return True

        subject = context.get("subject")
        session = context.get("session")

        if not subject:
            for image in (context.get("t1w_files") or []) + (context.get("t2w_files") or []):
                entities = getattr(image, "entities", {}) or {}
                subject = entities.get("sub") or subject
                session = session or entities.get("ses")
                if subject:
                    break

        if not subject:
            self.logger.warning(
                "Recon-all is enabled, but the subject could not be determined during the FreeSurfer completion check."
            )
            return False

        recon_cfg = pre_cfg.recon_all or {}
        fs_dir = Path(recon_cfg.get("subjects_dir")) if recon_cfg.get("subjects_dir") else (
            self.config.bids_dir / "derivatives" / "freesurfer"
        )

        fs_sub_id = f"sub-{subject}"
        if session:
            fs_sub_id += f"_ses-{session}"

        subj_dir = fs_dir / fs_sub_id
        if ReconAllStep.has_complete_recon(subj_dir, method=recon_cfg.get("method")):
            return True

        self.logger.info(
            f"Cached anatomical outputs found, but FreeSurfer recon is incomplete for {fs_sub_id}. "
            "Running the anatomical workflow so recon-all can be checked or rerun."
        )
        return False

    def run(self, output_dir: Path, context: dict, final_output_dir: Optional[Path] = None, reporter=None) -> dict:
        """
        Execute the anatomical preprocessing workflow.
        """
        self.reporter = reporter

        output_dir.mkdir(parents=True, exist_ok=True)
        if final_output_dir:
            final_output_dir.mkdir(parents=True, exist_ok=True)

        rerun_from_step = get_rerun_from_step(self.config, "anat.preprocessing", "anat")
        self._anat_rerun_from_step = rerun_from_step
        self._anat_force_from_step_active = False
        rerun_hits_anat = any_step_matches(self.steps, rerun_from_step)
        if self.config.get("skip_existing", True) and not self.config.get("force", False) and not rerun_hits_anat:
            cached = self._load_preprocessed_from_output(context, final_output_dir)
            if cached and self._recon_all_finished(context):
                self.logger.info(
                    f"⚡ FAST SKIP: Found preprocessed anatomical outputs in {final_output_dir}"
                )
                context = dict(context)
                context.update({k: v for k, v in cached.items() if v is not None})
                context["anat_preprocessing_skipped"] = True
                return context

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        context = dict(context)
        step_metrics: List[dict] = []

        if context.get("t1w_files"):
            context, step_metrics = self._preprocess_t1w(
                output_dir,
                context,
                final_output_dir,
                reporter,
                figures_dir,
                step_metrics
            )
        else:
            self.logger.info("Skipping T1w preprocessing (no T1w inputs selected).")

        if context.get("t2w_files"):
            context, step_metrics = self._preprocess_t2w(
                output_dir,
                context,
                final_output_dir,
                reporter,
                figures_dir,
                step_metrics
            )
        else:
            self.logger.info("Skipping T2w preprocessing (no T2w inputs selected).")

        context, step_metrics = self._run_coregistration(
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics
        )

        context, step_metrics = self._run_brain_masking(
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics
        )

        context, step_metrics = self._run_normalization(
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics
        )

        context, step_metrics = self._run_freesurfer(
            output_dir,
            context,
            final_output_dir,
            reporter,
            step_metrics
        )

        context, step_metrics = self._run_super_synth(
            output_dir,
            context,
            final_output_dir,
            reporter,
            step_metrics,
        )

        context, step_metrics = self._run_segmentation(
            output_dir,
            context,
            final_output_dir,
            reporter,
            step_metrics
        )

        try:
            target_output_dir = final_output_dir if final_output_dir else output_dir
            self.save_results(context, target_output_dir)
        except Exception as e:
            self.logger.warning(f"Failed to save final anatomical outputs: {e}")

        self._handle_reporting_and_outputs(context, final_output_dir, reporter, step_metrics)

        return context

    def _update_json_history(self, json_path: Path, steps: list):
        """Update JSON sidecar with processing history."""
        history_msg = "Pipeline Steps: " + ", ".join([s.__class__.__name__ for s in steps])

        data = {}
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            except Exception:
                pass

        prev_history = data.get("History", "")
        if prev_history:
            data["History"] = prev_history + "; " + history_msg
        else:
            data["History"] = history_msg

        data["ProcessingSteps"] = [s.__class__.__name__ for s in steps]

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def save_results(self, context, output_dir: Path):
        """Save final anatomical outputs to output_dir."""
        # Ensure output dir exists
        output_dir.mkdir(parents=True, exist_ok=True)

        errors = context.setdefault('errors', [])

        # Copy T1w
        pre_t1 = context.get("preprocessed_t1w")
        if pre_t1 and hasattr(pre_t1, 'img') and pre_t1.img.exists():
            # Build name with desc='preproc'
            entities = dict(pre_t1.entities)
            entities['desc'] = 'preproc'
            if 'suffix' not in entities:
                entities['suffix'] = 'T1w'

            fname = build_bids_name(entities)
            dest = output_dir / fname

            if not dest.exists():
                try:
                    self.logger.info(f"Saving Final Anat T1w: {dest}")
                    shutil.copy(pre_t1.img, dest)

                    dest_json = dest.with_suffix("").with_suffix(".json")
                    if pre_t1.json and pre_t1.json.exists():
                        shutil.copy(pre_t1.json, dest_json)

                    self._update_json_history(dest_json, self.steps)
                except Exception as e:
                    err_msg = f"Failed to save final T1w: {e}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
            else:
                self.logger.info(f"Final Anat T1w already exists, skipping copy: {dest}")

            context['preprocessed_t1w'] = ImageFile(entities=entities, img=dest)
        elif context.get("t1w_files"):
            msg = "Final T1w image missing/corrupted, cannot save result."
            self.logger.warning(msg)
            errors.append(msg)

        # Copy Brain Mask
        mask = context.get("brain_mask")
        if mask and hasattr(mask, 'img') and mask.img.exists():
            entities = dict(mask.entities)
            entities['desc'] = 'preproc'
            entities['suffix'] = 'mask'

            fname = build_bids_name(entities)
            dest = output_dir / fname

            if not dest.exists():
                try:
                    self.logger.info(f"Saving Final Brain Mask: {dest}")
                    shutil.copy(mask.img, dest)
                except Exception as e:
                    err_msg = f"Failed to save final brain mask: {e}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
            else:
                self.logger.info(f"Final Brain Mask already exists, skipping copy: {dest}")

            context['brain_mask'] = ImageFile(entities=entities, img=dest)
        else:
            msg = "Final brain mask missing/corrupted, cannot save result."
            self.logger.warning(msg)
            errors.append(msg)

        # Copy T2w (processed)
        pre_t2 = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
        if pre_t2 and hasattr(pre_t2, 'img') and pre_t2.img.exists():
            entities = dict(pre_t2.entities)
            entities['desc'] = 'preproc'
            if 'suffix' not in entities:
                entities['suffix'] = 'T2w'

            fname = build_bids_name(entities)
            dest = output_dir / fname

            if not dest.exists():
                try:
                    self.logger.info(f"Saving Final Anat T2w: {dest}")
                    shutil.copy(pre_t2.img, dest)

                    dest_json = dest.with_suffix("").with_suffix(".json")
                    if pre_t2.json and pre_t2.json.exists():
                        shutil.copy(pre_t2.json, dest_json)

                    self._update_json_history(dest_json, self.steps)
                except Exception as e:
                    err_msg = f"Failed to save final T2w: {e}"
                    self.logger.warning(err_msg)
                    errors.append(err_msg)
            else:
                self.logger.info(f"Final Anat T2w already exists, skipping copy: {dest}")

            key = "preprocessed_t2w_coreg" if "preprocessed_t2w_coreg" in context else "preprocessed_t2w"
            context[key] = ImageFile(entities=entities, img=dest)
        elif context.get("t2w_files"):
            msg = "Final T2w image missing/corrupted, cannot save result."
            self.logger.warning(msg)
            errors.append(msg)


class AnatPipeline(BasePipeline):
    """
    Top-level Anatomical Pipeline.
    Finds T1w/T2w files and runs AnatPreprocessingWorkflow.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "anat-pipeline"

    @property
    def version(self) -> str:
        return "1.0.0"

    def _initialize_pipeline(self) -> None:
        """Initialize workflows."""
        self.preprocessing = AnatPreprocessingWorkflow(self.config, self.logger, self.provenance)

    def _get_anat_input_cfg(self) -> dict:
        anat_section = self.config.get('anat') or {}
        return anat_section.get('input') or self.config.get('anat_input') or {}

    def _get_primary_anat_modality(self) -> str:
        primary = str(self._get_anat_input_cfg().get("primary_modality", "auto")).strip().lower()
        if primary not in {"auto", "t1w", "t2w"}:
            self.logger.warning(
                f"Unknown anat.input.primary_modality='{primary}'. Falling back to 'auto'."
            )
            return "auto"
        return primary

    def _apply_anat_selectors(self, t1w: list[ImageFile], t2w: list[ImageFile]) -> tuple[list[ImageFile], list[ImageFile]]:
        anat_input_cfg = self._get_anat_input_cfg()
        t1w = select_anatomical_candidates(t1w, anat_input_cfg.get("t1w_match"), "T1w", logger=self.logger)
        t2w = select_anatomical_candidates(t2w, anat_input_cfg.get("t2w_match"), "T2w", logger=self.logger)
        primary = self._get_primary_anat_modality()
        if primary == "t1w":
            t2w = []
        elif primary == "t2w":
            t1w = []
        return t1w, t2w

    def _get_work_dir(self, subject: str, session: Optional[str] = None) -> Path:
        """Get working directory for subject/session."""
        work_root = Path(self.config.get('work_dir'))
        if session:
            return work_root / f'sub-{subject}' / f'ses-{session}' / 'anat'
        return work_root / f'sub-{subject}' / 'anat'

    def _get_output_dir(self, subject: str, session: Optional[str] = None) -> Path:
        """Get output directory for subject/session."""
        out_root = Path(self.config.get('output_dir'))
        if session:
            return out_root / f'sub-{subject}' / f'ses-{session}'
        return out_root / f'sub-{subject}'

    def process_subject(self, subject: str, session: Optional[str] = None):
        """Process a single subject/session."""
        ses = f"ses-{session}" if session else ""

        # Find inputs first so anatomical selectors can constrain downstream work.
        subj_dir = self.config.bids_dir / f"sub-{subject}"
        if session:
            subj_dir = subj_dir / f"ses-{session}"

        anat_dir = subj_dir / "anat"

        t1w = bids_find_t1w(anat_dir)
        t2w = bids_find_t2w(anat_dir)
        t1w, t2w = self._apply_anat_selectors(t1w, t2w)

        if not t1w and not t2w:
            self.logger.warning(f"No T1w or T2w found for sub-{subject}. Skipping.")
            return

        # QC: MRIQC
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        if qc_cfg.get("enabled"):
            pipeline_mods = set()
            if t1w:
                pipeline_mods.add('T1w')
            if t2w:
                pipeline_mods.add('T2w')
            cfg_mods = qc_cfg.get("modalities")

            if cfg_mods:
                target_mods = list(pipeline_mods.intersection(set(cfg_mods)))
            else:
                target_mods = list(pipeline_mods)

            if target_mods:
                self.logger.info(f"Running MRIQC for sub-{subject} {ses} (modalities={target_mods})")
                mriqc_out = self.config.output_dir.parent / "mriqc"
                try:
                    run_mriqc(
                        bids_dir=self.config.bids_dir,
                        output_dir=mriqc_out,
                        participant_label=subject,
                        session_id=session,
                        n_procs=self.config.n_cpus,
                        modalities=target_mods,
                        verbose_reports=self.config.verbose
                    )
                except Exception as e:
                    self.logger.warning(f"MRIQC failed: {e}")
                    if self.config.stop_on_error:
                        raise e
            else:
                self.logger.info(f"Skipping MRIQC for sub-{subject} (Computed modalities empty. Config: {cfg_mods})")

        subj_work_dir = self._get_work_dir(subject, session)

        context = {
            "subject": subject,
            "session": session,
            "t1w_files": t1w,
            "t2w_files": t2w
        }

        # Initialize Reporter
        final_anat_dir = self._get_output_dir(subject, session) / "anat"
        report_title = f"QMRI-Neuropipe Report: sub-{subject} {ses}"

        final_anat_dir.mkdir(parents=True, exist_ok=True)
        reporter = ReportGenerator(final_anat_dir.parent, title=report_title)

        # Participant Summary
        part_summ = f"Participant: sub-{subject}"
        if session:
            part_summ += f", Session: {session}"
        reporter.set_participant_summary(part_summ, details={
            "Subject": subject,
            "Session": session or "N/A",
            "BIDS Path": str(self.config.bids_dir),
            "Output Path": str(self.config.output_dir)
        })

        try:
            results = self.preprocessing.run(subj_work_dir, context, final_output_dir=final_anat_dir, reporter=reporter)

            reporter.generate()
            try:
                reporter.generate_pdf()
            except Exception:
                pass

            tracker = self.config.tracker
            if tracker and subject and session:
                study = self.config.get('study_name')
                tracker.update_status(subject, session, "Overall_Status", "Complete", study=study, modality="Anatomical")
                tracker.save()

        except Exception as e:
            self.logger.error(f"Error processing sub-{subject}: {e}")
            if self.config.stop_on_error:
                raise e

def run_anatomical_workflow(config: PipelineConfig, subject: str, session: Optional[str] = None, reporter=None) -> dict:
    """
    Convenience function to run the anatomical workflow standalone.

    Args:
        config (PipelineConfig): Pipeline configuration.
        subject (str): Subject identifier.
        session (Optional[str]): Session identifier.
        reporter (optional): Optional reporter instance.

    Returns:
        dict: Resulting processing context with outputs.
    """
    pipeline = AnatPipeline(config)
    pipeline._initialize_pipeline()

    # Find inputs
    subj_dir = config.bids_dir / f"sub-{subject}"
    if session:
        subj_dir = subj_dir / f"ses-{session}"
    anat_dir = subj_dir / "anat"

    t1w = bids_find_t1w(anat_dir)
    t2w = bids_find_t2w(anat_dir)
    t1w, t2w = pipeline._apply_anat_selectors(t1w, t2w)
    if not t1w and not t2w:
        raise RuntimeError(f"No T1w or T2w images found for subject {subject}, session {session}")

    work_dir = pipeline._get_work_dir(subject, session)
    final_output_dir = pipeline._get_output_dir(subject, session) / "anat"

    context = {
        "subject": subject,
        "session": session,
        "t1w_files": t1w,
        "t2w_files": t2w
    }

    results = pipeline.preprocessing.run(work_dir, context, final_output_dir=final_output_dir, reporter=reporter)

    # Validate outputs and add error summary if not already present
    errors = results.get('errors', [])
    if errors and reporter:
        error_rows = [{"Error": err} for err in errors]
        reporter.add_anat_summary("Final Validation Errors", error_rows)

    return results
