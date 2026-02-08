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

from ...core import BasePipeline, BaseWorkflow, PipelineConfig
from ...core.types import ImageFile
from ...io.anat.bids import bids_find_t1w, bids_find_t2w
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
from ...lib.common.sharpen import SharpeningStep
from ...lib.common.segmentation import SegmentationStep
from ...interfaces.mriqc import run_mriqc
from ...interfaces import ants  # For manual apply_transforms
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

@dataclass
class NormalizationConfig:
    enabled: bool = False
    template: Optional[str] = None

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
class AnatomicalConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    freesurfer: FreeSurferConfig = field(default_factory=FreeSurferConfig)


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
        
        self.anat_config = AnatomicalConfig(
            preprocessing=PreprocessingConfig(
                resample=preprocessing_cfg.get("resample", {}),
                reorient=preprocessing_cfg.get("reorient", {}),
                denoising=preprocessing_cfg.get("denoising", {}),
                degibbs=preprocessing_cfg.get("degibbs", {}),
                gibbs=preprocessing_cfg.get("gibbs", {}),
                bias_correction=preprocessing_cfg.get("bias_correction", {}),
                sharpen=preprocessing_cfg.get("sharpen", {}),
                brain_masking=preprocessing_cfg.get("brain_masking", {}),
                coregistration=preprocessing_cfg.get("coregistration", {}),
                normalization=normalization_cfg,
                recon_all=preprocessing_cfg.get("recon_all", {}),
                use_freesurfer=freesurfer_use,
                force_run=preprocessing_cfg.get("force_run", False),
            ),
            normalization=NormalizationConfig(
                enabled=normalization_cfg.get("enabled", False),
                template=normalization_cfg.get("template")
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
            )
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
            self.add_step(NonlinearRegistrationStep(self.config, self.logger, self.provenance, template=norm_cfg.get("template")))

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

        # Validate preprocessed T1w
        _check_image_file(context.get("preprocessed_t1w"), "Preprocessed_T1w")

        # Validate preprocessed T2w or coregistered T2w
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

        for step in self.steps:
            if isinstance(step, (CoregistrationStep, BrainMaskingStep, NonlinearRegistrationStep, SegmentationStep, FreeSurferStatsStep)):
                continue

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
            if final_output_dir and step_desc:
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
                    processed_t1 = step(processed_t1, output_dir=output_dir, force=force_run)
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

        current_t2 = t2w_files[0]
        processed_t2 = current_t2

        for step in self.steps:
            if isinstance(step, (ReconAllStep, NonlinearRegistrationStep, BrainMaskingStep, CoregistrationStep, SegmentationStep)):
                continue

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
            if final_output_dir and step_desc:
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
                    processed_t2 = step(processed_t2, output_dir=output_dir)
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

        try:
            coreg_step = CoregistrationStep(self.config, self.logger, self.provenance, method=coreg_cfg_run.get("method", "fsl"))
            coreg_step.modality = "Anatomical"

            ref_img = coreg_cfg_run.get("reference_image", "t1w").lower()

            # Flatten options logic (Top-level + Nested 'options')
            coreg_options = dict(coreg_cfg_run)
            if "options" in coreg_options:
                sub_opts = coreg_options.pop("options")
                if isinstance(sub_opts, dict):
                    coreg_options.update(sub_opts)

            st = time.time()
            if ref_img == 't2w':
                self.logger.info("Coregistration: Reference=T2w. Registering T1w -> T2w.")
                res_t1 = coreg_step(context["preprocessed_t1w"], output_dir=output_dir, target=context["preprocessed_t2w"].img, options=coreg_options)
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
                res_t2 = coreg_step(context["preprocessed_t2w"], output_dir=output_dir, target=context["preprocessed_t1w"].img, options=coreg_options)
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

        skipped_mask = False
        if final_output_dir and skip_existing:
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
                brain_masked, mask = mask_step(target_img, output_dir=output_dir, return_mask=True)
                dur = time.time() - st
                step_metrics.append({"Step": "BrainMasking", "Status": "Completed", "Duration": f"{dur:.2f}s"})

                context["brain_mask"] = mask

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

    def _run_normalization(self, output_dir: Path, context: dict, final_output_dir: Optional[Path], reporter, figures_dir: Path, step_metrics: List[dict]) -> (dict, List[dict]):
        """
        Run nonlinear normalization if enabled.

        Returns:
            Updated context and metrics list.
        """
        norm_step = next((s for s in self.steps if isinstance(s, NonlinearRegistrationStep)), None)
        if not norm_step:
            return context, step_metrics

        save_inter = self.config.get("save_intermediate", False)
        skip_existing = self.config.get("skip_existing", False)
        errors = context.setdefault('errors', [])

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.update(description=f"[cyan]Normalization ({norm_step.template})")
            except Exception:
                pass

        t1_img = context.get("preprocessed_t1w")
        t1_mask = context.get("brain_mask")

        if t1_img:
            self.logger.info("Running Normalization on T1w...")
            try:
                norm_step(context, output_dir=output_dir)
            except Exception as e:
                err_msg = f"Normalization step failed: {e}"
                self.logger.error(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Normalization", {"Status": "Failed"}, details={"error": str(e)})
                return context, step_metrics

            norm_t1 = context.get("current_image")
            context["preprocessed_t1w"] = norm_t1

            if not (norm_t1 and hasattr(norm_t1, 'img') and norm_t1.img.exists()):
                err_msg = "Normalization output T1w image missing or invalid."
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step("Normalization", {"Status": "Output Missing"}, details={"path": str(getattr(norm_t1, 'img', 'Unknown'))})

            # Apply to T2w
            transform = context.get("template_transform")
            t2_img = context.get("preprocessed_t2w")
            if "preprocessed_t2w_coreg" in context:
                t2_img = context["preprocessed_t2w_coreg"]

            if t2_img and transform and transform.exists():
                self.logger.info("Applying Normalization Warp to T2w...")
                ents = dict(t2_img.entities)
                ents['space'] = 'Standard'
                ents['desc'] = 'norm'
                if 'suffix' not in ents:
                    ents['suffix'] = 'T2w'

                norm_t2_path = output_dir / build_bids_name(ents)

                try:
                    if not norm_t2_path.exists() or not skip_existing:
                        template = norm_step.template or self.anat_config.preprocessing.normalization.get("template")

                        if template:
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
                                norm_t2_obj = ImageFile(norm_t2_path, self.config.bids_dir, entities=ents)
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
                        context["preprocessed_t2w"] = ImageFile(norm_t2_path, self.config.bids_dir, entities=ents)

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

        if reporter:
            try:
                from rich.progress import Progress
                if isinstance(reporter, Progress):
                    reporter.advance()
            except Exception:
                pass

        # Save intermediate normalized images
        if save_inter and final_output_dir:
            def _save_norm_img(img_obj, suffix: str):
                if img_obj and hasattr(img_obj, 'entities') and hasattr(img_obj, 'img') and img_obj.img.exists():
                    if img_obj.entities.get('space') == 'Standard':
                        ents = dict(img_obj.entities)
                        ents['desc'] = 'norm'
                        if 'suffix' not in ents:
                            ents['suffix'] = suffix
                        fname = build_bids_name(ents)
                        dest = final_output_dir / fname
                        try:
                            if not dest.exists():
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy(img_obj.img, dest)
                                if img_obj.json and img_obj.json.exists():
                                    shutil.copy(img_obj.json, dest.with_suffix("").with_suffix(".json"))
                        except Exception as e:
                            err_msg = f"Failed to save normalized intermediate {suffix}: {e}"
                            self.logger.warning(err_msg)
                            errors.append(err_msg)
                            if reporter:
                                self._add_anat_step(f"Normalization_{suffix}", {"Status": "Failed to Save Intermediate"}, details={"error": str(e)})

            _save_norm_img(context.get("preprocessed_t1w"), 'T1w')
            _save_norm_img(context.get("preprocessed_t2w"), 'T2w')

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
            fs_stats_step(context, output_dir=output_dir)
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
            seg_step(context, output_dir=output_dir)
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

        # T2w Path
        pre_t2 = context.get("preprocessed_t2w_coreg") or context.get("preprocessed_t2w")
        if pre_t2:
            ents = dict(pre_t2.entities)
            ents['desc'] = 'preproc'
            if 'suffix' not in ents:
                ents['suffix'] = 'T2w'
            t2_path = final_output_dir / build_bids_name(ents)
            anat_outputs["Final Preprocessed Images"].append({"key": "Preprocessed T2w", "path": str(t2_path)})

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

        if not t1w_files:
            return None

        preprocessed_t1w = None
        preprocessed_t2w = None
        brain_mask = None

        t1w = t1w_files[0]
        t1_entities = dict(getattr(t1w, "entities", {}) or {})
        t1_entities.setdefault("suffix", "T1w")
        t1_entities["desc"] = "preproc"
        t1_name = build_bids_name(t1_entities)
        t1_path = final_output_dir / t1_name

        if not t1_path.exists():
            return None

        preprocessed_t1w = ImageFile(entities=t1_entities, img=t1_path, json=None)

        mask_entities = dict(t1_entities)
        mask_entities["suffix"] = "mask"
        mask_name = build_bids_name(mask_entities)
        mask_path = final_output_dir / mask_name
        if mask_path.exists():
            brain_mask = ImageFile(entities=mask_entities, img=mask_path, json=None)

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

        return {
            "preprocessed_t1w": preprocessed_t1w,
            "preprocessed_t2w": preprocessed_t2w,
            "brain_mask": brain_mask
        }

    def run(self, output_dir: Path, context: dict, final_output_dir: Optional[Path] = None, reporter=None) -> dict:
        """
        Execute the anatomical preprocessing workflow.
        """
        self.reporter = reporter

        output_dir.mkdir(parents=True, exist_ok=True)
        if final_output_dir:
            final_output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.get("skip_existing", True) and not self.config.get("force", False):
            cached = self._load_preprocessed_from_output(context, final_output_dir)
            if cached:
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

        context, step_metrics = self._preprocess_t1w(
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics
        )

        context, step_metrics = self._preprocess_t2w(
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics
        )

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
        else:
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
        else:
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

        # QC: MRIQC
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        if qc_cfg.get("enabled"):
            pipeline_mods = {'T1w', 'T2w'}
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

        # Find inputs
        subj_dir = self.config.bids_dir / f"sub-{subject}"
        if session:
            subj_dir = subj_dir / f"ses-{session}"

        anat_dir = subj_dir / "anat"

        t1w = bids_find_t1w(anat_dir)
        t2w = bids_find_t2w(anat_dir)

        if not t1w and not t2w:
            self.logger.warning(f"No T1w or T2w found for sub-{subject}. Skipping.")
            return

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

