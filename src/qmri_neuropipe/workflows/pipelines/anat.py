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
from typing import Optional, List, Any
import logging
import shutil
import json
import time

import nibabel as nib
import numpy as np

try:
    from rich.progress import Progress
except ImportError:  # Rich is optional for programmatic workflow use.
    Progress = None

from ...core import (
    BasePipeline,
    BaseWorkflow,
    PipelineConfig,
    PipelineContext,
    ProcessingError,
)
from ...core.caching import reuse_if_exists, reuse_path_if_exists
from ...core.step_control import get_rerun_from_step, step_force_active, step_matches, any_step_matches
from ...core.tracking import flush_tracker, mark_tracker_dirty, update_step_status
from ...core.types import ImageFile
from ...io.anat.bids import bids_find_t1w, bids_find_t2w, select_anatomical_candidates
from ...io.bids import build_bids_name

# Steps
from ...lib.common.resample import ResampleStep
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
from .anat_config import (
    AnatomicalConfig as AnatomicalConfig,
    FreeSurferConfig as FreeSurferConfig,
    NormalizationConfig as NormalizationConfig,
    PreprocessingConfig as PreprocessingConfig,
    QCConfig as QCConfig,
    SegmentationConfig as SegmentationConfig,
    SuperSynthConfig as SuperSynthConfig,
    parse_anatomical_config,
)


def _update_progress(reporter, *, description: Optional[str] = None, advance: bool = False) -> None:
    """Update a Rich progress reporter without hiding reporter failures."""
    if Progress is None or not isinstance(reporter, Progress):
        return
    if description is not None:
        reporter.update(description=description)
    if advance:
        reporter.advance()

STEP_DESC = {
    ResampleStep: "resample",
    ReorientStep: "reorient",
    DenoisingStep: "denoise",
    GibbsUnringingStep: "gibbs",
    BiasCorrectionStep: "bias",
    SharpeningStep: "sharpen",
    NonlinearRegistrationStep: "normalize",
}

T1W_SKIP = (
    CoregistrationStep,
    BrainMaskingStep,
    NonlinearRegistrationStep,
    SegmentationStep,
    FreeSurferStatsStep,
)
T2W_SKIP = (
    ReconAllStep,
    NonlinearRegistrationStep,
    BrainMaskingStep,
    CoregistrationStep,
    SegmentationStep,
    FreeSurferStatsStep,
)
FREESURFER_T1W_SKIP = (
    ResampleStep,
    ReorientStep,
    DenoisingStep,
    GibbsUnringingStep,
    BiasCorrectionStep,
    SharpeningStep,
)

class AnatPreprocessingWorkflow(BaseWorkflow):
    """
    Workflow for preprocessing T1w (and optionally T2w) images.
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger, provenance: Any):
        self.anat_config = AnatomicalConfig()
        super().__init__(config, logger, provenance)
        self.anat_config = parse_anatomical_config(self.config)

        self.steps = []
        self._initialize_steps()

    def _add_anat_step(self, *args, **kwargs) -> None:
        """Forward one anatomical processing step to the reporter.

        Canonical call shape: ``self._add_anat_step(step_name, details, figures=...)``.
        ``self.reporter`` (set once per run() in ``run()``) is used unless an
        explicit ``reporter=`` keyword override is supplied. No other positional
        call shape is supported — passing a reporter positionally as the first
        argument is no longer accepted, since it silently produced wrong results
        when ``reporter`` was falsy.
        """
        figures = kwargs.pop("figures", None)
        extra_details = kwargs.pop("details", None)
        reporter = kwargs.pop("reporter", None) or getattr(self, "reporter", None)

        if len(args) != 2:
            raise TypeError(
                "_add_anat_step expects exactly two positional arguments "
                "(step_name, details); got "
                f"{len(args)}. Pass the reporter via the 'reporter' keyword "
                "if an override is needed."
            )
        step_name, base_details = args

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

    def _preprocess_t1w(
        self,
        output_dir: Path,
        context: dict,
        final_output_dir: Optional[Path],
        reporter,
        figures_dir: Path,
        step_metrics: Optional[List[dict]] = None,
    ) -> tuple[dict, List[dict]]:
        """Preprocess the required T1w input using the shared modality engine."""
        return self._preprocess_modality(
            "T1w",
            T1W_SKIP,
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics,
            required=True,
        )

    def _preprocess_t2w(
        self,
        output_dir: Path,
        context: dict,
        final_output_dir: Optional[Path],
        reporter,
        figures_dir: Path,
        step_metrics: Optional[List[dict]] = None,
    ) -> tuple[dict, List[dict]]:
        """Preprocess optional T2w input using the shared modality engine."""
        return self._preprocess_modality(
            "T2w",
            T2W_SKIP,
            output_dir,
            context,
            final_output_dir,
            reporter,
            figures_dir,
            step_metrics,
            required=False,
        )

    def _preprocess_modality(
        self,
        suffix: str,
        skip_types: tuple[type, ...],
        output_dir: Path,
        context: dict,
        final_output_dir: Optional[Path],
        reporter,
        figures_dir: Path,
        step_metrics: Optional[List[dict]],
        *,
        required: bool,
    ) -> tuple[dict, List[dict]]:
        """Run the behavior-compatible anatomical preprocessing loop."""
        if step_metrics is None:
            step_metrics = []

        errors = context.setdefault("errors", [])
        modality_key = suffix.lower()
        image_files = context.get(f"{modality_key}_files", [])
        if not image_files:
            if required:
                msg = f"No {suffix} image found."
                self.logger.error(msg)
                errors.append(msg)
                raise ValueError(msg)
            return context, step_metrics

        is_t1w = suffix == "T1w"
        use_freesurfer = (
            is_t1w and self.anat_config.preprocessing.use_freesurfer
        )
        save_inter = self.config.get(
            "save_intermediates",
            self.config.get("save_intermediate", False),
        )
        skip_existing = self.config.get("skip_existing", False)
        force_run = self.anat_config.preprocessing.force_run
        if is_t1w and force_run:
            self.logger.info(
                "Anatomical force_run enabled: Ignoring existing outputs."
            )
            skip_existing = False

        processed = image_files[0]
        if is_t1w:
            context["current_image"] = processed

        rerun_from_step = (
            getattr(self, "_anat_rerun_from_step", None)
            or get_rerun_from_step(
                self.config,
                "anat.preprocessing",
                "anat",
            )
        )
        force_from_step_active = bool(
            getattr(self, "_anat_force_from_step_active", False)
        )

        for step in self.steps:
            if isinstance(step, skip_types):
                continue

            force_from_step_active = step_force_active(
                force_from_step_active,
                step,
                rerun_from_step,
            )
            self._anat_force_from_step_active = force_from_step_active

            if use_freesurfer and isinstance(step, FREESURFER_T1W_SKIP):
                update_step_status(
                    self.config,
                    context,
                    step,
                    "completed (FreeSurfer)",
                    modality="Anatomical",
                )
                continue

            processed = self._run_one_anat_step(
                suffix=suffix,
                step=step,
                processed=processed,
                output_dir=output_dir,
                context=context,
                final_output_dir=final_output_dir,
                reporter=reporter,
                figures_dir=figures_dir,
                step_metrics=step_metrics,
                errors=errors,
                save_inter=save_inter,
                skip_existing=skip_existing,
                force_run=force_run,
                force_from_step_active=force_from_step_active,
            )

        context[f"preprocessed_{modality_key}"] = processed
        self.logger.info(f"{suffix} processing complete: {processed.img}")

        try:
            self._run_mriqc_qc(
                [processed],
                output_dir,
                context,
                reporter,
            )
        except Exception as exc:
            err_msg = f"{suffix} QC step failed: {exc}"
            self.logger.warning(err_msg)
            context.setdefault("errors", []).append(err_msg)

        self._validate_outputs(context, step_metrics, reporter)
        return context, step_metrics

    def _run_one_anat_step(
        self,
        *,
        suffix: str,
        step,
        processed: ImageFile,
        output_dir: Path,
        context: dict,
        final_output_dir: Optional[Path],
        reporter,
        figures_dir: Path,
        step_metrics: List[dict],
        errors: List[str],
        save_inter: bool,
        skip_existing: bool,
        force_run: bool,
        force_from_step_active: bool,
    ) -> ImageFile:
        """Run, reuse, publish, and report one modality preprocessing step."""
        is_t1w = suffix == "T1w"
        step_name = step.__class__.__name__
        self.logger.info(f"Running {suffix} step: {step_name}")
        step_desc = next(
            (
                desc
                for step_type, desc in STEP_DESC.items()
                if isinstance(step, step_type)
            ),
            None,
        )

        skipped = False
        if final_output_dir and step_desc and not force_from_step_active:
            entities = dict(processed.entities)
            entities["desc"] = step_desc
            if "suffix" not in entities:
                entities["suffix"] = suffix

            filename = build_bids_name(entities)
            expected_path = final_output_dir / filename
            cached_image = reuse_path_if_exists(expected_path, entities)

            if (
                is_t1w
                and cached_image is None
                and skip_existing
                and step == self.steps[-1]
            ):
                preproc_entities = dict(processed.entities)
                preproc_entities["desc"] = "preproc"
                if "suffix" not in preproc_entities:
                    preproc_entities["suffix"] = suffix
                preproc_name = build_bids_name(preproc_entities)
                preproc_path = final_output_dir / preproc_name
                if preproc_path.exists():
                    expected_path = preproc_path
                    filename = preproc_name
                    cached_image = reuse_path_if_exists(
                        expected_path,
                        entities,
                    )

            if cached_image is not None and (skip_existing or save_inter):
                self.logger.info(
                    f"Skipping {step_name} "
                    f"(Found existing output: {filename})"
                )
                try:
                    processed = cached_image
                    skipped = True
                    step_metrics.append({
                        "Step": f"{suffix}_{step_name}",
                        "Status": "Skipped (Found)",
                        "Duration": "0s",
                    })

                    update_step_status(
                        self.config,
                        context,
                        step,
                        "completed (cached)",
                        modality="Anatomical",
                    )
                except Exception as exc:
                    if is_t1w:
                        self.logger.warning(
                            "Failed to load existing intermediate "
                            f"{filename}: {exc}. Re-running."
                        )

        if not skipped:
            started = time.time()
            try:
                step_force = force_run or force_from_step_active
                if force_from_step_active:
                    self.logger.info(
                        f"Forcing {suffix} {step_name} because "
                        "rerun_from_step has been reached."
                    )
                if is_t1w and isinstance(step, ReconAllStep):
                    context["current_image"] = processed
                    processed = step(
                        context,
                        output_dir=output_dir,
                        force=step_force,
                    )
                else:
                    processed = step(
                        processed,
                        output_dir=output_dir,
                        force=step_force,
                    )
            except Exception as exc:
                err_msg = f"Error during {suffix} {step_name}: {exc}"
                self.logger.error(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step(
                        f"{suffix}_{step_name}",
                        {"Status": "Failed"},
                        details={"error": str(exc)},
                    )
                return processed

            duration = time.time() - started
            step_metrics.append({
                "Step": f"{suffix}_{step_name}",
                "Status": "Completed",
                "Duration": f"{duration:.2f}s",
            })

            if isinstance(processed, dict):
                if is_t1w:
                    context.update(processed)
                    processed = context.get("current_image")
                else:
                    processed = processed.get("current_image", processed)

            if (
                hasattr(processed, "img")
                and not processed.img.exists()
            ):
                err_msg = (
                    f"Output file missing after {suffix} {step_name}: "
                    f"{processed.img}"
                )
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step(
                        f"{suffix}_{step_name}",
                        {"Status": "Output Missing"},
                        details={"path": str(processed.img)},
                    )

        if (
            save_inter
            and not skipped
            and final_output_dir
            and step_desc
        ):
            try:
                entities = dict(processed.entities)
                entities["desc"] = step_desc
                if "suffix" not in entities:
                    entities["suffix"] = suffix
                destination = final_output_dir / build_bids_name(entities)

                if not destination.exists():
                    destination.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    shutil.copy(processed.img, destination)
                    if processed.json and processed.json.exists():
                        shutil.copy(
                            processed.json,
                            destination.with_suffix("")
                            .with_suffix(".json"),
                        )
            except Exception as exc:
                self.logger.warning(
                    f"Failed to save intermediate {suffix} "
                    f"{step_name} output: {exc}"
                )

        if reporter and (not is_t1w or not skipped):
            self._report_anat_preprocessing_step(
                suffix,
                step,
                processed,
                figures_dir,
            )

        return processed

    def _report_anat_preprocessing_step(
        self,
        suffix: str,
        step,
        processed: ImageFile,
        figures_dir: Path,
    ) -> None:
        """Add the existing modality-specific preprocessing report content."""
        is_t1w = suffix == "T1w"
        step_name = step.__class__.__name__
        try:
            from ...lib.reporting.viz import create_ortho_view

            figures = []
            if isinstance(step, DenoisingStep):
                figure_path = (
                    figures_dir / f"{suffix.lower()}_denoised.png"
                )
                create_ortho_view(
                    processed.img,
                    figure_path,
                    title=f"Denoised {suffix}",
                )
                figures.append({
                    "path": str(figure_path),
                    "title": "Denoising",
                    "caption": f"Denoised {suffix}",
                })
            elif is_t1w and isinstance(step, GibbsUnringingStep):
                figure_path = figures_dir / "t1w_gibbs.png"
                create_ortho_view(
                    processed.img,
                    figure_path,
                    title="Gibbs Corrected T1w",
                )
                figures.append({
                    "path": str(figure_path),
                    "title": "Gibbs",
                    "caption": "Gibbs Corrected T1w",
                })

            details = {"Modality": suffix}
            if hasattr(step, "method"):
                details["Method"] = step.method
            if is_t1w and hasattr(step, "patch_radius"):
                details["Patch Radius"] = step.patch_radius

            self._add_anat_step(
                step_name,
                details,
                figures=figures,
            )
        except ImportError as exc:
            if not is_t1w:
                self.logger.warning(
                    f"Failed to plot {suffix} report figure: {exc}"
                )
        except Exception as exc:
            self.logger.warning(
                f"Failed to plot {suffix} report figure: {exc}"
            )

    def _prepare_t2w_for_coregistration(
        self,
        context: dict,
        output_dir: Path,
    ) -> ImageFile:
        """Prepare the T2w moving image for a skull-stripped FS target."""
        if not self.anat_config.preprocessing.use_freesurfer:
            return context["preprocessed_t2w"]

        mask_step = next(
            (s for s in self.steps if isinstance(s, BrainMaskingStep)),
            None,
        )
        if not mask_step:
            self.logger.warning(
                "FreeSurfer anatomical coregistration requested, but no "
                "BrainMaskingStep is configured. Proceeding with the "
                "unmasked T2w image."
            )
            return context["preprocessed_t2w"]

        self.logger.info(
            "FreeSurfer coregistration uses a skull-stripped T1w target. "
            "Generating a skull-stripped T2w moving image before T2w -> "
            "T1w registration."
        )
        masked_t2w, _ = mask_step(
            context["preprocessed_t2w"],
            output_dir=output_dir,
            return_mask=True,
        )
        return masked_t2w

    @staticmethod
    def _flatten_coregistration_options(config: dict) -> dict:
        """Flatten the legacy nested options block without mutating config."""
        options = dict(config)
        nested = options.pop("options", None)
        if isinstance(nested, dict):
            options.update(nested)
        return options

    def _make_coregistration_step(
        self,
        config: dict,
        options: dict,
        *,
        method: Optional[str] = None,
    ) -> CoregistrationStep:
        step = CoregistrationStep(
            self.config,
            self.logger,
            self.provenance,
            method=method or config.get("method", "fsl"),
            options={
                key: value
                for key, value in options.items()
                if key not in {"enabled", "method"}
            },
        )
        step.modality = "Anatomical"
        return step

    @staticmethod
    def _do_coreg(
        step: CoregistrationStep,
        moving: ImageFile,
        fixed: ImageFile,
        output_dir: Path,
        options: dict,
        force: bool,
    ) -> Optional[ImageFile]:
        """Execute one registration and normalize its result contract."""
        result = step(
            moving,
            output_dir=output_dir,
            target=fixed.img,
            options=options,
            force=force,
        )
        if isinstance(result, dict):
            result = result.get("current_image")
        return result

    def _report_coregistration(
        self,
        reporter,
        fixed: ImageFile,
        result: ImageFile,
        figures_dir: Path,
        *,
        filename: str,
        title: str,
        reference: str,
        moving: str,
        method: str,
        caption: str,
    ) -> None:
        if not reporter:
            return
        try:
            from ...lib.reporting.viz import plot_comparison

            figure_path = figures_dir / filename
            plot_comparison(
                fixed.img,
                result.img,
                figure_path,
                title=title,
            )
            self._add_anat_step(
                "Coregistration",
                {
                    "Method": method,
                    "Reference": reference,
                    "Moving": moving,
                },
                figures=[{
                    "path": str(figure_path),
                    "title": "Coregistration",
                    "caption": caption,
                }],
            )
        except Exception as exc:
            self.logger.warning(
                f"Failed to plot Coregistration report: {exc}"
            )

    def _coregister_to_supersynth(
        self,
        ref_img: str,
        coreg_config: dict,
        coreg_options: dict,
        coreg_step: CoregistrationStep,
        force_coreg: bool,
        context: dict,
        output_dir: Path,
        reporter,
        figures_dir: Path,
    ) -> None:
        """Register anatomical images using matched synthetic contrasts."""
        from ...lib.anat.super_synth import (
            ensure_matched_supersynth_registration_inputs,
        )

        t1w = context.get("preprocessed_t1w")
        t2w = context.get("preprocessed_t2w")
        preference = str(coreg_options.get("supersynth_input", "auto")).lower()
        if preference == "t2w":
            fixed_source, moving = t2w, t1w
        else:
            fixed_source, moving = t1w, t2w
        if fixed_source is None or moving is None:
            raise ProcessingError(
                "Matched SuperSynth anatomical coregistration requires both "
                "preprocessed T1w and T2w images."
            )

        use_multivariate = (
            ref_img == "supersynth_multivariate"
            or str(coreg_options.get("supersynth_registration", "")).lower() == "multivariate"
        )

        if use_multivariate and coreg_step.method != "ants":
            self.logger.warning(
                "SuperSynth multivariate anatomical registration requires ANTs; "
                "using matched synthetic T1w images with %s instead.",
                coreg_step.method,
            )
            use_multivariate = False

        ss_root = (
            output_dir
            / "coregistration"
            / (
                "supersynth_multivariate"
                if use_multivariate
                else "supersynth_matched"
            )
        )
        required = (
            ("synth_t1w", "synth_t2w")
            if use_multivariate
            else ("synth_t1w",)
        )
        try:
            fixed_outputs, moving_outputs = (
                ensure_matched_supersynth_registration_inputs(
                    fixed_source,
                    moving,
                    ss_root,
                    self.config,
                    self.logger,
                    required_contrasts=required,
                    mode=coreg_options.get("supersynth_mode"),
                    device=coreg_options.get("supersynth_device"),
                    sharpen_synths=coreg_options.get(
                        "supersynth_sharpen_synths"
                    ),
                    force=bool(coreg_options.get("force", False))
                    or force_coreg,
                )
            )
        except Exception as exc:
            raise ProcessingError(
                f"Matched SuperSynth anatomical coregistration failed: {exc}"
            ) from exc

        synth_ref = ImageFile(
            entities={**fixed_source.entities, "desc": "synthT1w"},
            img=fixed_outputs["synth_t1w"],
        )
        coreg_options.update(
            {
                "registration_fixed": fixed_outputs["synth_t1w"],
                "registration_moving": moving_outputs["synth_t1w"],
                "application_fixed": fixed_source.img,
            }
        )

        if use_multivariate:
            coreg_options.update(
                {
                    "registration_fixed_extras": [
                        fixed_outputs["synth_t2w"]
                    ],
                    "registration_moving_extras": [
                        moving_outputs["synth_t2w"]
                    ],
                    "transform_type": coreg_options.get(
                        "transform_type", "SyNOnly"
                    ),
                }
            )
            coreg_step = self._make_coregistration_step(
                coreg_config,
                coreg_options,
                method="ants",
            )
            contrast_label = "T1w+T2w"
        else:
            coreg_options.pop("registration_fixed_extras", None)
            coreg_options.pop("registration_moving_extras", None)
            contrast_label = "T1w"

        self.logger.info(
            "Coregistration: matched SuperSynth %s registration "
            "(moving=%s -> fixed=%s); applying the transform to the original "
            "anatomical images.",
            contrast_label,
            moving.entities.get("suffix", "anatomical"),
            fixed_source.entities.get("suffix", "anatomical"),
        )

        res_anat = self._do_coreg(
            coreg_step,
            moving,
            synth_ref,
            output_dir,
            coreg_options,
            force_coreg,
        )
        context["preprocessed_anat_coreg"] = res_anat
        context["preprocessed_t1w_synth"] = synth_ref

        if moving is context.get("preprocessed_t2w"):
            context["preprocessed_t2w_coreg"] = res_anat
        else:
            context["preprocessed_t1w"] = res_anat

        self._report_coregistration(
            reporter,
            fixed_source,
            res_anat,
            figures_dir,
            filename="coreg_anat_on_supersynth.png",
            title="Coregistration using matched SuperSynth contrasts",
            reference=fixed_source.entities.get("suffix", "Anatomical"),
            moving=moving.entities.get("suffix", "Anat"),
            method=coreg_step.method,
            caption=(
                "Original anatomical moving image resampled onto the original "
                "fixed-image grid after matched SuperSynth registration"
            ),
        )

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

        _update_progress(reporter, description="[cyan]Coregistration")

        errors = context.setdefault('errors', [])

        try:
            # Flatten options logic (Top-level + Nested 'options')
            coreg_options = self._flatten_coregistration_options(
                coreg_cfg_run
            )
            coreg_step = self._make_coregistration_step(
                coreg_cfg_run,
                coreg_options,
            )
            rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
            force_coreg = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(coreg_step, rerun_from_step)
            self._anat_force_from_step_active = force_coreg
            if force_coreg:
                self.logger.info("Forcing Coregistration because rerun_from_step has been reached.")

            ref_img = coreg_cfg_run.get("reference_image", "t1w").lower()

            st = time.time()
            if ref_img in {"supersynth", "syntht1w", "synthetic_t1w", "supersynth_multivariate"}:
                self._coregister_to_supersynth(
                    ref_img,
                    coreg_cfg_run,
                    coreg_options,
                    coreg_step,
                    force_coreg,
                    context,
                    output_dir,
                    reporter,
                    figures_dir,
                )

            elif ref_img == 't2w':
                self.logger.info("Coregistration: Reference=T2w. Registering T1w -> T2w.")
                res_t1 = self._do_coreg(
                    coreg_step,
                    context["preprocessed_t1w"],
                    context["preprocessed_t2w"],
                    output_dir,
                    coreg_options,
                    force_coreg,
                )
                context["preprocessed_t1w"] = res_t1

                self._report_coregistration(
                    reporter,
                    context["preprocessed_t2w"],
                    res_t1,
                    figures_dir,
                    filename="coreg_t1_on_t2.png",
                    title="Coreg T1w -> T2w",
                    reference="T2w",
                    moving="T1w",
                    method=coreg_step.method,
                    caption="T1w (overlay) on T2w",
                )

            else:
                self.logger.info("Coregistration: Reference=T1w. Registering T2w -> T1w.")
                moving_t2 = self._prepare_t2w_for_coregistration(
                    context,
                    output_dir,
                )
                res_t2 = self._do_coreg(
                    coreg_step,
                    moving_t2,
                    context["preprocessed_t1w"],
                    output_dir,
                    coreg_options,
                    force_coreg,
                )
                context["preprocessed_t2w_coreg"] = res_t2

                self._report_coregistration(
                    reporter,
                    context["preprocessed_t1w"],
                    res_t2,
                    figures_dir,
                    filename="coreg_t2_on_t1.png",
                    title="Coreg T2w -> T1w",
                    reference="T1w",
                    moving="T2w",
                    method=coreg_step.method,
                    caption="T2w (overlay) on T1w",
                )

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
        save_inter = self.config.get(
            "save_intermediates",
            self.config.get("save_intermediate", False),
        )
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

        _update_progress(reporter, description="[cyan]Brain Masking")

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

        save_inter = self.config.get(
            "save_intermediates",
            self.config.get("save_intermediate", False),
        )
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

            cached_mask = reuse_path_if_exists(
                m_dest,
                m_ents,
                force=force_masking,
            )
            if cached_mask is not None:
                self.logger.info(f"Skipping Brain Masking (Found existing mask: {m_fname})")
                context["brain_mask"] = cached_mask
                _store_skull_stripped_preproc(ref_img_key, target_img, context["brain_mask"])
                skipped_mask = True
                step_metrics.append({"Step": "BrainMasking", "Status": "Skipped (Found)", "Duration": "0s"})

                update_step_status(
                    self.config,
                    context,
                    mask_step,
                    "completed (cached)",
                    modality="Anatomical",
                )

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

        _update_progress(reporter, advance=True)

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
        cached = reuse_if_exists(
            ents,
            output_dir,
            force=force or not self.config.get("skip_existing", False),
            json_path=getattr(img_obj, "json", None),
        )
        if cached is not None:
            return cached

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

    def apply_spatial_transform(
        self,
        in_img: Path,
        template: Path,
        transform: Path,
        *,
        transform_type: str,
        interp: str,
        out_path: Path,
    ) -> Path:
        """Apply an FSL affine or ANTs warp/affine transform pair."""
        transform = Path(transform)
        if transform_type == "fsl" or transform.suffix == ".mat":
            fsl.applywarp(
                in_file=in_img,
                ref_file=Path(template),
                out_file=out_path,
                premat=transform,
                interp=interp,
                force=True,
            )
            return out_path

        warp = transform.with_suffix("").parent / (
            transform.name + "1Warp.nii.gz"
        )
        affine = transform.with_suffix("").parent / (
            transform.name + "0GenericAffine.mat"
        )
        if not (warp.exists() and affine.exists()):
            raise FileNotFoundError(
                f"Missing {warp.name} or {affine.name}"
            )
        ants.apply_transforms(
            fixed_file=Path(template),
            moving_file=in_img,
            out_file=out_path,
            transforms=[warp, affine],
            interpolator=(
                "nearestNeighbor" if interp == "nn" else interp
            ),
        )
        return out_path

    def _normalize_primary(
        self,
        context: dict,
        norm_step: NonlinearRegistrationStep,
        output_dir: Path,
        *,
        force_norm: bool,
        errors: List[str],
        reporter,
    ) -> tuple[dict, Optional[str], bool]:
        primary_img = context.get("preprocessed_t1w")
        primary_key = "preprocessed_t1w"
        primary_suffix = "T1w"
        if primary_img is None:
            primary_img = context.get("preprocessed_t2w_coreg") or context.get(
                "preprocessed_t2w"
            )
            primary_key = "preprocessed_t2w"
            primary_suffix = "T2w"
        if primary_img is None:
            return context, None, True

        self.logger.info("Running Normalization on %s...", primary_suffix)
        try:
            context["current_image"] = primary_img
            normalization_result = norm_step(
                context,
                output_dir=output_dir,
                force=force_norm,
            )
            if isinstance(normalization_result, dict):
                context = normalization_result
        except Exception as exc:
            err_msg = f"Normalization step failed: {exc}"
            self.logger.error(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step(
                    "Normalization",
                    {"Status": "Failed"},
                    details={"error": str(exc)},
                )
            return context, primary_suffix, False

        normalized_primary = context.get("current_image")
        context[primary_key] = normalized_primary
        if not (
            normalized_primary
            and hasattr(normalized_primary, "img")
            and normalized_primary.img.exists()
        ):
            err_msg = (
                f"Normalization output {primary_suffix} image missing or invalid."
            )
            self.logger.warning(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step(
                    "Normalization",
                    {"Status": "Output Missing"},
                    details={
                        "path": str(
                            getattr(normalized_primary, "img", "Unknown")
                        )
                    },
                )
        return context, primary_suffix, True

    def _apply_warp_to_secondary(
        self,
        context: dict,
        norm_step: NonlinearRegistrationStep,
        norm_space: str,
        output_dir: Path,
        *,
        primary_suffix: Optional[str],
        force_norm: bool,
        skip_existing: bool,
        errors: List[str],
        reporter,
    ) -> None:
        transform = context.get("template_transform")
        t2_img = context.get("preprocessed_t2w")
        if "preprocessed_t2w_coreg" in context:
            t2_img = context["preprocessed_t2w_coreg"]
        if not (
            primary_suffix == "T1w"
            and t2_img
            and transform
            and transform.exists()
        ):
            return

        self.logger.info("Applying Normalization Warp to T2w...")
        entities = dict(t2_img.entities)
        entities["space"] = norm_space
        entities["desc"] = "norm"
        entities.setdefault("suffix", "T2w")
        norm_t2_path = output_dir / build_bids_name(entities)

        try:
            cached_norm_t2 = reuse_path_if_exists(
                norm_t2_path,
                entities,
                force=force_norm or not skip_existing,
            )
            if cached_norm_t2 is not None:
                self.logger.info(
                    "Skipping T2w Normalization (Exists): %s", norm_t2_path
                )
                context["preprocessed_t2w"] = cached_norm_t2
                update_step_status(
                    self.config,
                    context,
                    "Coregistration",
                    "completed (cached)",
                    modality="Anatomical",
                )
                return

            template = norm_step.template or (
                self.anat_config.preprocessing.normalization.get("template")
            )
            if not template:
                warn_msg = "No template reference found for T2 normalization apply."
                self.logger.warning(warn_msg)
                errors.append(warn_msg)
                if reporter:
                    self._add_anat_step(
                        "Normalization_T2w", {"Status": "Template Missing"}
                    )
                return

            transform_type = str(
                context.get("template_transform_type", "ants") or "ants"
            ).lower()
            is_fsl = transform_type == "fsl" or Path(transform).suffix == ".mat"
            interp = (
                self.anat_config.preprocessing.normalization
                .get("options", {})
                .get("interpolation")
                or "trilinear"
            ) if is_fsl else "linear"
            try:
                self.apply_spatial_transform(
                    t2_img.img,
                    Path(template),
                    Path(transform),
                    transform_type=transform_type,
                    interp=interp,
                    out_path=norm_t2_path,
                )
            except FileNotFoundError:
                prefix = Path(transform)
                warp = prefix.with_suffix("").parent / (
                    prefix.name + "1Warp.nii.gz"
                )
                affine = prefix.with_suffix("").parent / (
                    prefix.name + "0GenericAffine.mat"
                )
                warn_msg = "Could not find warp/affine files for T2 normalization."
                self.logger.warning(warn_msg)
                errors.append(warn_msg)
                if reporter:
                    self._add_anat_step(
                        "Normalization_T2w",
                        {"Status": "Warp Files Missing"},
                        details={"warp": str(warp), "affine": str(affine)},
                    )
                return

            norm_t2_obj = ImageFile(entities=entities, img=norm_t2_path)
            context["preprocessed_t2w"] = norm_t2_obj
            if not norm_t2_obj.img.exists():
                backend = "FLIRT apply" if is_fsl else "apply_transforms"
                err_msg = (
                    "T2w normalized output file missing after "
                    f"{backend}: {norm_t2_obj.img}"
                )
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step(
                        "Normalization_T2w",
                        {"Status": "Output Missing"},
                        details={"path": str(norm_t2_obj.img)},
                    )
        except Exception as exc:
            err_msg = f"Failed to apply normalization warp to T2w: {exc}"
            self.logger.warning(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step(
                    "Normalization_T2w",
                    {"Status": "Failed"},
                    details={"error": str(exc)},
                )

    def _normalize_brain_mask(
        self,
        context: dict,
        norm_step: NonlinearRegistrationStep,
        norm_space: str,
        output_dir: Path,
        *,
        force_norm: bool,
        skip_existing: bool,
        errors: List[str],
        reporter,
    ) -> Optional[ImageFile]:
        """Normalize the anatomical brain mask using the primary transform."""
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
        cached_norm_mask = reuse_path_if_exists(
            norm_mask_path,
            ents,
            force=force_norm or not skip_existing,
        )
        if cached_norm_mask is not None:
            return cached_norm_mask

        try:
            transform_type = str(context.get("template_transform_type", "ants") or "ants").lower()
            is_fsl = (
                transform_type == "fsl"
                or Path(transform).suffix == ".mat"
            )
            if is_fsl:
                if not Path(transform).exists():
                    warn_msg = f"Normalization transform does not exist: {transform}"
                    self.logger.warning(warn_msg)
                    errors.append(warn_msg)
                    return None
            try:
                self.apply_spatial_transform(
                    mask_obj.img,
                    Path(template),
                    Path(transform),
                    transform_type=transform_type,
                    interp="nn",
                    out_path=norm_mask_path,
                )
            except FileNotFoundError:
                prefix = Path(transform)
                warp = prefix.with_suffix("").parent / (
                    prefix.name + "1Warp.nii.gz"
                )
                affine = prefix.with_suffix("").parent / (
                    prefix.name + "0GenericAffine.mat"
                )
                if not is_fsl:
                    warn_msg = (
                        "Could not normalize the brain mask: "
                        f"missing {warp.name} or {affine.name}."
                    )
                    self.logger.warning(warn_msg)
                    errors.append(warn_msg)
                    return None
                raise

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



    def _publish_normalized_image(
        self,
        image: Optional[ImageFile],
        suffix: str,
        final_output_dir: Path,
        errors: List[str],
        reporter,
    ) -> None:
        if not (
            image
            and hasattr(image, "entities")
            and hasattr(image, "img")
            and image.img.exists()
            and image.entities.get("space")
        ):
            return

        entities = dict(image.entities)
        entities["desc"] = "norm"
        entities.setdefault("suffix", suffix)
        destination = final_output_dir / build_bids_name(entities)
        try:
            if image.img.resolve() != destination.resolve():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image.img, destination)
                if image.json and image.json.exists():
                    shutil.copy2(
                        image.json,
                        destination.with_suffix("").with_suffix(".json"),
                    )
            image.img = destination
        except Exception as exc:
            err_msg = f"Failed to publish normalized {suffix}: {exc}"
            self.logger.warning(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step(
                    f"Normalization_{suffix}",
                    {"Status": "Failed to Save"},
                    details={"error": str(exc)},
                )

    def _publish_normalization_transforms(
        self,
        context: dict,
        final_output_dir: Path,
        errors: List[str],
        reporter,
    ) -> None:
        transform = context.get("template_transform")
        if not transform:
            return

        transform = Path(transform)
        transform_type = str(
            context.get("template_transform_type", "ants") or "ants"
        ).lower()
        if transform_type == "ants":
            candidates = [
                transform.parent / f"{transform.name}0GenericAffine.mat",
                transform.parent / f"{transform.name}1Warp.nii.gz",
                transform.parent / f"{transform.name}1InverseWarp.nii.gz",
            ]
            if transform.exists():
                candidates.append(transform)
        else:
            candidates = [transform]

        for source in candidates:
            if not source.exists():
                continue
            destination = final_output_dir / source.name
            try:
                if source.resolve() != destination.resolve():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            except Exception as exc:
                err_msg = (
                    "Failed to publish normalization transform "
                    f"{source.name}: {exc}"
                )
                self.logger.warning(err_msg)
                errors.append(err_msg)
                if reporter:
                    self._add_anat_step(
                        "Normalization_Transform",
                        {"Status": "Failed to Save"},
                        details={
                            "error": str(exc),
                            "file": str(source),
                        },
                    )

    def _publish_normalization_outputs(
        self,
        context: dict,
        final_output_dir: Optional[Path],
        errors: List[str],
        reporter,
    ) -> None:
        if not final_output_dir:
            return
        self._publish_normalized_image(
            context.get("preprocessed_t1w"),
            "T1w",
            final_output_dir,
            errors,
            reporter,
        )
        self._publish_normalized_image(
            context.get("preprocessed_t2w"),
            "T2w",
            final_output_dir,
            errors,
            reporter,
        )
        self._publish_normalization_transforms(
            context,
            final_output_dir,
            errors,
            reporter,
        )

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

        save_inter = self.config.get(
            "save_intermediates",
            self.config.get("save_intermediate", False),
        )
        skip_existing = self.config.get("skip_existing", False)
        errors = context.setdefault('errors', [])
        rerun_from_step = getattr(self, "_anat_rerun_from_step", None) or get_rerun_from_step(self.config, "anat.preprocessing", "anat")
        force_norm = bool(getattr(self, "_anat_force_from_step_active", False)) or step_matches(norm_step, rerun_from_step)
        self._anat_force_from_step_active = force_norm
        if force_norm:
            self.logger.info("Forcing Normalization because rerun_from_step has been reached.")

        _update_progress(
            reporter,
            description=f"[cyan]Normalization ({norm_step.template})",
        )

        context, primary_suffix, normalization_ok = self._normalize_primary(
            context,
            norm_step,
            output_dir,
            force_norm=force_norm,
            errors=errors,
            reporter=reporter,
        )
        if not normalization_ok:
            return context, step_metrics
        self._apply_warp_to_secondary(
            context,
            norm_step,
            norm_space,
            output_dir,
            primary_suffix=primary_suffix,
            force_norm=force_norm,
            skip_existing=skip_existing,
            errors=errors,
            reporter=reporter,
        )

        if self.anat_config.normalization.skull_stripped_outputs:
            normalized_mask = self._normalize_brain_mask(
                context,
                norm_step,
                norm_space,
                output_dir,
                force_norm=force_norm,
                skip_existing=skip_existing,
                errors=errors,
                reporter=reporter,
            )
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
                        desc="brain",
                        suffix=suffix,
                        errors=errors,
                        reporter=reporter,
                        label=f"Normalization_{suffix}_Brain",
                        force=force_norm,
                    )
                    if brain_obj:
                        context[context_key] = brain_obj
                        self.logger.info(f"Saved normalized skull-stripped {suffix}: {brain_obj.img}")

        _update_progress(reporter, advance=True)

        # Publish normalized images and transforms to the final anat directory.
        self._publish_normalization_outputs(
            context,
            final_output_dir,
            errors,
            reporter,
        )

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

        _update_progress(reporter, description="[cyan]FreeSurfer Stats")

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

        _update_progress(reporter, advance=True)

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
                    mark_tracker_dirty(self.config)

            if reporter:
                details = {"Status": "Complete", "Regions": len(context.get("super_synth_volumes", {}))}
                csv = context.get("super_synth_volumes_csv")
                if csv:
                    details["Volumes_CSV"] = str(csv)
                self._add_anat_step("SuperSynth", details)

        except Exception as e:
            err_msg = f"SuperSynth failed: {e}"
            self.logger.error(err_msg)
            errors.append(err_msg)
            if reporter:
                self._add_anat_step("SuperSynth", {"Status": "Failed", "error": str(e)})

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

        _update_progress(reporter, description="[cyan]Segmentation")

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

        _update_progress(reporter, advance=True)

        # Save intermediates
        save_inter = self.config.get(
            "save_intermediates",
            self.config.get("save_intermediate", False),
        )
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

        for f in final_output_dir.glob("*space-*_desc-brain_*.nii.gz"):
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
            preprocessed_t1w = reuse_if_exists(
                t1_entities,
                final_output_dir,
            )
            if preprocessed_t1w is None:
                return None

        if t2w_files:
            t2w = t2w_files[0]
            t2_entities = dict(getattr(t2w, "entities", {}) or {})
            t2_entities.setdefault("suffix", "T2w")
            t2_entities["desc"] = "preproc"
            preprocessed_t2w = reuse_if_exists(
                t2_entities,
                final_output_dir,
            )
            if preprocessed_t2w is None:
                return None

        if preprocessed_t1w is None and preprocessed_t2w is None:
            return None

        mask_ref = preprocessed_t1w or preprocessed_t2w
        mask_entities = dict(getattr(mask_ref, "entities", {}) or {})
        mask_entities.pop("space", None)
        mask_entities["suffix"] = "mask"
        brain_mask = reuse_if_exists(mask_entities, final_output_dir)

        if brain_mask and self.anat_config.preprocessing.skull_stripped_outputs:
            if preprocessed_t1w:
                t1_brain_entities = dict(preprocessed_t1w.entities)
                t1_brain_entities["desc"] = "preproc-brain"
                preprocessed_t1w_brain = reuse_if_exists(
                    t1_brain_entities,
                    final_output_dir,
                )
                if preprocessed_t1w_brain is None:
                    return None

            if preprocessed_t2w:
                t2_brain_entities = dict(preprocessed_t2w.entities)
                t2_brain_entities["desc"] = "preproc-brain"
                preprocessed_t2w_brain = reuse_if_exists(
                    t2_brain_entities,
                    final_output_dir,
                )
                if preprocessed_t2w_brain is None:
                    return None

        norm_cfg = self.anat_config.preprocessing.normalization or {}
        if self.anat_config.normalization.skull_stripped_outputs and norm_cfg.get("enabled"):
            norm_space = norm_cfg.get("space_entity", norm_cfg.get("space_name", norm_cfg.get("space", "Standard")))

            if preprocessed_t1w:
                norm_t1_entities = dict(preprocessed_t1w.entities)
                norm_t1_entities["space"] = norm_space
                norm_t1_entities["desc"] = "brain"
                normalized_t1w_brain = reuse_if_exists(
                    norm_t1_entities,
                    final_output_dir,
                )
                if normalized_t1w_brain is None:
                    return None

            if preprocessed_t2w:
                norm_t2_entities = dict(preprocessed_t2w.entities)
                norm_t2_entities["space"] = norm_space
                norm_t2_entities["desc"] = "brain"
                normalized_t2w_brain = reuse_if_exists(
                    norm_t2_entities,
                    final_output_dir,
                )
                if normalized_t2w_brain is None:
                    return None

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

    def run(self, output_dir: Path, context: dict, final_output_dir: Optional[Path] = None, reporter=None) -> PipelineContext:
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
                context = PipelineContext(context)
                context.update({k: v for k, v in cached.items() if v is not None})
                context["anat_preprocessing_skipped"] = True
                return context

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        context = PipelineContext(context)
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

        return PipelineContext.ensure(context)

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

    def __init__(
        self,
        config: PipelineConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, logger=logger)

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

        context = PipelineContext({
            "subject": subject,
            "session": session,
            "t1w_files": t1w,
            "t2w_files": t2w
        })

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

            if update_step_status(
                self.config,
                context,
                "Overall_Status",
                "Complete",
                modality="Anatomical",
            ):
                flush_tracker(self.config)

        except Exception as e:
            self.logger.error(f"Error processing sub-{subject}: {e}")
            if self.config.stop_on_error:
                raise e

def run_anatomical_workflow(config: PipelineConfig, subject: str, session: Optional[str] = None, reporter=None) -> PipelineContext:
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

    context = PipelineContext({
        "subject": subject,
        "session": session,
        "t1w_files": t1w,
        "t2w_files": t2w
    })

    results = pipeline.preprocessing.run(work_dir, context, final_output_dir=final_output_dir, reporter=reporter)

    # Validate outputs and add error summary if not already present
    errors = results.get('errors', [])
    if errors and reporter:
        error_rows = [{"Error": err} for err in errors]
        reporter.add_anat_summary("Final Validation Errors", error_rows)

    return results
