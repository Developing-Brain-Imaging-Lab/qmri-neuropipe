"""
FreeSurfer SuperSynth (mri_super_synth) processing step.
"""

from pathlib import Path
from typing import Any, Optional

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageFile, ImageLike
from ...interfaces import freesurfer


# SuperSynth output filenames as produced by mri_super_synth.
# These follow the naming conventions observed for FreeSurfer Synth tools;
# adjust if the tool uses different names in your FreeSurfer build.
_OUTPUT_STEMS = {
    "seg": "seg.nii.gz",
    "synth_t1w": "T1w.nii.gz",
    "synth_t2w": "T2w.nii.gz",
    "synth_flair": "FLAIR.nii.gz",
}


class SuperSynthStep(BaseProcessingStep):
    """
    Run FreeSurfer mri_super_synth (SuperSynth) on an anatomical image.

    SuperSynth is a U-Net trained on synthetic data that produces:
    - Brain region segmentation
    - MNI atlas registration
    - Synthetic 1 mm isotropic T1w, T2w, and FLAIR images
    - Dice scores for segmentation QC

    It accepts inputs of any resolution/contrast and supports in vivo, ex vivo,
    single-hemisphere, and cerebrum-only acquisitions.

    Requires a FreeSurfer development build newer than October 2025.

    Configuration (``anat.super_synth``):
        enabled (bool): Whether to run this step. Default False.
        mode (str): Input type — ``invivo``, ``exvivo``, ``cerebrum``,
            ``left-hemi``, or ``right-hemi``. Default ``"invivo"``.
        sharpen_synths (bool): Sharpen synthetic predictions. Default False.
        device (str|None): ``"cpu"`` or ``"cuda"``. Omit to use tool default.
    """

    def __init__(self, config, logger=None, provenance=None):
        super().__init__(config, logger, provenance)
        ss_cfg = config.get("anat", {}).get("super_synth", {})
        self.enabled = ss_cfg.get("enabled", False)
        self.mode = ss_cfg.get("mode", "invivo")
        self.sharpen_synths = ss_cfg.get("sharpen_synths", False)
        self.device: Optional[str] = ss_cfg.get("device", None)

    # ------------------------------------------------------------------
    # BaseProcessingStep interface
    # ------------------------------------------------------------------

    def run(self, first_arg: Any, output_dir: Path, **kwargs) -> Any:
        if not self.enabled and not kwargs.get("force", False):
            self.logger.info("SuperSynthStep disabled — skipping.")
            return first_arg

        context, input_image = self.unpack_input(first_arg)

        # Resolve input path
        if input_image is None:
            raise ValidationError("SuperSynthStep requires an input image.")

        in_path = (
            input_image.img
            if isinstance(input_image, ImageFile)
            else input_image
        )
        in_path = Path(in_path)
        if not in_path.exists():
            raise ValidationError(f"SuperSynthStep: input not found: {in_path}")

        # Build a subject/session-specific output subdirectory so that
        # concurrent runs for different subjects do not collide.
        sub = context.get("subject") if context else None
        ses = context.get("session") if context else None
        if not sub and isinstance(input_image, ImageFile):
            sub = getattr(input_image, "entities", {}).get("sub")

        step_dir = output_dir / "super_synth"
        if sub:
            step_dir = step_dir / f"sub-{sub}"
        if ses:
            step_dir = step_dir / f"ses-{ses}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # Resolve runtime overrides
        mode = kwargs.get("mode", self.mode)
        threads = kwargs.get("nthreads", getattr(self.config, "n_cpus", -1))
        if threads is None:
            threads = -1
        device = kwargs.get("device", self.device)
        sharpen = kwargs.get("sharpen_synths", self.sharpen_synths)

        self.logger.info(
            f"Running mri_super_synth (mode={mode}) on {in_path.name} → {step_dir}"
        )

        freesurfer.mri_super_synth(
            in_file=in_path,
            out_dir=step_dir,
            mode=mode,
            threads=threads,
            device=device,
            sharpen_synths=sharpen,
            overwrite=kwargs.get("force", False),
        )

        # Collect outputs and update context
        outputs: dict[str, Path] = {}
        for key, fname in _OUTPUT_STEMS.items():
            candidate = step_dir / fname
            if candidate.exists():
                outputs[key] = candidate

        if not outputs:
            self.logger.warning(
                f"mri_super_synth finished but no recognised output files were "
                f"found in {step_dir}. Check the FreeSurfer output naming for "
                f"your build version."
            )
        else:
            self.logger.info(
                f"mri_super_synth outputs: {[k for k in outputs]}"
            )

        if context is not None:
            context["super_synth_dir"] = step_dir
            context["super_synth_outputs"] = outputs

            # If a synthetic T1w was produced and no preprocessed T1w is set
            # yet, inject it so downstream steps can use it.
            if "synth_t1w" in outputs and "preprocessed_t1w" not in context:
                entities = {"sub": sub, "ses": ses, "suffix": "T1w", "desc": "synthT1w"}
                context["preprocessed_t1w"] = ImageFile(
                    img=outputs["synth_t1w"], entities={k: v for k, v in entities.items() if v}
                )

            return context

        # Standalone (no context) — return the output directory
        return step_dir
