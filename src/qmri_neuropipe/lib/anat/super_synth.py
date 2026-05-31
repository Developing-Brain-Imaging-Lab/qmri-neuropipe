"""
FreeSurfer SuperSynth (mri_super_synth) processing step.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageFile, ImageLike
from ...interfaces import freesurfer


# SuperSynth output filenames vary across FreeSurfer builds. Prefer the current
# mri_super_synth names first, but keep the older expected names as aliases.
_OUTPUT_CANDIDATES = {
    "seg": ("segmentation.mgz", "seg.nii.gz", "seg.mgz", "segmentation.nii.gz"),
    "synth_t1w": ("SynthT1.mgz", "T1w.nii.gz", "SynthT1.nii.gz", "T1w.mgz"),
    "synth_t2w": ("SynthT2.mgz", "T2w.nii.gz", "SynthT2.nii.gz", "T2w.mgz"),
    "synth_flair": ("SynthFLAIR.mgz", "FLAIR.nii.gz", "SynthFLAIR.nii.gz", "FLAIR.mgz"),
    "ribbon": ("ribbon.mgz", "ribbon.nii.gz"),
    "input_resampled": ("input_resampled.mgz", "input_resampled.nii.gz"),
    "mni_deformation": ("mni_deformation.mgz", "mni_deformation.nii.gz"),
    "mni_affine": ("mni_affine.txt",),
    "qc_csv": ("qc.csv",),
    "volumes_csv": ("volumes.csv",),
}

_log = logging.getLogger(__name__)


def find_supersynth_outputs(out_dir: Path) -> dict[str, Path]:
    """Return recognized SuperSynth outputs for known naming conventions."""
    out_dir = Path(out_dir)
    outputs: dict[str, Path] = {}
    for key, candidates in _OUTPUT_CANDIDATES.items():
        for fname in candidates:
            candidate = out_dir / fname
            if candidate.exists():
                outputs[key] = candidate
                break
    return outputs


def expected_supersynth_output(out_dir: Path, key: str) -> Path:
    """Return the preferred output path for a SuperSynth product."""
    return Path(out_dir) / _OUTPUT_CANDIDATES[key][0]


def _load_freesurfer_lut() -> Dict[int, str]:
    """Load FreeSurferColorLUT.txt from $FREESURFER_HOME.

    Returns a dict mapping integer label IDs to region name strings.
    Returns an empty dict when the LUT cannot be found.
    """
    fs_home = os.environ.get("FREESURFER_HOME", "")
    if not fs_home:
        return {}
    lut_path = Path(fs_home) / "FreeSurferColorLUT.txt"
    if not lut_path.exists():
        return {}
    lut: Dict[int, str] = {}
    with open(lut_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    lut[int(parts[0])] = parts[1]
                except ValueError:
                    pass
    return lut


def _compute_volumes(seg_path: Path) -> Dict[int, Tuple[str, float, int]]:
    """Compute per-region volumes from a segmentation NIfTI.

    Returns a dict keyed by integer label ID:
        {label_id: (label_name, volume_mm3, n_voxels)}

    Label 0 (background / unknown) is always excluded.
    Region names fall back to ``ROI_{label_id}`` when the FreeSurfer LUT is
    not available.
    """
    img = nib.load(str(seg_path))
    data = np.round(img.get_fdata()).astype(int)
    vox_vol_mm3 = float(np.prod(img.header.get_zooms()[:3]))

    lut = _load_freesurfer_lut()

    result: Dict[int, Tuple[str, float, int]] = {}
    labels, counts = np.unique(data, return_counts=True)
    for label, count in zip(labels.tolist(), counts.tolist()):
        if label == 0:
            continue
        name = lut.get(label, f"ROI_{label}")
        result[label] = (name, float(count) * vox_vol_mm3, int(count))
    return result


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
        compute_volumes (bool): Extract and save per-region anatomical volumes
            from the segmentation output. Default False.
    """

    def __init__(self, config, logger=None, provenance=None):
        super().__init__(config, logger, provenance)
        ss_cfg = config.get("anat", {}).get("super_synth", {})
        self.enabled = ss_cfg.get("enabled", False)
        self.mode = ss_cfg.get("mode", "invivo")
        self.sharpen_synths = ss_cfg.get("sharpen_synths", False)
        self.device: Optional[str] = ss_cfg.get("device", None)
        self.compute_volumes: bool = bool(ss_cfg.get("compute_volumes", False))

    # ------------------------------------------------------------------
    # BaseProcessingStep interface
    # ------------------------------------------------------------------

    def run(self, first_arg: Any, output_dir: Path, **kwargs) -> Any:
        if not self.enabled and not kwargs.get("force", False):
            self.logger.info("SuperSynthStep disabled — skipping.")
            return first_arg

        context, input_image = self.unpack_input(first_arg)

        # Resolve input image — fall back through context keys when
        # current_image is not set (e.g. called right after preprocessing).
        if input_image is None and context is not None:
            for key in ("preprocessed_t1w", "t1w", "current_t1w"):
                candidate = context.get(key)
                if candidate is not None:
                    input_image = candidate
                    break

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

        # Build a subject/session-specific output subdirectory.
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

        # Collect recognised output files.
        outputs = find_supersynth_outputs(step_dir)

        if not outputs:
            self.logger.warning(
                f"mri_super_synth finished but no recognised output files were "
                f"found in {step_dir}. Check the FreeSurfer output naming for "
                f"your build version."
            )
        else:
            self.logger.info(f"mri_super_synth outputs: {list(outputs)}")

        # ----------------------------------------------------------------
        # Optional: compute and save anatomical volumes from segmentation
        # ----------------------------------------------------------------
        compute_vols = kwargs.get("compute_volumes", self.compute_volumes)
        volumes_csv: Optional[Path] = None
        volumes_dict: Dict[str, float] = {}

        if compute_vols and "seg" in outputs:
            try:
                vol_data = _compute_volumes(outputs["seg"])

                prefix = f"sub-{sub}" if sub else "subject"
                if ses:
                    prefix += f"_ses-{ses}"
                volumes_csv = step_dir / f"{prefix}_desc-supersynth_volumes.csv"

                rows = [
                    {
                        "label_id": lid,
                        "label_name": name,
                        "n_voxels": n_vox,
                        "volume_mm3": round(vol_mm3, 4),
                    }
                    for lid, (name, vol_mm3, n_vox) in sorted(vol_data.items())
                ]
                pd.DataFrame(rows).to_csv(volumes_csv, index=False)

                volumes_dict = {name: vol_mm3 for _, (name, vol_mm3, _) in vol_data.items()}
                self.logger.info(
                    f"Saved SuperSynth volumes ({len(rows)} regions) → {volumes_csv.name}"
                )
            except Exception as e:
                self.logger.warning(f"SuperSynth volume computation failed: {e}")

        # ----------------------------------------------------------------
        # Update context
        # ----------------------------------------------------------------
        if context is not None:
            context["super_synth_dir"] = step_dir
            context["super_synth_outputs"] = outputs

            if volumes_dict:
                context["super_synth_volumes"] = volumes_dict
            if volumes_csv and volumes_csv.exists():
                context["super_synth_volumes_csv"] = volumes_csv

            # If a synthetic T1w was produced and no preprocessed T1w is set
            # yet, inject it so downstream steps can use it.
            if "synth_t1w" in outputs and "preprocessed_t1w" not in context:
                entities = {
                    "sub": sub, "ses": ses,
                    "suffix": "T1w", "desc": "synthT1w",
                }
                context["preprocessed_t1w"] = ImageFile(
                    img=outputs["synth_t1w"],
                    entities={k: v for k, v in entities.items() if v},
                )

            return context

        # Standalone (no context) — return the output directory.
        return step_dir


def ensure_supersynth_t1w(
    context: dict,
    output_dir: Path,
    config,
    logger: logging.Logger,
    *,
    input_preference: str = "auto",
    mode: Optional[str] = None,
    device: Optional[str] = None,
    sharpen_synths: Optional[bool] = None,
    force: bool = False,
    subdir: str = "supersynth",
) -> Optional[ImageFile]:
    """
    Generate or reuse a SuperSynth synthetic T1w image for downstream steps.

    ``input_preference`` selects the anatomical source used by SuperSynth:
    ``"T1w"``, ``"T2w"``, or ``"auto"`` (T1w first, then T2w).
    """
    outputs = context.get("super_synth_outputs", {}) or {}
    existing = outputs.get("synth_t1w")
    entities = {
        k: v for k, v in {
            "sub": context.get("subject"),
            "ses": context.get("session"),
            "desc": "synthT1w",
            "suffix": "T1w",
        }.items() if v
    }

    if existing and Path(existing).exists() and not force:
        logger.info(f"Using existing SuperSynth T1w: {existing}")
        return ImageFile(entities=entities, img=Path(existing), json=None)

    t1w_files = [
        img for img in [context.get("preprocessed_t1w"), *(context.get("t1w_files") or [])]
        if img is not None
    ]
    t2w_files = [
        img for img in [
            context.get("preprocessed_t2w_coreg"),
            context.get("preprocessed_t2w"),
            *(context.get("t2w_files") or []),
        ]
        if img is not None
    ]
    preference = str(input_preference or "auto").lower()
    anat_input = None
    if preference == "t2w":
        anat_input = t2w_files[0] if t2w_files else None
    elif preference == "t1w":
        anat_input = t1w_files[0] if t1w_files else None
    else:
        anat_input = t1w_files[0] if t1w_files else (t2w_files[0] if t2w_files else None)

    if anat_input is None:
        logger.warning("SuperSynth T1w requested but no anatomical input was available.")
        return None

    ss_dir = Path(output_dir) / subdir
    ss_dir.mkdir(parents=True, exist_ok=True)
    current_outputs = find_supersynth_outputs(ss_dir)
    synth_path = current_outputs.get("synth_t1w", expected_supersynth_output(ss_dir, "synth_t1w"))

    if not synth_path.exists() or force:
        logger.info(f"Generating SuperSynth T1w from {Path(anat_input.img).name}")
        freesurfer.mri_super_synth(
            in_file=anat_input.img,
            out_dir=ss_dir,
            mode=mode or config.get("anat.super_synth.mode", "invivo"),
            threads=getattr(config, "n_cpus", -1),
            device=device if device is not None else config.get("anat.super_synth.device"),
            sharpen_synths=(
                bool(sharpen_synths)
                if sharpen_synths is not None
                else bool(config.get("anat.super_synth.sharpen_synths", False))
            ),
            overwrite=force,
        )
        current_outputs = find_supersynth_outputs(ss_dir)
        synth_path = current_outputs.get("synth_t1w", synth_path)

    if not synth_path.exists():
        found = sorted(path.name for path in ss_dir.glob("*"))
        logger.warning(
            f"SuperSynth did not produce a recognized T1w output in {ss_dir}. "
            f"Found: {found}"
        )
        return None

    context.setdefault("super_synth_outputs", {}).update(current_outputs or {"synth_t1w": synth_path})
    context["super_synth_dir"] = ss_dir
    return ImageFile(entities=entities, img=synth_path, json=None)


def ensure_supersynth_outputs_for_image(
    input_image: ImageLike,
    output_dir: Path,
    config,
    logger: logging.Logger,
    *,
    mode: Optional[str] = None,
    device: Optional[str] = None,
    sharpen_synths: Optional[bool] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """Run SuperSynth for a specific image and return recognized outputs."""
    in_path = Path(input_image.img if hasattr(input_image, "img") else input_image)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = find_supersynth_outputs(out_dir)
    if force or "synth_t1w" not in outputs or "synth_t2w" not in outputs:
        logger.info(f"Generating SuperSynth contrast set from {in_path.name}")
        freesurfer.mri_super_synth(
            in_file=in_path,
            out_dir=out_dir,
            mode=mode or config.get("anat.super_synth.mode", "invivo"),
            threads=getattr(config, "n_cpus", -1),
            device=device if device is not None else config.get("anat.super_synth.device"),
            sharpen_synths=(
                bool(sharpen_synths)
                if sharpen_synths is not None
                else bool(config.get("anat.super_synth.sharpen_synths", False))
            ),
            overwrite=force,
        )
        outputs = find_supersynth_outputs(out_dir)

    return outputs
