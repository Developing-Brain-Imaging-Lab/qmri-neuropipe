"""
Synb0 Distortion Correction module (Deep Learning).

Uses dipy.nn.tf.synb0 to estimate an undistorted (or reverse PE) b0 signal
from a distorted b0 and a T1w image.
"""

from pathlib import Path
from typing import Optional, Literal, Tuple
from dataclasses import dataclass
import numpy as np
import nibabel as nib
import logging
import json
import gzip

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageFile, DWIFile
from ...interfaces import freesurfer, fsl, c3d, ants
from ..common.mask import mask_brain    
import multiprocessing
import sys
import traceback


@dataclass(frozen=True)
class _LinearTransform:
    """An ANTs-compatible linear transform and its application direction."""

    path: Path
    invert: bool = False


def _run_synb0_worker(in_file, t1_file, out_file, gpu_ids=None, device="cpu"):
    """
    Worker function to run Synb0 estimation in a separate process.
    This ensures GPU memory is released when the process terminates.
    """
    try:
        import os

        device = str(device or "cpu").lower()
        if device == "cpu":
             os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
             print("Synb0 Worker: Running TensorFlow on CPU (CUDA_VISIBLE_DEVICES=-1)")
        elif gpu_ids is not None:
             gpus = gpu_ids
             if isinstance(gpus, int):
                 gpus = [gpus]
             os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
             print(f"Synb0 Worker: Setting CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # Import inside the worker process to avoid initializing TF in main process
        from ...interfaces import dipy
        
        print(f"Synb0 Worker: Running estimation...")
        dipy.synb0_estimation(
            in_file=in_file,
            t1_file=t1_file,
            out_file=out_file
        )
        print("Synb0 Worker: Finished successfully.")
        
    except Exception:
        print("Synb0 Worker failed with error:")
        traceback.print_exc()
        sys.exit(1)

def _validate_nifti(path: Path, logger: logging.Logger, label: str) -> None:
    if not path.exists():
        raise ProcessingError(f"{label} does not exist: {path}")
    if path.stat().st_size == 0:
        raise ProcessingError(f"{label} is empty (0 bytes): {path}")
    if path.suffix == ".gz":
        try:
            with gzip.open(path, "rb") as f:
                _ = f.read(2)
        except Exception as e:
            raise ProcessingError(f"{label} is not valid gzip: {path} ({e})")

class Synb0EstimationStep(BaseProcessingStep):
    """
    Synb0 estimation step.
    
    Generates a synthetic b0 image (representing the reverse phase encoding direction)
    using a Deep Learning model (DIPY Synb0).
    
    This synthetic b0 is then paired with the real b0 to form a Topup config,
    allowing Topup to estimate the susceptibility field.
    
    Attributes:
        method: 'dipy-dl'
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
    ):
        super().__init__(config, logger, provenance)
        self.method = 'dipy-dl'
        self.synb0_cfg = config.get("dmri.preprocessing.distcorr.synb0", {}) or {}
        self.logger.info(f"Initialized Synb0 estimation (Deep Learning).")

    @staticmethod
    def _is_dwi_supersynth_source(source: str, preference: str) -> bool:
        return (
            source in {"dwi_supersynth", "supersynth_dwi", "diffusion_supersynth", "b0_supersynth"}
            or (source in {"supersynth", "prefer_supersynth"} and preference in {"dwi", "b0", "diffusion", "mean_b0"})
        )

    def _bias_correct_t1(self, t1w_mgz: Path, output_dir: Path) -> Path:
        """Bias-correct the Synb0 T1, using ANTs when FreeSurfer lacks MINC."""
        t1w_n3 = output_dir / "t1w_n3.mgz"
        try:
            freesurfer.mri_nu_correct(in_file=t1w_mgz, out_file=t1w_n3)
        except Exception as exc:
            self.logger.warning(
                "FreeSurfer mri_nu_correct failed during Synb0 T1 preprocessing "
                "(%s). Falling back to ANTs N4 bias correction.",
                exc,
            )
            if t1w_n3.exists():
                t1w_n3.unlink()
            t1w_n4_input = output_dir / "t1w_n4_input.nii.gz"
            t1w_n4 = output_dir / "t1w_n4.nii.gz"
            freesurfer.mri_convert(in_file=t1w_mgz, out_file=t1w_n4_input)
            ants.n4bias(
                in_file=t1w_n4_input,
                out_file=t1w_n4,
                nthreads=int(getattr(self.config, "n_cpus", 1) or 1),
            )
            freesurfer.mri_convert(in_file=t1w_n4, out_file=t1w_n3)
        return t1w_n3

    def _normalize_t1(self, t1w_bias_corrected: Path, output_dir: Path) -> Path:
        """Normalize T1 intensities when FreeSurfer can identify control points."""
        t1w_norm = output_dir / "t1w_norm.mgz"
        try:
            freesurfer.mri_normalize(
                in_file=t1w_bias_corrected,
                out_file=t1w_norm,
            )
        except Exception as exc:
            self.logger.warning(
                "FreeSurfer mri_normalize failed during Synb0 T1 preprocessing "
                "(%s). Continuing with the bias-corrected T1.",
                exc,
            )
            if t1w_norm.exists():
                t1w_norm.unlink()
            freesurfer.mri_convert(
                in_file=t1w_bias_corrected,
                out_file=t1w_norm,
            )
        return t1w_norm

    def _register_t1w_to_dwi(
        self,
        t1w_registration_path: Path,
        registration_ref_path: Path,
        output_dir: Path,
        *,
        moving_apply_path: Optional[Path] = None,
    ) -> Tuple[_LinearTransform, _LinearTransform]:
        """Estimate rigid T1w↔DWI transforms using the selected fixed contrast."""
        # SuperSynth commonly writes SynthT1.mgz. Normalize both sides of the
        # registration boundary to NIfTI so FSL and ANTs see identical inputs.
        if t1w_registration_path.suffix.lower() == ".mgz":
            registration_moving_nii = output_dir / "registration_moving.nii.gz"
            freesurfer.mri_convert(
                in_file=t1w_registration_path,
                out_file=registration_moving_nii,
            )
            t1w_registration_path = registration_moving_nii

        if registration_ref_path.suffix.lower() == ".mgz":
            registration_ref_nii = output_dir / "registration_ref.nii.gz"
            freesurfer.mri_convert(
                in_file=registration_ref_path,
                out_file=registration_ref_nii,
            )
            registration_ref_path = registration_ref_nii

        moving_registration_path = t1w_registration_path
        fixed_registration_path = registration_ref_path
        if self._skull_strip_registration_enabled():
            moving_registration_path = self._skull_strip_registration_image(
                t1w_registration_path,
                output_dir / "registration_masks" / "t1w_moving",
            )
            fixed_registration_path = self._skull_strip_registration_image(
                registration_ref_path,
                output_dir / "registration_masks" / "dwi_fixed",
            )

        return self._estimate_linear_registration(
            moving_registration_path=moving_registration_path,
            fixed_registration_path=fixed_registration_path,
            moving_apply_path=moving_apply_path or t1w_registration_path,
            fixed_apply_path=registration_ref_path,
            output_dir=output_dir,
            stem="t1w_2_dwi",
            transform_type="Rigid",
            dof=6,
            output_image=output_dir / "t1w_brain_reg.nii.gz",
            artifact_tag=(
                "supersynth" if self._uses_supersynth_registration() else None
            ),
        )

    def _registration_backend(self) -> str:
        backend = str(
            self.synb0_cfg.get(
                "registration_backend",
                self.synb0_cfg.get("registration_tool", "fsl"),
            )
        ).strip().lower()
        if backend not in {"fsl", "ants"}:
            raise ValidationError(
                "synb0.registration_backend must be either 'fsl' or 'ants'."
            )
        return backend

    def _skull_strip_registration_enabled(self) -> bool:
        return bool(
            self.synb0_cfg.get(
                "skull_strip_registration",
                self.synb0_cfg.get("skull_strip_registration_inputs", False),
            )
        )

    def _skull_strip_registration_image(
        self,
        image_path: Path,
        output_dir: Path,
    ) -> Path:
        """Create a brain-only proxy used solely to estimate a transform."""
        method = str(
            self.synb0_cfg.get(
                "skull_strip_method",
                self.synb0_cfg.get(
                    "registration_skull_strip_method",
                    "synthstrip",
                ),
            )
        ).lower()
        brain = mask_brain(
            input_image=image_path,
            output_dir=output_dir,
            method=method,
            nthreads=int(getattr(self.config, "n_cpus", 1) or 1),
            use_gpu=bool(getattr(self.config, "use_gpu", False)),
        )
        return Path(brain.img)

    def _estimate_linear_registration(
        self,
        *,
        moving_registration_path: Path,
        fixed_registration_path: Path,
        moving_apply_path: Path,
        fixed_apply_path: Path,
        output_dir: Path,
        stem: str,
        transform_type: Literal["Rigid", "Affine"],
        dof: int,
        output_image: Path,
        artifact_tag: Optional[str] = None,
    ) -> Tuple[_LinearTransform, _LinearTransform]:
        """
        Estimate on registration proxies and resample the original moving image.

        The returned transforms are always directly consumable by
        ``ants.apply_transforms``. For ANTs linear transforms, the inverse is
        represented by the same affine matrix with ``invert=True``.
        """
        backend = self._registration_backend()
        nthreads = int(getattr(self.config, "n_cpus", 1) or 1)
        suffix_parts = [
            part
            for part in (
                artifact_tag,
                "brain" if self._skull_strip_registration_enabled() else None,
            )
            if part
        ]
        artifact_suffix = (
            f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
        )
        proxy_output = (
            output_dir
            / f"{stem}{artifact_suffix}_{backend}_registration_proxy.nii.gz"
        )

        self.logger.info(
            "Estimating %s registration with %s%s",
            stem,
            backend.upper(),
            " on skull-stripped proxies"
            if self._skull_strip_registration_enabled()
            else "",
        )

        if backend == "ants":
            prefix = output_dir / f"{stem}{artifact_suffix}_ants_"
            _, transforms = ants.registration(
                fixed_file=fixed_registration_path,
                moving_file=moving_registration_path,
                out_prefix=prefix,
                transform_type=transform_type,
                interpolator="linear",
                nthreads=nthreads,
            )
            transforms = [Path(transform) for transform in transforms]
            if len(transforms) != 1:
                raise ProcessingError(
                    f"ANTs {transform_type} registration for {stem} returned "
                    f"{len(transforms)} transforms; expected one affine transform."
                )
            forward = _LinearTransform(transforms[0], False)
            inverse = _LinearTransform(transforms[0], True)
        else:
            forward_fsl = output_dir / f"{stem}{artifact_suffix}.mat"
            _, forward_fsl = fsl.flirt(
                in_file=moving_registration_path,
                ref_file=fixed_registration_path,
                out_file=proxy_output,
                omat=forward_fsl,
                cost="normmi",
                dof=dof,
                extra_args=(
                    "-searchcost normmi -searchrx -180 180 "
                    "-searchry -180 180 -searchrz -180 180"
                ),
            )
            forward_ants = output_dir / f"{stem}{artifact_suffix}.txt"
            c3d.fsl2ants(
                ref_file=fixed_registration_path,
                in_file=moving_registration_path,
                transform_file=forward_fsl,
                out_file=forward_ants,
            )

            moving_name, fixed_name = stem.split("_2_", 1)
            inverse_stem = f"{fixed_name}_2_{moving_name}"
            inverse_fsl = output_dir / f"{inverse_stem}{artifact_suffix}.mat"
            fsl.convert_xfm(
                in_file=forward_fsl,
                out_file=inverse_fsl,
                inverse=True,
            )
            inverse_ants = output_dir / f"{inverse_stem}{artifact_suffix}.txt"
            c3d.fsl2ants(
                ref_file=moving_registration_path,
                in_file=fixed_registration_path,
                transform_file=inverse_fsl,
                out_file=inverse_ants,
            )
            forward = _LinearTransform(forward_ants, False)
            inverse = _LinearTransform(inverse_ants, False)

        ants.apply_transforms(
            fixed_file=fixed_apply_path,
            moving_file=moving_apply_path,
            out_file=output_image,
            transforms=[forward.path],
            invert_transforms=[forward.invert],
            interpolator="linear",
            nthreads=nthreads,
        )
        return forward, inverse

    def _register_t1w_to_mni(
        self,
        t1w_norm_path: Path,
        t1w_brain_path: Path,
        mni_atlas_path: Path,
        output_dir: Path,
    ) -> Tuple[_LinearTransform, _LinearTransform]:
        """Estimate the affine T1w↔MNI transform with the selected backend."""
        moving_registration_path = t1w_norm_path
        fixed_registration_path = mni_atlas_path
        if self._skull_strip_registration_enabled():
            moving_registration_path = t1w_brain_path
            fixed_registration_path = self._skull_strip_registration_image(
                mni_atlas_path,
                output_dir / "registration_masks" / "mni_fixed",
            )

        return self._estimate_linear_registration(
            moving_registration_path=moving_registration_path,
            fixed_registration_path=fixed_registration_path,
            moving_apply_path=t1w_norm_path,
            fixed_apply_path=mni_atlas_path,
            output_dir=output_dir,
            stem="t1w_2_mni",
            transform_type="Affine",
            dof=12,
            output_image=output_dir / "t1w_mni.nii.gz",
        )

    def _extract_mean_b0(self, input_dwi: DWIFile, b0_path: Path, force: bool = False, as_4d: bool = True) -> Path:
        if b0_path.exists() and not force:
            return b0_path

        _validate_nifti(input_dwi.img, self.logger, "Input DWI")
        try:
            img = nib.load(str(input_dwi.img))
        except Exception as e:
            size = input_dwi.img.stat().st_size
            raise ProcessingError(
                f"Failed to read input DWI NIfTI: {input_dwi.img} (size={size} bytes). "
                f"Original error: {e}"
            )

        data = img.get_fdata()
        b0_vol = data[..., 0]
        if input_dwi.bval and input_dwi.bval.exists():
            try:
                bvals = np.loadtxt(str(input_dwi.bval))
                if bvals.ndim == 2:
                    bvals = bvals[0]
                b0_indices = np.where(bvals < 50)[0]
                if len(b0_indices) > 0:
                    b0_vol = data[..., b0_indices]
            except Exception as e:
                self.logger.warning(f"Could not parse bvals for Synb0 b0 extraction; using first volume: {e}")

        if b0_vol.ndim == 4:
            b0_vol = b0_vol.mean(axis=-1)
        if as_4d and b0_vol.ndim == 3:
            b0_vol = b0_vol[..., np.newaxis]

        nib.Nifti1Image(b0_vol, img.affine, img.header).to_filename(b0_path)
        return b0_path

    def _prepare_anatomical_series(
        self,
        images: list[ImageFile],
        output_dir: Path,
        *,
        force: bool = False,
    ) -> ImageFile:
        """Collapse a VFA-like or 4-D anatomical series to one 3-D image."""
        first = images[0]
        suffix = str(first.entities.get("suffix", "")).lower()
        vfa_like = suffix in {"vfa", "spgr", "ssfp", "flash"}
        selected = images if vfa_like else [first]

        first_img = nib.load(str(first.img))
        if len(selected) == 1 and first_img.ndim == 3:
            return first

        mode = str(
            self.synb0_cfg.get("anatomical_series_mode", "mean")
        ).lower()
        if mode not in {"mean", "representative", "first"}:
            raise ProcessingError(
                "synb0.anatomical_series_mode must be 'mean' or 'representative'."
            )

        prepared_path = output_dir / (
            f"anatomical_desc-{'mean' if mode == 'mean' else 'representative'}"
            "_supersynthInput.nii.gz"
        )
        if prepared_path.exists() and not force:
            return ImageFile(
                entities={**first.entities, "desc": f"{mode}SupersynthInput"},
                img=prepared_path,
                json=None,
            )

        volumes: list[np.ndarray] = []
        reference_img = None
        for image in selected:
            loaded = nib.load(str(image.img))
            data = loaded.get_fdata()
            image_volumes = (
                [data[..., index] for index in range(data.shape[-1])]
                if data.ndim == 4
                else [data]
            )
            if reference_img is None:
                reference_img = loaded
            elif (
                loaded.shape[:3] != reference_img.shape[:3]
                or not np.allclose(loaded.affine, reference_img.affine)
            ):
                raise ProcessingError(
                    "VFA/SPGR/SSFP anatomical images must share geometry "
                    "before they can be averaged for SuperSynth."
                )
            volumes.extend(image_volumes)

        if mode == "mean":
            prepared_data = np.mean(np.stack(volumes, axis=-1), axis=-1)
        else:
            index = int(self.synb0_cfg.get("anatomical_series_index", 0))
            if index < 0 or index >= len(volumes):
                raise ProcessingError(
                    f"anatomical_series_index={index} is outside the available "
                    f"range 0..{len(volumes) - 1}."
                )
            prepared_data = volumes[index]

        assert reference_img is not None
        nib.Nifti1Image(
            prepared_data,
            reference_img.affine,
            reference_img.header,
        ).to_filename(prepared_path)
        return ImageFile(
            entities={**first.entities, "desc": f"{mode}SupersynthInput"},
            img=prepared_path,
            json=None,
        )

    def _prepare_anatomical_t1w(
        self,
        context: dict,
        output_dir: Path,
        *,
        force: bool = False,
    ) -> Optional[ImageFile]:
        """Return an acquired T1w or synthesize one from another anatomy scan."""
        t1w_files = context.get("t1w_files", [])
        if t1w_files:
            context["synb0_t1w_source"] = "acquired_t1w"
            return t1w_files[0]

        candidates = list(context.get("anatomical_files") or [])
        if not candidates:
            candidates = list(context.get("t2w_files") or [])
        if not candidates:
            return None

        preference = str(self.synb0_cfg.get("anatomical_input", "auto")).lower()
        if preference != "auto":
            matches = [
                image
                for image in candidates
                if str(image.entities.get("suffix", "")).lower() == preference
            ]
            if not matches:
                raise ProcessingError(
                    f"Synb0 anatomical_input={preference!r} did not match an "
                    "available anatomical scan."
                )
        else:
            first_suffix = str(candidates[0].entities.get("suffix", "")).lower()
            matches = [
                image
                for image in candidates
                if str(image.entities.get("suffix", "")).lower() == first_suffix
            ]

        anatomical = self._prepare_anatomical_series(
            matches,
            output_dir,
            force=force,
        )

        ss_dir = output_dir / "supersynth_from_anatomical"
        from ..anat.super_synth import expected_supersynth_output, find_supersynth_outputs

        ss_outputs = find_supersynth_outputs(ss_dir)
        synth_path = ss_outputs.get(
            "synth_t1w", expected_supersynth_output(ss_dir, "synth_t1w")
        )
        if not synth_path.exists() or force:
            self.logger.info(
                "Generating the Synb0 T1w input with SuperSynth from %s",
                anatomical.img.name,
            )
            freesurfer.mri_super_synth(
                in_file=anatomical.img,
                out_dir=ss_dir,
                mode=self.synb0_cfg.get(
                    "supersynth_mode",
                    self.config.get("anat.super_synth.mode", "invivo"),
                ),
                threads=getattr(self.config, "n_cpus", -1),
                device=self.synb0_cfg.get(
                    "supersynth_device",
                    self.config.get("anat.super_synth.device"),
                ),
                sharpen_synths=bool(
                    self.synb0_cfg.get(
                        "supersynth_sharpen_synths",
                        self.config.get("anat.super_synth.sharpen_synths", False),
                    )
                ),
                overwrite=force,
            )
            ss_outputs = find_supersynth_outputs(ss_dir)
            synth_path = ss_outputs.get("synth_t1w", synth_path)

        _validate_nifti(synth_path, self.logger, "Anatomical SuperSynth T1w")
        context["synb0_anatomical_input"] = anatomical
        context["synb0_t1w_source"] = "supersynth_anatomical"
        return ImageFile(
            entities={
                **anatomical.entities,
                "desc": "synthT1w",
                "suffix": "T1w",
            },
            img=synth_path,
            json=None,
        )

    def _prepare_supersynth_registration_reference(
        self,
        context: dict,
        output_dir: Path,
        *,
        input_dwi: Optional[DWIFile] = None,
        b0_path: Optional[Path] = None,
        force: bool = False,
    ) -> Optional[Path]:
        """
        Generate a DWI-derived T1-like image used only for T1w-to-DWI registration.

        The returned image must never replace the supplied T1w as the anatomical
        input to the Synb0 model.
        """
        enabled = self._uses_supersynth_registration()
        if not enabled:
            return None

        supersynth_b0_path = output_dir / "real_b0_desc-supersynthInput.nii.gz"
        if input_dwi is not None:
            supersynth_b0_path = self._extract_mean_b0(
                input_dwi,
                supersynth_b0_path,
                force=force,
                as_4d=False,
            )
        elif b0_path is not None and Path(b0_path).exists():
            supersynth_b0_path = Path(b0_path)
        else:
            raise ProcessingError(
                "SuperSynth-assisted Synb0 registration requires an extracted DWI b0."
            )

        synth_path = self._run_supersynth_registration_proxy(
            supersynth_b0_path,
            output_dir / "supersynth_from_dwi",
            force=force,
            label="DWI-derived",
        )
        context["synb0_registration_reference"] = synth_path
        context["synb0_registration_method"] = "supersynth"
        return synth_path

    def _uses_supersynth_registration(self) -> bool:
        """Return whether contrast-matched SuperSynth registration is enabled."""
        registration_method = str(
            self.synb0_cfg.get(
                "registration",
                self.synb0_cfg.get("registration_method", "direct"),
            )
        ).lower()
        source = str(self.synb0_cfg.get("t1w_source", "raw")).lower()
        preference = str(self.synb0_cfg.get("supersynth_input", "auto")).lower()
        return (
            registration_method in {"supersynth", "super_synth"}
            or bool(self.synb0_cfg.get("use_supersynth_registration", False))
            # Backward compatibility for configurations written before
            # SuperSynth was restricted to registration assistance.
            or source in {"supersynth", "prefer_supersynth"}
            or self._is_dwi_supersynth_source(source, preference)
        )

    def _run_supersynth_registration_proxy(
        self,
        input_path: Path,
        ss_dir: Path,
        *,
        force: bool,
        label: str,
    ) -> Path:
        """Generate one T1-like image used only for transform estimation."""
        from ..anat.super_synth import expected_supersynth_output, find_supersynth_outputs

        ss_outputs = find_supersynth_outputs(ss_dir)
        synth_path = ss_outputs.get(
            "synth_t1w", expected_supersynth_output(ss_dir, "synth_t1w")
        )
        if not synth_path.exists() or force:
            self.logger.info(
                "Generating a %s SuperSynth T1w registration proxy",
                label,
            )
            freesurfer.mri_super_synth(
                in_file=input_path,
                out_dir=ss_dir,
                mode=self.synb0_cfg.get(
                    "supersynth_mode",
                    self.config.get("anat.super_synth.mode", "invivo"),
                ),
                threads=getattr(self.config, "n_cpus", -1),
                device=self.synb0_cfg.get(
                    "supersynth_device",
                    self.config.get("anat.super_synth.device"),
                ),
                sharpen_synths=bool(
                    self.synb0_cfg.get(
                        "supersynth_sharpen_synths",
                        self.config.get("anat.super_synth.sharpen_synths", False),
                    )
                ),
                overwrite=force,
            )
            ss_outputs = find_supersynth_outputs(ss_dir)
            synth_path = ss_outputs.get("synth_t1w", synth_path)

        _validate_nifti(synth_path, self.logger, f"{label} SuperSynth T1w")
        return synth_path

    def _prepare_supersynth_registration_moving(
        self,
        context: dict,
        output_dir: Path,
        *,
        t1w_path: Path,
        force: bool = False,
    ) -> Optional[Path]:
        """Create a T1-like moving proxy to match the fixed SuperSynth contrast."""
        if not self._uses_supersynth_registration():
            return None

        synth_path = self._run_supersynth_registration_proxy(
            t1w_path,
            output_dir / "supersynth_from_t1w_registration",
            force=force,
            label="anatomical-derived",
        )
        context["synb0_registration_moving"] = synth_path
        return synth_path

    def validate_inputs(self, first_arg, output_dir: Path, **kwargs) -> None:
        context, _ = self.unpack_input(first_arg)
        if context is None:
             raise ValidationError("Synb0EstimationStep requires pipeline context.")
        self._registration_backend()
        
        has_anatomical = bool(
            context.get("t1w_files")
            or context.get("anatomical_files")
            or context.get("t2w_files")
        )
        if not has_anatomical:
            raise ValidationError(
                "Synb0 estimation requires an undistorted anatomical image. "
                "A non-T1w anatomical scan will be converted to T1w with SuperSynth."
            )
        
        dwi_files = context.get("dwi_files", [])
        if not dwi_files:
            raise ValidationError("No DWI files found for Synb0 estimation.")

    def run(self, first_arg, output_dir: Path, **kwargs) -> dict:
        """
        Run Synb0 estimation.
        """
        context, _ = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("Synb0EstimationStep must run in pipeline context mode.")
        dwi_files: list[DWIFile] = context.get("dwi_files", [])
        output_dir = self.get_step_output_dir(output_dir)
        force = bool(kwargs.get("force", False))
        
        # We need to generate a synthetic b0 for each distinct acquisition group? 
        # Or usually just one per session if they share geometry?
        # Let's assume we do it for the first DWI file and assume others share distortion if valid.
        # Ideally, we should check for different PE directions.
        
        # For simplicity, pick the first valid DWI as the "forward" b0 source.
        input_dwi = None
        for candidate in dwi_files:
            try:
                _validate_nifti(candidate.img, self.logger, "Input DWI")
                _ = nib.load(str(candidate.img))
            except Exception as e:
                self.logger.warning(f"Skipping invalid DWI for Synb0: {candidate.img} ({e})")
                continue
            input_dwi = candidate
            break

        if input_dwi is None:
            raise ProcessingError("No valid DWI files found for Synb0 estimation.")

        real_json = input_dwi.json
        
        should_skip = False
        # 1. Check if outputs exist
        syn_b0_path = output_dir / "syn_b0_desc-synthetic.nii.gz"
        syn_b0_native_path = output_dir / "syn_b0_native.nii.gz"
        syn_json_path = syn_b0_path.with_suffix(".json")
        b0_path = output_dir / "real_b0.nii.gz"
        dummy_bval_path = output_dir / "b0.bval"

        self._extract_mean_b0(input_dwi, b0_path, force=force)

        t1w_ref = self._prepare_anatomical_t1w(
            context,
            output_dir,
            force=force,
        )
        if t1w_ref is None:
             raise ProcessingError("Synb0 estimation requires an anatomical image.")

        t1w_path = t1w_ref.img
        registration_ref_path = self._prepare_supersynth_registration_reference(
            context,
            output_dir,
            input_dwi=input_dwi,
            b0_path=b0_path,
            force=force,
        ) or b0_path
        
        if syn_b0_path.exists() and b0_path.exists() and syn_b0_native_path.exists() and syn_json_path.exists() and dummy_bval_path.exists() and not kwargs.get('force', False):
            # Check timestamps
            out_mtime = syn_b0_path.stat().st_mtime
            t1_mtime = t1w_path.stat().st_mtime
            dwi_mtime = input_dwi.img.stat().st_mtime
            registration_ref_mtime = registration_ref_path.stat().st_mtime
            missing_matched_moving = False
            if self._uses_supersynth_registration():
                from ..anat.super_synth import find_supersynth_outputs

                moving_outputs = find_supersynth_outputs(
                    output_dir / "supersynth_from_t1w_registration"
                )
                missing_matched_moving = "synth_t1w" not in moving_outputs
            
            if (
                t1_mtime > out_mtime
                or dwi_mtime > out_mtime
                or registration_ref_mtime > out_mtime
                or missing_matched_moving
            ):
                 self.logger.info(
                     "Synb0 inputs or registration proxies changed. Re-running."
                 )
            else:
                 self.logger.info(f"Skipping Synb0 estimation (outputs exist and are up-to-date): {syn_b0_path}")
                 should_skip = True
        
        if not should_skip:
            #Preprocess T1w
            #Normalize T1w
            t1w_mgz = output_dir / "t1w.mgz"
            freesurfer.mri_convert(in_file=t1w_path, out_file=t1w_mgz)
    
            t1w_n3 = self._bias_correct_t1(t1w_mgz, output_dir)
    
            t1w_norm = self._normalize_t1(t1w_n3, output_dir)
    
            t1w_norm_nii = output_dir / "t1w_norm.nii.gz"
            freesurfer.mri_convert(in_file=t1w_norm, out_file=t1w_norm_nii)
    
            #Skull-strip T1w
            t1w_brain, t1w_mask = mask_brain(
                input_image=t1w_norm_nii,
                output_dir=output_dir,
                method="synthstrip",
                return_mask=True,
                use_gpu=getattr(self.config, 'use_gpu', False)
            )
            
            # Get paths
            t1w_brain_path = t1w_brain.img
            # t1w_mask_path = t1w_mask.img

            # With SuperSynth-assisted registration, synthesize the same T1-like
            # contrast from both sides of the registration pair. These images
            # are used only to estimate the transform.
            registration_moving_path = (
                self._prepare_supersynth_registration_moving(
                    context,
                    output_dir,
                    t1w_path=t1w_norm_nii,
                    force=force,
                )
                or (
                    t1w_norm_nii
                    if self._skull_strip_registration_enabled()
                    else t1w_brain_path
                )
            )
    
            # Register the anatomical image to DWI space using direct images or
            # matched SuperSynth proxies. The resulting transform is applied to
            # the original normalized T1w, never to the inference input itself.
            t1w_2_dwi, dwi_2_t1w = self._register_t1w_to_dwi(
                registration_moving_path,
                registration_ref_path,
                output_dir,
                moving_apply_path=t1w_norm_nii,
            )
    
            #REGISTER T1 to Atlas
            mni_atlas_img = Path(__file__).parent / "data" / "mni_icbm152_t1_tal_nlin_asym_09c_2_5.nii.gz"
            t1w_2_mni, mni_2_t1w = self._register_t1w_to_mni(
                t1w_norm_nii,
                t1w_brain_path,
                mni_atlas_img,
                output_dir,
            )
    
           
            #Apply linear registration to normalized T1w to get into Atlas space
            t1w_norm_atlas = output_dir / "t1w_norm_mni.nii.gz"
            b0_in_mni = output_dir / "b0_in_mni.nii.gz"
    
            ants.apply_transforms(fixed_file=mni_atlas_img,
                                  moving_file=t1w_norm_nii,
                                  out_file=t1w_norm_atlas,
                                  transforms=[t1w_2_mni.path],
                                  invert_transforms=[t1w_2_mni.invert],
                                  interpolator="bSpline")
                                  
            #Apply series of registrations to dwi
            # Chain: DWI -> T1w -> MNI
            ants.apply_transforms(fixed_file=mni_atlas_img,
                                  moving_file=b0_path,
                                  out_file=b0_in_mni,
                                  transforms=[t1w_2_mni.path, dwi_2_t1w.path],
                                  invert_transforms=[t1w_2_mni.invert, dwi_2_t1w.invert],
                                  interpolator="bSpline")   
                                  
    
            # 2. Run Synb0 Estimation (Real b0 + T1w -> Synthetic Reverse b0)
            try:
                # Prepare arguments
                synb0_device = str(self.synb0_cfg.get("device", "cpu")).lower()
                gpu_ids = self.synb0_cfg.get("gpu_ids", self.config.gpu_ids)
                
                self.logger.info(f"Launching Synb0 estimation in a separate process to manage GPU memory...")
                
                # Launch process
                p = multiprocessing.Process(
                    target=_run_synb0_worker,
                    kwargs={
                        'in_file': b0_in_mni,
                        't1_file': t1w_norm_atlas, 
                        'out_file': syn_b0_path,
                        'gpu_ids': gpu_ids,
                        'device': synb0_device,
                    }
                )
                p.start()
                p.join()
                
                if p.exitcode != 0:
                    raise ProcessingError(f"Synb0 estimation subprocess failed with exit code {p.exitcode}")
                
            except Exception as e:
                 raise ProcessingError(f"Synb0 estimation execution failed: {e}")
                 
            #Now inverse warp the synthetic to native b0 space
            # Chain: Synthetic -> T1w -> DWI 
            # syn_b0_native_path declared above
            ants.apply_transforms(fixed_file=b0_path,
                                  moving_file=syn_b0_path,
                                  out_file=syn_b0_native_path,
                                  transforms=[t1w_2_dwi.path, mni_2_t1w.path],
                                  invert_transforms=[t1w_2_dwi.invert, mni_2_t1w.invert],
                                  interpolator="bSpline")

            #Warp T1w mask to DWI space
            t1w_mask_2_dwi = output_dir / "t1w_mask_2_dwi.nii.gz"
            ants.apply_transforms(fixed_file=b0_path,
                                  moving_file=t1w_mask,
                                  out_file=t1w_mask_2_dwi,
                                  transforms=[t1w_2_dwi.path],
                                  invert_transforms=[t1w_2_dwi.invert],
                                  interpolator="nearestNeighbor")



            
            #Force 4D for syn_b0_native
            img_syn = nib.load(str(syn_b0_native_path))
            if img_syn.ndim == 3:
                 nib.Nifti1Image(img_syn.get_fdata()[..., np.newaxis], img_syn.affine, img_syn.header).to_filename(syn_b0_native_path)
            
            #Force 4D for b0_path (real b0)
            img_b0 = nib.load(str(b0_path))
            if img_b0.ndim == 3:
                 nib.Nifti1Image(img_b0.get_fdata()[..., np.newaxis], img_b0.affine, img_b0.header).to_filename(b0_path)

            
            

            real_meta = {}
            if real_json and real_json.exists():
                with open(real_json, "r") as f:
                    real_meta = json.load(f)
            
            real_pe = real_meta.get("PhaseEncodingDirection", "j-")
            

                
            syn_meta = {
                "PhaseEncodingDirection": real_pe,
                "TotalReadoutTime": 0.0, # For synthetic b0
                "Synthesized": True
            }
            with open(syn_json_path, "w") as f:
                json.dump(syn_meta, f)
                       # Create dummy bval for single b0
            # dummy_bval_path defined above
            with open(dummy_bval_path, "w") as f:
                f.write("0\n")

        syn_dwi_file = DWIFile(
            entities={**input_dwi.entities, "desc": "synthetic"},
            img=syn_b0_native_path,
            json=syn_json_path,
            bval=dummy_bval_path,
            bvec=input_dwi.bvec  # Optional
        )
        
        # 4. Create a Topup Group
        # [Real b0, Synthetic b0]
        # We need to represent the real b0 as a DWIFile too if we want to add it to the group?
        # The input_dwi is the whole 4D file. We extracted real_b0.
        # Let's wrap real_b0 as well to be clean.
        real_b0_obj = DWIFile(
            entities={**input_dwi.entities, "desc": "realb0"},
            img=b0_path,
            json=input_dwi.json, # Shares metadata
            bval=dummy_bval_path,
            bvec=None
        )
        
        new_group = {
            "inputs": [real_b0_obj, syn_dwi_file],
            "targets": dwi_files
        }
        
        # Add to context
        # Check if 'topup_groups' exists. 
        topup_groups = context.get("topup_groups", [])
        topup_groups.append(new_group)
        context["topup_groups"] = topup_groups
        
        # Also need to ensure the main DWI is mapped to this group for applying corrections?
        # TopupStep usually handles mapping based on matching acquisition params.
        # We might need to manually update topup_map in TopupStep or ensure parameters match.
        # If we copy TotalReadoutTime, it should match.
        
        self.logger.info("Synb0 synthetic image generated and added to Topup groups.")
        return context
