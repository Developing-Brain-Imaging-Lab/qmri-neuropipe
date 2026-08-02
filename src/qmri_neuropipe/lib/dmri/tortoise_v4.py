"""TORTOISEV4 motion and eddy-current correction pipeline step."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import nibabel as nib

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...interfaces import freesurfer
from ...io.bids import build_bids_name
from ...interfaces.tortoise import tortoise_v4_motion_eddy
from ...io.dmri.bids import infer_phase_encoding_direction
from ..anat.super_synth import ensure_supersynth_outputs_for_image
from .b0_reference import select_optimal_b0


def _nifti_json_path(path: Path) -> Path:
    return Path(str(path).split(".nii", 1)[0] + ".json")


def _image_grid(path: Path) -> tuple[list[float], list[int], str]:
    """Return the TORTOISE resolution, matrix, and orientation for an image."""
    image = nib.load(str(path))
    return (
        [float(value) for value in image.header.get_zooms()[:3]],
        [int(value) for value in image.shape[:3]],
        "".join(nib.aff2axcodes(image.affine)),
    )


class TortoiseV4CorrectionStep(BaseProcessingStep):
    """Feed the current pipeline DWI into TORTOISEV4's correction engine."""

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance=None,
        **options,
    ):
        super().__init__(config, logger, provenance)
        self.options = dict(options)

    def validate_inputs(self, first_arg, output_dir: Path, **kwargs) -> None:
        _, image = self.unpack_input(first_arg)
        if not isinstance(image, DWIFile):
            raise ValidationError("TORTOISEV4 correction requires a DWIFile")
        for label, path in (("image", image.img), ("bval", image.bval), ("bvec", image.bvec)):
            if not path or not Path(path).exists():
                raise ValidationError(f"TORTOISEV4 input {label} is missing: {path}")
        if len(nib.load(str(image.img)).shape) != 4:
            raise ValidationError("TORTOISEV4 correction requires a 4D DWI")

    def validate_outputs(self, result) -> None:
        image = result.get("current_image") if isinstance(result, dict) else result
        if not isinstance(image, DWIFile):
            raise ProcessingError("TORTOISEV4 did not return a DWIFile")
        for path in (image.img, image.bval, image.bvec):
            if not path or not Path(path).exists():
                raise ProcessingError(f"Missing TORTOISEV4 output: {path}")

    def _coregistration_config(self) -> dict:
        nested = self.options.get("coregistration_to_anatomy") or {}
        result = dict(nested) if isinstance(nested, dict) else {}
        result.setdefault(
            "enabled",
            bool(nested or self.options.get("coregister_to_anatomical", False)),
        )
        result.setdefault(
            "reference", self.options.get("anatomical_reference", "auto")
        )
        return result

    @staticmethod
    def _first_path(context: Optional[dict], key: str) -> Optional[Path]:
        values = (context or {}).get(key) or []
        if not values:
            return None
        return Path(getattr(values[0], "img", values[0]))

    def _select_anatomical_reference(self, context: Optional[dict]) -> Optional[Path]:
        coreg = self._coregistration_config()
        configured = coreg.get("reference_file") or self.options.get("reorientation_file")
        if configured:
            return Path(configured)

        preference = str(coreg.get("reference", "auto")).strip().lower()
        if preference in {"synthesized", "synthetic", "supersynth"}:
            if (context or {}).get("tortoise_t2w_source") == "mri_super_synth":
                selected = (context or {}).get("tortoise_t2w")
                return Path(selected) if selected else None
            return None
        key_orders = {
            "t1w": ("t1w_files", "t2w_files", "anatomical_files"),
            "t2w": ("t2w_files", "t1w_files", "anatomical_files"),
            "other": ("anatomical_files", "t2w_files", "t1w_files"),
            "anatomical": ("anatomical_files", "t2w_files", "t1w_files"),
            "auto": ("t2w_files", "t1w_files", "anatomical_files"),
        }
        for key in key_orders.get(preference, key_orders["auto"]):
            selected = self._first_path(context, key)
            if selected:
                return selected
        return None

    def _synthesize_t2w(
        self,
        context: Optional[dict],
        output_dir: Path,
        force: bool,
    ) -> Optional[Path]:
        configured = self.options.get("t2w_fallback", {})
        if configured is False or (
            isinstance(configured, dict) and not configured.get("enabled", True)
        ):
            return None
        fallback_cfg = dict(configured) if isinstance(configured, dict) else {}
        source_preference = str(fallback_cfg.get("anatomical_input", "auto")).lower()
        source_keys = {
            "t1w": ("t1w_files",),
            "t2w": ("t2w_files",),
            "other": ("anatomical_files",),
            "anatomical": ("anatomical_files",),
            "auto": ("t1w_files", "anatomical_files", "t2w_files"),
        }.get(
            source_preference,
            ("t1w_files", "anatomical_files", "t2w_files"),
        )
        source = next(
            (candidate for key in source_keys if (candidate := self._first_path(context, key))),
            None,
        )
        if source is None:
            return None

        synth_dir = output_dir / "supersynth_t2w"
        synth_dir.mkdir(parents=True, exist_ok=True)
        outputs = ensure_supersynth_outputs_for_image(
            source,
            synth_dir,
            self.config,
            self.logger,
            mode=fallback_cfg.get("mode"),
            device=fallback_cfg.get("device"),
            sharpen_synths=fallback_cfg.get("sharpen_synths"),
            force=force,
        )
        synth_t2w = outputs.get("synth_t2w")
        if not synth_t2w:
            raise ProcessingError(
                f"mri_super_synth did not create a synthesized T2w from {source}"
            )
        synth_t2w = Path(synth_t2w)
        if synth_t2w.suffix.lower() == ".mgz":
            converted = synth_dir / "desc-supersynth_T2w.nii.gz"
            freesurfer.mri_convert(in_file=synth_t2w, out_file=converted)
            synth_t2w = converted
        if context is not None:
            context["tortoise_t2w"] = synth_t2w
            context["tortoise_t2w_source"] = "mri_super_synth"
        return synth_t2w

    def _select_t2w_reference(
        self,
        context: Optional[dict],
        output_dir: Path,
        force: bool,
    ) -> Optional[Path]:
        """Select an acquired or SuperSynth T2w for TORTOISE EPI correction."""
        configured = self.options.get("t2w_fallback", {})
        fallback_cfg = dict(configured) if isinstance(configured, dict) else {}
        source = str(fallback_cfg.get("source", "auto")).strip().lower()
        if source not in {"auto", "acquired", "synthesized"}:
            raise ValidationError(
                "TORTOISEV4 t2w_fallback.source must be auto, acquired, or synthesized"
            )

        acquired = self._first_path(context, "t2w_files")
        if source in {"auto", "acquired"} and acquired:
            if context is not None:
                context["tortoise_t2w"] = acquired
                context["tortoise_t2w_source"] = "acquired"
            return acquired
        if source == "acquired":
            return None
        if source == "synthesized" and (
            configured is False or not fallback_cfg.get("enabled", True)
        ):
            raise ValidationError(
                "TORTOISEV4 t2w_fallback.source=synthesized requires "
                "t2w_fallback.enabled=true"
            )
        return self._synthesize_t2w(context, output_dir, force)

    def _select_structural(
        self,
        context: Optional[dict],
        output_dir: Path,
        force: bool,
    ) -> Optional[list[Path]]:
        configured = self.options.get("structural_file")
        if configured:
            values = configured if isinstance(configured, (list, tuple)) else [configured]
            return [Path(value) for value in values]
        epi = str(self.options.get("epi", "off")).lower()
        if epi == "t2wreg":
            selected = self._select_t2w_reference(context, output_dir, force)
            return [selected] if selected else None
        t2w_cfg = self.options.get("t2w_fallback")
        if (
            epi == "drbuddi"
            and isinstance(t2w_cfg, dict)
            and bool(t2w_cfg.get("use_for_drbuddi", False))
        ):
            selected = self._select_t2w_reference(context, output_dir, force)
            if not selected:
                raise ValidationError(
                    "TORTOISEV4 DRBUDDI was configured to use a T2w structural "
                    "image, but no acquired or synthesized T2w is available"
                )
            return [selected]
        wants_structural = (
            self.options.get("use_structural", False)
            or bool(self._coregistration_config().get("enabled", False))
        )
        if not wants_structural or not context:
            return None
        for key in ("t2w_files", "anatomical_files", "t1w_files"):
            values = context.get(key) or []
            if values:
                if self.options.get("all_structurals", False):
                    return [Path(getattr(value, "img", value)) for value in values]
                return [Path(getattr(values[0], "img", values[0]))]
        return None

    def _select_reorientation(
        self,
        context: Optional[dict],
        structurals,
        output_dir: Optional[Path] = None,
        force: bool = False,
    ) -> Optional[Path]:
        if not self._coregistration_config().get("enabled", False):
            return None
        preference = str(
            self._coregistration_config().get("reference", "auto")
        ).strip().lower()
        if preference in {"synthesized", "synthetic", "supersynth"}:
            selected = self._select_anatomical_reference(context)
            if selected is None and output_dir is not None:
                selected = self._synthesize_t2w(context, output_dir, force)
            if selected is None:
                raise ValidationError(
                    "TORTOISEV4 coregistration reference=synthesized requires an "
                    "anatomical input and t2w_fallback.enabled=true"
                )
            return selected
        return self._select_anatomical_reference(context) or (structurals[0] if structurals else None)

    def _resolve_output_grid(
        self,
        input_dwi: DWIFile,
        reorientation: Optional[Path],
    ) -> tuple[Optional[list[float]], Optional[list[int]], str]:
        output_res = self.options.get("output_res")
        output_voxels = self.options.get("output_voxels")
        output_orientation = self.options.get("output_orientation")
        coreg = self._coregistration_config()
        resolution_mode = str(coreg.get("output_resolution", "")).strip().lower()
        grid_reference: Optional[Path] = None
        if resolution_mode in {"native", "dwi"}:
            grid_reference = Path(input_dwi.img)
        elif resolution_mode == "anatomical":
            if reorientation is None:
                raise ValidationError(
                    "TORTOISEV4 anatomical output resolution requires an anatomical reference"
                )
            grid_reference = reorientation
        elif resolution_mode:
            raise ValidationError(
                "coregistration_to_anatomy.output_resolution must be 'native' or 'anatomical'"
            )

        if grid_reference:
            derived_res, derived_voxels, derived_orientation = _image_grid(grid_reference)
            output_res = output_res or derived_res
            output_voxels = output_voxels or derived_voxels
            output_orientation = output_orientation or derived_orientation
        if not output_orientation:
            output_orientation = _image_grid(Path(input_dwi.img))[2]
        return output_res, output_voxels, output_orientation

    def _resolve_nthreads(self, runtime_nthreads=None) -> int:
        """Resolve the per-step override before the pipeline-wide CPU count."""
        requested = self.options.get(
            "nthreads",
            self.options.get(
                "threads",
                runtime_nthreads if runtime_nthreads is not None else self.config.n_cpus,
            ),
        )
        try:
            requested = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValidationError("TORTOISEV4 nthreads must be a positive integer") from exc
        if requested < 1:
            raise ValidationError("TORTOISEV4 nthreads must be a positive integer")
        return requested

    def _resolve_repol(self) -> bool:
        """Avoid TORTOISEV4's final WLLS crash for b0-only Synb0 down data."""
        requested = bool(self.options.get("repol", False))
        if not requested or not self.options.get("use_synb0", False):
            return requested

        policy = str(self.options.get("synb0_repol_policy", "disable")).strip().lower()
        if policy == "allow":
            self.logger.warning(
                "TORTOISEV4 repol is explicitly allowed with b0-only Synb0 down_data; "
                "TORTOISE V4.1.1 may segfault during final WLLS fitting."
            )
            return True
        if policy == "error":
            raise ValidationError(
                "TORTOISEV4 repol with b0-only Synb0 down_data is unsafe because "
                "TORTOISE V4.1.1 segfaults during final WLLS fitting. Set repol: false, "
                "use synb0_repol_policy: disable, or use native reverse-PE DWI data."
            )
        if policy != "disable":
            raise ValidationError(
                "TORTOISEV4 synb0_repol_policy must be disable, error, or allow"
            )
        self.logger.warning(
            "Disabling TORTOISEV4 repol for b0-only Synb0 down_data to avoid the "
            "TORTOISE V4.1.1 final-WLLS segmentation fault. Configure the pipeline's "
            "post-TORTOISE outliers step if additional outlier rejection is required."
        )
        return False

    @staticmethod
    def _prepare_temp_folder(output_dir: Path, force: bool) -> Path:
        """Discard failed TORTOISE state when an explicit rerun is requested."""
        temp_folder = output_dir / "tortoise_work"
        if force and temp_folder.exists():
            shutil.rmtree(temp_folder)
        return temp_folder

    def _find_synb0_down(self, context: dict) -> Optional[DWIFile]:
        for group_item in context.get("topup_groups", []):
            inputs = group_item.get("inputs", []) if isinstance(group_item, dict) else group_item
            for candidate in inputs:
                desc = str(getattr(candidate, "entities", {}).get("desc", "")).lower()
                if isinstance(candidate, DWIFile) and (desc == "synthetic" or "synb0" in desc):
                    return candidate
        return None

    def _select_up_down(self, context: Optional[dict], fallback: DWIFile) -> tuple[DWIFile, Optional[DWIFile]]:
        if not context:
            return fallback, None
        files = list(context.get("dwi_files") or [fallback])
        if self.options.get("use_synb0", False):
            synthetic = self._find_synb0_down(context)
            if synthetic is None:
                raise ValidationError("TORTOISEV4 use_synb0 requested but no Synb0 image was generated")
            return files[0], synthetic
        if not self.options.get("use_reverse_pe", False):
            return files[0], None

        preferred = self.options.get("up_phase_encoding")
        up = next(
            (dwi for dwi in files if preferred and infer_phase_encoding_direction(dwi) == preferred),
            files[0],
        )
        up_pe = infer_phase_encoding_direction(up)
        if not up_pe:
            raise ValidationError("Cannot identify PhaseEncodingDirection for TORTOISEV4 up_data")
        down = next(
            (
                dwi for dwi in files
                if dwi is not up
                and infer_phase_encoding_direction(dwi)
                and infer_phase_encoding_direction(dwi)[0] == up_pe[0]
                and infer_phase_encoding_direction(dwi).endswith("-") != up_pe.endswith("-")
            ),
            None,
        )
        if down is None:
            raise ValidationError(
                f"TORTOISEV4 reverse-PE processing requested but no opposite {up_pe[0]} direction was found"
            )
        return up, down

    def run(self, first_arg, output_dir: Path, **kwargs):
        context, fallback_dwi = self.unpack_input(first_arg)
        input_dwi, down_dwi = self._select_up_down(context, fallback_dwi)
        output_dir = self.get_step_output_dir(output_dir)
        entities = {**input_dwi.entities, "desc": "tortoisev4corrected"}
        if down_dwi:
            entities.pop("dir", None)
        out_file = output_dir / build_bids_name(entities)
        out_base = Path(str(out_file).split(".nii", 1)[0])
        out_bval = Path(f"{out_base}.bval")
        out_bvec = Path(f"{out_base}.bvec")

        force = bool(kwargs.get("force", False))
        if out_file.exists() and out_bval.exists() and out_bvec.exists() and not force:
            try:
                cached_shape = nib.load(str(out_file)).shape
                if len(cached_shape) != 4:
                    raise ValueError(f"expected a 4D DWI, got {cached_shape}")
            except Exception as exc:
                self.logger.warning(
                    "Ignoring unreadable cached TORTOISEV4 output %s: %s",
                    out_file,
                    exc,
                )
            else:
                result = DWIFile(
                    entities=entities,
                    img=out_file,
                    json=_nifti_json_path(out_file) if _nifti_json_path(out_file).exists() else input_dwi.json,
                    bval=out_bval,
                    bvec=out_bvec,
                    Delta=getattr(input_dwi, "Delta", None),
                    delta=getattr(input_dwi, "delta", None),
                )
                return self._return_result(context, result)

        reference_cfg = dict(self.options.get("reference_selection") or {})
        selection = None
        b0_id = int(self.options.get("b0_id", -1))
        if reference_cfg.get("enabled", True) and reference_cfg.get("method", "native") == "native":
            selection = select_optimal_b0(
                input_dwi,
                output_dir / "b0_reference",
                threshold=float(reference_cfg.get("b0_threshold", 50.0)),
                local_radius=int(reference_cfg.get("local_radius", 3)),
                force=force,
            )
            b0_id = selection.index

        structurals = self._select_structural(context, output_dir, force)
        reorientation = self._select_reorientation(
            context, structurals, output_dir=output_dir, force=force
        )
        output_res, output_voxels, orientation = self._resolve_output_grid(
            input_dwi, reorientation
        )
        epi = str(self.options.get("epi", "off"))
        if epi.lower() == "drbuddi" and down_dwi is None:
            raise ValidationError("TORTOISEV4 epi=DRBUDDI requires reverse-PE or Synb0 down_data")
        if epi.lower() == "t2wreg" and not structurals:
            raise ValidationError("TORTOISEV4 epi=T2Wreg requires an undistorted T2-weighted image")
        if self.options.get("use_synb0", False) and epi.lower() != "drbuddi":
            raise ValidationError("TORTOISEV4 use_synb0 is only meaningful with epi=DRBUDDI")
        output_combination = self.options.get("output_data_combination")
        if self.options.get("use_synb0", False) and not output_combination:
            # Keep the synthetic reverse-PE b0 out of the downstream DWI. In
            # JacSep mode TORTOISE writes the corrected up series at --output.
            output_combination = "JacSep"
        requested_threads = self._resolve_nthreads(kwargs.get("nthreads"))
        self._resolved_nthreads = requested_threads
        effective_repol = self._resolve_repol()
        self._effective_repol = effective_repol
        temp_folder = self._prepare_temp_folder(output_dir, force)
        result = tortoise_v4_motion_eddy(
            input_dwi,
            out_file,
            down_file=down_dwi,
            structural_file=structurals,
            reorientation_file=reorientation,
            b0_id=b0_id,
            correction_mode=self.options.get("correction_mode", "quadratic"),
            slice_to_volume=bool(self.options.get("slice_to_volume", False)),
            repol=effective_repol,
            niter=int(self.options.get("niter", 3)),
            denoising=self.options.get("denoising", "off"),
            gibbs=bool(self.options.get("gibbs", False)),
            drift=self.options.get("drift", "off"),
            epi=epi,
            output_orientation=self.options.get("output_orientation", orientation),
            output_res=output_res,
            output_voxels=output_voxels,
            output_data_combination=output_combination,
            output_signal_redist_method=self.options.get("output_signal_redist_method"),
            temp_folder=temp_folder,
            executable=self.options.get("executable"),
            use_gpu=bool(self.options.get("use_gpu", self.config.use_gpu)),
            nthreads=requested_threads,
            do_qc=bool(self.options.get("do_qc", True)),
            extra_options=self.options.get("options", {}),
        )
        result.entities = entities
        result.json = self._write_json(input_dwi, result, selection)
        return self._return_result(context, result, selection)

    def _write_json(self, source: DWIFile, result: DWIFile, selection) -> Path:
        payload = {}
        if source.json and Path(source.json).exists():
            try:
                payload = json.loads(Path(source.json).read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}
        payload["MotionCorrection"] = "TORTOISEV4"
        payload["EddyCurrentCorrection"] = "TORTOISEV4"
        payload["SliceToVolumeCorrection"] = bool(self.options.get("slice_to_volume", False))
        payload["SignalDriftCorrection"] = self.options.get("drift", "off")
        payload["Denoising"] = self.options.get("denoising", "off")
        payload["GibbsRingingCorrection"] = bool(self.options.get("gibbs", False))
        payload["SusceptibilityDistortionCorrection"] = self.options.get("epi", "off")
        payload["CoregistrationToAnatomy"] = bool(
            self._coregistration_config().get("enabled", False)
        )
        payload["TORTOISEThreads"] = int(
            getattr(self, "_resolved_nthreads", self.config.n_cpus)
        )
        payload["OutlierReplacementRequested"] = bool(self.options.get("repol", False))
        payload["OutlierReplacementApplied"] = bool(
            getattr(self, "_effective_repol", self.options.get("repol", False))
        )
        if selection:
            payload["MotionCorrectionReferenceVolume"] = selection.index
            payload["MotionCorrectionReferenceImage"] = str(selection.reference_image)
        destination = _nifti_json_path(result.img)
        destination.write_text(json.dumps(payload, indent=2) + "\n")
        return destination

    def _return_result(self, context, result: DWIFile, selection=None):
        if context is None:
            return result
        context["current_image"] = result
        context["dwi_files"] = [result]
        context.setdefault("preprocessed_dwis", []).append(result)
        if selection:
            context["b0_reference_selection"] = selection
            context["motion_reference"] = selection.pair_average_image
        context["spatial_transform"] = {
            "type": "motion_eddy_correction",
            "method": "tortoise_v4",
            "slice_to_volume": bool(self.options.get("slice_to_volume", False)),
            "usable_for_gnl_mapping": False,
        }
        return context


__all__ = ["TortoiseV4CorrectionStep", "_image_grid"]
