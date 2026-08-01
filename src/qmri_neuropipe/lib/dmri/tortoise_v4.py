"""TORTOISEV4 motion and eddy-current correction pipeline step."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import nibabel as nib

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...io.bids import build_bids_name
from ...interfaces.tortoise import tortoise_v4_motion_eddy
from .b0_reference import select_optimal_b0


def _nifti_json_path(path: Path) -> Path:
    return Path(str(path).split(".nii", 1)[0] + ".json")


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

    def _select_structural(self, context: Optional[dict]) -> Optional[Path]:
        configured = self.options.get("structural_file")
        if configured:
            return Path(configured)
        if not self.options.get("use_structural", False) or not context:
            return None
        for key in ("t2w_files", "anatomical_files"):
            values = context.get(key) or []
            if values:
                return Path(getattr(values[0], "img", values[0]))
        return None

    def run(self, first_arg, output_dir: Path, **kwargs):
        context, input_dwi = self.unpack_input(first_arg)
        output_dir = self.get_step_output_dir(output_dir)
        entities = {**input_dwi.entities, "desc": "tortoisev4corrected"}
        out_file = output_dir / build_bids_name(entities)
        out_base = Path(str(out_file).split(".nii", 1)[0])
        out_bval = Path(f"{out_base}.bval")
        out_bvec = Path(f"{out_base}.bvec")

        force = bool(kwargs.get("force", False))
        if out_file.exists() and out_bval.exists() and out_bvec.exists() and not force:
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

        orientation = "".join(nib.aff2axcodes(nib.load(str(input_dwi.img)).affine))
        result = tortoise_v4_motion_eddy(
            input_dwi,
            out_file,
            structural_file=self._select_structural(context),
            b0_id=b0_id,
            correction_mode=self.options.get("correction_mode", "quadratic"),
            slice_to_volume=bool(self.options.get("slice_to_volume", False)),
            repol=bool(self.options.get("repol", False)),
            niter=int(self.options.get("niter", 3)),
            output_orientation=self.options.get("output_orientation", orientation),
            temp_folder=output_dir / "tortoise_work",
            executable=self.options.get("executable"),
            use_gpu=bool(self.options.get("use_gpu", self.config.use_gpu)),
            nthreads=int(kwargs.get("nthreads", self.config.n_cpus)),
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


__all__ = ["TortoiseV4CorrectionStep"]
