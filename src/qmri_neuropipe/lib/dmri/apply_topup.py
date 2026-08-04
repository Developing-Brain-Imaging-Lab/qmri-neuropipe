"""Apply an FSL topup field after TORTOISEV4 motion/eddy correction."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...core.utils import get_nifti_stem
from ...interfaces import fsl
from ...io.bids import build_bids_name
from ..common.json_metadata import copy_json_with_metadata


class ApplyTopupStep(BaseProcessingStep):
    """Unwarp a single post-TORTOISE DWI with a Synb0-derived topup field."""

    def __init__(self, config, logger=None, provenance=None, *, method: str = "jac"):
        super().__init__(config, logger, provenance)
        self.method = str(method or "jac").lower()
        if self.method != "jac":
            raise ValidationError(
                "Post-TORTOISE ApplyTopupStep currently requires method 'jac'; "
                "least-squares restoration requires multiple PE input streams"
            )

    def validate_inputs(self, first_arg, **kwargs) -> None:
        context, current = self.unpack_input(first_arg)
        if context is None:
            raise ValidationError("ApplyTopupStep requires pipeline context mode")
        dwis = list(context.get("dwi_files") or ([current] if current else []))
        if len(dwis) != 1 or not isinstance(dwis[0], DWIFile):
            raise ValidationError(
                "Post-TORTOISE applytopup currently requires exactly one DWI stream"
            )

    def validate_outputs(self, result) -> None:
        context, current = self.unpack_input(result)
        output = current if context is not None else result
        if not isinstance(output, DWIFile):
            raise ProcessingError("ApplyTopupStep did not return a DWIFile")
        for path in (output.img, output.bval, output.bvec):
            if path is None or not Path(path).exists():
                raise ProcessingError(f"Missing post-TORTOISE applytopup output: {path}")
        if len(nib.load(str(output.img)).shape) != 4:
            raise ProcessingError("Post-TORTOISE applytopup output must be a 4D DWI")

    @staticmethod
    def _topup_base(context: dict, dwi: DWIFile) -> Path:
        mapping = context.get("topup_map", {})
        value = mapping.get(dwi.img) or mapping.get(str(dwi.img)) or context.get("topup_base")
        if not value:
            raise ProcessingError(
                "No topup result was mapped to the post-TORTOISE DWI"
            )
        return Path(value)

    def run(self, first_arg, output_dir: Path, **kwargs):
        context, current = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("ApplyTopupStep requires pipeline context mode")
        dwi = list(context.get("dwi_files") or [current])[0]
        topup_base = self._topup_base(context, dwi)
        datain = topup_base.with_name(topup_base.name + "_topup_datain.txt")

        step_dir = self.get_step_output_dir(output_dir)
        entities = {**dwi.entities, "desc": "tortoisev4topupcorrected"}
        out_img = step_dir / build_bids_name(entities)
        stem = get_nifti_stem(out_img)
        out_bval = step_dir / f"{stem}.bval"
        out_bvec = step_dir / f"{stem}.bvec"
        out_json = step_dir / f"{stem}.json"

        fsl.applytopup(
            dwi,
            out_img,
            topup_base=topup_base,
            datain=datain,
            in_index=1,
            method=self.method,
            force=bool(kwargs.get("force", False)),
        )
        if dwi.bval:
            shutil.copy2(dwi.bval, out_bval)
        if dwi.bvec:
            shutil.copy2(dwi.bvec, out_bvec)

        copy_json_with_metadata(dwi.json, out_json)
        try:
            payload = json.loads(out_json.read_text()) if out_json.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        payload.update({
            "SusceptibilityDistortionCorrection": "FSL TOPUP (post-TORTOISEV4, Synb0)",
            "TopupApplicationMethod": self.method,
            "TopupReferenceRow": 1,
        })
        out_json.write_text(json.dumps(payload, indent=2) + "\n")

        result = DWIFile(
            entities=entities,
            img=out_img,
            json=out_json,
            bval=out_bval,
            bvec=out_bvec,
            Delta=getattr(dwi, "Delta", None),
            delta=getattr(dwi, "delta", None),
        )
        result.spatial_transform = {
            "type": "susceptibility_distortion_correction",
            "method": "topup",
            "topup_base": str(topup_base),
            "usable_for_gnl_mapping": False,
        }
        context["current_image"] = result
        context["dwi_files"] = [result]
        context["post_tortoise_topup"] = str(topup_base)
        context["spatial_transform"] = result.spatial_transform
        return context


__all__ = ["ApplyTopupStep"]
