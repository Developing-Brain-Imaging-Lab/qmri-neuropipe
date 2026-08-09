"""Experimental Synb0/TOPUP-derived synthetic reverse-PE generation."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...interfaces import fsl
from ...io.dmri.bids import infer_phase_encoding_direction


_OPPOSITE_PE = {"i": "i-", "i-": "i", "j": "j-", "j-": "j", "k": "k-", "k-": "k"}
_FUGUE_DIRECTION = {"i": "x", "i-": "x-", "j": "y", "j-": "y-", "k": "z", "k-": "z-"}


class SyntheticReversePEStep(BaseProcessingStep):
    """Forward-distort Synb0 with a TOPUP field for experimental DRBUDDI use."""

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance=None,
        options: Optional[dict] = None,
    ):
        super().__init__(config, logger, provenance)
        self.options = dict(options or {})

    def validate_inputs(self, first_arg, **kwargs) -> None:
        context, _ = self.unpack_input(first_arg)
        if context is None:
            raise ValidationError("SyntheticReversePEStep requires pipeline context mode")
        if not context.get("synb0_undistorted_reference"):
            raise ValidationError("Synthetic reverse-PE generation requires a Synb0 reference")
        if not context.get("topup_base"):
            raise ValidationError("Synthetic reverse-PE generation requires a TOPUP field")

    def validate_outputs(self, result) -> None:
        generated = result.get("tortoise_synthetic_reverse_pe") if isinstance(result, dict) else None
        if not isinstance(generated, DWIFile) or not all(
            path and Path(path).exists()
            for path in (
                getattr(generated, "img", None),
                getattr(generated, "bval", None),
                getattr(generated, "bvec", None),
            )
        ):
            raise ProcessingError("Synthetic reverse-PE generation did not produce a DWIFile")

    @staticmethod
    def _metadata(dwi: DWIFile) -> dict:
        if dwi.json and Path(dwi.json).exists():
            try:
                return json.loads(Path(dwi.json).read_text())
            except (OSError, json.JSONDecodeError):
                return {}
        return {}

    def _resolve_dwell_time(self, acquired: DWIFile, pe_direction: str) -> tuple[float, float]:
        metadata = self._metadata(acquired)
        configured_dwell = self.options.get("effective_echo_spacing")
        configured_readout = self.options.get("total_readout_time")
        dwell = configured_dwell or metadata.get("EffectiveEchoSpacing")
        total_readout = configured_readout or metadata.get("TotalReadoutTime")
        axis = {"i": 0, "j": 1, "k": 2}[pe_direction[0]]
        pe_samples = int(nib.load(str(acquired.img)).shape[axis])

        if dwell is None:
            if total_readout is None:
                raise ValidationError(
                    "Synthetic reverse-PE generation requires EffectiveEchoSpacing or "
                    "TotalReadoutTime in the acquired DWI JSON (or an explicit override)"
                )
            if pe_samples < 2:
                raise ValidationError("Cannot derive echo spacing from a one-voxel PE axis")
            dwell = float(total_readout) / float(pe_samples - 1)
        dwell = float(dwell)
        if dwell <= 0:
            raise ValidationError("EffectiveEchoSpacing must be positive")
        if total_readout is None:
            total_readout = dwell * float(max(pe_samples - 1, 1))
        return dwell, float(total_readout)

    def _series_mode(self) -> str:
        """Return the requested synthetic reverse-PE payload type."""
        requested = str(self.options.get("series_mode", "b0_duplicates")).strip().lower()
        aliases = {
            "b0": "b0_duplicates",
            "b0_only": "b0_duplicates",
            "duplicate_b0": "b0_duplicates",
            "duplicated_b0": "b0_duplicates",
            "full": "full_dwi",
            "dwi": "full_dwi",
            "full_series": "full_dwi",
        }
        mode = aliases.get(requested, requested)
        if mode not in {"b0_duplicates", "full_dwi"}:
            raise ValidationError(
                "synthetic_reverse_pe.series_mode must be b0_duplicates or full_dwi"
            )
        return mode

    def run(self, first_arg, output_dir: Path, **kwargs) -> dict:
        context, fallback = self.unpack_input(first_arg)
        if context is None:
            raise ProcessingError("SyntheticReversePEStep requires pipeline context mode")
        acquired_files = list(context.get("dwi_files") or ([fallback] if fallback else []))
        if not acquired_files:
            raise ProcessingError("No acquired DWI is available for synthetic reverse-PE generation")
        acquired = acquired_files[0]
        acquired_pe = infer_phase_encoding_direction(acquired)
        if acquired_pe not in _OPPOSITE_PE:
            raise ValidationError(
                "Synthetic reverse-PE generation requires a valid BIDS PhaseEncodingDirection"
            )
        synthetic_pe = _OPPOSITE_PE[acquired_pe]
        dwell, total_readout = self._resolve_dwell_time(acquired, acquired_pe)

        override = str(self.options.get("fugue_unwarpdir", "auto")).strip().lower()
        fugue_direction = _FUGUE_DIRECTION[synthetic_pe] if override == "auto" else override
        backend = str(self.options.get("forward_warp_backend", "fugue")).strip().lower()
        if backend != "fugue":
            raise ValidationError("Only forward_warp_backend: fugue is currently supported")

        step_dir = self.get_step_output_dir(output_dir)
        force = bool(kwargs.get("force", False))
        series_mode = self._series_mode()
        undistorted = Path(context["synb0_undistorted_reference"])
        topup_base = Path(context["topup_base"])
        field_hz = topup_base.with_name(topup_base.name + "_field.nii.gz")
        if not undistorted.exists():
            raise ValidationError(f"Synb0 undistorted reference does not exist: {undistorted}")
        if not field_hz.exists():
            raise ValidationError(f"TOPUP field map does not exist: {field_hz}")
        if series_mode == "full_dwi":
            if not acquired.bval or not acquired.bvec or not all(
                Path(path).exists() for path in (acquired.bval, acquired.bvec)
            ):
                raise ValidationError(
                    "Full-DWI synthetic reverse-PE generation requires acquired gradients"
                )
            datain = topup_base.with_name(topup_base.name + "_topup_datain.txt")
            if not datain.exists():
                raise ValidationError(
                    f"Full-DWI synthetic reverse-PE generation requires TOPUP datain: {datain}"
                )
            required_topup = [
                topup_base.with_name(topup_base.name + "_fieldcoef.nii.gz"),
                topup_base.with_name(topup_base.name + "_movpar.txt"),
            ]
            missing_topup = [path for path in required_topup if not path.exists()]
            if missing_topup:
                raise ValidationError(
                    "Full-DWI synthetic reverse-PE generation is missing TOPUP "
                    f"outputs: {', '.join(str(path) for path in missing_topup)}"
                )
            undistorted_source = step_dir / "desc-topupUndistorted_sourceDWI.nii.gz"
            force_unwarp = (
                force
                or not undistorted_source.exists()
                or max(
                    Path(acquired.img).stat().st_mtime,
                    required_topup[0].stat().st_mtime,
                    required_topup[1].stat().st_mtime,
                    datain.stat().st_mtime,
                ) > undistorted_source.stat().st_mtime
            )
            fsl.applytopup(
                acquired,
                undistorted_source,
                topup_base=topup_base,
                datain=datain,
                in_index=int(self.options.get("topup_in_index", 1)),
                method="jac",
                force=force_unwarp,
            )
            warp_source = undistorted_source
            warped_3d = step_dir / "desc-forwardDistortedReversePE_fullDWI.nii.gz"
            warp_sidecar = step_dir / "desc-forwardDistortedReversePE_fullDWI.json"
        else:
            warp_source = undistorted
            warped_3d = step_dir / "synb0_desc-forwardDistortedReversePE_b0.nii.gz"
            warp_sidecar = step_dir / "synb0_desc-forwardDistortedReversePE_b0.json"
        intensity_correction = bool(self.options.get("intensity_correction", True))
        warp_config = {
            "Source": str(warp_source),
            "Field": str(field_hz),
            "DwellTime": dwell,
            "FUGUEUnwarpDirection": fugue_direction,
            "IntensityCorrection": intensity_correction,
            "SeriesMode": series_mode,
        }
        try:
            cached_warp_config = (
                json.loads(warp_sidecar.read_text()) if warp_sidecar.exists() else {}
            )
        except (OSError, json.JSONDecodeError):
            cached_warp_config = {}
        force_forward = (
            force
            or not warped_3d.exists()
            or not field_hz.exists()
            or max(warp_source.stat().st_mtime, field_hz.stat().st_mtime)
            > warped_3d.stat().st_mtime
            or cached_warp_config != warp_config
        )
        fsl.forward_distort_with_fugue(
            warp_source,
            field_hz,
            warped_3d,
            dwell_time=dwell,
            unwarp_direction=fugue_direction,
            intensity_correction=intensity_correction,
            force=force_forward,
        )
        warp_sidecar.write_text(json.dumps(warp_config, indent=2) + "\n")

        duplicates = None
        if series_mode == "b0_duplicates":
            try:
                duplicates = int(self.options.get("duplicate_volumes", 2))
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "synthetic_reverse_pe.duplicate_volumes must be a positive integer"
                ) from exc
            if duplicates < 1:
                raise ValidationError("synthetic_reverse_pe.duplicate_volumes must be positive")
        series_path = step_dir / (
            "desc-syntheticReversePE_fullDWI.nii.gz"
            if series_mode == "full_dwi"
            else "synb0_desc-syntheticReversePE_dwi.nii.gz"
        )
        base = Path(str(series_path).split(".nii", 1)[0])
        bval = Path(f"{base}.bval")
        bvec = Path(f"{base}.bvec")
        sidecar = Path(f"{base}.json")
        metadata = self._metadata(acquired) if series_mode == "full_dwi" else {}
        metadata.update({
            "PhaseEncodingDirection": synthetic_pe,
            "EffectiveEchoSpacing": dwell,
            "TotalReadoutTime": total_readout,
            "Synthesized": True,
            "SyntheticReversePE": True,
            "Experimental": True,
            "GeneratedFrom": [str(acquired.img), str(undistorted), str(field_hz)],
            "ForwardWarpBackend": "FSL FUGUE",
            "FUGUEUnwarpDirection": fugue_direction,
            "IntensityCorrection": bool(self.options.get("intensity_correction", True)),
            "SyntheticReversePESeriesMode": series_mode,
            "StatisticallyIndependentAcquisition": False,
        })
        if duplicates is not None:
            metadata["DuplicateVolumes"] = duplicates
        try:
            cached_metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        except (OSError, json.JSONDecodeError):
            cached_metadata = {}
        rebuild_series = (
            force
            or not series_path.exists()
            or not bval.exists()
            or not bvec.exists()
            or warped_3d.stat().st_mtime > series_path.stat().st_mtime
            or cached_metadata != metadata
        )
        if series_mode == "full_dwi":
            if not bval.exists() or not bvec.exists():
                rebuild_series = True
            else:
                source_gradient_mtime = max(
                    Path(acquired.bval).stat().st_mtime,
                    Path(acquired.bvec).stat().st_mtime,
                )
                output_gradient_mtime = min(
                    bval.stat().st_mtime,
                    bvec.stat().st_mtime,
                )
                rebuild_series = rebuild_series or (
                    source_gradient_mtime > output_gradient_mtime
                )
        if rebuild_series:
            warped_img = nib.load(str(warped_3d))
            warped_data = np.asanyarray(warped_img.dataobj)
            if series_mode == "full_dwi":
                if warped_data.ndim != 4:
                    raise ProcessingError(
                        f"Synthetic full reverse-PE DWI must be 4D, got {warped_data.shape}"
                    )
                if warped_data.shape[3] != nib.load(str(acquired.img)).shape[3]:
                    raise ProcessingError(
                        "Synthetic full reverse-PE DWI changed the acquired volume count"
                    )
                acquired_image = nib.load(str(acquired.img))
                if (
                    warped_img.shape[:3] != acquired_image.shape[:3]
                    or not np.allclose(warped_img.affine, acquired_image.affine, atol=1e-4)
                ):
                    raise ProcessingError(
                        "Synthetic full reverse-PE DWI changed the acquired spatial grid"
                    )
                shutil.copy2(warped_3d, series_path)
                shutil.copy2(acquired.bval, bval)
                shutil.copy2(acquired.bvec, bvec)
            else:
                if warped_data.ndim == 4:
                    warped_data = warped_data[..., 0]
                series = np.repeat(warped_data[..., np.newaxis], duplicates, axis=3)
                nib.save(
                    nib.Nifti1Image(series, warped_img.affine, warped_img.header.copy()),
                    series_path,
                )
                bval.write_text(" ".join("0" for _ in range(duplicates)) + "\n")
                bvec.write_text(
                    "\n".join(" ".join("0" for _ in range(duplicates)) for _ in range(3))
                    + "\n"
                )
            sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
        generated_entities = {**acquired.entities, "desc": "syntheticReversePE"}
        generated_entities.pop("dir", None)
        generated = DWIFile(
            entities=generated_entities,
            img=series_path,
            json=sidecar,
            bval=bval,
            bvec=bvec,
            Delta=getattr(acquired, "Delta", None),
            delta=getattr(acquired, "delta", None),
        )
        context["tortoise_synthetic_reverse_pe"] = generated
        context["synthetic_reverse_pe_provenance"] = json.loads(sidecar.read_text())
        self.logger.warning(
            "Generated experimental %s synthetic reverse-PE data for TORTOISE "
            "DRBUDDI; this is not an independently acquired reverse-PE series",
            series_mode,
        )
        return context


__all__ = ["SyntheticReversePEStep"]
