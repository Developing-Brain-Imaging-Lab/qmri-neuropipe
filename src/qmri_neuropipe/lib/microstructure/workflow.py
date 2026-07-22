"""Reusable aggregate g-ratio image workflow."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from ...io.bids import build_bids_name
from .gratio import (
    Calibration,
    compute_aggregate_gratio,
    compute_conduction_measures,
    compute_myelin_thickness,
)
from .inputs import GRatioInputs


def _load_3d(path: Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Aggregate g-ratio input must be 3D: {path} has shape {data.shape}")
    return image, data


def _same_grid(first, second) -> bool:
    return first.shape[:3] == second.shape[:3] and np.allclose(first.affine, second.affine, atol=1e-5, rtol=1e-5)


def _resample_to(image, reference, order: int = 1):
    if _same_grid(image, reference):
        return image
    return resample_from_to(image, (reference.shape[:3], reference.affine), order=order)


def _write_nifti(data: np.ndarray, reference, path: Path, dtype=np.float32) -> Path:
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=dtype), reference.affine, header), str(path))
    return path


class AggregateGRatioWorkflow:
    """Calculate aggregate g-ratio derivatives from already aligned inputs.

    Cross-modal registration is performed by the pipeline adapter before this
    numerical workflow. A guarded affine-aware resampling fallback makes direct
    generic inputs usable when they already share world coordinates.
    """

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.cfg = dict(config.get("gratio", {}) or {})

    def run(self, inputs: GRatioInputs, output_dir: Path) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        reference, _ = _load_3d(inputs.diffusion_reference)
        myelin_img, _ = _load_3d(inputs.myelin)
        intra_img, _ = _load_3d(inputs.intracellular)
        myelin = np.asarray(_resample_to(myelin_img, reference).dataobj, dtype=np.float64)
        intracellular = np.asarray(_resample_to(intra_img, reference).dataobj, dtype=np.float64)
        isotropic = None
        if inputs.isotropic:
            fiso_img, _ = _load_3d(inputs.isotropic)
            isotropic = np.asarray(_resample_to(fiso_img, reference).dataobj, dtype=np.float64)

        calibration_cfg = dict(self.cfg.get("calibration", {}) or {})
        calibration = Calibration(
            mode=calibration_cfg.get("mode", "identity"),
            slope=calibration_cfg.get("slope"),
            intercept=calibration_cfg.get("intercept"),
        )
        validity_cfg = dict(self.cfg.get("validity", {}) or {})
        result = compute_aggregate_gratio(
            myelin,
            intracellular,
            isotropic,
            calibration=calibration,
            axonal_input_is_avf=inputs.axonal_input_is_avf,
            epsilon=float(validity_cfg.get("epsilon", 1e-6)),
            clipping_tolerance=float(validity_cfg.get("clipping_tolerance", 1e-6)),
        )

        entities = {k: v for k, v in inputs.entities.items() if k not in {"suffix", "desc", "model", "space"}}
        entities["model"] = "gratioAggregate"
        outputs = {}
        arrays = {"MVF": result.mvf, "AVF": result.avf, "FVF": result.fvf, "gratio": result.gratio}
        for metric, data in arrays.items():
            path = output_dir / build_bids_name({**entities, "suffix": metric})
            outputs[metric] = _write_nifti(data, reference, path)

        valid_path = output_dir / build_bids_name({**entities, "desc": "valid", "suffix": "mask"})
        outputs["valid_mask"] = _write_nifti(result.valid, reference, valid_path, np.uint8)

        recommended = result.valid.copy()
        recommended_cfg = dict(self.cfg.get("recommended_mask", {}) or {})
        if recommended_cfg.get("enabled", True):
            if isotropic is not None:
                recommended &= np.isfinite(isotropic) & (isotropic <= float(recommended_cfg.get("fiso_max", 0.5)))
            recommended &= np.nan_to_num(result.avf) >= float(recommended_cfg.get("avf_min", 0.0))
            recommended &= np.nan_to_num(result.fvf) >= float(recommended_cfg.get("fvf_min", 0.0))
            if inputs.recommended_mask:
                wm_img, _ = _load_3d(inputs.recommended_mask)
                wm = np.asarray(_resample_to(wm_img, reference, order=0).dataobj, dtype=np.float64)
                recommended &= np.isfinite(wm) & (wm >= float(recommended_cfg.get("wm_probability_min", 0.5)))
            erosion_voxels = int(recommended_cfg.get("erosion_voxels", 0) or 0)
            if erosion_voxels > 0:
                from scipy.ndimage import binary_erosion
                recommended = binary_erosion(recommended, iterations=erosion_voxels)
        recommended_path = output_dir / build_bids_name({**entities, "desc": "recommended", "suffix": "mask"})
        outputs["recommended_mask"] = _write_nifti(recommended, reference, recommended_path, np.uint8)

        diameter = None
        diameter_units = None
        if inputs.axon_diameter:
            diameter_img, diameter = _load_3d(inputs.axon_diameter)
            diameter = np.asarray(_resample_to(diameter_img, reference).dataobj, dtype=np.float64)
            diameter_cfg = dict((self.cfg.get("inputs", {}) or {}).get("axon_diameter", {}) or {})
            diameter_units = str(diameter_cfg.get("units", "um")).lower()
            if diameter_units in {"mm", "millimeter", "millimeters"}:
                diameter *= 1000.0
                diameter_units = "um"
            elif diameter_units not in {"um", "µm", "micrometer", "micrometers"}:
                raise ValueError(f"Unsupported inner axon diameter units: {diameter_units}")
            thickness = compute_myelin_thickness(diameter, result.gratio)
            path = output_dir / build_bids_name({**entities, "suffix": "myelinThickness"})
            outputs["myelinThickness"] = _write_nifti(thickness, reference, path)

        conduction_cfg = dict(self.cfg.get("conduction", {}) or {})
        measures = compute_conduction_measures(
            result.gratio,
            diameter,
            rushton_coefficient=(conduction_cfg.get("rushton", {}) or {}).get("calibration_coefficient"),
            waxman_bennett_coefficient=(conduction_cfg.get("waxman_bennett", {}) or {}).get("calibration_coefficient"),
        )
        if not (conduction_cfg.get("rushton", {}) or {}).get("enabled", True):
            measures = {key: value for key, value in measures.items() if not key.startswith("Rushton")}
        if not (conduction_cfg.get("waxman_bennett", {}) or {}).get("enabled", True):
            measures = {key: value for key, value in measures.items() if not key.startswith("WaxmanBennett")}
        for metric, data in measures.items():
            if metric == "ConductionFactor" and not conduction_cfg.get("conduction_factor", True):
                continue
            model = "gratioAggregate"
            suffix = metric
            if metric.startswith("Rushton"):
                model, suffix = "Rushton", metric.replace("Rushton", "")
            elif metric.startswith("WaxmanBennett"):
                model, suffix = "WaxmanBennett", metric.replace("WaxmanBennett", "")
            path = output_dir / build_bids_name({**entities, "model": model, "suffix": suffix})
            outputs[metric] = _write_nifti(data, reference, path)

        sidecar_common = {
            "MetricDefinition": "MRI-derived aggregate g-ratio and component measures",
            "MyelinInput": str(inputs.myelin),
            "AxonalInput": str(inputs.intracellular),
            "IsotropicInput": str(inputs.isotropic) if inputs.isotropic else None,
            "DiffusionReference": str(inputs.diffusion_reference),
            "MyelinCalibration": {
                "Mode": calibration.mode,
                "Slope": calibration.slope,
                "Intercept": calibration.intercept,
                "Calibrated": calibration.calibrated,
            },
            "AVFFormula": "AVF" if inputs.axonal_input_is_avf else "(1-FISO)*ICVF",
            "AggregateGRatioFormula": "sqrt(AVF/(MVF+AVF))",
            "InnerAxonDiameter": str(inputs.axon_diameter) if inputs.axon_diameter else None,
            "DiameterUnitsAfterConversion": diameter_units,
            "RecommendedMaskInput": str(inputs.recommended_mask) if inputs.recommended_mask else None,
            "Registration": inputs.registration_metadata or {},
        }
        for key, path in outputs.items():
            if path.name.endswith(".nii.gz"):
                json_path = Path(str(path).replace(".nii.gz", ".json"))
                json_path.write_text(json.dumps({**sidecar_common, "OutputMetric": key}, indent=2) + "\n")

        valid_values = result.gratio[result.valid]
        qc = {
            "TotalVoxels": int(result.valid.size),
            "ValidVoxels": int(np.count_nonzero(result.valid)),
            "RecommendedVoxels": int(np.count_nonzero(recommended)),
            "ClippedVoxels": int(np.count_nonzero(result.clipped)),
            "InvalidVoxels": int(result.valid.size - np.count_nonzero(result.valid)),
            "AggregateGRatioRange": (
                [float(np.min(valid_values)), float(np.max(valid_values))]
                if valid_values.size else None
            ),
            "CalibrationMode": calibration.mode,
            "Registration": inputs.registration_metadata or {},
        }
        qc_path = output_dir / build_bids_name(
            {**entities, "desc": "qc", "suffix": "gratio"}, extension=".json"
        )
        qc_path.write_text(json.dumps(qc, indent=2) + "\n")
        outputs["qc"] = qc_path

        summary = output_dir / build_bids_name(
            {**entities, "desc": "summary", "suffix": "gratio"},
            extension=".tsv",
        )
        with summary.open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["metric", "n", "mean", "std", "median", "min", "max"])
            for metric, data in {**arrays, **measures}.items():
                values = np.asarray(data)[recommended & np.isfinite(data)]
                writer.writerow([metric, values.size, *( [float(np.mean(values)), float(np.std(values)), float(np.median(values)), float(np.min(values)), float(np.max(values))] if values.size else ["nan"] * 5 )])
        outputs["summary"] = summary
        return outputs
