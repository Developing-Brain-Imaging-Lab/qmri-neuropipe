"""Standalone and reusable aggregate g-ratio pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np

from ...core import BasePipeline
from ...io.bids import get_entities_from_path
from ...lib.microstructure.inputs import (
    GRatioInputs,
    discover_mcdespot_vfm,
    discover_noddi_pairs,
    discover_spgr_reference,
    expand_subject_path,
)
from ...lib.microstructure.registration import grids_match, register_myelin_to_b0
from ...lib.microstructure.workflow import AggregateGRatioWorkflow


def _identity(entities: dict) -> tuple:
    return tuple((key, entities.get(key)) for key in ("sub", "ses", "acq", "rec", "run", "dir", "task") if entities.get(key))


def _extract_mean_b0(dwi: Path, bval: Path, output: Path, threshold: float = 100.0) -> Path:
    image = nib.load(str(dwi))
    data = np.asarray(image.dataobj)
    if data.ndim == 3:
        mean = data
    else:
        bvalues = np.atleast_1d(np.loadtxt(bval))
        if bvalues.size != data.shape[3]:
            raise ValueError(f"DWI/bval count mismatch for aggregate g-ratio reference: {dwi}")
        indices = bvalues <= threshold
        if not np.any(indices):
            raise ValueError(f"No b<={threshold:g} volumes found in {dwi}")
        mean = np.mean(data[..., indices], axis=3)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(mean, dtype=np.float32), image.affine, image.header), str(output))
    return output


def _find_matching_dwi(dwi_root: Path, entities: dict) -> tuple[Path, Path]:
    wanted = _identity(entities)
    candidates = []
    for path in dwi_root.glob("**/*_dwi.nii.gz"):
        if _identity(get_entities_from_path(path)) != wanted:
            continue
        bval = Path(str(path).replace(".nii.gz", ".bval"))
        if not bval.exists():
            continue
        lower = path.name.lower()
        score = sum(token in lower for token in ("preproc", "eddy", "topup", "coreg"))
        candidates.append((score, path, bval))
    if not candidates:
        raise FileNotFoundError(f"No modeled DWI with bval matched NODDI entities {dict(wanted)}")
    best_score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_score]
    if len(best) != 1:
        raise ValueError("Ambiguous diffusion reference for NODDI result: " + ", ".join(str(item[1]) for item in best))
    return best[0][1], best[0][2]


def _find_subject_wm_mask(subject_root: Path) -> Optional[Path]:
    patterns = ("**/*label-WM*_probseg.nii.gz", "**/*_wmseg.nii.gz", "**/*desc-wm*_mask.nii.gz")
    candidates = []
    for pattern in patterns:
        candidates.extend(subject_root.glob(pattern))
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


class AggregateGRatioPipeline(BasePipeline):
    @property
    def name(self):
        return "gratio-pipeline"

    @property
    def version(self):
        return "1.0.0"

    def _initialize_pipeline(self):
        self.workflow = AggregateGRatioWorkflow(self.config, self.logger)

    def _should_skip(self, subject: str, session: Optional[str]) -> bool:
        return False

    def process_subject(self, subject: str, session: Optional[str]):
        return run_aggregate_gratio_subject(self.config, subject, session, logger=self.logger)


def run_aggregate_gratio_subject(
    config,
    subject: str,
    session: Optional[str] = None,
    logger=None,
    entity_filter: Optional[dict] = None,
):
    logger = logger or logging.getLogger("AggregateGRatioWorkflow")
    cfg = dict(config.get("gratio", {}) or {})
    subject_root = Path(config.output_dir) / f"sub-{subject}"
    if session:
        subject_root /= f"ses-{session}"
    anat_root = subject_root / "anat"
    dwi_root = subject_root / "dwi"
    input_cfg = dict(cfg.get("inputs", {}) or {})

    myelin_cfg = dict(input_cfg.get("myelin", {}) or {})
    myelin = expand_subject_path(myelin_cfg["path"], subject, session) if myelin_cfg.get("path") else discover_mcdespot_vfm(anat_root)
    spgr_value = myelin_cfg.get("reference") or input_cfg.get("spgr_reference")
    spgr = expand_subject_path(spgr_value, subject, session) if spgr_value else discover_spgr_reference(anat_root)

    axonal_cfg = dict(input_cfg.get("axonal", {}) or {})
    if axonal_cfg.get("path"):
        intracellular = expand_subject_path(axonal_cfg["path"], subject, session)
        isotropic = expand_subject_path(axonal_cfg["isotropic_path"], subject, session) if axonal_cfg.get("isotropic_path") else None
        pairs = [(intracellular, isotropic, get_entities_from_path(intracellular))]
    else:
        pairs = discover_noddi_pairs(dwi_root)
    if entity_filter:
        wanted = _identity(entity_filter)
        pairs = [pair for pair in pairs if _identity(pair[2]) == wanted]
        if not pairs:
            raise FileNotFoundError(
                f"No NODDI result matched active diffusion entities {dict(wanted)}"
            )

    diameter_cfg = dict(input_cfg.get("axon_diameter", {}) or {})
    diameter = expand_subject_path(diameter_cfg["path"], subject, session) if diameter_cfg.get("enabled") and diameter_cfg.get("path") else None
    recommended_cfg = dict(cfg.get("recommended_mask", {}) or {})
    if recommended_cfg.get("path"):
        recommended_mask = expand_subject_path(recommended_cfg["path"], subject, session)
    elif recommended_cfg.get("prefer_subject_native_wm", True):
        recommended_mask = _find_subject_wm_mask(subject_root)
    else:
        recommended_mask = None
    outputs = []
    for icvf, fiso, entities in pairs:
        reference_value = axonal_cfg.get("reference")
        if reference_value:
            b0 = expand_subject_path(reference_value, subject, session)
        else:
            dwi, bval = _find_matching_dwi(dwi_root, entities)
            reference_dir = Path(config.work_dir) / f"sub-{subject}"
            if session:
                reference_dir /= f"ses-{session}"
            b0 = _extract_mean_b0(dwi, bval, reference_dir / "gratio" / (dwi.name.replace(".nii.gz", "_meanb0.nii.gz")))

        registration_cfg = dict(cfg.get("registration", {}) or {})
        registered_myelin = myelin
        registration_metadata = {"Skipped": True, "Reason": "matching-grid"}
        assume_aligned = bool(registration_cfg.get("assume_aligned", False))
        if not assume_aligned:
            registration_dir = Path(config.work_dir) / f"sub-{subject}"
            if session:
                registration_dir /= f"ses-{session}"
            identity_label = "_".join(f"{k}-{v}" for k, v in _identity(entities)) or "default"
            registered_myelin, registration_metadata = register_myelin_to_b0(
                config, myelin, spgr, b0, registration_dir / "gratio" / identity_label / "registration", registration_cfg, logger
            )
        elif not grids_match(myelin, b0):
            raise ValueError(
                "gratio.registration.assume_aligned is true, but myelin and diffusion grids differ."
            )
        logger.info("Aggregate g-ratio registration: %s", registration_metadata)

        output_dir = dwi_root / "gratio"
        result = AggregateGRatioWorkflow(config, logger).run(
            GRatioInputs(
                myelin=registered_myelin,
                spgr_reference=spgr,
                intracellular=icvf,
                isotropic=fiso,
                diffusion_reference=b0,
                entities=entities,
                axonal_input_is_avf=str(axonal_cfg.get("interpretation", "")).lower() == "axonal_volume_fraction",
                axon_diameter=diameter,
                recommended_mask=recommended_mask,
                registration_metadata=registration_metadata,
            ),
            output_dir,
        )
        outputs.append(result)
    return outputs
