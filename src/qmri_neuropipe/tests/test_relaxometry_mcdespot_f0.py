import logging
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.workflows.pipelines import relaxometry as relax
from qmri_neuropipe.workflows.pipelines.relaxometry import RelaxometryWorkflow
from qmri_neuropipe.workflows.pipelines.relaxometry_config import (
    RelaxometryConfig,
    RelaxometryModelingConfig,
)


def _image(path: Path, shape=(3, 3, 3), affine=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(
            np.zeros(shape, dtype=np.float32),
            np.eye(4) if affine is None else affine,
        ),
        path,
    )
    return path


def _workflow(tmp_path: Path) -> RelaxometryWorkflow:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
        work_dir=tmp_path / "work",
    )
    return RelaxometryWorkflow(config, logging.getLogger("mcdespot-f0"), {})


def test_explicit_f0_path_expands_subject_session_and_roots(tmp_path):
    workflow = _workflow(tmp_path)
    reference = _image(tmp_path / "ssfp.nii.gz", shape=(3, 3, 3, 2))
    f0 = _image(
        tmp_path
        / "derivatives"
        / "sub-007"
        / "ses-02"
        / "anat"
        / "models"
        / "sub-007_ses-02_model-DESPOT2FM_F0.nii.gz"
    )
    template = (
        "{output_dir}/{sub}/{ses}/anat/models/"
        "{subject_session}_model-DESPOT2FM_F0.nii.gz"
    )

    resolved = workflow._resolve_mcdespot_f0(
        {"fix-f0": True, "f0": template},
        {},
        subject="007",
        session="02",
        reference_file=reference,
    )

    assert resolved == f0


def test_fix_f0_reuses_existing_despot2fm_result(tmp_path):
    workflow = _workflow(tmp_path)
    reference = _image(tmp_path / "ssfp.nii.gz", shape=(3, 3, 3, 4))
    f0 = _image(tmp_path / "models" / "sub-001_model-DESPOT2FM_F0.nii.gz")

    resolved = workflow._resolve_mcdespot_f0(
        {"fix-f0": True},
        {"DESPOT2FM": {"F0": f0}},
        subject="001",
        session=None,
        reference_file=reference,
    )

    assert resolved == f0


def test_fix_f0_without_any_map_fails_before_fitter(tmp_path):
    workflow = _workflow(tmp_path)
    reference = _image(tmp_path / "ssfp.nii.gz", shape=(3, 3, 3, 2))

    with pytest.raises(ValueError, match="no F0 map was configured"):
        workflow._resolve_mcdespot_f0(
            {"fix-f0": True},
            {},
            subject="001",
            session=None,
            reference_file=reference,
        )


def test_f0_grid_must_match_ssfp(tmp_path):
    workflow = _workflow(tmp_path)
    reference = _image(tmp_path / "ssfp.nii.gz", shape=(3, 3, 3, 2))
    f0 = _image(tmp_path / "f0.nii.gz", shape=(4, 3, 3))

    with pytest.raises(ValueError, match="must match the SSFP spatial grid"):
        workflow._resolve_mcdespot_f0(
            {"fix-f0": True, "f0": f0},
            {},
            subject="001",
            session=None,
            reference_file=reference,
        )


def test_newly_computed_despot2fm_f0_is_connected_to_mcdespot(
    tmp_path, monkeypatch
):
    modeling = RelaxometryModelingConfig(
        despot1={"enabled": True},
        despot2fm={"enabled": True},
        mcdespot={"enabled": True, "fix-f0": True},
    )
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "derivatives",
    )
    workflow = RelaxometryWorkflow(
        config,
        logging.getLogger("mcdespot-f0-integration"),
        {},
        RelaxometryConfig(modeling=modeling),
    )
    spgr = _image(tmp_path / "spgr.nii.gz", shape=(3, 3, 3, 2))
    ssfp = _image(tmp_path / "ssfp.nii.gz", shape=(3, 3, 3, 4))
    captured = {}

    def fit_despot2fm(**kwargs):
        return {
            "f0": _image(
                kwargs["out_dir"] / "sub-001_model-DESPOT2FM_F0.nii.gz"
            )
        }

    def fit_mcdespot(**kwargs):
        captured.update(kwargs)
        return {"vfm": _image(kwargs["out_dir"] / "sub-001_model-mcDESPOT_VFm.nii.gz")}

    fitters = {"DESPOT2FM": fit_despot2fm, "mcDESPOT": fit_mcdespot}
    monkeypatch.setattr(
        relax,
        "DESPOT_SPECS",
        tuple(
            replace(spec, fit_fn=fitters.get(spec.name, spec.fit_fn))
            for spec in relax.DESPOT_SPECS
        ),
    )

    workflow._fit_downstream_despot_models(
        modeling,
        spgr,
        ssfp,
        tmp_path / "b1.nii.gz",
        {"t1": tmp_path / "t1.nii.gz", "b1": tmp_path / "b1.nii.gz"},
        tmp_path / "params.json",
        tmp_path / "models",
        None,
        "sub-001",
        False,
        {},
        {},
        {"subject": "001", "session": None},
    )

    assert captured["f0_file"] == (
        tmp_path / "models" / "sub-001_model-DESPOT2FM_F0.nii.gz"
    )
    assert captured["extra_options"]["fix-f0"] is True
    assert "f0" not in captured["extra_options"]
