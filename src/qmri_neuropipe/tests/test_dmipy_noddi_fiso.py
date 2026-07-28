from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from qmri_neuropipe import tools
from qmri_neuropipe.interfaces import dmipy
from qmri_neuropipe.interfaces.dmipy_backend import (
    DmipyFitExecution,
    DmipyRuntime,
)


def _save_image(path, data, affine=None):
    nib.save(
        nib.Nifti1Image(
            np.asarray(data, dtype=np.float32),
            np.eye(4) if affine is None else affine,
        ),
        path,
    )
    return path


def test_external_fiso_requires_dwi_grid_and_bounded_finite_values(tmp_path):
    dwi = nib.Nifti1Image(np.ones((2, 2, 2, 3)), np.eye(4))
    valid = _save_image(tmp_path / "fiso.nii.gz", np.full((2, 2, 2), 0.25))

    values = dmipy._load_external_fiso(valid, dwi, np.ones((2, 2, 2), dtype=bool))
    assert values == pytest.approx(np.full(8, 0.25))

    wrong_affine = _save_image(
        tmp_path / "wrong-affine.nii.gz",
        np.full((2, 2, 2), 0.25),
        np.diag([2.0, 2.0, 2.0, 1.0]),
    )
    with pytest.raises(ValueError, match="affine"):
        dmipy._load_external_fiso(wrong_affine, dwi, None)

    invalid = np.full((2, 2, 2), 0.25)
    invalid[0, 0, 0] = 1.1
    invalid_path = _save_image(tmp_path / "invalid.nii.gz", invalid)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        dmipy._load_external_fiso(invalid_path, dwi, None)

    invalid[0, 0, 0] = np.nan
    nonfinite_path = _save_image(tmp_path / "nonfinite.nii.gz", invalid)
    with pytest.raises(ValueError, match="finite"):
        dmipy._load_external_fiso(nonfinite_path, dwi, None)


def test_jax_chunk_fixes_voxelwise_fiso_before_fitting(monkeypatch):
    captured = {}
    runtime = DmipyRuntime(
        version="2.1.0",
        solver="jax",
        requested_device="auto",
        backend="cpu",
    )

    def fake_build(config, fixed_params):
        captured["fixed_params"] = fixed_params
        return "noddi-model"

    def fake_execute(request):
        captured["request"] = request
        fitted = SimpleNamespace(
            fitted_parameters={"partial_volume_1": np.asarray([0.2, 0.4])}
        )
        return DmipyFitExecution(
            fitted=fitted,
            runtime=runtime,
            model_name="noddi",
            voxel_count=2,
            used_gradient_nonlinearity=False,
        )

    monkeypatch.setattr(dmipy, "_build_noddi_model", fake_build)
    monkeypatch.setattr(dmipy, "execute_dmipy_fit", fake_execute)
    monkeypatch.setattr(
        dmipy.DmipyRuntime,
        "resolve",
        classmethod(lambda cls, *args, **kwargs: runtime),
    )
    monkeypatch.setattr(
        dmipy, "install_dmipy_jax_postprocessing_workaround", lambda: False
    )

    result = dmipy._fit_chunk(
        (
            0,
            np.ones((2, 4)),
            "scheme",
            {"model_type": "standard"},
            {"partial_volume_1": np.asarray([0.2, 0.4])},
            None,
            "jax",
            {"batch_size": 2},
        )
    )

    assert captured["fixed_params"]["partial_volume_1"] == pytest.approx([0.2, 0.4])
    assert captured["request"].runtime.uses_jax
    assert captured["request"].solver_options == {"batch_size": 2}
    assert result["partial_volume_1"] == pytest.approx([0.2, 0.4])


def test_fit_noddi_cli_forwards_external_fiso_to_dmipy(tmp_path, monkeypatch):
    dwi = _save_image(tmp_path / "sub-01_dwi.nii.gz", np.ones((1, 1, 1, 2)))
    fiso = _save_image(tmp_path / "sub-01_FISO.nii.gz", np.full((1, 1, 1), 0.2))
    bval = tmp_path / "sub-01_dwi.bval"
    bvec = tmp_path / "sub-01_dwi.bvec"
    bval.write_text("0 1000\n")
    bvec.write_text("0 1\n0 0\n0 0\n")
    captured = {}

    def fake_fit(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(dmipy, "fit_noddi", fake_fit)
    result = CliRunner().invoke(
        tools.app,
        [
            "fit-noddi",
            "--input",
            str(dwi),
            "--output-dir",
            str(tmp_path / "out"),
            "--bval",
            str(bval),
            "--bvec",
            str(bvec),
            "--solver",
            "jax",
            "--fiso",
            str(fiso),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["solver"] == "jax"
    assert captured["fiso_file"] == fiso
