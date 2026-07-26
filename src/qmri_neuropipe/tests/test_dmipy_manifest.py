import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.interfaces.dmipy_backend import DmipyRuntime
from qmri_neuropipe.interfaces.dmipy_generic import dmipy_run_spec_for_fit
from qmri_neuropipe.interfaces.dmipy_manifest import (
    MANIFEST_FILENAME,
    build_dmipy_run_spec,
    completed_outputs,
    invalidate_completion_manifest,
    write_completion_manifest,
)
from qmri_neuropipe.lib.dmri.fitting import DmipyModelFittingStep


def _write(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _run_spec(tmp_path, *, solver="jax", solver_options=None):
    return build_dmipy_run_spec(
        model_name="ball",
        in_file=_write(tmp_path / "dwi.nii.gz", b"dwi-input"),
        bval_file=_write(tmp_path / "dwi.bval", b"0 1000"),
        bvec_file=_write(tmp_path / "dwi.bvec", b"0 1\n0 0\n0 0"),
        mask_file=_write(tmp_path / "mask.nii.gz", b"mask-input"),
        grad_nonlin=None,
        delta_file=None,
        Delta_file=None,
        TE_file=None,
        solver=solver,
        device="cpu" if solver == "jax" else "auto",
        solver_options=solver_options or {"Ns": 5},
        factory_options={},
    )


def _completed_run(tmp_path, run_spec):
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    image = _write(out_dir / "sub-01_model-Ball_MD.nii.gz", b"x" * 128)
    sidecar = _write(
        out_dir / "sub-01_model-Ball_MD.json",
        json.dumps({"Parameter": "lambda_iso"}).encode(),
    )
    runtime = DmipyRuntime(
        version="2.1.0",
        solver=run_spec["solver"],
        requested_device=run_spec["device"],
        backend="cpu" if run_spec["solver"] == "jax" else "native-cpu",
    )
    write_completion_manifest(
        out_dir,
        run_spec=run_spec,
        outputs={"lambda_iso": image},
        runtime=runtime,
    )
    return out_dir, image, sidecar


def test_completion_manifest_reuses_only_declared_valid_outputs(tmp_path):
    run_spec = _run_spec(tmp_path)
    out_dir, image, _ = _completed_run(tmp_path, run_spec)

    outputs = completed_outputs(
        out_dir,
        run_spec=run_spec,
        validate_image=lambda path: path.stat().st_size >= 100,
    )

    assert outputs == {"lambda_iso": image.resolve()}
    manifest = json.loads((out_dir / MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "complete"
    assert manifest["request"]["solver_options"] == {"Ns": 5}


def test_changed_solver_options_or_inputs_invalidate_completion(tmp_path):
    run_spec = _run_spec(tmp_path)
    out_dir, _, _ = _completed_run(tmp_path, run_spec)

    changed_options = {
        **run_spec,
        "solver_options": {"Ns": 9},
    }
    assert completed_outputs(out_dir, run_spec=changed_options) is None

    (tmp_path / "dwi.bval").write_text("0 2000")
    changed_input = _run_spec(tmp_path)
    assert completed_outputs(out_dir, run_spec=changed_input) is None


def test_missing_or_modified_derivative_invalidates_completion(tmp_path):
    run_spec = _run_spec(tmp_path)
    out_dir, image, sidecar = _completed_run(tmp_path, run_spec)

    image.write_bytes(b"changed-size")
    assert completed_outputs(out_dir, run_spec=run_spec) is None

    image.write_bytes(b"x" * 128)
    sidecar.unlink()
    assert completed_outputs(out_dir, run_spec=run_spec) is None


def test_incomplete_declared_parameter_set_invalidates_completion(tmp_path):
    run_spec = _run_spec(tmp_path)
    out_dir, _, _ = _completed_run(tmp_path, run_spec)
    path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(path.read_text())
    manifest["expected_parameters"].append("missing_parameter")
    path.write_text(json.dumps(manifest))

    assert completed_outputs(out_dir, run_spec=run_spec) is None


def test_invalidation_removes_only_completion_marker(tmp_path):
    run_spec = _run_spec(tmp_path)
    out_dir, image, _ = _completed_run(tmp_path, run_spec)

    invalidate_completion_manifest(out_dir)

    assert not (out_dir / MANIFEST_FILENAME).exists()
    assert image.exists()


def test_pipeline_skips_only_matching_completed_request(tmp_path, monkeypatch):
    dwi_path = tmp_path / "sub-01_dwi.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2)), np.eye(4)), dwi_path)
    bval = _write(tmp_path / "sub-01_dwi.bval", b"0 1000")
    bvec = _write(tmp_path / "sub-01_dwi.bvec", b"0 1\n0 0\n0 0")
    dwi = DWIFile({"sub": "01"}, dwi_path, bval=bval, bvec=bvec)
    config = PipelineConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "derivatives",
        n_cpus=1,
    )
    step = DmipyModelFittingStep(
        config,
        logging.getLogger("dmipy-manifest-test"),
        None,
        model_name="ball",
        solver_options={"Ns": 5},
    )
    output_root = tmp_path / "modeling"
    model_out = output_root / "dmipy-ball"
    model_out.mkdir(parents=True)
    image = model_out / "sub-01_model-Ball_MD.nii.gz"
    nib.save(
        nib.Nifti1Image(np.arange(1000, dtype=float).reshape(10, 10, 10), np.eye(4)),
        image,
    )
    _write(
        model_out / "sub-01_model-Ball_MD.json",
        json.dumps({"Parameter": "lambda_iso"}).encode(),
    )
    run_spec = dmipy_run_spec_for_fit(
        dwi,
        model_name="ball",
        solver_kwargs={"Ns": 5},
    )
    runtime = DmipyRuntime(
        "2.1.0", "brute2fine", "auto", "native-cpu"
    )
    write_completion_manifest(
        model_out,
        run_spec=run_spec,
        outputs={"lambda_iso": image},
        runtime=runtime,
    )

    from qmri_neuropipe.interfaces import dmipy_generic

    called = []
    monkeypatch.setattr(
        dmipy_generic,
        "fit_dmipy_reference",
        lambda *args, **kwargs: called.append(kwargs) or {},
    )
    context = {"current_image": dwi}
    result = step.run(context, output_root)

    assert called == []
    assert result["modeling_results"]["dmipy:ball"] == {
        "lambda_iso": image.resolve()
    }

    changed_step = DmipyModelFittingStep(
        config,
        logging.getLogger("dmipy-manifest-test"),
        None,
        model_name="ball",
        solver_options={"Ns": 9},
    )
    changed_step.run({"current_image": dwi}, output_root)
    assert called[0]["solver_kwargs"] == {"Ns": 9}
