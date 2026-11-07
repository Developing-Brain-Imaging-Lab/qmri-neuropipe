"""Contract tests for external anatomy execution; no models/GPU required."""
import json
import logging
from pathlib import Path
import shlex

import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.interfaces import deep_anatomical as interface
from qmri_neuropipe.lib.anat import deep_anatomical as module
from qmri_neuropipe.workflows.pipelines.anat import AnatPreprocessingWorkflow
from qmri_neuropipe.workflows.pipelines.anat_config import parse_anatomical_config


def setup(tmp_path, backend="fastsurfer", **settings):
    t1 = tmp_path / "input with spaces" / "sub-01_ses-02_T1w.nii.gz"
    t1.parent.mkdir(exist_ok=True)
    t1.write_bytes(b"input")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    settings = {"enabled": True, "fs_license": str(license_file), **settings}
    config = PipelineConfig(bids_dir=tmp_path / "bids", output_dir=tmp_path / "out",
                            config_data={"anat": {backend: settings}})
    step = {"fastsurfer": module.FastSurferStep, "deepprep": module.DeepPrepStep}[backend](config)
    context = {"subject": "01", "session": "02", "t1w_files": [ImageFile(img=t1, entities={"sub": "01"})]}
    return step, context


def products(step, context):
    subject_dir = step._paths(context)[3]
    for path in step._products(subject_dir).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"output")


@pytest.mark.parametrize("backend,mode", [("fastsurfer", "segmentation"), ("fastsurfer", "full"), ("deepprep", "full")])
def test_execution_completion_cache_and_input_changes(tmp_path, monkeypatch, backend, mode):
    step, context = setup(tmp_path, backend, mode=mode)
    calls = []
    def execute(args, settings, mounts, **kwargs):
        calls.append(args)
        products(step, context)
    monkeypatch.setattr(module, "execute_anatomical", execute)
    result = step.run(context, tmp_path / "work")
    assert step.is_complete(context)
    assert result[f"{backend}_subject_dir"].is_dir()
    step.run(context, tmp_path / "work")
    assert len(calls) == 1
    context["t1w_files"][0].img.write_bytes(b"changed input")
    assert not step.is_complete(context)
    step.run(context, tmp_path / "work")
    assert len(calls) == 2
    if backend == "fastsurfer":
        assert ("--seg_only" in calls[0]) == (mode == "segmentation")
    else:
        assert "--anat_only" in calls[0]
        stage = Path(calls[0][1])
        assert list(stage.glob("sub-*/anat/*T1w.nii.gz"))
        assert not list(stage.rglob("ses-*"))
        assert json.loads((stage / "dataset_description.json").read_text())["BIDSVersion"]
        assert step._paths({**context, "session": "03"})[2] != step._paths(context)[2]


def test_missing_products_and_failed_force_invalidate_cache(tmp_path, monkeypatch):
    step, context = setup(tmp_path)
    monkeypatch.setattr(module, "execute_anatomical", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="required outputs"):
        step.run(context, tmp_path / "work")
    assert not step.is_complete(context)
    monkeypatch.setattr(module, "execute_anatomical", lambda *a, **k: products(step, context))
    step.run(context, tmp_path / "work")
    assert step.is_complete(context)
    def fail(*a, **k):
        raise RuntimeError("tool failed")
    monkeypatch.setattr(module, "execute_anatomical", fail)
    with pytest.raises(RuntimeError, match="tool failed"):
        step.run(context, tmp_path / "work", force=True)
    assert not step.is_complete(context)


def test_incomplete_surface_does_not_skip(tmp_path, monkeypatch):
    step, context = setup(tmp_path)
    monkeypatch.setattr(module, "execute_anatomical", lambda *a, **k: products(step, context))
    step.run(context, tmp_path / "work")
    step._products(step._paths(context)[3])["rh_pial"].unlink()
    assert not step.is_complete(context)


def test_segmentation_needs_no_license_but_full_does(tmp_path, monkeypatch):
    step, context = setup(tmp_path, mode="segmentation", fs_license=None)
    monkeypatch.setattr(module, "execute_anatomical", lambda *a, **k: products(step, context))
    step.run(context, tmp_path / "work")
    step.settings["mode"] = "full"
    with pytest.raises(ValueError, match="fs_license"):
        step.run(context, tmp_path / "work")


def test_requires_t1_and_unambiguous_selection(tmp_path):
    step, context = setup(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        step.run({**context, "t1w_files": []}, tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        step.run({**context, "t1w_files": context["t1w_files"] * 2}, tmp_path)


@pytest.mark.parametrize("runtime", ["native", "docker", "apptainer", "singularity"])
def test_command_quoting_and_gpu_mounts(tmp_path, monkeypatch, runtime):
    captured = []
    monkeypatch.setattr(interface.shutil, "which", lambda x: x)
    monkeypatch.setattr(interface, "run_cmd", lambda command, **kw: captured.append(shlex.split(command)))
    settings = {"runtime": runtime, "image": "tool:version"}
    args = ["tool", "--input", str(tmp_path / "file with spaces.nii.gz")]
    mount = tmp_path / "input with spaces"
    interface.execute_anatomical(args, settings, [(mount, True)], label="tool", gpu=True)
    command = captured[0]
    assert command[-2:] == args[-2:]
    if runtime != "native":
        assert f"{mount}:{mount}:ro" in command
        assert ("--gpus" if runtime == "docker" else "--nv") in command


def test_native_missing_dependency_is_actionable(monkeypatch):
    monkeypatch.setattr(interface.shutil, "which", lambda x: None)
    with pytest.raises(FileNotFoundError, match="install fastsurfer"):
        interface.execute_anatomical(["run_fastsurfer.sh"], {}, [], label="fastsurfer")


@pytest.mark.parametrize("settings", [{"mode": "typo"}, {"input": "normalized"}, {"runtime": "unknown"}, {"runtime": "docker"}])
def test_config_validation(settings):
    with pytest.raises(ValueError):
        parse_anatomical_config({"anat": {"fastsurfer": {"enabled": True, **settings}}})


def test_workflow_dedicated_stage_and_error_policy(tmp_path, monkeypatch):
    step, context = setup(tmp_path)
    workflow = AnatPreprocessingWorkflow(step.config, logging.getLogger(__name__), None)
    deep = next(s for s in workflow.steps if isinstance(s, module.FastSurferStep))
    calls = []
    monkeypatch.setattr(deep, "run", lambda context, **kw: calls.append(kw) or {**context, "fastsurfer_subject_dir": tmp_path})
    monkeypatch.setattr(workflow, "_run_one_anat_step", lambda **kw: pytest.fail("external stage ran per-image"))
    workflow._preprocess_t1w(tmp_path, context, None, None, tmp_path, [])
    assert calls == []
    result, _ = workflow._run_deep_anatomical(tmp_path, context, None, [])
    assert result["fastsurfer_subject_dir"] == tmp_path
    def fail(*a, **kw):
        raise RuntimeError("failure")
    monkeypatch.setattr(deep, "run", fail)
    result, _ = workflow._run_deep_anatomical(tmp_path, context, None, [])
    assert "fastsurfer failed" in result["errors"][-1]
    workflow.config.stop_on_error = True
    with pytest.raises(Exception, match="failure"):
        workflow._run_deep_anatomical(tmp_path, context, None, [])


def test_cached_anatomy_requires_backend_completion_and_restores_outputs(tmp_path, monkeypatch):
    step, context = setup(tmp_path)
    workflow = AnatPreprocessingWorkflow(step.config, logging.getLogger(__name__), None)
    deep = next(s for s in workflow.steps if isinstance(s, module.FastSurferStep))
    cached = {"preprocessed_t1w": context["t1w_files"][0]}
    monkeypatch.setattr(workflow, "_load_preprocessed_from_output", lambda *a: cached)
    monkeypatch.setattr(workflow, "_recon_all_finished", lambda *a: True)
    monkeypatch.setattr(deep, "is_complete", lambda ctx: True)
    result = workflow.run(tmp_path / "work", context, tmp_path / "final")
    assert result["anat_preprocessing_skipped"]
    assert "fastsurfer_outputs" in result
    monkeypatch.setattr(deep, "is_complete", lambda ctx: False)
    def preprocess(*args):
        raise RuntimeError("cache bypassed")
    monkeypatch.setattr(workflow, "_preprocess_t1w", preprocess)
    with pytest.raises(RuntimeError, match="cache bypassed"):
        workflow.run(tmp_path / "work", context, tmp_path / "final")


def test_rerun_target_forces_backend_and_later_backend(tmp_path, monkeypatch):
    step, context = setup(tmp_path)
    step.config.config_data["anat"]["deepprep"] = {"enabled": True}
    workflow = AnatPreprocessingWorkflow(step.config, logging.getLogger(__name__), None)
    workflow._anat_rerun_from_step = "fastsurfer"
    calls = []
    for deep in workflow.steps:
        if isinstance(deep, module.DeepAnatomicalStep):
            monkeypatch.setattr(deep, "run", lambda ctx, **kw: calls.append(kw["force"]) or ctx)
    workflow._run_deep_anatomical(tmp_path, context, None, [])
    assert calls == [True, True]


def test_changed_input_starts_clean_and_preserves_previous(tmp_path, monkeypatch):
    step, context = setup(tmp_path)
    def execute(*a, **k):
        assert not (step._paths(context)[3] / "surf/lh.white").exists()
        products(step, context)
    monkeypatch.setattr(module, "execute_anatomical", execute)
    step.run(context, tmp_path / "work")
    context["t1w_files"][0].img.write_bytes(b"different")
    step.run(context, tmp_path / "work")
    assert list(step._paths(context)[2].glob("sub-01_ses-02.previous-*"))
