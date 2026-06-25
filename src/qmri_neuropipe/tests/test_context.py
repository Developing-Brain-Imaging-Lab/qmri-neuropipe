"""R12 tests for typed pipeline context compatibility and return contracts."""

import logging
import pickle

import pytest

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.context import PipelineContext
from qmri_neuropipe.core.types import ImageFile


def test_known_fields_are_dictionary_views_without_eager_membership():
    context = PipelineContext({"subject": "01"})

    assert isinstance(context, dict)
    assert context.subject == context["subject"] == "01"
    assert context.session is None
    assert "session" not in context

    context.session = "A"
    assert context["session"] == "A"
    assert "session" in context

    context["current_image"] = "image"
    assert context.current_image == "image"


def test_standard_mapping_operations_remain_compatible():
    context = PipelineContext(subject="01")

    assert context.setdefault("errors", []) == []
    context["errors"].append("problem")
    context.update({"custom": 3, "models_fitted": ["DESPOT1"]})

    copied = context.copy()
    assert isinstance(copied, PipelineContext)
    assert dict(copied) == dict(context)
    assert context.pop("custom") == 3
    assert "custom" not in context


def test_extra_is_a_live_view_of_unknown_keys():
    context = PipelineContext({"subject": "01", "custom_metric": 4})

    assert dict(context.extra) == {"custom_metric": 4}
    context.extra["vendor_value"] = "x"
    assert context["vendor_value"] == "x"

    context["another_value"] = 7
    assert context.extra["another_value"] == 7
    with pytest.raises(KeyError, match="typed context field"):
        context.extra["subject"] = "02"


def test_ensure_and_pickle_preserve_context_type_and_unknown_keys():
    context = PipelineContext.ensure(
        {"subject": "01", "unknown": {"nested": True}}
    )
    assert PipelineContext.ensure(context) is context

    restored = pickle.loads(pickle.dumps(context))
    assert isinstance(restored, PipelineContext)
    assert restored.subject == "01"
    assert restored.extra["unknown"] == {"nested": True}


def test_relaxometry_result_contract_returns_context_directly(tmp_path):
    from qmri_neuropipe.workflows.pipelines.relaxometry import (
        RelaxometryWorkflow,
    )

    context = PipelineContext(
        {
            "subject": "01",
            "normalized_results": {"T1": tmp_path / "norm.nii.gz"},
        }
    )
    reference = ImageFile(
        {"sub": "01", "suffix": "VFA"},
        tmp_path / "reference.nii.gz",
    )

    result = RelaxometryWorkflow._compose_run_context(
        context,
        {"T1": reference},
        {"DESPOT1": {"T1": reference.img}},
        {"atlas": tmp_path / "stats.csv"},
        None,
        reference,
    )

    assert result is context
    assert isinstance(result, PipelineContext)
    assert "context" not in result
    assert result.fitted_maps == {"T1": reference}
    assert result.modeling_results == {"DESPOT1": {"T1": reference.img}}
    assert result.reference_image is reference


def test_anatomical_workflow_returns_typed_context(tmp_path, monkeypatch):
    from qmri_neuropipe.workflows.pipelines.anat import (
        AnatPreprocessingWorkflow,
    )

    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        skip_existing=False,
    )
    workflow = AnatPreprocessingWorkflow(
        config,
        logging.getLogger("context-test"),
        None,
    )
    workflow.steps = []
    passthrough = lambda output, context, *args, **kwargs: (context, args[-1])
    for method in (
        "_run_coregistration",
        "_run_brain_masking",
        "_run_normalization",
        "_run_freesurfer",
        "_run_super_synth",
        "_run_segmentation",
    ):
        monkeypatch.setattr(workflow, method, passthrough)
    monkeypatch.setattr(workflow, "save_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        workflow,
        "_handle_reporting_and_outputs",
        lambda *args, **kwargs: None,
    )

    result = workflow.run(tmp_path / "work", {"subject": "01"})

    assert isinstance(result, PipelineContext)
    assert result.subject == "01"


def test_dmri_initial_context_and_noop_subworkflows_are_typed(
    tmp_path,
    monkeypatch,
):
    from qmri_neuropipe.workflows.pipelines import dmri
    from qmri_neuropipe.workflows.pipelines.normalization_workflow import (
        NormalizationWorkflow,
    )
    from qmri_neuropipe.workflows.pipelines.segmentation_workflow import (
        SegmentationWorkflow,
    )

    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
    )
    stub = type("Stub", (), {"config": config})()
    dwi = ImageFile({"sub": "01", "suffix": "dwi"}, tmp_path / "dwi.nii.gz")
    monkeypatch.setattr(dmri, "find_reversed_phase_groups", lambda files: [])

    initial = dmri.DMRIPipeline._build_initial_context(
        stub,
        "01",
        None,
        [dwi],
        [],
        [],
    )
    assert isinstance(initial, PipelineContext)
    assert initial.current_image is dwi

    for workflow_cls in (NormalizationWorkflow, SegmentationWorkflow):
        workflow = workflow_cls(config, logging.getLogger("context-test"), None)
        result = workflow.run(tmp_path / "work", {"subject": "01"})
        assert isinstance(result, PipelineContext)
        assert result.subject == "01"
