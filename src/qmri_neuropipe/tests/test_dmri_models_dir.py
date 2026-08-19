import logging

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline


class _ModelingStub:
    def __init__(self):
        self.run_args = None

    def build_pipeline(self, context):
        pass

    def run(self, work_dir, context, reporter=None, final_output_dir=None):
        self.run_args = (work_dir, context, reporter, final_output_dir)
        return context


def _pipeline(config):
    pipeline = DMRIPipeline.__new__(DMRIPipeline)
    pipeline.config = config
    pipeline.logger = logging.getLogger(__name__)
    pipeline.modeling = _ModelingStub()
    return pipeline


def test_modeling_uses_separate_models_root_with_bids_hierarchy(tmp_path):
    output_root = tmp_path / "dmri-preproc"
    models_root = tmp_path / "dmri-models"
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=output_root,
        models_dir=models_root,
    )
    pipeline = _pipeline(config)
    subject_output = output_root / "sub-01" / "ses-02" / "dwi"

    context = {"subject": "01", "session": "02"}
    result = pipeline._run_modeling(context, tmp_path / "work", subject_output, None)

    expected = models_root / "sub-01" / "ses-02" / "dwi"
    assert result is context
    assert pipeline.modeling.run_args[-1] == expected
    assert expected.is_dir()


def test_modeling_defaults_to_preprocessing_models_subdirectory(tmp_path):
    output_root = tmp_path / "dmri-preproc"
    config = PipelineConfig(bids_dir=tmp_path / "bids", output_dir=output_root)
    pipeline = _pipeline(config)
    subject_output = output_root / "sub-01" / "dwi"

    pipeline._run_modeling({}, tmp_path / "work", subject_output, None)

    expected = subject_output / "models"
    assert pipeline.modeling.run_args[-1] == expected
    assert expected.is_dir()
