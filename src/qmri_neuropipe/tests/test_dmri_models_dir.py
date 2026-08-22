import logging
from types import SimpleNamespace
from unittest.mock import Mock

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile, ImageFile
from qmri_neuropipe.workflows.pipelines.dmri import DMRIPipeline


class _ModelingStub:
    def __init__(self):
        self.run_args = None
        self.steps = []

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


def test_modeling_only_reuses_derivatives_and_skips_other_stages(tmp_path, monkeypatch):
    from qmri_neuropipe.workflows.pipelines import dmri

    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        config_data={
            "dmri": {
                "modeling_only": "auto",
                "modeling": {"dti": {"enabled": True}},
                "normalization": {"enabled": True},
                "analysis": {"enabled": True},
            },
            "qc": {"mriqc": {"enabled": True}},
            "anat": {"preprocessing": {"enabled": True}},
            "gratio": {"enabled": True},
        },
    )
    pipeline = _pipeline(config)
    pipeline.preprocessing = SimpleNamespace(steps=[])
    pipeline.normalization = SimpleNamespace(steps=[])
    pipeline.segmentation = SimpleNamespace(steps=[])

    raw = DWIFile(
        img=tmp_path / "sub-01_ses-02_dwi.nii.gz",
        bval=tmp_path / "sub-01_ses-02_dwi.bval",
        bvec=tmp_path / "sub-01_ses-02_dwi.bvec",
        entities={"sub": "01", "ses": "02", "suffix": "dwi"},
    )
    preprocessed = DWIFile(
        img=tmp_path / "sub-01_ses-02_desc-preproc_dwi.nii.gz",
        bval=raw.bval,
        bvec=raw.bvec,
        entities={"sub": "01", "ses": "02", "desc": "preproc", "suffix": "dwi"},
    )
    preprocessed.img.write_bytes(b"preprocessed")
    preprocessed.bval.write_text("0\n")
    preprocessed.bvec.write_text("0 0 0\n")
    anat = ImageFile(
        img=tmp_path / "sub-01_ses-02_T1w.nii.gz",
        entities={"sub": "01", "ses": "02", "suffix": "T1w"},
    )

    class _Reporter:
        def __init__(self, *args, **kwargs):
            pass

        def set_participant_summary(self, *args, **kwargs):
            pass

        def generate(self):
            pass

        def generate_pdf(self):
            pass

    monkeypatch.setattr(dmri, "ReportGenerator", _Reporter)
    monkeypatch.setattr(dmri, "bids_find_dwi", lambda path: [raw])
    monkeypatch.setattr(dmri, "_validate_dwi_gradient_tables", lambda files: None)
    pipeline._find_anat_files = lambda *args: [anat]
    pipeline._find_other_anat_files = lambda *args: []
    pipeline._load_preprocessed_from_output = lambda *args: {
        "preprocessed_dwis": [preprocessed],
        "preprocessed_masks": [None],
        "current_image": preprocessed,
    }
    pipeline._run_mriqc = Mock()
    pipeline._run_anatomical_preprocessing = Mock()
    pipeline._run_preprocessing = Mock(side_effect=AssertionError("preprocessing ran"))
    pipeline._run_modeling = Mock()
    pipeline._run_normalization = Mock()
    pipeline._run_segmentation = Mock()
    pipeline._update_study_tracker = Mock()

    staged = tmp_path / "out" / "sub-01" / "ses-02" / "dwi"
    staged.mkdir(parents=True)
    (staged / "sub-01_ses-02_desc-preproc_dwi.nii.gz").write_bytes(b"staged")

    pipeline.process_subject("01", "02")

    pipeline._run_modeling.assert_called_once()
    pipeline._run_mriqc.assert_not_called()
    pipeline._run_anatomical_preprocessing.assert_not_called()
    pipeline._run_preprocessing.assert_not_called()
    pipeline._run_normalization.assert_not_called()
    pipeline._run_segmentation.assert_not_called()
    pipeline._update_study_tracker.assert_not_called()


def test_modeling_auto_runs_preprocessing_when_derivatives_are_absent(tmp_path, monkeypatch):
    from qmri_neuropipe.workflows.pipelines import dmri

    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        config_data={
            "dmri": {
                "modeling_only": "auto",
                "modeling": {"dti": {"enabled": True}},
                "normalization": {"enabled": True},
                "analysis": {"enabled": True},
            },
            "gratio": {"enabled": True},
        },
    )
    pipeline = _pipeline(config)
    pipeline.preprocessing = SimpleNamespace(steps=[])
    pipeline.normalization = SimpleNamespace(steps=[])
    pipeline.segmentation = SimpleNamespace(steps=[])
    raw = DWIFile(
        img=tmp_path / "sub-01_dwi.nii.gz",
        bval=tmp_path / "sub-01_dwi.bval",
        bvec=tmp_path / "sub-01_dwi.bvec",
        entities={"sub": "01", "suffix": "dwi"},
    )
    reporter = SimpleNamespace(
        set_participant_summary=lambda *args, **kwargs: None,
        generate=lambda: None,
        generate_pdf=lambda: None,
    )
    monkeypatch.setattr(dmri, "ReportGenerator", lambda *args, **kwargs: reporter)
    monkeypatch.setattr(dmri, "bids_find_dwi", lambda path: [raw])
    monkeypatch.setattr(dmri, "_validate_dwi_gradient_tables", lambda files: None)
    pipeline._find_anat_files = lambda *args: []
    pipeline._find_other_anat_files = lambda *args: []
    pipeline._load_preprocessed_from_output = Mock(return_value=None)
    pipeline._run_preprocessing = Mock(side_effect=lambda context, *args: context)
    pipeline._run_modeling = Mock()
    pipeline._run_normalization = Mock()
    pipeline._run_segmentation = Mock()
    pipeline._update_study_tracker = Mock()

    pipeline.process_subject("01", None)

    pipeline._load_preprocessed_from_output.assert_called_once()
    pipeline._run_preprocessing.assert_called_once()
    pipeline._run_modeling.assert_called_once()
    pipeline._run_normalization.assert_not_called()
    pipeline._run_segmentation.assert_not_called()
    pipeline._update_study_tracker.assert_not_called()
