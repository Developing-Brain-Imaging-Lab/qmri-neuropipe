import logging
from pathlib import Path
from types import SimpleNamespace

from qmri_neuropipe.interfaces import ants, freesurfer
from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep


def test_synb0_falls_back_to_ants_when_freesurfer_minc_is_missing(
    tmp_path: Path, monkeypatch
):
    step = Synb0EstimationStep.__new__(Synb0EstimationStep)
    step.logger = logging.getLogger(__name__)
    step.config = SimpleNamespace(n_cpus=6)
    t1w_mgz = tmp_path / "t1w.mgz"
    t1w_mgz.touch()
    calls = []

    def fail_nu_correct(**kwargs):
        Path(kwargs["out_file"]).touch()
        raise RuntimeError("nu_correct: Command not found.")

    def record_convert(**kwargs):
        calls.append(("convert", kwargs["in_file"], kwargs["out_file"]))

    def record_n4(**kwargs):
        calls.append(
            ("n4", kwargs["in_file"], kwargs["out_file"], kwargs["nthreads"])
        )

    monkeypatch.setattr(freesurfer, "mri_nu_correct", fail_nu_correct)
    monkeypatch.setattr(freesurfer, "mri_convert", record_convert)
    monkeypatch.setattr(ants, "n4bias", record_n4)

    result = step._bias_correct_t1(t1w_mgz, tmp_path)

    assert result == tmp_path / "t1w_n3.mgz"
    assert not result.exists()
    assert calls == [
        (
            "convert",
            t1w_mgz,
            tmp_path / "t1w_n4_input.nii.gz",
        ),
        (
            "n4",
            tmp_path / "t1w_n4_input.nii.gz",
            tmp_path / "t1w_n4.nii.gz",
            6,
        ),
        (
            "convert",
            tmp_path / "t1w_n4.nii.gz",
            tmp_path / "t1w_n3.mgz",
        ),
    ]


def test_synb0_uses_bias_corrected_t1_when_normalization_fails(
    tmp_path: Path, monkeypatch
):
    step = Synb0EstimationStep.__new__(Synb0EstimationStep)
    step.logger = logging.getLogger(__name__)
    t1w_n3 = tmp_path / "t1w_n3.mgz"
    t1w_n3.touch()
    calls = []

    def fail_normalize(**kwargs):
        Path(kwargs["out_file"]).touch()
        raise RuntimeError("could not find enough control points")

    def record_convert(**kwargs):
        calls.append((kwargs["in_file"], kwargs["out_file"]))

    monkeypatch.setattr(freesurfer, "mri_normalize", fail_normalize)
    monkeypatch.setattr(freesurfer, "mri_convert", record_convert)

    result = step._normalize_t1(t1w_n3, tmp_path)

    assert result == tmp_path / "t1w_norm.mgz"
    assert not result.exists()
    assert calls == [(t1w_n3, tmp_path / "t1w_norm.mgz")]
