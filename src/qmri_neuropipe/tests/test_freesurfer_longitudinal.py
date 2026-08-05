import logging
from pathlib import Path

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.core.utils import resolve_freesurfer_subjects_dir
from qmri_neuropipe.interfaces import freesurfer
from qmri_neuropipe.lib.anat.recon import ReconAllStep


def _config(tmp_path: Path, longitudinal: dict) -> PipelineConfig:
    return PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
        n_cpus=6,
        config_data={
            "anat": {
                "preprocessing": {
                    "recon_all": {
                        "enabled": True,
                        "method": "standard",
                        "subjects_dir": str(tmp_path / "subjects"),
                        "longitudinal": longitudinal,
                    }
                }
            }
        },
    )


def _complete(subject_dir: Path) -> None:
    for path in ReconAllStep.critical_outputs(subject_dir):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_default_subjects_dir_uses_writable_output_root(tmp_path, monkeypatch):
    config = PipelineConfig(
        bids_dir=tmp_path / "read-only-bids",
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        config_data={
            "anat": {
                "preprocessing": {
                    "recon_all": {"enabled": True, "method": "standard"}
                }
            }
        },
    )
    current = ImageFile(
        {"sub": "01", "ses": "01", "suffix": "T1w"},
        config.bids_dir / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz",
    )
    current.img.parent.mkdir(parents=True)
    current.img.touch()
    calls = []

    def fake_cross(**kwargs):
        calls.append(kwargs)
        _complete(Path(kwargs["subjects_dir"]) / kwargs["subject_id"])

    monkeypatch.setattr(freesurfer, "recon_all", fake_cross)

    step = ReconAllStep(config, logging.getLogger("test-fs-output-root"))
    result = step.run(
        {"subject": "01", "session": "01", "t1w_files": [current]},
        config.work_dir,
    )

    expected = config.output_dir / "freesurfer"
    assert resolve_freesurfer_subjects_dir(config) == expected
    assert calls[0]["subjects_dir"] == expected
    assert result["freesurfer_dir"] == expected / "sub-01_ses-01"


def test_explicit_subjects_dir_is_preserved(tmp_path):
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "output",
    )
    explicit = tmp_path / "shared-freesurfer"
    assert resolve_freesurfer_subjects_dir(config, explicit) == explicit


def test_longitudinal_interface_builds_base_and_long_commands(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(freesurfer, "run_cmd", lambda cmd, **kwargs: commands.append((cmd, kwargs)))

    freesurfer.recon_all_base(
        ["sub-01_ses-01", "sub-01_ses-02"],
        "sub-01_base",
        tmp_path / "subjects dir",
        openmp=4,
    )
    freesurfer.recon_all_longitudinal(
        "sub-01_ses-01",
        "sub-01_base",
        tmp_path / "subjects dir",
        openmp=4,
    )

    assert commands[0][0] == (
        "recon-all -base sub-01_base -tp sub-01_ses-01 -tp sub-01_ses-02 "
        f"-sd '{tmp_path / 'subjects dir'}' -all -openmp 4"
    )
    assert commands[0][1]["label"] == "recon-all-base"
    assert commands[1][0] == (
        "recon-all -long sub-01_ses-01 sub-01_base "
        f"-sd '{tmp_path / 'subjects dir'}' -all -openmp 4"
    )
    assert commands[1][1]["label"] == "recon-all-long"


def test_configured_sessions_resolve_bids_inputs_and_current_preprocessed_input(tmp_path):
    for session in ("baseline", "followup"):
        anat_dir = tmp_path / "bids" / "sub-01" / f"ses-{session}" / "anat"
        anat_dir.mkdir(parents=True)
        (anat_dir / f"sub-01_ses-{session}_T1w.nii.gz").touch()

    step = ReconAllStep(
        _config(
            tmp_path,
            {"enabled": True, "timepoints": ["baseline", "ses-followup"]},
        ),
        logging.getLogger("test-fs-long"),
    )
    current = ImageFile(
        {"sub": "01", "ses": "followup", "suffix": "T1w"},
        tmp_path / "preprocessed-followup.nii.gz",
    )

    specs = step._timepoint_specs("01", "followup", current)

    assert [spec["id"] for spec in specs] == [
        "sub-01_ses-baseline",
        "sub-01_ses-followup",
    ]
    assert specs[0]["input"].img.name == "sub-01_ses-baseline_T1w.nii.gz"
    assert specs[1]["input"] is current


def test_longitudinal_orchestration_runs_cross_base_then_long(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        {"enabled": True, "timepoints": ["01", "02"], "base_id": "sub-01_template"},
    )
    for session in ("01", "02"):
        anat_dir = config.bids_dir / "sub-01" / f"ses-{session}" / "anat"
        anat_dir.mkdir(parents=True)
        (anat_dir / f"sub-01_ses-{session}_T1w.nii.gz").touch()

    calls = []

    def fake_cross(**kwargs):
        calls.append(("cross", kwargs["subject_id"]))
        _complete(Path(kwargs["subjects_dir"]) / kwargs["subject_id"])

    def fake_base(**kwargs):
        calls.append(("base", tuple(kwargs["timepoint_ids"])))
        _complete(Path(kwargs["subjects_dir"]) / kwargs["base_id"])

    def fake_long(**kwargs):
        calls.append(("long", kwargs["timepoint_id"]))
        long_id = ReconAllStep.longitudinal_subject_id(
            kwargs["timepoint_id"], kwargs["base_id"]
        )
        _complete(Path(kwargs["subjects_dir"]) / long_id)

    monkeypatch.setattr(freesurfer, "recon_all", fake_cross)
    monkeypatch.setattr(freesurfer, "recon_all_base", fake_base)
    monkeypatch.setattr(freesurfer, "recon_all_longitudinal", fake_long)

    step = ReconAllStep(config, logging.getLogger("test-fs-long"))
    current = ImageFile(
        {"sub": "01", "ses": "01", "suffix": "T1w"},
        config.bids_dir / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz",
    )
    long_dir, cross_dir, base_dir, timepoints = step._run_longitudinal(
        subject="01",
        session="01",
        current_input=current,
        subjects_dir=tmp_path / "subjects",
        n_threads=6,
        force=False,
    )

    assert calls == [
        ("cross", "sub-01_ses-01"),
        ("cross", "sub-01_ses-02"),
        ("base", ("sub-01_ses-01", "sub-01_ses-02")),
        ("long", "sub-01_ses-01"),
        ("long", "sub-01_ses-02"),
    ]
    assert timepoints == ["sub-01_ses-01", "sub-01_ses-02"]
    assert cross_dir.name == "sub-01_ses-01"
    assert base_dir.name == "sub-01_template"
    assert long_dir.name == "sub-01_ses-01.long.sub-01_template"

    # A completed stream is fully resumable and performs no new commands.
    calls.clear()
    step._run_longitudinal(
        subject="01",
        session="01",
        current_input=current,
        subjects_dir=tmp_path / "subjects",
        n_threads=6,
        force=False,
    )
    assert calls == []


def test_longitudinal_base_id_expands_subject_placeholder():
    assert ReconAllStep.longitudinal_base_id("07") == "sub-07_base"
    assert (
        ReconAllStep.longitudinal_base_id("07", "study-{subject}-template")
        == "study-07-template"
    )
