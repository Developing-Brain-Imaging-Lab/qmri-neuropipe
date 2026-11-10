from pathlib import Path
import shlex

import pytest

from qmri_neuropipe.interfaces.relaxometry import (
    _append_cli_options,
    _get_qmri_fit_command,
    fit_despot1,
    fit_despot1_hifi,
    fit_despot2,
    fit_despot2_fm,
    fit_mcdespot,
)
from qmri_neuropipe.workflows.pipelines.relaxometry_config import (
    RelaxometryModelingConfig,
)


def _make_executable(path: Path) -> None:
    path.write_text("#!/usr/bin/env sh\nexit 0\n")
    path.chmod(0o755)


def test_get_qmri_fit_command_prefers_hyphenated_cli(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qmri_fit = bin_dir / "qmri-fit"
    legacy_name = bin_dir / "qmri_fit"
    _make_executable(qmri_fit)
    _make_executable(legacy_name)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert _get_qmri_fit_command("despot1") == [str(qmri_fit), "despot1"]


def test_get_qmri_fit_command_accepts_underscore_cli(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qmri_fit = bin_dir / "qmri_fit"
    _make_executable(qmri_fit)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert _get_qmri_fit_command("mcdespot") == [str(qmri_fit), "mcdespot"]


def test_get_qmri_fit_command_prefers_v4_when_both_are_installed(tmp_path, monkeypatch):
    _make_executable(tmp_path / "qmri")
    _make_executable(tmp_path / "qmri_fit")
    monkeypatch.setenv("PATH", str(tmp_path))

    assert _get_qmri_fit_command("despot1_hifi") == [str(tmp_path / "qmri"), "despot1_hifi"]


@pytest.mark.parametrize(
    "name,help_text,expected",
    [
        ("qmri_cuda", "", ["mcdespot", "--device=gpu"]),
        ("qmr_cuda", "", ["mcdespot", "--device=gpu"]),
        ("qmri_fit_cuda", "mcdespot_cuda    mcDESPOT CUDA GPU", ["mcdespot_cuda"]),
        ("qmri_fit", "mcdespot_cuda    mcDESPOT CUDA GPU", ["mcdespot_cuda"]),
        ("qmri_fit_mcdespot_cuda", "", []),
    ],
)
def test_mcdespot_cuda_selects_gpu_entry_point(tmp_path, monkeypatch, name, help_text, expected):
    bin_dir = tmp_path / "fitter bin"
    bin_dir.mkdir()
    binary = bin_dir / name
    binary.write_text(f"#!/bin/sh\nprintf '%s\\n' '{help_text}'\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    captured = []
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd",
        lambda command, label: captured.extend(shlex.split(command)),
    )
    mask = tmp_path / "brain mask.nii.gz"
    f0 = tmp_path / "off resonance.nii.gz"
    fit_mcdespot(
        spgr_file=tmp_path / "spgr.nii.gz",
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=tmp_path / "t1.nii.gz",
        b1_file=tmp_path / "b1.nii.gz",
        params_file=tmp_path / "params.json",
        out_dir=tmp_path / "out",
        mask_file=mask,
        f0_file=f0,
        cuda=True,
        extra_options={"model": "3pool", "fix-f0": True},
    )

    assert captured[:1 + len(expected)] == [str(binary), *expected]
    assert not any(arg.startswith("--t1=") for arg in captured)
    assert {f"--mask={mask}", f"--f0={f0}", "--model=3pool", "--fix-f0"} <= set(captured)


def test_cuda_request_rejects_cpu_only_fitter(tmp_path, monkeypatch):
    binary = tmp_path / "qmri_fit"
    binary.write_text("#!/bin/sh\nprintf 'mcdespot    CPU\\n'\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(RuntimeError, match="CUDA mcDESPOT requires"):
        _get_qmri_fit_command("mcdespot", cuda=True)


def test_value_boolean_options_emit_explicit_false():
    command = ["qmri_fit", "mcdespot"]

    _append_cli_options(
        command,
        {
            "t1f-despot-bounds": False,
            "scale-to-mean": False,
            "progress": False,
            "verbose": True,
        },
    )

    assert command == [
        "qmri_fit",
        "mcdespot",
        "--t1f-despot-bounds=false",
        "--scale-to-mean=false",
        "--verbose",
    ]


def test_model_config_accepts_save_rates_snake_case_alias():
    config = RelaxometryModelingConfig(
        despot1={"enabled": True, "save_rates": True},
        despot2={"enabled": True, "save-rates": True},
    )

    assert config.despot1 == {"enabled": True, "save-rates": True}
    assert config.despot2 == {"enabled": True, "save-rates": True}


def test_despot_wrappers_return_native_rate_outputs(tmp_path, monkeypatch):
    commands = []
    metrics_by_label = {
        "despot1_fit": ("T1", "M0", "R1"),
        "despot1_hifi_fit": ("T1", "M0", "B1", "R1"),
        "despot2_fit": ("T2", "M0", "F0", "R2"),
        "despot2fm_fit": ("T2", "M0", "F0", "R2"),
    }

    def fake_run(command, label):
        commands.append(command)
        out_dir = next(
            Path(token.split("=", 1)[1])
            for token in command.split()
            if token.startswith("--out_dir=")
        )
        out_base = next(
            token.split("=", 1)[1]
            for token in command.split()
            if token.startswith("--out_base=")
        )
        for metric in metrics_by_label[label]:
            (out_dir / f"{out_base}{metric}.nii.gz").touch()

    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry._get_qmri_fit_command",
        lambda subcommand, **kwargs: ["qmri_fit", subcommand],
    )
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd",
        fake_run,
    )

    common = {
        "params_file": tmp_path / "params.json",
        "out_dir": tmp_path / "out",
        "extra_options": {"save-rates": True},
    }
    despot1 = fit_despot1(
        spgr_file=tmp_path / "spgr.nii.gz",
        out_base="despot1",
        **common,
    )
    hifi = fit_despot1_hifi(
        spgr_file=tmp_path / "spgr.nii.gz",
        irspgr_file=tmp_path / "irspgr.nii.gz",
        out_base="hifi",
        **common,
    )
    despot2 = fit_despot2(
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=tmp_path / "t1.nii.gz",
        b1_file=tmp_path / "b1.nii.gz",
        out_base="despot2",
        **common,
    )
    despot2fm = fit_despot2_fm(
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=tmp_path / "t1.nii.gz",
        b1_file=tmp_path / "b1.nii.gz",
        out_base="despot2fm",
        **common,
    )

    assert despot1["r1"].name == "despot1_R1.nii.gz"
    assert hifi["r1"].name == "hifi_R1.nii.gz"
    assert not any(arg.startswith("--b1") for arg in shlex.split(commands[1]))
    assert despot2["r2"].name == "despot2_R2.nii.gz"
    assert despot2fm["r2"].name == "despot2fm_R2.nii.gz"
    assert all("--save-rates" in command for command in commands)
    assert all(
        path.exists()
        for outputs in (despot1, hifi, despot2, despot2fm)
        for path in outputs.values()
    )


def test_unified_mcdespot_command_does_not_emit_retired_t1_option(
    tmp_path, monkeypatch
):
    captured = {}
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry._get_qmri_fit_command",
        lambda *args, **kwargs: ["qmri_fit", "mcdespot"],
    )
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd",
        lambda command, label: captured.update(command=command, label=label),
    )

    fit_mcdespot(
        spgr_file=tmp_path / "spgr.nii.gz",
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=tmp_path / "t1.nii.gz",
        b1_file=tmp_path / "b1.nii.gz",
        params_file=tmp_path / "params.json",
        out_dir=tmp_path / "out",
    )

    assert captured["command"].startswith("qmri_fit mcdespot ")
    assert "--t1=" not in captured["command"]
    assert "--out_base=mcdespot_" in captured["command"]


def test_legacy_mcdespot_command_keeps_required_t1_option(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry._get_qmri_fit_command",
        lambda *args, **kwargs: ["qmri_fit_mcdespot"],
    )
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd",
        lambda command, label: captured.update(command=command, label=label),
    )

    t1_file = tmp_path / "t1.nii.gz"
    fit_mcdespot(
        spgr_file=tmp_path / "spgr.nii.gz",
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=t1_file,
        b1_file=tmp_path / "b1.nii.gz",
        params_file=tmp_path / "params.json",
        out_dir=tmp_path / "out",
    )

    assert f"--t1={t1_file}" in captured["command"]


def test_mcdespot_returns_native_3pool_outputs_with_single_separator(
    tmp_path, monkeypatch
):
    def fake_run(command, label):
        out_dir = tmp_path / "out"
        for metric in (
            "VFm", "T1m", "T2m", "T1f", "T2f", "Tau", "F0",
            "VFcsf", "T1csf", "T2csf",
        ):
            (out_dir / f"sub-01_model-mcDESPOT_{metric}.nii.gz").touch()
        (out_dir / "sub-01_model-mcDESPOTrun_metadata.json").touch()

    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry._get_qmri_fit_command",
        lambda *args, **kwargs: ["qmri_fit", "mcdespot"],
    )
    monkeypatch.setattr(
        "qmri_neuropipe.interfaces.relaxometry.run_cmd", fake_run
    )

    outputs = fit_mcdespot(
        spgr_file=tmp_path / "spgr.nii.gz",
        ssfp_file=tmp_path / "ssfp.nii.gz",
        t1_file=tmp_path / "t1.nii.gz",
        b1_file=tmp_path / "b1.nii.gz",
        params_file=tmp_path / "params.json",
        out_dir=tmp_path / "out",
        out_base="sub-01_model-mcDESPOT_",
        extra_options={"model": "3pool"},
    )

    assert set(outputs) == {
        "vfm", "t1m", "t2m", "t1f", "t2f", "tau", "f0",
        "vfcsf", "t1csf", "t2csf",
    }
    assert outputs["vfm"].name == "sub-01_model-mcDESPOT_VFm.nii.gz"
    assert all(path.exists() for path in outputs.values())
    assert (tmp_path / "out/sub-01_model-mcDESPOT_run_metadata.json").exists()


@pytest.mark.parametrize("retain_work", [False, True])
def test_hifi_smooths_b1_and_refits_spgr(tmp_path, monkeypatch, retain_work):
    import json
    import nibabel as nib
    import numpy as np
    from qmri_neuropipe.interfaces import relaxometry as module

    calls = []
    affine = np.diag([1., 2., 3., 1.])
    data = np.ones((7, 7, 7), dtype=np.float32)
    data[0] = 0
    data[3, 3, 3] = 2

    monkeypatch.setattr(module, "_get_qmri_fit_command", lambda *args, **kwargs: ["qmri-fit", "despot1_hifi"])
    def fake_run(command, label):
        calls.append(command)
        fit_dir = Path(next(token.split("=", 1)[1] for token in command.split() if token.startswith("--out_dir=")))
        assert fit_dir != tmp_path
        for metric in ("T1", "M0", "B1"):
            nib.save(nib.Nifti1Image(data, affine), fit_dir / f"hifi_desc-unsmoothed_{metric}.nii.gz")
    monkeypatch.setattr(module, "run_cmd", fake_run)
    def fake_refit(**kwargs):
        calls.append(kwargs)
        assert "b1" not in kwargs["extra_options"]
        b1 = nib.load(kwargs["b1_file"])
        assert b1.shape == data.shape
        np.testing.assert_allclose(b1.affine, affine)
        assert 1 < b1.get_fdata()[3, 3, 3] < 2
        assert b1.get_fdata()[1, 1, 1] >= 0.999
        assert np.all(b1.get_fdata()[0] == 0)
        assert "irspgr_file" not in kwargs
        return {"t1": tmp_path / "refit_T1.nii.gz", "m0": tmp_path / "refit_M0.nii.gz"}
    monkeypatch.setattr(module, "fit_despot1", fake_refit)
    outputs = module.fit_despot1_hifi(
        tmp_path / "spgr.nii.gz", tmp_path / "ir.nii.gz", tmp_path / "params.json",
        tmp_path, out_base="hifi", b1_smoothing_fwhm=4,
        extra_options={"b1": tmp_path / "external_b1.nii.gz"},
        work_dir=tmp_path / "work" if retain_work else None,
    )
    assert len(calls) == 2
    assert not any(arg.startswith("--b1") for arg in shlex.split(calls[0]))
    assert outputs["t1"].name == "refit_T1.nii.gz"
    assert outputs["b1"].name == "hifi_B1.nii.gz"
    assert json.loads((tmp_path / "hifi_B1.json").read_text())["B1SmoothingFWHM"] == 4
    assert not list(tmp_path.glob("*desc-unsmoothed*"))
    fit_dir = Path(next(token.split("=", 1)[1] for token in calls[0].split() if token.startswith("--out_dir=")))
    if retain_work:
        np.testing.assert_array_equal(nib.load(fit_dir / "hifi_desc-unsmoothed_B1.nii.gz").get_fdata(), data)
    else:
        assert not fit_dir.exists()


def test_hifi_smoothing_mask_preserves_constant_and_grid(tmp_path):
    import nibabel as nib
    import numpy as np
    from qmri_neuropipe.interfaces.relaxometry import _smooth_hifi_b1

    affine = np.diag([1., 2., 3., 1.])
    data = np.full((7, 7, 7), 1.2, dtype=np.float32)
    mask = np.zeros_like(data)
    mask[2:5, 2:5, 2:5] = 1
    source, mask_file, output = [tmp_path / name for name in ("b1.nii.gz", "mask.nii.gz", "smooth.nii.gz")]
    nib.save(nib.Nifti1Image(data, affine), source)
    nib.save(nib.Nifti1Image(mask, affine), mask_file)
    _smooth_hifi_b1(source, output, 5, mask_file)
    result = nib.load(output)
    np.testing.assert_allclose(result.get_fdata()[mask > 0], 1.2)
    assert np.all(result.get_fdata()[mask == 0] == 0)
    np.testing.assert_allclose(result.affine, affine)
