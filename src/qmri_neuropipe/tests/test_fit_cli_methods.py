from typer.testing import CliRunner

from qmri_neuropipe import tools
from qmri_neuropipe.interfaces import dipy


def _dwi_inputs(tmp_path):
    dwi = tmp_path / "sub-01_dwi.nii.gz"
    bval = tmp_path / "sub-01_dwi.bval"
    bvec = tmp_path / "sub-01_dwi.bvec"
    dwi.touch()
    bval.write_text("0 1000 2000\n")
    bvec.write_text("0 1 1\n0 0 0\n0 0 0\n")
    return dwi, bval, bvec


def test_fit_dki_cli_forwards_method(tmp_path, monkeypatch):
    dwi, bval, bvec = _dwi_inputs(tmp_path)
    captured = {}

    def fake_fit_dki(*args, **kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(dipy, "fit_dki", fake_fit_dki)
    result = CliRunner().invoke(
        tools.app,
        [
            "fit-dki",
            "--input",
            str(dwi),
            "--output-dir",
            str(tmp_path / "out"),
            "--bval",
            str(bval),
            "--bvec",
            str(bvec),
            "--method",
            "NLLS",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["fit_method"] == "NLLS"


def test_dki_nlls_normalization_preserves_nonlinear_solver():
    assert dipy._normalize_dki_fit_method("NLLS") == "NLLS"
    assert dipy._normalize_dki_fit_method("nls") == "NLLS"


def test_fit_commands_expose_supported_solver_selection():
    runner = CliRunner()
    option_by_command = {
        "fit-dti": "--method",
        "fit-dki": "--method",
        "fit-noddi": "--solver",
        "fit-sandi": "--solver",
        "fit-microglia": "--solver",
        "fit-dmipy": "--solver",
        "fit-fwe-dti": "--method",
        "fit-csd": "--method",
    }

    for command, option in option_by_command.items():
        result = runner.invoke(tools.app, [command, "--help"])
        assert result.exit_code == 0, f"{command}: {result.stdout}"
        assert option in result.stdout, command
