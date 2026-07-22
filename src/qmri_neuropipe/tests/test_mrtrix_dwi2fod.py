from pathlib import Path


def _capture(monkeypatch):
    from qmri_neuropipe.interfaces import mrtrix

    calls = []
    monkeypatch.setattr(
        mrtrix, "_run_mrtrix", lambda parts, **kwargs: calls.append((parts, kwargs))
    )
    return mrtrix, calls


def _responses(tmp_path: Path, *tissues: str) -> dict[str, Path]:
    return {tissue: tmp_path / f"response_{tissue}.txt" for tissue in tissues}


def test_dwi2fod_expands_scalar_lmax_for_three_tissue_msmt_csd(tmp_path, monkeypatch):
    mrtrix, calls = _capture(monkeypatch)

    mrtrix.dwi2fod(
        tmp_path / "dwi.nii.gz",
        _responses(tmp_path, "wm", "gm", "csf"),
        tmp_path / "out",
        algorithm="msmt_csd",
        lmax=8,
    )

    cmd = calls[0][0]
    assert cmd[cmd.index("-lmax") + 1] == "8,0,0"


def test_dwi2fod_preserves_explicit_multi_tissue_lmax(tmp_path, monkeypatch):
    mrtrix, calls = _capture(monkeypatch)

    mrtrix.dwi2fod(
        tmp_path / "dwi.nii.gz",
        _responses(tmp_path, "wm", "gm", "csf"),
        tmp_path / "out",
        algorithm="msmt_csd",
        lmax=[8, 2, 0],
    )

    cmd = calls[0][0]
    assert cmd[cmd.index("-lmax") + 1] == "8,2,0"


def test_dwi2fod_keeps_scalar_lmax_for_single_tissue_csd(tmp_path, monkeypatch):
    mrtrix, calls = _capture(monkeypatch)

    mrtrix.dwi2fod(
        tmp_path / "dwi.nii.gz",
        {"response": tmp_path / "response.txt"},
        tmp_path / "out",
        algorithm="csd",
        lmax=8,
    )

    cmd = calls[0][0]
    assert cmd[cmd.index("-lmax") + 1] == "8"
