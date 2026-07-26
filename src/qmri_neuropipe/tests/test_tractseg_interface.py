def test_bundle_names_do_not_add_unsupported_tractseg_flag(tmp_path, monkeypatch):
    from qmri_neuropipe.interfaces import tractseg

    calls = []
    monkeypatch.setattr(
        tractseg,
        "run_cmd",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    tractseg.run_tractseg(
        input_file=tmp_path / "peaks.nii.gz",
        output_dir=tmp_path / "tractseg",
        bundle_names=["CST_left", "CST_right"],
        gpu=False,
    )

    command, kwargs = calls[0]
    assert command.startswith("TractSeg ")
    assert "--bundles" not in command
    assert "CST_left" not in command
    assert kwargs == {"label": "TractSeg"}
