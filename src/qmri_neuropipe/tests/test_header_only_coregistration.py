from pathlib import Path

from qmri_neuropipe.lib.common.registration import _preserve_header_only_bvec


def test_header_only_coregistration_copies_bvec_without_modification(tmp_path: Path):
    source = tmp_path / "input.bvec"
    source.write_text("1 0 0\n0 1 0\n0 0 1\n")

    output = _preserve_header_only_bvec(
        source,
        tmp_path,
        {"sub": "01", "ses": "02", "suffix": "dwi"},
        "coreg",
    )

    assert output == tmp_path / "sub-01_ses-02_desc-coreg_bvec.bvec"
    assert output.read_bytes() == source.read_bytes()
