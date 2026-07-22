from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from qmri_neuropipe.lib.common.registration import (
    _ants_affine_to_ras_matrix,
    _preserve_header_only_bvec,
)


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


def test_ants_pull_affine_is_inverted_for_header_composition(monkeypatch, tmp_path: Path):
    # ITK parameters describe fixed -> moving. Here that mapping translates
    # fixed points +4 mm in LPS x, so moving -> fixed must translate -4 mm in
    # LPS x, which is +4 mm in RAS x.
    transform = SimpleNamespace(
        parameters=[1, 0, 0, 0, 1, 0, 0, 0, 1, 4, 0, 0],
        fixed_parameters=[0, 0, 0],
    )
    monkeypatch.setitem(
        sys.modules,
        "ants",
        SimpleNamespace(read_transform=lambda _: transform),
    )

    matrix = _ants_affine_to_ras_matrix(tmp_path / "affine.mat")

    assert np.allclose(matrix[:3, :3], np.eye(3))
    assert np.allclose(matrix[:3, 3], [4, 0, 0])
