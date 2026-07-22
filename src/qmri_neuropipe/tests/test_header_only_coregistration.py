from pathlib import Path
import sys
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from qmri_neuropipe.lib.common.registration import (
    _apply_mrtrix_header_transform,
    _ants_affine_to_ras_matrix,
    _preserve_header_only_bvec,
)
from qmri_neuropipe.interfaces import mrtrix
from qmri_neuropipe.lib.common import registration


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


def test_mrtrix_header_mode_uses_linear_transform_without_template(monkeypatch, tmp_path: Path):
    input_path = tmp_path / "dwi.nii.gz"
    output_path = tmp_path / "coreg.nii.gz"
    itk_path = tmp_path / "affine.mat"
    mrtrix_path = tmp_path / "affine.txt"
    data = np.arange(4 * 5 * 6 * 2, dtype=np.float32).reshape(4, 5, 6, 2)
    nib.save(nib.Nifti1Image(data, np.eye(4)), input_path)
    calls = {}

    def fake_transform(**kwargs):
        calls["transform"] = kwargs
        image = nib.load(str(input_path))
        affine = image.affine.copy()
        affine[:3, 3] = [3, -2, 1]
        nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), output_path)

    monkeypatch.setattr(mrtrix, "mrtransform", fake_transform)
    monkeypatch.setattr(
        registration,
        "_ants_affine_to_ras_matrix",
        lambda _: np.diag([1, 1, 1, 1]),
    )

    world_transform = _apply_mrtrix_header_transform(
        input_path, output_path, itk_path, mrtrix_path, nthreads=4
    )

    assert calls["transform"]["linear_transform"] == mrtrix_path
    assert calls["transform"]["template"] is None
    assert calls["transform"]["interp"] is None
    assert np.allclose(np.loadtxt(mrtrix_path), np.eye(4))
    assert np.array_equal(nib.load(str(output_path)).get_fdata(), data)
    assert np.allclose(world_transform[:3, 3], [3, -2, 1])
