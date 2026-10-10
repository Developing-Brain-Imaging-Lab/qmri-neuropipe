from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.interfaces import dipy as dipy_interface


@pytest.mark.parametrize("preserve", [False, True])
def test_mppca_can_preserve_original_values_outside_mask(
    tmp_path: Path, monkeypatch, preserve: bool
):
    import dipy.denoise.localpca as localpca

    data = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2) + 1
    mask = np.zeros((2, 2, 2), dtype=np.uint8)
    mask[0, 0, 0] = 1
    source = tmp_path / "source.nii.gz"
    mask_file = tmp_path / "mask.nii.gz"
    output = tmp_path / "denoised.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)

    def fake_mppca(array, **kwargs):
        return np.zeros_like(array), np.ones(array.shape[:3], dtype=np.float32)

    monkeypatch.setattr(localpca, "mppca", fake_mppca)

    dipy_interface.mppca(
        source,
        output,
        mask=mask_file,
        preserve_outside_mask=preserve,
    )

    result = nib.load(output).get_fdata()
    np.testing.assert_array_equal(result[0, 0, 0], [0, 0])
    if preserve:
        np.testing.assert_array_equal(result[mask == 0], data[mask == 0])
    else:
        np.testing.assert_array_equal(result[mask == 0], 0)
