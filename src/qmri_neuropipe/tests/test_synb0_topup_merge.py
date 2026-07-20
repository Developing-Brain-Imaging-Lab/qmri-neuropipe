import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.interfaces import fsl


def _single_b0(tmp_path: Path, name: str, value: float, readout: float) -> DWIFile:
    image = tmp_path / f"{name}.nii.gz"
    sidecar = tmp_path / f"{name}.json"
    bval = tmp_path / f"{name}.bval"
    nib.save(nib.Nifti1Image(np.full((3, 4, 5, 1), value), np.eye(4)), image)
    sidecar.write_text(json.dumps({
        "PhaseEncodingDirection": "j-",
        "TotalReadoutTime": readout,
    }))
    # A near-zero scanner b-value must still be treated as b0.
    bval.write_text("5\n")
    return DWIFile(entities={}, img=image, json=sidecar, bval=bval)


def test_synb0_pair_is_merged_in_input_order_with_matching_datain(tmp_path, monkeypatch):
    real = _single_b0(tmp_path, "real", 1.0, 0.075)
    synthetic = _single_b0(tmp_path, "synthetic", 2.0, 0.0)
    monkeypatch.setattr(fsl, "run_cmd", lambda *args, **kwargs: None)

    base = tmp_path / "topup"
    fsl.topup([real, synthetic], base)

    merged = nib.load(str(tmp_path / "topup_topup_imain.nii.gz")).get_fdata()
    assert merged.shape == (3, 4, 5, 2)
    np.testing.assert_allclose(merged[..., 0], 1.0)
    np.testing.assert_allclose(merged[..., 1], 2.0)
    assert (tmp_path / "topup_topup_datain.txt").read_text().splitlines() == [
        "0 -1 0 0.075000",
        "0 -1 0 0.000000",
    ]
