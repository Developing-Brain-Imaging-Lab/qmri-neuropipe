import json
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.interfaces import fsl
from qmri_neuropipe.lib.dmri.apply_topup import ApplyTopupStep


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


def test_applytopup_uses_first_topup_row_for_post_tortoise_dwi(tmp_path, monkeypatch):
    dwi = _single_b0(tmp_path, "tortoise", 1.0, 0.075)
    out_file = tmp_path / "corrected.nii.gz"
    base = tmp_path / "topup_group0"
    datain = tmp_path / "topup_group0_topup_datain.txt"
    datain.write_text("0 -1 0 0.075\n0 1 0 0.000\n")
    (tmp_path / "topup_group0_fieldcoef.nii.gz").write_bytes(b"field")
    (tmp_path / "topup_group0_movpar.txt").write_text("0 0 0 0 0 0\n")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        shutil.copy2(dwi.img, out_file)

    monkeypatch.setattr(fsl, "run_cmd", fake_run)

    result = fsl.applytopup(
        dwi,
        out_file,
        topup_base=base,
        datain=datain,
    )

    assert result == out_file
    assert "--inindex=1" in commands[0]
    assert "--method=jac" in commands[0]
    assert f"--topup={base}" in commands[0]


def test_applytopup_step_preserves_tortoise_gradients_and_updates_context(
    tmp_path, monkeypatch
):
    dwi = _single_b0(tmp_path, "tortoise", 1.0, 0.075)
    dwi.entities = {"sub": "01", "suffix": "dwi", "desc": "tortoisev4corrected"}
    dwi.bvec = tmp_path / "tortoise.bvec"
    dwi.bvec.write_text("0\n0\n0\n")
    base = tmp_path / "topup" / "topup_group0"
    base.parent.mkdir()
    datain = base.with_name(base.name + "_topup_datain.txt")
    datain.write_text("0 -1 0 0.075\n0 1 0 0.000\n")

    def fake_applytopup(in_file, out_file, **kwargs):
        shutil.copy2(in_file.img, out_file)
        return out_file

    monkeypatch.setattr(fsl, "applytopup", fake_applytopup)
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        config_data={},
    )
    step = ApplyTopupStep(config, logging.getLogger(__name__))
    context = {
        "current_image": dwi,
        "dwi_files": [dwi],
        "topup_map": {dwi.img: str(base)},
    }

    result = step(context, output_dir=tmp_path / "work")
    corrected = result["current_image"]

    assert corrected.img.exists()
    assert corrected.bval.read_text() == dwi.bval.read_text()
    assert corrected.bvec.read_text() == dwi.bvec.read_text()
    metadata = json.loads(corrected.json.read_text())
    assert metadata["SusceptibilityDistortionCorrection"].startswith("FSL TOPUP")
    assert result["dwi_files"] == [corrected]
