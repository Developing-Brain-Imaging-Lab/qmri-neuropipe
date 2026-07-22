import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.lib.microstructure.inputs import GRatioInputs
from qmri_neuropipe.lib.microstructure.workflow import AggregateGRatioWorkflow


def _image(path: Path, values):
    nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.eye(4)), path)
    return path


def test_workflow_writes_maps_masks_sidecars_and_summary(tmp_path):
    shape = (2, 2, 2)
    myelin = _image(tmp_path / "sub-01_model-mcDESPOT_VFm.nii.gz", np.full(shape, 0.2))
    icvf = _image(tmp_path / "sub-01_model-NODDI_ICVF.nii.gz", np.full(shape, 0.5))
    fiso = _image(tmp_path / "sub-01_model-NODDI_FISO.nii.gz", np.full(shape, 0.2))
    reference = _image(tmp_path / "sub-01_desc-b0_dwi.nii.gz", np.zeros(shape))
    spgr = _image(tmp_path / "sub-01_desc-spgrref_VFA.nii.gz", np.ones(shape))

    config = PipelineConfig(config_data={
        "gratio": {
            "calibration": {"mode": "identity"},
            "recommended_mask": {"enabled": True, "fiso_max": 0.5},
        }
    })
    outputs = AggregateGRatioWorkflow(config).run(
        GRatioInputs(
            myelin=myelin,
            spgr_reference=spgr,
            intracellular=icvf,
            isotropic=fiso,
            diffusion_reference=reference,
            entities={"sub": "01", "acq": "test"},
        ),
        tmp_path / "out",
    )

    assert {"MVF", "AVF", "FVF", "gratio", "valid_mask", "recommended_mask", "ConductionFactor", "summary"} <= set(outputs)
    gratio = nib.load(outputs["gratio"]).get_fdata()
    assert np.allclose(gratio, np.sqrt(2 / 3))
    sidecar = json.loads(Path(str(outputs["gratio"]).replace(".nii.gz", ".json")).read_text())
    assert sidecar["MyelinCalibration"]["Calibrated"] is False
    assert sidecar["AVFFormula"] == "(1-FISO)*ICVF"
    assert outputs["summary"].suffix == ".tsv"
