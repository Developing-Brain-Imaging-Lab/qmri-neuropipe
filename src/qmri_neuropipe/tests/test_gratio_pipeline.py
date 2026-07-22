from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.workflows.pipelines.gratio import run_aggregate_gratio_subject


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), np.eye(4)), path)
    return path


def test_standalone_pipeline_discovers_mcdespot_and_noddi(tmp_path):
    root = tmp_path / "derivatives"
    subject = root / "sub-01"
    shape = (2, 2, 2)
    _save(subject / "anat" / "models" / "sub-01_model-mcDESPOT_VFm.nii.gz", np.full(shape, 0.2))
    _save(subject / "anat" / "sub-01_desc-spgrref_VFA.nii.gz", np.ones(shape))
    _save(subject / "dwi" / "NODDI" / "sub-01_acq-a_model-NODDI_ICVF.nii.gz", np.full(shape, 0.5))
    _save(subject / "dwi" / "NODDI" / "sub-01_acq-a_model-NODDI_FISO.nii.gz", np.full(shape, 0.2))
    dwi = _save(subject / "dwi" / "sub-01_acq-a_desc-preproc_dwi.nii.gz", np.ones((*shape, 2)))
    Path(str(dwi).replace(".nii.gz", ".bval")).write_text("0 1000\n")

    config = PipelineConfig(
        output_dir=root,
        work_dir=tmp_path / "work",
        config_data={"gratio": {
            "enabled": True,
            "calibration": {"mode": "identity"},
            "registration": {"assume_aligned": True},
        }},
    )
    results = run_aggregate_gratio_subject(config, "01")
    assert len(results) == 1
    assert results[0]["gratio"].exists()
    assert "acq-a" in results[0]["gratio"].name


def test_pipeline_rejects_ambiguous_noddi_pair(tmp_path):
    root = tmp_path / "derivatives"
    dwi = root / "sub-01" / "dwi" / "NODDI"
    _save(dwi / "sub-01_model-NODDI_ICVF.nii.gz", np.ones((1, 1, 1)))
    _save(dwi / "sub-01_model-NODDI_FISO.nii.gz", np.ones((1, 1, 1)))
    _save(dwi / "sub-01_desc-copy_model-NODDI_FISO.nii.gz", np.ones((1, 1, 1)))
    from qmri_neuropipe.lib.microstructure.inputs import discover_noddi_pairs
    import pytest
    with pytest.raises(ValueError, match="Ambiguous"):
        discover_noddi_pairs(root / "sub-01" / "dwi")
