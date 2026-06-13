import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.io.bids import build_bids_name
from qmri_neuropipe.utils.data_io import DataIOManager


def _save_nifti(path: Path, shape: tuple[int, ...]) -> None:
    nib.save(
        nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)),
        str(path),
    )


def test_anatomical_space_gnl_tensor_is_saved_with_final_dwi_derivatives(tmp_path: Path):
    derivatives_dir = tmp_path / "derivatives"
    work_dir = tmp_path / "work"
    work_dir.mkdir()

    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=derivatives_dir,
        work_dir=work_dir,
        config_data={
            "skip_existing": False,
            "dmri": {
                "preprocessing": {
                    "coregistration": {
                        "enabled": True,
                        "output_resolution": "anatomical",
                    },
                    "grad_nonlin": {"enabled": True},
                }
            },
        },
    )

    dwi_entities = {
        "sub": "01",
        "ses": "02",
        "run": "1",
        "space": "T1w",
        "desc": "coreg",
        "suffix": "dwi",
    }
    final_dwi_path = work_dir / build_bids_name(dwi_entities)
    final_gnl_path = work_dir / "grad_nonlin" / "aligned_tensor.nii.gz"
    final_gnl_path.parent.mkdir()
    _save_nifti(final_dwi_path, (4, 5, 6, 2))
    _save_nifti(final_gnl_path, (4, 5, 6, 9))

    final_dwi = DWIFile(img=final_dwi_path, entities=dwi_entities)
    context = {
        "subject": "01",
        "session": "02",
        "preprocessed_dwis": [final_dwi],
        "preprocessed_masks": [None],
        "gnl_map": final_gnl_path,
        "gnl_maps": [final_gnl_path],
        "gnl_map_by_image": {
            final_dwi_path: final_gnl_path,
            str(final_dwi_path): final_gnl_path,
        },
    }

    DataIOManager(config, logging.getLogger(__name__)).save_final_outputs(
        context,
        derivatives_dir,
        skip_existing=False,
    )

    expected_entities = dict(dwi_entities)
    expected_entities["desc"] = "gnl_tensor"
    expected_tensor = (
        derivatives_dir
        / "sub-01"
        / "ses-02"
        / "dwi"
        / build_bids_name(expected_entities)
    )

    assert expected_tensor.exists()
    assert nib.load(str(expected_tensor)).shape == (4, 5, 6, 9)
