import json

import nibabel as nib
import numpy as np

from qmri_neuropipe.interfaces.dmipy import _save_noddi_outputs
from qmri_neuropipe.interfaces.dmipy_backend import DmipyRuntime


def test_noddi_outputs_use_bids_derivative_names_and_sidecars(tmp_path):
    input_path = (
        tmp_path
        / "sub-311140_ses-01_acq-mb4_run-2_desc-preproc_dwi.nii.gz"
    )
    output_dir = tmp_path / "noddi"
    output_dir.mkdir()
    runtime = DmipyRuntime(
        version="2.1.0",
        solver="jax",
        requested_device="gpu",
        backend="gpu",
        devices=("cuda:0",),
        gpu_device=0,
    )
    arrays = {
        "odi": np.full((2, 2, 2), 0.2),
        "fiso": np.full((2, 2, 2), 0.1),
        "vf_intra": np.full((2, 2, 2), 0.5),
        "vf_extra": np.full((2, 2, 2), 0.4),
    }

    outputs = _save_noddi_outputs(
        output_dir,
        input_path,
        np.eye(4),
        arrays,
        runtime,
        model_type="standard",
        distribution="Watson",
        parallel_diffusivity=1.7e-9,
        iso_diffusivity=3.0e-9,
        solver_kwargs={"batch_size": 4000},
        fiso_constrained=False,
    )

    expected_suffixes = {
        "odi": "ODI",
        "fiso": "FISO",
        "vf_intra": "ICVF",
        "vf_extra": "EXVF",
    }
    for key, suffix in expected_suffixes.items():
        expected_name = (
            "sub-311140_ses-01_acq-mb4_run-2_"
            f"model-NODDI_{suffix}.nii.gz"
        )
        output_path = outputs[key]
        assert output_path.name == expected_name
        assert "desc-preproc" not in output_path.name
        assert nib.load(output_path).get_data_dtype() == np.dtype(np.float32)

        sidecar = output_path.with_name(output_path.name[:-7] + ".json")
        metadata = json.loads(sidecar.read_text())
        assert metadata["ModelName"].startswith("NODDI")
        assert metadata["Metric"] == suffix
        assert metadata["MetricUnits"] == "unitless"
        assert metadata["FittingSoftware"] == "dmipy-fit"
        assert metadata["FittingSoftwareVersion"] == "2.1.0"
        assert metadata["ExecutionBackend"] == "gpu"
        assert metadata["SolverOptions"]["batch_size"] == 4000
