import json
from pathlib import Path

from qmri_neuropipe.utils.data_io import DataIOManager


def test_processing_history_serializes_path_parameters(tmp_path: Path):
    sidecar = tmp_path / "sub-01_dwi.json"
    sidecar.write_text("{}")
    manager = object.__new__(DataIOManager)

    manager._update_json_history(
        sidecar,
        ["CoregistrationStep"],
        [
            {
                "step": "CoregistrationStep",
                "parameters": {"registration_fixed": tmp_path / "SynthT1.mgz"},
            }
        ],
    )

    payload = json.loads(sidecar.read_text())
    parameter = payload["ProcessingStepsDetail"][0]["parameters"][
        "registration_fixed"
    ]
    assert parameter == str(tmp_path / "SynthT1.mgz")
