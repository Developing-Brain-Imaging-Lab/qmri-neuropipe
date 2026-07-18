import json
from pathlib import Path

import pytest

from qmri_neuropipe.lib.common.spatial_transforms import (
    write_transform_chain_to_sidecar,
)


def test_write_transform_chain_treats_null_sidecar_as_empty_object(tmp_path: Path):
    sidecar = tmp_path / "sub-01_dwi.json"
    sidecar.write_text("null\n")
    transform = {"type": "linear", "operation": "reorient"}

    write_transform_chain_to_sidecar(sidecar, [transform])

    assert json.loads(sidecar.read_text()) == {
        "SpatialTransformChain": [transform]
    }


def test_write_transform_chain_preserves_existing_metadata(tmp_path: Path):
    sidecar = tmp_path / "sub-01_dwi.json"
    sidecar.write_text(json.dumps({"PhaseEncodingDirection": "j-"}))

    write_transform_chain_to_sidecar(sidecar, [])

    assert json.loads(sidecar.read_text()) == {
        "PhaseEncodingDirection": "j-",
        "SpatialTransformChain": [],
    }


def test_write_transform_chain_rejects_non_object_sidecar(tmp_path: Path):
    sidecar = tmp_path / "sub-01_dwi.json"
    sidecar.write_text("[]\n")

    with pytest.raises(TypeError, match="Expected a JSON object"):
        write_transform_chain_to_sidecar(sidecar, [])
