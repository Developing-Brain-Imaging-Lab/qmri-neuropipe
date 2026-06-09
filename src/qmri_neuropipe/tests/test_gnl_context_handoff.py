from pathlib import Path

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.utils.execution_engine import ExecutionEngine


class _RegisteredDWIProducer:
    def __call__(self, context, output_dir: Path, **kwargs):
        current = context["current_image"]
        gnl_map = context["gnl_map"]
        registered = DWIFile(
            img=output_dir / "sub-01_desc-coreg_dwi.nii.gz",
            bval=current.bval,
            bvec=current.bvec,
            json=current.json,
            entities={**current.entities, "desc": "coreg", "suffix": "dwi"},
        )
        return {
            "current_image": registered,
            "gnl_source_map": {str(gnl_map): str(current.img)},
        }


def test_gnl_map_survives_string_keyed_per_image_handoff(tmp_path: Path):
    input_dwi = DWIFile(
        img=tmp_path / "sub-01_dwi.nii.gz",
        entities={"sub": "01", "suffix": "dwi"},
    )
    gnl_map = tmp_path / "sub-01_desc-gnl_tensor_dwi.nii.gz"
    transform = {"type": "linear", "transforms": ["xfm.mat"]}

    context = {
        "dwi_files": [input_dwi],
        "gnl_map": gnl_map,
        "gnl_map_by_image": {str(input_dwi.img): gnl_map},
        "gnl_native_reference_map": {str(input_dwi.img): input_dwi},
        "gnl_transform_map": {str(input_dwi.img): transform},
    }

    engine = ExecutionEngine(config={}, logger=None)
    result = engine.execute_steps([_RegisteredDWIProducer()], context, tmp_path)

    registered = result["preprocessed_dwis"][0]
    assert result["gnl_map_by_image"][registered.img] == gnl_map
    assert result["gnl_map_by_image"][str(registered.img)] == gnl_map
    assert result["gnl_source_map"][str(gnl_map)] == str(input_dwi.img)
    assert result["gnl_native_reference_map"][registered.img] == input_dwi
    assert result["gnl_native_reference_map"][str(registered.img)] == input_dwi
    assert result["gnl_transform_map"][registered.img] == [transform]
    assert result["gnl_transform_map"][str(registered.img)] == [transform]
