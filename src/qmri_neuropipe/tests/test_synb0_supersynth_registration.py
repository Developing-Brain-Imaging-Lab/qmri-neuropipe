import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core import ValidationError
from qmri_neuropipe.core.types import DWIFile, ImageFile
from qmri_neuropipe.interfaces import c3d, freesurfer, fsl
from qmri_neuropipe.io.anat.bids import bids_find_other_anat
from qmri_neuropipe.lib.dmri.synb0 import Synb0EstimationStep


class _Config(dict):
    n_cpus = 4

    def get(self, key, default=None):
        current = self
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def _step(synb0_config: dict) -> Synb0EstimationStep:
    step = Synb0EstimationStep.__new__(Synb0EstimationStep)
    step.logger = logging.getLogger(__name__)
    step.config = _Config(
        {
            "anat": {"super_synth": {}},
            "dmri": {"preprocessing": {"distcorr": {"synb0": synb0_config}}},
        }
    )
    step.synb0_cfg = synb0_config
    return step


def test_acquired_t1w_is_preferred_for_synb0(tmp_path: Path):
    supplied_t1w = ImageFile(img=tmp_path / "supplied_T1w.nii.gz", entities={"suffix": "T1w"})
    context = {
        "t1w_files": [supplied_t1w],
        "anatomical_files": [
            ImageFile(img=tmp_path / "anatomical_FLAIR.nii.gz", entities={"suffix": "FLAIR"})
        ],
    }

    selected = _step({"registration": "supersynth"})._prepare_anatomical_t1w(
        context, tmp_path
    )

    assert selected is supplied_t1w
    assert context["synb0_t1w_source"] == "acquired_t1w"


def test_non_t1w_anatomical_is_synthesized_for_synb0(tmp_path: Path, monkeypatch):
    flair = ImageFile(
        img=tmp_path / "anatomical_FLAIR.nii.gz",
        entities={"suffix": "FLAIR"},
    )
    nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)).to_filename(flair.img)
    calls = []

    def fake_super_synth(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["out_dir"]) / "SynthT1.mgz"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"synth")
        return kwargs["out_dir"]

    monkeypatch.setattr(freesurfer, "mri_super_synth", fake_super_synth)
    context = {"t1w_files": [], "anatomical_files": [flair]}

    selected = _step({"anatomical_input": "FLAIR"})._prepare_anatomical_t1w(
        context, tmp_path
    )

    assert calls[0]["in_file"] == flair.img
    assert selected.img == tmp_path / "supersynth_from_anatomical" / "SynthT1.mgz"
    assert selected.entities["suffix"] == "T1w"
    assert context["synb0_anatomical_input"] is flair
    assert context["synb0_t1w_source"] == "supersynth_anatomical"


def test_supersynth_registration_reference_is_generated_from_dwi_b0(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration": "supersynth"})
    dwi = DWIFile(img=tmp_path / "dwi.nii.gz", entities={"suffix": "dwi"})
    extracted_b0 = tmp_path / "real_b0_desc-supersynthInput.nii.gz"
    calls = []

    def fake_extract(input_dwi, output_path, force=False, as_4d=True):
        assert input_dwi is dwi
        assert output_path == extracted_b0
        assert as_4d is False
        output_path.write_bytes(b"b0")
        return output_path

    def fake_super_synth(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["out_dir"]) / "SynthT1.mgz"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"synth")
        return kwargs["out_dir"]

    monkeypatch.setattr(step, "_extract_mean_b0", fake_extract)
    monkeypatch.setattr(freesurfer, "mri_super_synth", fake_super_synth)

    context = {}
    result = step._prepare_supersynth_registration_reference(
        context, tmp_path, input_dwi=dwi
    )

    assert result == tmp_path / "supersynth_from_dwi" / "SynthT1.mgz"
    assert calls[0]["in_file"] == extracted_b0
    assert context["synb0_registration_reference"] == result
    assert context["synb0_registration_method"] == "supersynth"


def test_synb0_rejects_session_without_any_anatomical_scan(tmp_path: Path):
    step = _step({"t1w_source": "dwi_supersynth"})
    context = {
        "t1w_files": [],
        "anatomical_files": [],
        "dwi_files": [DWIFile(img=tmp_path / "dwi.nii.gz", entities={"suffix": "dwi"})],
    }

    with pytest.raises(ValidationError, match="requires an undistorted anatomical"):
        step.validate_inputs(context, tmp_path)


def test_supersynth_image_is_used_only_as_registration_fixed_image(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration": "supersynth"})
    supplied_t1w_brain = tmp_path / "supplied_t1w_brain.nii.gz"
    dwi_synth_t1w = tmp_path / "dwi_SynthT1.mgz"
    calls = []

    def fake_flirt(**kwargs):
        calls.append(("flirt", kwargs))
        return kwargs["out_file"], kwargs["omat"]

    def fake_convert_xfm(**kwargs):
        calls.append(("convert_xfm", kwargs))
        return kwargs["out_file"]

    def fake_fsl2ants(**kwargs):
        calls.append(("fsl2ants", kwargs))
        return kwargs["out_file"]

    monkeypatch.setattr(fsl, "flirt", fake_flirt)
    monkeypatch.setattr(fsl, "convert_xfm", fake_convert_xfm)
    monkeypatch.setattr(c3d, "fsl2ants", fake_fsl2ants)

    step._register_t1w_to_dwi(supplied_t1w_brain, dwi_synth_t1w, tmp_path)

    flirt_call = calls[0][1]
    assert flirt_call["in_file"] == supplied_t1w_brain
    assert flirt_call["ref_file"] == dwi_synth_t1w
    assert calls[1][1]["ref_file"] == dwi_synth_t1w
    assert calls[1][1]["in_file"] == supplied_t1w_brain


def test_other_anatomical_discovery_excludes_t1_t2_and_segmentations(tmp_path: Path):
    for suffix in ("FLAIR", "PD", "T1w", "T2w", "dseg", "mask"):
        (tmp_path / f"sub-01_{suffix}.nii.gz").touch()

    found = bids_find_other_anat(tmp_path)

    assert [image.entities["suffix"] for image in found] == ["FLAIR", "PD"]


def test_vfa_series_is_averaged_for_supersynth(tmp_path: Path):
    affine = np.eye(4)
    images = []
    for index, value in enumerate((1.0, 3.0, 5.0)):
        path = tmp_path / f"sub-01_flip-{index + 1}_VFA.nii.gz"
        nib.Nifti1Image(np.full((3, 3, 3), value), affine).to_filename(path)
        images.append(ImageFile(img=path, entities={"suffix": "VFA", "flip": str(index + 1)}))

    prepared = _step({"anatomical_series_mode": "mean"})._prepare_anatomical_series(
        images, tmp_path
    )

    assert prepared.img.name == "anatomical_desc-mean_supersynthInput.nii.gz"
    assert np.allclose(nib.load(str(prepared.img)).get_fdata(), 3.0)


def test_representative_volume_can_be_selected_from_4d_vfa(tmp_path: Path):
    path = tmp_path / "sub-01_VFA.nii.gz"
    data = np.stack(
        [np.full((3, 3, 3), value) for value in (2.0, 4.0, 8.0)],
        axis=-1,
    )
    nib.Nifti1Image(data, np.eye(4)).to_filename(path)
    image = ImageFile(img=path, entities={"suffix": "VFA"})

    prepared = _step(
        {
            "anatomical_series_mode": "representative",
            "anatomical_series_index": 1,
        }
    )._prepare_anatomical_series([image], tmp_path)

    assert np.allclose(nib.load(str(prepared.img)).get_fdata(), 4.0)
