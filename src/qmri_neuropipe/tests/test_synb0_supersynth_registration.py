import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core import ValidationError
from qmri_neuropipe.core.types import DWIFile, ImageFile
from qmri_neuropipe.interfaces import ants, c3d, freesurfer, fsl
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


def test_supersynth_registration_moving_is_generated_from_normalized_t1w(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration": "supersynth"})
    t1w_norm = tmp_path / "t1w_norm.nii.gz"
    calls = []

    def fake_super_synth(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["out_dir"]) / "SynthT1.mgz"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"synth")
        return kwargs["out_dir"]

    monkeypatch.setattr(freesurfer, "mri_super_synth", fake_super_synth)
    context = {}

    result = step._prepare_supersynth_registration_moving(
        context,
        tmp_path,
        t1w_path=t1w_norm,
    )

    expected = (
        tmp_path / "supersynth_from_t1w_registration" / "SynthT1.mgz"
    )
    assert result == expected
    assert calls[0]["in_file"] == t1w_norm
    assert calls[0]["out_dir"] == tmp_path / "supersynth_from_t1w_registration"
    assert context["synb0_registration_moving"] == expected


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

    def fake_mri_convert(**kwargs):
        calls.append(("mri_convert", kwargs))
        return kwargs["out_file"]

    def fake_flirt(**kwargs):
        calls.append(("flirt", kwargs))
        return kwargs["out_file"], kwargs["omat"]

    def fake_convert_xfm(**kwargs):
        calls.append(("convert_xfm", kwargs))
        return kwargs["out_file"]

    def fake_fsl2ants(**kwargs):
        calls.append(("fsl2ants", kwargs))
        return kwargs["out_file"]

    def fake_apply_transforms(**kwargs):
        calls.append(("apply_transforms", kwargs))
        return kwargs["out_file"]

    monkeypatch.setattr(freesurfer, "mri_convert", fake_mri_convert)
    monkeypatch.setattr(fsl, "flirt", fake_flirt)
    monkeypatch.setattr(fsl, "convert_xfm", fake_convert_xfm)
    monkeypatch.setattr(c3d, "fsl2ants", fake_fsl2ants)
    monkeypatch.setattr(ants, "apply_transforms", fake_apply_transforms)

    forward, inverse = step._register_t1w_to_dwi(
        supplied_t1w_brain, dwi_synth_t1w, tmp_path
    )

    assert calls[0][0] == "mri_convert"
    assert calls[0][1]["in_file"] == dwi_synth_t1w
    assert calls[0][1]["out_file"] == tmp_path / "registration_ref.nii.gz"

    flirt_call = calls[1][1]
    assert flirt_call["in_file"] == supplied_t1w_brain
    assert flirt_call["ref_file"] == tmp_path / "registration_ref.nii.gz"
    assert calls[2][1]["ref_file"] == tmp_path / "registration_ref.nii.gz"
    assert calls[2][1]["in_file"] == supplied_t1w_brain
    assert forward.path == tmp_path / "t1w_2_dwi_supersynth.txt"
    assert forward.invert is False
    assert inverse.path == tmp_path / "dwi_2_t1w_supersynth.txt"
    assert inverse.invert is False
    assert calls[-1][1]["moving_file"] == supplied_t1w_brain
    assert calls[-1][1]["fixed_file"] == tmp_path / "registration_ref.nii.gz"


def test_nifti_registration_reference_does_not_require_conversion(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration": "supersynth"})
    supplied_t1w_brain = tmp_path / "supplied_t1w_brain.nii.gz"
    dwi_synth_t1w = tmp_path / "dwi_SynthT1.nii.gz"

    def unexpected_convert(**kwargs):
        pytest.fail("A NIfTI registration reference must not be reconverted")

    monkeypatch.setattr(freesurfer, "mri_convert", unexpected_convert)
    monkeypatch.setattr(
        fsl, "flirt", lambda **kwargs: (kwargs["out_file"], kwargs["omat"])
    )
    monkeypatch.setattr(
        fsl, "convert_xfm", lambda **kwargs: kwargs["out_file"]
    )
    monkeypatch.setattr(c3d, "fsl2ants", lambda **kwargs: kwargs["out_file"])
    monkeypatch.setattr(
        ants, "apply_transforms", lambda **kwargs: kwargs["out_file"]
    )

    step._register_t1w_to_dwi(supplied_t1w_brain, dwi_synth_t1w, tmp_path)


def test_supersynth_mgz_registration_pair_is_converted_to_nifti(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration": "supersynth"})
    moving = tmp_path / "moving_SynthT1.mgz"
    fixed = tmp_path / "fixed_SynthT1.mgz"
    conversions = []

    def fake_convert(**kwargs):
        conversions.append((kwargs["in_file"], kwargs["out_file"]))
        return kwargs["out_file"]

    monkeypatch.setattr(freesurfer, "mri_convert", fake_convert)
    monkeypatch.setattr(
        fsl, "flirt", lambda **kwargs: (kwargs["out_file"], kwargs["omat"])
    )
    monkeypatch.setattr(
        fsl, "convert_xfm", lambda **kwargs: kwargs["out_file"]
    )
    monkeypatch.setattr(c3d, "fsl2ants", lambda **kwargs: kwargs["out_file"])
    monkeypatch.setattr(
        ants, "apply_transforms", lambda **kwargs: kwargs["out_file"]
    )

    step._register_t1w_to_dwi(moving, fixed, tmp_path)

    assert conversions == [
        (moving, tmp_path / "registration_moving.nii.gz"),
        (fixed, tmp_path / "registration_ref.nii.gz"),
    ]


def test_ants_backend_estimates_linear_transform_and_inverts_same_matrix(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration_backend": "ants"})
    t1w_brain = tmp_path / "t1w_brain.nii.gz"
    t1w_norm = tmp_path / "t1w_norm.nii.gz"
    b0 = tmp_path / "b0.nii.gz"
    calls = []

    def fake_registration(**kwargs):
        calls.append(("registration", kwargs))
        return kwargs["out_prefix"].parent / "warped.nii.gz", [
            tmp_path / "t1w_2_dwi_ants_0GenericAffine.mat"
        ]

    def fake_apply(**kwargs):
        calls.append(("apply", kwargs))
        return kwargs["out_file"]

    monkeypatch.setattr(ants, "registration", fake_registration)
    monkeypatch.setattr(ants, "apply_transforms", fake_apply)

    forward, inverse = step._register_t1w_to_dwi(
        t1w_brain,
        b0,
        tmp_path,
        moving_apply_path=t1w_norm,
    )

    registration_call = calls[0][1]
    assert registration_call["moving_file"] == t1w_brain
    assert registration_call["fixed_file"] == b0
    assert registration_call["transform_type"] == "Rigid"
    assert forward.path == inverse.path
    assert forward.invert is False
    assert inverse.invert is True
    assert calls[1][1]["moving_file"] == t1w_norm
    assert calls[1][1]["fixed_file"] == b0


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("synthmorph", "synthmorph"), ("mri_synthmorph", "synthmorph")],
)
def test_synthmorph_registration_backend_is_accepted(backend, expected):
    assert _step({"registration_backend": backend})._registration_backend() == expected


def test_synthmorph_backend_estimates_rigid_forward_and_inverse_transforms(
    tmp_path: Path, monkeypatch
):
    step = _step({"registration_backend": "synthmorph"})
    t1w_brain = tmp_path / "t1w_brain.nii.gz"
    t1w_norm = tmp_path / "t1w_norm.nii.gz"
    b0 = tmp_path / "b0.nii.gz"
    calls = []

    def fake_register(**kwargs):
        calls.append(("register", kwargs))
        return kwargs["transform_out"]

    def fake_lta_to_itk(in_lta, out_file, src, trg, invert=False):
        calls.append(
            (
                "lta_to_itk",
                {
                    "in_lta": in_lta,
                    "out_file": out_file,
                    "src": src,
                    "trg": trg,
                    "invert": invert,
                },
            )
        )
        return out_file

    def fake_apply(**kwargs):
        calls.append(("apply", kwargs))
        return kwargs["out_file"]

    monkeypatch.setattr(freesurfer, "mri_synthmorph_register", fake_register)
    monkeypatch.setattr(freesurfer, "lta_to_itk", fake_lta_to_itk)
    monkeypatch.setattr(ants, "apply_transforms", fake_apply)

    forward, inverse = step._register_t1w_to_dwi(
        t1w_brain,
        b0,
        tmp_path,
        moving_apply_path=t1w_norm,
    )

    register = calls[0][1]
    assert register["moving"] == t1w_brain
    assert register["target"] == b0
    assert register["model"] == "rigid"
    assert register["transform_out"].name == "t1w_2_dwi_synthmorph_rigid.lta"
    assert register["inverse_transform_out"].name == "dwi_2_t1w_synthmorph_rigid.lta"
    assert register["overwrite"] is True

    forward_conversion = calls[1][1]
    assert forward_conversion["src"] == t1w_brain
    assert forward_conversion["trg"] == b0
    inverse_conversion = calls[2][1]
    assert inverse_conversion["src"] == b0
    assert inverse_conversion["trg"] == t1w_brain

    assert forward.path.name == "t1w_2_dwi_synthmorph_rigid.txt"
    assert inverse.path.name == "dwi_2_t1w_synthmorph_rigid.txt"
    assert forward.invert is False
    assert inverse.invert is False
    assert calls[3][1]["moving_file"] == t1w_norm
    assert calls[3][1]["fixed_file"] == b0


def test_synthmorph_backend_uses_affine_model_for_mni_registration(
    tmp_path: Path, monkeypatch
):
    step = _step(
        {
            "registration_backend": "synthmorph",
            "synthmorph_register_args": "--threads 4",
        }
    )
    t1w_norm = tmp_path / "t1w_norm.nii.gz"
    t1w_brain = tmp_path / "t1w_brain.nii.gz"
    atlas = tmp_path / "atlas.nii.gz"
    calls = []

    monkeypatch.setattr(
        freesurfer,
        "mri_synthmorph_register",
        lambda **kwargs: calls.append(kwargs) or kwargs["transform_out"],
    )
    monkeypatch.setattr(
        freesurfer,
        "lta_to_itk",
        lambda in_lta, out_file, src, trg, invert=False: out_file,
    )
    monkeypatch.setattr(
        ants, "apply_transforms", lambda **kwargs: kwargs["out_file"]
    )

    step._register_t1w_to_mni(t1w_norm, t1w_brain, atlas, tmp_path)

    assert calls[0]["model"] == "affine"
    assert calls[0]["extra_args"] == "--threads 4"
    assert calls[0]["transform_out"].name == "t1w_2_mni_synthmorph_affine.lta"


def test_synb0_synthmorph_rejects_deformable_model():
    step = _step(
        {"registration_backend": "synthmorph", "synthmorph_model": "deform"}
    )

    with pytest.raises(ValidationError, match="linear model"):
        step._synthmorph_model("Rigid")


def test_unknown_registration_backend_is_rejected():
    step = _step({"registration_backend": "niftyreg"})
    with pytest.raises(ValidationError, match="registration_backend"):
        step._registration_backend()


def test_skull_stripping_is_used_only_for_transform_estimation(
    tmp_path: Path, monkeypatch
):
    step = _step(
        {
            "registration_backend": "ants",
            "skull_strip_registration": True,
            "skull_strip_method": "synthstrip",
        }
    )
    t1w_norm = tmp_path / "t1w_norm.nii.gz"
    t1w_brain = tmp_path / "t1w_brain.nii.gz"
    atlas = tmp_path / "atlas.nii.gz"
    atlas_brain = tmp_path / "atlas_brain.nii.gz"
    calls = []

    def fake_strip(image_path, output_dir):
        calls.append(("strip", image_path, output_dir))
        return atlas_brain

    def fake_registration(**kwargs):
        calls.append(("registration", kwargs))
        return tmp_path / "warped.nii.gz", [tmp_path / "affine.mat"]

    def fake_apply(**kwargs):
        calls.append(("apply", kwargs))
        return kwargs["out_file"]

    monkeypatch.setattr(step, "_skull_strip_registration_image", fake_strip)
    monkeypatch.setattr(ants, "registration", fake_registration)
    monkeypatch.setattr(ants, "apply_transforms", fake_apply)

    step._register_t1w_to_mni(t1w_norm, t1w_brain, atlas, tmp_path)

    assert calls[0][0:2] == ("strip", atlas)
    assert calls[1][1]["moving_file"] == t1w_brain
    assert calls[1][1]["fixed_file"] == atlas_brain
    assert calls[2][1]["moving_file"] == t1w_norm
    assert calls[2][1]["fixed_file"] == atlas


def test_dwi_registration_skull_strips_both_proxies_but_applies_to_originals(
    tmp_path: Path, monkeypatch
):
    step = _step(
        {
            "registration_backend": "ants",
            "registration": "supersynth",
            "skull_strip_registration": True,
        }
    )
    moving_proxy = tmp_path / "moving_SynthT1.nii.gz"
    fixed_proxy = tmp_path / "fixed_SynthT1.nii.gz"
    original_t1w = tmp_path / "t1w_norm.nii.gz"
    stripped_moving = tmp_path / "moving_brain.nii.gz"
    stripped_fixed = tmp_path / "fixed_brain.nii.gz"
    calls = []

    def fake_strip(image_path, output_dir):
        calls.append(("strip", image_path, output_dir))
        return stripped_moving if image_path == moving_proxy else stripped_fixed

    def fake_registration(**kwargs):
        calls.append(("registration", kwargs))
        return tmp_path / "warped.nii.gz", [tmp_path / "affine.mat"]

    def fake_apply(**kwargs):
        calls.append(("apply", kwargs))
        return kwargs["out_file"]

    monkeypatch.setattr(step, "_skull_strip_registration_image", fake_strip)
    monkeypatch.setattr(ants, "registration", fake_registration)
    monkeypatch.setattr(ants, "apply_transforms", fake_apply)

    step._register_t1w_to_dwi(
        moving_proxy,
        fixed_proxy,
        tmp_path,
        moving_apply_path=original_t1w,
    )

    assert calls[0][0:2] == ("strip", moving_proxy)
    assert calls[1][0:2] == ("strip", fixed_proxy)
    assert calls[2][1]["moving_file"] == stripped_moving
    assert calls[2][1]["fixed_file"] == stripped_fixed
    assert calls[3][1]["moving_file"] == original_t1w
    assert calls[3][1]["fixed_file"] == fixed_proxy


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
