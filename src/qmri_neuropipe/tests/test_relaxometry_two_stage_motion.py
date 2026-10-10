import logging
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.interfaces import freesurfer, fsl
from qmri_neuropipe.lib.relax.motion import RelaxometryMotionCorrectionStep
from qmri_neuropipe.lib.relax import motion as motion_module


def _step(tmp_path: Path, options=None) -> RelaxometryMotionCorrectionStep:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
    )
    return RelaxometryMotionCorrectionStep(
        config,
        logging.getLogger("test-two-stage-ssfp"),
        {},
        method="ants",
        options=options or {"ssfp_two_stage": {"enabled": True}},
    )


def test_normalized_median_ssfp_reference_preserves_geometry(tmp_path):
    first = np.ones((3, 3, 3), dtype=np.float32)
    second = np.full((3, 3, 3), 10, dtype=np.float32)
    first[0, 0, 0] = 0
    second[0, 0, 0] = 20
    data = np.stack([first, second], axis=3)
    affine = np.diag([1.2, 1.3, 1.4, 1.0])
    source = tmp_path / "ssfp.nii.gz"
    output = tmp_path / "ssfp_reference.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), source)

    RelaxometryMotionCorrectionStep._build_ssfp_reference(
        source, output, mode="median", normalize=True
    )

    result = nib.load(output)
    assert result.shape == (3, 3, 3)
    np.testing.assert_allclose(result.affine, affine)
    np.testing.assert_allclose(result.get_fdata()[1:, 1:, 1:], 1.0)
    assert result.get_fdata()[0, 0, 0] == 1.0


def test_aligned_template_combines_multiple_series_and_normalizes_signal(tmp_path):
    affine = np.diag([1.1, 1.2, 1.3, 1.0])
    first = tmp_path / "first.nii.gz"
    second = tmp_path / "second.nii.gz"
    output = tmp_path / "template.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.stack([np.ones((2, 2, 2)), np.full((2, 2, 2), 2)], axis=3),
            affine,
        ),
        first,
    )
    nib.save(
        nib.Nifti1Image(np.full((2, 2, 2), 10, dtype=np.float32), affine),
        second,
    )

    RelaxometryMotionCorrectionStep._build_aligned_template(
        [first, second], output, mode="mean", normalize=True
    )

    result = nib.load(output)
    assert result.shape == (2, 2, 2)
    np.testing.assert_allclose(result.affine, affine)
    np.testing.assert_allclose(result.get_fdata(), 1.0)


def test_stage_specific_options_override_shared_registration_options(tmp_path):
    step = _step(
        tmp_path,
        {
            "transform_type": "DenseRigid",
            "aff_metric": "mattes",
            "ssfp_two_stage": {
                "enabled": True,
                "within_options": {"aff_metric": "GC"},
                "cross_options": {"aff_sampling": 64},
            },
        },
    )

    assert step._stage_options("within") == {
        "transform_type": "DenseRigid",
        "aff_metric": "GC",
    }
    assert step._stage_options("cross") == {
        "transform_type": "DenseRigid",
        "aff_metric": "mattes",
        "aff_sampling": 64,
        "initial_transform": None,
    }


def test_explicit_cross_initialization_overrides_template_default(tmp_path):
    step = _step(
        tmp_path,
        {
            "initial_transform": "Identity",
            "ssfp_two_stage": {
                "enabled": True,
                "cross_options": {"initial_transform": "Identity"},
            },
        },
    )

    assert step._stage_options("within")["initial_transform"] == "Identity"
    assert step._stage_options("cross")["initial_transform"] == "Identity"


def test_two_stage_ssfp_uses_templates_and_one_final_composed_resampling(
    tmp_path, monkeypatch
):
    step = _step(tmp_path)
    ssfp_reference = tmp_path / "ssfp_reference.nii.gz"
    spgr_reference = tmp_path / "spgr_reference.nii.gz"
    spgr_template = tmp_path / "spgr_aligned_template.nii.gz"
    volumes = [tmp_path / "vol0000.nii.gz", tmp_path / "vol0001.nii.gz"]
    calls = []

    def estimate(moving, fixed, prefix, options):
        if Path(moving).name == "ssfp_aligned_template.nii.gz":
            assert Path(fixed) == spgr_template
            return [tmp_path / "ssfp_to_spgr.mat"]
        return [tmp_path / f"{Path(moving).stem}_to_ssfp.mat"]

    def apply(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["out_file"])
        if output.name.endswith("_within_aligned.nii.gz"):
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), output)
        else:
            output.touch()

    monkeypatch.setattr(step, "_estimate_ants_transforms", estimate)
    monkeypatch.setattr(step, "_cleanup_ants_outputs", lambda prefix: None)
    monkeypatch.setattr(motion_module.ants, "apply_transforms", apply)

    outputs = step._run_two_stage_ssfp(
        volumes,
        ssfp_reference,
        spgr_reference,
        tmp_path / "split",
        cross_reference=spgr_template,
    )

    assert len(outputs) == 2
    assert len(calls) == 4
    temporary_calls = calls[:2]
    final_calls = calls[2:]
    for index, call in enumerate(temporary_calls):
        assert call["fixed_file"] == ssfp_reference
        assert call["moving_file"] == volumes[index]
        assert call["transforms"] == [
            tmp_path / f"{volumes[index].stem}_to_ssfp.mat"
        ]
    for index, call in enumerate(final_calls):
        assert call["fixed_file"] == spgr_reference
        assert call["moving_file"] == volumes[index]
        assert call["transforms"] == [
            tmp_path / "ssfp_to_spgr.mat",
            tmp_path / f"{volumes[index].stem}_to_ssfp.mat",
        ]
    assert all(output.exists() for output in outputs)


def test_ssfp_run_routes_through_two_stage_strategy_and_records_metadata(
    tmp_path, monkeypatch
):
    step = _step(tmp_path)
    data = np.stack(
        [np.ones((2, 2, 2), dtype=np.float32), np.full((2, 2, 2), 2)],
        axis=3,
    )
    source = tmp_path / "sub-01_acq-SSFP_VFA.nii.gz"
    source_json = tmp_path / "sub-01_acq-SSFP_VFA.json"
    reference = tmp_path / "spgr_reference.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), reference)
    source_json.write_text(json.dumps({"FlipAngle": [10, 20]}))
    image = ImageFile(
        img=source,
        json=source_json,
        entities={"sub": "01", "acq": "SSFP", "suffix": "VFA"},
    )
    reference_image = ImageFile(
        img=reference,
        entities={"sub": "01", "desc": "spgrref", "suffix": "VFA"},
    )
    captured = {}

    def fake_split(in_file, prefix):
        outputs = []
        input_nii = nib.load(str(in_file))
        for index in range(input_nii.shape[3]):
            output = prefix.parent / f"vol{index:04d}.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.asanyarray(input_nii.dataobj[..., index]),
                    input_nii.affine,
                    input_nii.header,
                ),
                output,
            )
            outputs.append(output)
        return outputs

    def fake_two_stage(
        volumes,
        ssfp_reference,
        spgr_reference,
        split_dir,
        cross_reference=None,
    ):
        captured["volumes"] = volumes
        captured["ssfp_reference"] = ssfp_reference
        captured["spgr_reference"] = spgr_reference
        outputs = []
        for index, volume in enumerate(volumes):
            output = split_dir / f"corrected{index:04d}.nii.gz"
            nii = nib.load(str(volume))
            nib.save(nib.Nifti1Image(np.asanyarray(nii.dataobj), nii.affine, nii.header), output)
            outputs.append(output)
        return outputs

    def fake_merge(images, out_file, dimension="t"):
        niis = [nib.load(str(image)) for image in images]
        merged = np.stack([np.asanyarray(nii.dataobj) for nii in niis], axis=3)
        nib.save(nib.Nifti1Image(merged, niis[0].affine, niis[0].header), out_file)
        return out_file

    monkeypatch.setattr(fsl, "split", fake_split)
    monkeypatch.setattr(fsl, "merge", fake_merge)
    monkeypatch.setattr(step, "_run_two_stage_ssfp", fake_two_stage)

    result = step.run(
        [image],
        tmp_path / "work",
        force=True,
        reference_image=reference_image,
        modality="SSFP",
    )

    assert len(captured["volumes"]) == 2
    assert captured["spgr_reference"] == reference
    assert nib.load(str(result[0].img)).shape == (2, 2, 2, 2)
    metadata = json.loads(Path(result[0].json).read_text())
    assert metadata["MotionCorrection"] == {
        "strategy": "ssfp_two_stage_aligned_templates",
        "backend": "ants",
        "ssfp_reference_mode": "median",
        "ssfp_reference_normalized": True,
        "transform_application": "composed_single_resampling",
        "cross_modality_templates": "within_aligned",
        "template_mode": "median",
        "template_normalized": True,
    }

    def fail_if_rerun(*args, **kwargs):
        raise AssertionError("cached two-stage output should be reused")

    monkeypatch.setattr(step, "_run_two_stage_ssfp", fail_if_rerun)
    cached = step.run(
        [image],
        tmp_path / "work",
        force=False,
        reference_image=reference_image,
        modality="SSFP",
    )
    cached_metadata = json.loads(Path(cached[0].json).read_text())
    assert (
        cached_metadata["MotionCorrection"]["strategy"]
        == "ssfp_two_stage_aligned_templates"
    )


def test_two_stage_ssfp_composes_flirt_matrices(tmp_path, monkeypatch):
    step = _step(
        tmp_path,
        {
            "cost": "normmi",
            "interpolation": "sinc",
            "ssfp_two_stage": {"enabled": True},
        },
    )
    step.method = "fsl"
    ssfp_reference = tmp_path / "ssfp_reference.nii.gz"
    spgr_reference = tmp_path / "spgr_reference.nii.gz"
    volumes = [tmp_path / "vol0000.nii.gz", tmp_path / "vol0001.nii.gz"]
    estimates = []
    applications = []
    compositions = []

    def estimate(moving, fixed, prefix, options):
        transform = tmp_path / f"{Path(prefix).name}transform.mat"
        estimates.append((Path(moving), Path(fixed), transform))
        return transform

    def fake_flirt(**kwargs):
        applications.append(kwargs)
        output = Path(kwargs["out_file"])
        if output.name.endswith("_within_aligned.nii.gz"):
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), output)
        else:
            output.touch()
        return output, kwargs.get("omat")

    def fake_convert(in_file, out_file, inverse=False, concat_mat=None):
        compositions.append((Path(in_file), Path(concat_mat), Path(out_file)))
        Path(out_file).touch()
        return Path(out_file)

    monkeypatch.setattr(step, "_estimate_fsl_transform", estimate)
    monkeypatch.setattr(fsl, "flirt", fake_flirt)
    monkeypatch.setattr(fsl, "convert_xfm", fake_convert)

    outputs = step._run_two_stage_ssfp(
        volumes, ssfp_reference, spgr_reference, tmp_path / "fsl_split"
    )

    assert len(estimates) == 3
    assert len(compositions) == 2
    cross_transform = estimates[-1][2]
    for index, (within, cross, composed) in enumerate(compositions):
        assert within == estimates[index][2]
        assert cross == cross_transform
        assert composed.name == f"vol{index:04d}_to_spgr_composed.mat"
    assert len(applications) == 4
    assert all("-applyxfm" in call["extra_args"] for call in applications)
    assert all("-interp sinc" in call["extra_args"] for call in applications)
    assert all(output.exists() for output in outputs)


def test_two_stage_ssfp_composes_synthmorph_ltas(tmp_path, monkeypatch):
    step = _step(
        tmp_path,
        {
            "synthmorph_model": "rigid",
            "synthmorph_apply_args": "-m linear",
            "ssfp_two_stage": {"enabled": True},
        },
    )
    step.method = "synthmorph"
    ssfp_reference = tmp_path / "ssfp_reference.nii.gz"
    spgr_reference = tmp_path / "spgr_reference.nii.gz"
    volumes = [tmp_path / "vol0000.nii.gz", tmp_path / "vol0001.nii.gz"]
    estimates = []
    applications = []
    compositions = []

    def estimate(moving, fixed, prefix, options):
        transform = tmp_path / f"{Path(prefix).name}transform.lta"
        estimates.append((Path(moving), Path(fixed), transform))
        return transform

    def fake_apply(**kwargs):
        applications.append(kwargs)
        output = Path(kwargs["out_file"])
        if output.name.endswith("_within_aligned.nii.gz"):
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), output)
        else:
            output.touch()
        return output

    def fake_concatenate(first, second, output, overwrite=False):
        compositions.append((Path(first), Path(second), Path(output), overwrite))
        Path(output).touch()
        return Path(output)

    monkeypatch.setattr(step, "_estimate_synthmorph_transform", estimate)
    monkeypatch.setattr(freesurfer, "mri_synthmorph_apply", fake_apply)
    monkeypatch.setattr(freesurfer, "mri_concatenate_lta", fake_concatenate)

    outputs = step._run_two_stage_ssfp(
        volumes, ssfp_reference, spgr_reference, tmp_path / "synth_split"
    )

    assert len(estimates) == 3
    assert len(compositions) == 2
    cross_transform = estimates[-1][2]
    for index, (within, cross, composed, overwrite) in enumerate(compositions):
        assert within == estimates[index][2]
        assert cross == cross_transform
        assert composed.name == f"vol{index:04d}_to_spgr_composed.lta"
        assert overwrite is True
    assert len(applications) == 4
    assert all(call["extra_args"] == "-m linear" for call in applications)
    assert all(output.exists() for output in outputs)
