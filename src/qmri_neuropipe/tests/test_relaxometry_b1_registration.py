import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.interfaces import ants, fsl
from qmri_neuropipe.lib.relax.b1 import B1MappingStep


def _step(tmp_path: Path, registration: dict) -> B1MappingStep:
    config = PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
    )
    return B1MappingStep(
        config,
        logging.getLogger("test-b1-registration"),
        {},
        method="afi",
        registration=registration,
    )


def _image(path: Path, shape, zoom, value=1.0) -> Path:
    affine = np.diag([zoom, zoom, zoom, 1.0])
    nib.save(
        nib.Nifti1Image(np.full(shape, value, dtype=np.float32), affine),
        path,
    )
    return path


def test_assume_aligned_is_backend_independent_header_resampling(tmp_path):
    source = _image(tmp_path / "sub-01_TB1map.nii.gz", (4, 4, 4), 6.0)
    reference = _image(tmp_path / "spgr_reference.nii.gz", (10, 10, 10), 2.0)
    step = _step(
        tmp_path,
        {"method": "future-backend", "assume_aligned": True},
    )

    result = step.run(
        ImageFile(img=source, entities={"sub": "01", "suffix": "TB1map"}),
        ImageFile(img=reference, entities={"sub": "01", "suffix": "VFA"}),
        tmp_path / "work",
        force=True,
    )

    output = nib.load(result.img)
    fixed = nib.load(reference)
    assert output.shape == fixed.shape
    np.testing.assert_allclose(output.affine, fixed.affine)


def test_assume_aligned_raw_afi_resamples_both_volumes_without_backend(
    tmp_path, monkeypatch
):
    source = tmp_path / "sub-01_acq-afi_TB1AFI.nii.gz"
    affine = np.diag([6.0, 6.0, 6.0, 1.0])
    nib.save(
        nib.Nifti1Image(
            np.stack(
                [
                    np.ones((4, 4, 4), dtype=np.float32),
                    np.full((4, 4, 4), 2, dtype=np.float32),
                ],
                axis=3,
            ),
            affine,
        ),
        source,
    )
    reference = _image(tmp_path / "spgr_reference.nii.gz", (10, 10, 10), 2.0)
    step = _step(
        tmp_path,
        {"method": "future-backend", "assume_aligned": True},
    )
    captured = {}

    def fake_split(in_file, prefix):
        image = nib.load(str(in_file))
        outputs = []
        for index in range(image.shape[3]):
            output = prefix.parent / f"vol{index:04d}.nii.gz"
            nib.save(
                nib.Nifti1Image(
                    np.asanyarray(image.dataobj[..., index]),
                    image.affine,
                    image.header,
                ),
                output,
            )
            outputs.append(output)
        return outputs

    def fake_merge(images, out_file, dimension="t"):
        inputs = [nib.load(str(image)) for image in images]
        merged = np.stack(
            [np.asanyarray(image.dataobj) for image in inputs], axis=3
        )
        nib.save(nib.Nifti1Image(merged, inputs[0].affine), out_file)
        return out_file

    def fake_compute(image, output_dir):
        captured["shape"] = nib.load(str(image.img)).shape
        output = output_dir / "computed_b1.nii.gz"
        _image(output, (10, 10, 10), 2.0)
        return output

    monkeypatch.setattr(fsl, "split", fake_split)
    monkeypatch.setattr(fsl, "merge", fake_merge)
    monkeypatch.setattr(step, "_compute_afi", fake_compute)

    step.run(
        ImageFile(img=source, entities={"sub": "01", "suffix": "TB1AFI"}),
        ImageFile(img=reference, entities={"sub": "01", "suffix": "VFA"}),
        tmp_path / "work",
        force=True,
    )

    assert captured["shape"] == (10, 10, 10, 2)


def test_resolution_matching_preserves_fixed_center_and_uses_moving_voxel_size(
    tmp_path,
):
    moving = _image(tmp_path / "afi_reference.nii.gz", (4, 4, 4), 6.0)
    fixed = _image(tmp_path / "spgr_reference.nii.gz", (10, 10, 10), 2.0)
    step = _step(tmp_path, {"match_resolution": True})

    proxy_path = step._fixed_resolution_proxy(moving, fixed, tmp_path / "work")

    proxy = nib.load(proxy_path)
    fixed_img = nib.load(fixed)
    np.testing.assert_allclose(voxel_sizes(proxy.affine), (6.0, 6.0, 6.0))
    fixed_center = nib.affines.apply_affine(
        fixed_img.affine, (np.asarray(fixed_img.shape) - 1) / 2
    )
    proxy_center = nib.affines.apply_affine(
        proxy.affine, (np.asarray(proxy.shape) - 1) / 2
    )
    np.testing.assert_allclose(proxy_center, fixed_center)


def test_ants_estimates_on_matched_proxy_but_outputs_on_original_spgr_grid(
    tmp_path, monkeypatch
):
    b1_map = _image(tmp_path / "sub-01_TB1map.nii.gz", (4, 4, 4), 6.0)
    b1_ref = _image(tmp_path / "afi_reference.nii.gz", (4, 4, 4), 6.0)
    spgr_ref = _image(tmp_path / "spgr_reference.nii.gz", (10, 10, 10), 2.0)
    step = _step(
        tmp_path,
        {
            "method": "ants",
            "match_resolution": True,
            "transform_type": "DenseRigid",
        },
    )
    calls = {}

    def fake_registration(**kwargs):
        calls["registration"] = kwargs
        return tmp_path / "warped.nii.gz", [tmp_path / "transform.mat"]

    def fake_apply(**kwargs):
        calls["apply"] = kwargs
        fixed = nib.load(str(kwargs["fixed_file"]))
        nib.save(
            nib.Nifti1Image(
                np.ones(fixed.shape, dtype=np.float32), fixed.affine
            ),
            kwargs["out_file"],
        )

    monkeypatch.setattr(ants, "registration", fake_registration)
    monkeypatch.setattr(ants, "apply_transforms", fake_apply)

    result = step.run(
        ImageFile(img=b1_map, entities={"sub": "01", "suffix": "TB1map"}),
        ImageFile(img=spgr_ref, entities={"sub": "01", "suffix": "VFA"}),
        tmp_path / "work",
        force=True,
        b1_ref_image=ImageFile(img=b1_ref, entities={}),
    )

    estimation_fixed = nib.load(str(calls["registration"]["fixed_file"]))
    np.testing.assert_allclose(voxel_sizes(estimation_fixed.affine), (6, 6, 6))
    assert Path(calls["apply"]["fixed_file"]) == spgr_ref
    assert nib.load(result.img).shape == (10, 10, 10)


def test_fsl_estimates_on_matched_proxy_but_outputs_on_original_spgr_grid(
    tmp_path, monkeypatch
):
    b1_map = _image(tmp_path / "sub-01_TB1map.nii.gz", (4, 4, 4), 6.0)
    spgr_ref = _image(tmp_path / "spgr_reference.nii.gz", (10, 10, 10), 2.0)
    step = _step(
        tmp_path,
        {"method": "fsl", "match_resolution": True, "interpolation": "linear"},
    )
    calls = []

    def fake_flirt(**kwargs):
        calls.append(kwargs)
        fixed = nib.load(str(kwargs["ref_file"]))
        nib.save(
            nib.Nifti1Image(
                np.ones(fixed.shape, dtype=np.float32), fixed.affine
            ),
            kwargs["out_file"],
        )
        if kwargs.get("omat"):
            Path(kwargs["omat"]).write_text("identity")
        return kwargs["out_file"], kwargs.get("omat")

    monkeypatch.setattr(fsl, "flirt", fake_flirt)

    result = step.run(
        ImageFile(img=b1_map, entities={"sub": "01", "suffix": "TB1map"}),
        ImageFile(img=spgr_ref, entities={"sub": "01", "suffix": "VFA"}),
        tmp_path / "work",
        force=True,
    )

    estimation_fixed = nib.load(str(calls[0]["ref_file"]))
    np.testing.assert_allclose(voxel_sizes(estimation_fixed.affine), (6, 6, 6))
    assert Path(calls[1]["ref_file"]) == spgr_ref
    assert "-interp trilinear" in calls[1]["extra_args"]
    assert nib.load(result.img).shape == (10, 10, 10)
