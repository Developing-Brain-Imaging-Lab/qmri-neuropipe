from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.workflows.pipelines.dmri import (
    _partition_dwi_inputs,
    _select_dwi_inputs,
    _validate_dwi_gradient_tables,
)


def _dwi(name, **entities):
    return DWIFile(img=Path(name), entities={"sub": "101", "suffix": "dwi", **entities})


def test_desc_groups_are_separated_and_preserved_as_acquisition():
    prisma_ap = _dwi("sub-101_desc-prisma_dir-AP_dwi.nii.gz", desc="prisma", dir="AP")
    prisma_pa = _dwi("sub-101_desc-prisma_dir-PA_dwi.nii.gz", desc="prisma", dir="PA")
    trio = _dwi("sub-101_desc-trio_dwi.nii.gz", desc="trio")

    groups = _partition_dwi_inputs(
        [trio, prisma_pa, prisma_ap],
        {"inputs": {"separate_by": "desc"}},
    )

    assert [(label, len(files)) for label, files in groups] == [
        ("prisma", 2),
        ("trio", 1),
    ]
    assert prisma_ap.entities["acq"] == "prisma"
    assert prisma_pa.entities["acq"] == "prisma"
    assert trio.entities["acq"] == "trio"


def test_existing_bids_acquisition_can_be_used_directly_for_grouping():
    prisma = _dwi("sub-101_acq-prisma_dwi.nii.gz", acq="prisma")
    trio = _dwi("sub-101_acq-trio_dwi.nii.gz", acq="trio")

    groups = _partition_dwi_inputs(
        [trio, prisma],
        {"inputs": {"separate_by": "acq"}},
    )

    assert [label for label, _ in groups] == ["prisma", "trio"]
    assert prisma.entities["acq"] == "prisma"


def test_grouping_rejects_inputs_missing_the_requested_entity():
    with pytest.raises(ValueError, match="missing BIDS entity"):
        _partition_dwi_inputs(
            [_dwi("sub-101_dwi.nii.gz")],
            {"inputs": {"separate_by": "desc"}},
        )


def test_grouping_is_disabled_by_default():
    files = [_dwi("sub-101_desc-prisma_dwi.nii.gz", desc="prisma")]
    assert _partition_dwi_inputs(files, {}) == [(None, files)]


def test_input_selector_can_choose_one_acquisition():
    prisma = _dwi("sub-101_acq-prisma_dwi.nii.gz", acq="prisma")
    trio = _dwi("sub-101_acq-trio_dwi.nii.gz", acq="trio")

    selected = _select_dwi_inputs(
        [prisma, trio],
        {"inputs": {"select": {"acq": "prisma"}}},
    )

    assert selected == [prisma]


def test_input_selector_accepts_multiple_values_and_entities():
    prisma_ap = _dwi("prisma_ap.nii.gz", acq="prisma", dir="AP")
    prisma_pa = _dwi("prisma_pa.nii.gz", acq="prisma", dir="PA")
    trio_ap = _dwi("trio_ap.nii.gz", acq="trio", dir="AP")

    selected = _select_dwi_inputs(
        [prisma_ap, prisma_pa, trio_ap],
        {"inputs": {"select": {"acq": ["prisma", "trio"], "dir": "AP"}}},
    )

    assert selected == [prisma_ap, trio_ap]


def _write_dwi_with_gradients(tmp_path, volumes, bvals, bvec_columns):
    img = tmp_path / "sub-101_ses-01_dir-PA_desc-trio_dwi.nii.gz"
    bval = tmp_path / "sub-101_ses-01_dir-PA_desc-trio_dwi.bval"
    bvec = tmp_path / "sub-101_ses-01_dir-PA_desc-trio_dwi.bvec"
    nib.Nifti1Image(np.zeros((2, 2, 2, volumes), dtype=np.float32), np.eye(4)).to_filename(img)
    np.savetxt(bval, np.arange(bvals)[None, :], fmt="%d")
    np.savetxt(bvec, np.zeros((3, bvec_columns)), fmt="%.1f")
    return DWIFile(
        img=img,
        bval=bval,
        bvec=bvec,
        entities={"sub": "101", "ses": "01", "dir": "PA", "desc": "trio", "suffix": "dwi"},
    )


def test_dwi_gradient_validation_accepts_matching_counts(tmp_path):
    dwi = _write_dwi_with_gradients(tmp_path, volumes=13, bvals=13, bvec_columns=13)

    _validate_dwi_gradient_tables([dwi])


def test_dwi_gradient_validation_rejects_mismatched_counts(tmp_path):
    dwi = _write_dwi_with_gradients(tmp_path, volumes=13, bvals=52, bvec_columns=52)

    with pytest.raises(ValueError, match="image has 13 volume\\(s\\).*52 b-value"):
        _validate_dwi_gradient_tables([dwi])
