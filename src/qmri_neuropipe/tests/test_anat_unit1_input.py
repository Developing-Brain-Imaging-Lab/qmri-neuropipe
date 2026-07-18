from pathlib import Path

from qmri_neuropipe.io.anat.bids import bids_find_t1w, select_anatomical_candidates


def test_bids_find_t1w_includes_unit1_as_t1w_input(tmp_path: Path):
    anat_dir = tmp_path / "sub-01" / "anat"
    anat_dir.mkdir(parents=True)
    unit1 = anat_dir / "sub-01_acq-mp2rage_UNIT1.nii.gz"
    unit1.touch()
    unit1.with_name("sub-01_acq-mp2rage_UNIT1.json").write_text(
        '{"SeriesDescription": "MP2RAGE"}'
    )

    images = bids_find_t1w(anat_dir)

    assert len(images) == 1
    assert images[0].img == unit1
    assert images[0].entities["suffix"] == "T1w"
    assert images[0].entities["acq"] == "mp2rage"
    assert images[0].json == unit1.with_name("sub-01_acq-mp2rage_UNIT1.json")


def test_unit1_can_be_selected_from_mprage_by_acquisition(tmp_path: Path):
    anat_dir = tmp_path / "anat"
    anat_dir.mkdir()
    (anat_dir / "sub-01_acq-mprage_T1w.nii.gz").touch()
    unit1 = anat_dir / "sub-01_acq-mp2rage_UNIT1.nii.gz"
    unit1.touch()

    images = bids_find_t1w(anat_dir)
    selected = select_anatomical_candidates(
        images,
        {"entities": {"acq": "mp2rage"}},
        "T1w",
    )

    assert [image.img for image in selected] == [unit1]
