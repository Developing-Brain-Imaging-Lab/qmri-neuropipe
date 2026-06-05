import json
from pathlib import Path

import nibabel as nib
import numpy as np

from qmri_neuropipe.core.types import DWIFile
from qmri_neuropipe.io.dmri.bids import (
    phase_encoding_transform_matrix,
    transform_acqparams_file,
    transform_phase_encoding_direction,
)
from qmri_neuropipe.lib.dmri.reorient import DMRIReorientStep


def test_phase_encoding_direction_follows_axis_permutation():
    source_affine = np.eye(4)
    target_affine = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    assert transform_phase_encoding_direction("i", source_affine, target_affine) == "j-"
    assert transform_phase_encoding_direction("j-", source_affine, target_affine) == "i-"


def test_phase_encoding_direction_follows_axis_flip():
    source_affine = np.eye(4)
    target_affine = np.diag([-1.0, 1.0, 1.0, 1.0])

    assert transform_phase_encoding_direction("i", source_affine, target_affine) == "i-"
    assert transform_phase_encoding_direction("j", source_affine, target_affine) == "j"


def test_acqparams_rows_follow_reorientation(tmp_path: Path):
    source_affine = np.eye(4)
    target_affine = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source = tmp_path / "acqparams.txt"
    destination = tmp_path / "reoriented_acqparams.txt"
    source.write_text("1 0 0 0.050000\n0 -1 0 0.075000\n", encoding="utf-8")

    transform_acqparams_file(
        source,
        destination,
        phase_encoding_transform_matrix(source_affine, target_affine),
    )

    assert destination.read_text() == "0 -1 0 0.050000\n-1 0 0 0.075000\n"


def test_reorient_step_updates_sidecar_and_pipeline_context(tmp_path: Path):
    source_affine = np.eye(4)
    target_affine = np.array(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source_img = tmp_path / "source.nii.gz"
    target_img = tmp_path / "target.nii.gz"
    source_json = tmp_path / "source.json"
    target_json = tmp_path / "target.json"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), source_affine), source_img)
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), target_affine), target_img)
    source_json.write_text('{"PhaseEncodingDirection": "i"}\n', encoding="utf-8")
    target_json.write_text("{}\n", encoding="utf-8")

    source = DWIFile(img=source_img, json=source_json, entities={})
    target = DWIFile(img=target_img, json=target_json, entities={})
    step = DMRIReorientStep(config={})
    step._update_output_phase_encoding(source, target)

    assert json.loads(target_json.read_text())["PhaseEncodingDirection"] == "j-"

    acqparams = tmp_path / "original_acqparams.txt"
    acqparams.write_text("1 0 0 0.050000\n", encoding="utf-8")
    context = {
        "acqp": acqparams,
        "merged_acqp": acqparams,
        "topup_groups": [{"acqp": acqparams}],
        "merge_source_info": [{"phase_encoding_direction": "i"}],
    }
    output_dir = tmp_path / "reorient"
    output_dir.mkdir()
    step._update_context_phase_encoding(context, [target], output_dir)

    expected_acqparams = output_dir / "acqparams.txt"
    assert context["acqp"] == expected_acqparams
    assert context["merged_acqp"] == expected_acqparams
    assert context["topup_groups"][0]["acqp"] == expected_acqparams
    assert expected_acqparams.read_text() == "0 -1 0 0.050000\n"
    assert context["merge_source_info"][0]["phase_encoding_direction"] == "j-"
