import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from qmri_neuropipe.interfaces.dmipy_backend import MODEL_REGISTRY, DmipyRuntime
from qmri_neuropipe.interfaces.dmipy_derivatives import (
    bids_safe_label,
    dmipy_model_label,
    write_dmipy_derivatives,
    write_dmipy_fit_result,
)


@pytest.fixture
def runtime():
    return DmipyRuntime(
        version="2.1.0",
        solver="brute2fine",
        requested_device="auto",
        backend="native-cpu",
    )


def _metadata(image_path):
    sidecar = image_path.with_name(image_path.name[:-7] + ".json")
    return json.loads(sidecar.read_text())


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_every_registry_model_can_write_generic_derivatives(
    tmp_path, runtime, model_name
):
    label = dmipy_model_label(model_name)

    assert label
    assert label.isalnum()
    outputs = write_dmipy_derivatives(
        tmp_path / model_name,
        tmp_path / "sub-01_desc-preproc_dwi.nii.gz",
        np.eye(4),
        {"raw_parameter": np.ones((1, 1, 1))},
        runtime,
        model_name=model_name,
    )
    output = outputs["raw_parameter"]
    assert f"_model-{label}_desc-RawParameter_parameter.nii.gz" in output.name
    metadata = _metadata(output)
    spec = MODEL_REGISTRY[model_name]
    assert metadata["ModelFamily"] == spec.family
    assert metadata["AcquisitionRequirements"] == list(
        spec.acquisition_requirements
    )


def test_registry_alias_is_used_as_metric_suffix(tmp_path, runtime):
    input_path = tmp_path / "sub-01_ses-02_desc-preproc_dwi.nii.gz"
    parameter = "SD1WatsonDistributed_1_SD1Watson_1_odi"

    outputs = write_dmipy_derivatives(
        tmp_path / "out",
        input_path,
        np.eye(4),
        {parameter: np.ones((2, 2, 2))},
        runtime,
        model_name="noddi",
    )

    output = outputs[parameter]
    assert output.name == "sub-01_ses-02_model-NODDI_ODI.nii.gz"
    metadata = _metadata(output)
    assert metadata["Parameter"] == parameter
    assert metadata["OutputAlias"] == "ODI"
    assert metadata["BIDSSuffix"] == "ODI"
    assert metadata["DerivativeKind"] == "metric"
    assert metadata["ModelFamily"] == "orientation-dispersion"


def test_unknown_parameter_uses_lossless_sidecar_and_safe_fallback_name(
    tmp_path, runtime
):
    input_path = tmp_path / "sub-01_acq-multishell_desc-preproc_dwi.nii.gz"
    parameter = "BundleModel_1_C1Stick_1_lambda_par"

    outputs = write_dmipy_derivatives(
        tmp_path / "out",
        input_path,
        np.eye(4),
        {parameter: np.ones((2, 2, 2))},
        runtime,
        model_name="ball_and_stick",
    )

    output = outputs[parameter]
    assert output.name == (
        "sub-01_acq-multishell_model-BallAndStick_"
        "desc-BundleModel1C1Stick1LambdaPar_parameter.nii.gz"
    )
    metadata = _metadata(output)
    assert metadata["Parameter"] == parameter
    assert metadata["OutputAlias"] is None
    assert metadata["BIDSSuffix"] == "parameter"
    assert metadata["DerivativeKind"] == "parameter"


def test_fit_result_writer_preserves_vector_parameters(tmp_path, runtime):
    parameter = "SD1Watson_1_mu"
    values = np.ones((2, 2, 2, 2))
    fit_result = SimpleNamespace(fitted_parameters={parameter: values})

    outputs = write_dmipy_fit_result(
        tmp_path / "out",
        tmp_path / "sub-01_dwi.nii.gz",
        np.eye(4),
        fit_result,
        runtime,
        model_name="bingham_noddi",
    )

    output = outputs[parameter]
    assert nib.load(output).shape == values.shape
    assert _metadata(output)["ParameterCardinality"] == 2


def test_fit_result_writer_rejects_missing_parameter_mapping(tmp_path, runtime):
    with pytest.raises(TypeError, match="fitted_parameters"):
        write_dmipy_fit_result(
            tmp_path / "out",
            tmp_path / "sub-01_dwi.nii.gz",
            np.eye(4),
            object(),
            runtime,
            model_name="ball",
        )


def test_writer_omits_all_nonfinite_passive_parameters(tmp_path, runtime):
    outputs = write_dmipy_derivatives(
        tmp_path / "out",
        tmp_path / "sub-01_dwi.nii.gz",
        np.eye(4),
        {
            "fitted": np.ones((1, 1, 1)),
            "passive_T2": np.full((1, 1, 1), np.nan),
        },
        runtime,
        model_name="nexi",
    )

    assert set(outputs) == {"fitted"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ODI", "ODI"),
        ("d_in", "DIn"),
        ("partial.volume-0", "PartialVolume0"),
        ("***", "Parameter"),
    ],
)
def test_bids_safe_label(raw, expected):
    assert bids_safe_label(raw, fallback="Parameter") == expected
