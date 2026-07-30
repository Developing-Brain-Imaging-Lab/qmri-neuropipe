"""Generic BIDS-style derivative writing for dmipy models."""

from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path
import re
from typing import Any, Mapping

import nibabel as nib
import numpy as np

from ..io.bids import build_bids_name, get_entities_from_path
from ..io.bids import _sidecar
from .dmipy_backend import DmipyRuntime, get_model_spec
from ..utils.serialization import json_ready


_MODEL_LABELS = {
    "active_ax": "ActiveAx",
    "axcaliber": "AxCaliber",
    "bingham_noddi": "BinghamNODDI",
    "charmed": "CHARMED",
    "fexi": "FEXI",
    "impulsed": "IMPULSED",
    "ivim": "IVIM",
    "mcsmt": "MCSMT",
    "mte_ball_stick": "MTEBallStick",
    "mte_impulsed": "MTEIMPULSED",
    "mte_noddi": "MTENODDI",
    "mte_sandi": "MTESANDI",
    "nexi": "NEXI",
    "noddi": "NODDI",
    "noddida": "NODDIDA",
    "noddida_mte": "NODDIDAMTE",
    "sandi": "SANDI",
    "sandix": "SANDIX",
    "verdict": "VERDICT",
    "wmti": "WMTI",
}


def bids_safe_label(value: str, *, fallback: str = "Value") -> str:
    """Return a non-empty alphanumeric label suitable for a BIDS entity."""
    raw_value = str(value)
    if re.fullmatch(r"[A-Za-z0-9]+", raw_value):
        return raw_value
    tokens = re.findall(r"[A-Za-z0-9]+", raw_value)
    if not tokens:
        return fallback
    return "".join(
        token if token.isupper() else token[:1].upper() + token[1:]
        for token in tokens
    )


def dmipy_model_label(model_name: str) -> str:
    """Return a stable BIDS model label for registry or custom models."""
    key = str(model_name).lower()
    if key in _MODEL_LABELS:
        return _MODEL_LABELS[key]
    label = bids_safe_label(model_name, fallback="Dmipy")
    return label[:1].upper() + label[1:]


def write_dmipy_derivatives(
    out_dir: Path,
    in_path: Path,
    affine: np.ndarray,
    parameter_maps: Mapping[str, np.ndarray | None],
    runtime: DmipyRuntime,
    *,
    model_name: str,
    model_label: str | None = None,
    output_aliases: Mapping[str, str] | None = None,
    base_metadata: Mapping[str, Any] | None = None,
    parameter_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write scalar or vector dmipy parameters as BIDS-style derivatives.

    Known aliases become compact metric suffixes, for example ``ODI``. Raw
    dmipy parameters without an alias use ``desc-<Parameter>_parameter`` and
    retain their complete dmipy name in the JSON sidecar.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = Path(in_path)
    entities = get_entities_from_path(in_path)
    entities.pop("desc", None)
    entities["model"] = model_label or dmipy_model_label(model_name)

    try:
        model_spec = get_model_spec(model_name)
    except ValueError:
        model_spec = None

    aliases = dict(model_spec.output_aliases if model_spec else {})
    aliases.update(output_aliases or {})
    per_parameter = parameter_metadata or {}
    common_metadata = {
        "ModelName": model_name,
        "ModelLabel": entities["model"],
        "ModelFamily": model_spec.family if model_spec else "custom",
        "AcquisitionRequirements": (
            list(model_spec.acquisition_requirements) if model_spec else []
        ),
        "ModelReferences": list(model_spec.references) if model_spec else [],
        "InputData": in_path.name,
        **dict(base_metadata or {}),
        **runtime.provenance(),
    }

    outputs: dict[str, Path] = {}
    used_names: dict[str, str] = {}
    for parameter_name, values in parameter_maps.items():
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float32)
        # dmipy exposes passive parameters such as T2 as all-NaN arrays when
        # the acquisition does not estimate them. Do not publish empty maps.
        if not np.any(np.isfinite(array)):
            continue
        parameter_entities = dict(entities)
        output_alias = aliases.get(parameter_name)
        if output_alias:
            suffix = bids_safe_label(output_alias, fallback="Parameter")
            derivative_kind = "metric"
        else:
            parameter_label = bids_safe_label(parameter_name, fallback="Parameter")
            suffix = "parameter"
            parameter_entities["desc"] = parameter_label
            derivative_kind = "parameter"

        filename = build_bids_name(parameter_entities, suffix=suffix)
        prior_parameter = used_names.get(filename)
        if prior_parameter is not None and prior_parameter != parameter_name:
            digest = sha1(parameter_name.encode("utf-8")).hexdigest()[:8]
            parameter_entities["desc"] = (
                parameter_entities.get("desc", suffix) + digest
            )
            filename = build_bids_name(parameter_entities, suffix=suffix)
        used_names[filename] = parameter_name

        out_path = out_dir / filename
        nib.save(nib.Nifti1Image(array, affine), out_path)
        metadata = {
            **common_metadata,
            "Parameter": parameter_name,
            "OutputAlias": output_alias,
            "BIDSSuffix": suffix,
            "DerivativeKind": derivative_kind,
            "ParameterCardinality": (
                int(np.prod(array.shape[3:])) if array.ndim > 3 else 1
            ),
            "ParameterComponentShape": list(array.shape[3:]),
            **dict(per_parameter.get(parameter_name, {})),
        }
        with _sidecar(out_path, ".json").open("w") as stream:
            json.dump(json_ready(metadata), stream, indent=2)
        outputs[parameter_name] = out_path
    return outputs


def write_dmipy_fit_result(
    out_dir: Path,
    in_path: Path,
    affine: np.ndarray,
    fit_result: Any,
    runtime: DmipyRuntime,
    *,
    model_name: str,
    model_label: str | None = None,
    output_aliases: Mapping[str, str] | None = None,
    base_metadata: Mapping[str, Any] | None = None,
    parameter_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write every parameter exposed by a dmipy ``fit`` result.

    This is the model-independent bridge between :func:`fit_model` and the
    derivative writer. Model-specific interfaces may instead pass derived maps
    directly to :func:`write_dmipy_derivatives`.
    """
    parameter_maps = getattr(fit_result, "fitted_parameters", None)
    if not isinstance(parameter_maps, Mapping):
        raise TypeError(
            "dmipy fit result must expose fitted_parameters as a mapping."
        )
    return write_dmipy_derivatives(
        out_dir,
        in_path,
        affine,
        parameter_maps,
        runtime,
        model_name=model_name,
        model_label=model_label,
        output_aliases=output_aliases,
        base_metadata=base_metadata,
        parameter_metadata=parameter_metadata,
    )
