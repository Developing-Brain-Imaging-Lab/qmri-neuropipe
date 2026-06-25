"""Typed, dictionary-compatible state passed between pipeline workflow steps."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional


_KNOWN_CONTEXT_KEYS = frozenset(
    {
        "subject",
        "session",
        "study_name",
        "current_image",
        "current_mask",
        "dwi_files",
        "t1w_files",
        "t2w_files",
        "relax_files",
        "t1w_file",
        "topup_groups",
        "topup_base",
        "topup_map",
        "acqp",
        "index",
        "preprocessed_dwis",
        "preprocessed_masks",
        "preprocessed_t1w",
        "preprocessed_t2w",
        "preprocessed_t2w_coreg",
        "preprocessed_t1w_brain",
        "preprocessed_t2w_brain",
        "processed_spgr",
        "processed_ssfp",
        "relax_reference",
        "spgr_ref",
        "brain_mask",
        "structural_mask",
        "b1_map",
        "fitted_maps",
        "modeling_results",
        "model_fits",
        "models_fitted",
        "normalized_results",
        "normalized_results_by_model",
        "template_transform",
        "template_transform_type",
        "spatial_transform",
        "normalization_transform_manifest",
        "gnl_map",
        "gnl_maps",
        "gnl_map_by_image",
        "gnl_transform_map",
        "outlier_stats",
        "super_synth_outputs",
        "super_synth_volumes",
        "super_synth_volumes_csv",
        "freesurfer_dir",
        "analysis_cfg",
        "analysis_modality",
        "spgr_exclude_indices",
        "ssfp_exclude_indices",
        "segmentation",
        "segmentations",
        "segmentation_stats",
        "roi_stats",
        "roi_stats_files",
        "roi_stats_combined_csv",
        "qc_metrics",
        "qc_registry",
        "qc_report",
        "processing_steps",
        "processing_steps_detail",
        "preprocessing_steps",
        "preprocessing_skipped",
        "anat_preprocessing_skipped",
        "errors",
        "reference_image",
    }
)


class _ExtraContextView(MutableMapping[str, Any]):
    """Live mapping view over keys not declared on PipelineContext."""

    def __init__(self, context: "PipelineContext") -> None:
        self._context = context

    def __getitem__(self, key: str) -> Any:
        if key in _KNOWN_CONTEXT_KEYS:
            raise KeyError(key)
        return self._context[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key in _KNOWN_CONTEXT_KEYS:
            raise KeyError(
                f"{key!r} is a typed context field; assign it on the context"
            )
        self._context[key] = value

    def __delitem__(self, key: str) -> None:
        if key in _KNOWN_CONTEXT_KEYS:
            raise KeyError(key)
        del self._context[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key in self._context if key not in _KNOWN_CONTEXT_KEYS)

    def __len__(self) -> int:
        return sum(key not in _KNOWN_CONTEXT_KEYS for key in self._context)


@dataclass(init=False, repr=False, eq=False)
class PipelineContext(dict[str, Any]):
    """Pipeline state with typed common fields and full ``dict`` compatibility."""

    subject: Optional[str]
    session: Optional[str]
    study_name: Optional[str]
    current_image: Any
    current_mask: Any
    dwi_files: Optional[list[Any]]
    t1w_files: Optional[list[Any]]
    t2w_files: Optional[list[Any]]
    relax_files: Optional[list[Any]]
    t1w_file: Any
    topup_groups: Optional[list[Any]]
    topup_base: Any
    topup_map: Any
    acqp: Any
    index: Any
    preprocessed_dwis: Optional[list[Any]]
    preprocessed_masks: Optional[list[Any]]
    preprocessed_t1w: Any
    preprocessed_t2w: Any
    preprocessed_t2w_coreg: Any
    preprocessed_t1w_brain: Any
    preprocessed_t2w_brain: Any
    processed_spgr: Optional[list[Any]]
    processed_ssfp: Optional[list[Any]]
    relax_reference: Any
    spgr_ref: Any
    brain_mask: Any
    structural_mask: Any
    b1_map: Any
    fitted_maps: Optional[dict[str, Any]]
    modeling_results: Optional[dict[str, dict[str, Path]]]
    model_fits: Optional[dict[str, Any]]
    models_fitted: Optional[list[str]]
    normalized_results: Optional[dict[str, Any]]
    normalized_results_by_model: Optional[dict[str, Any]]
    template_transform: Any
    template_transform_type: Optional[str]
    spatial_transform: Any
    normalization_transform_manifest: Any
    gnl_map: Any
    gnl_maps: Optional[list[Any]]
    gnl_map_by_image: Optional[dict[Any, Any]]
    gnl_transform_map: Optional[dict[Any, Any]]
    outlier_stats: Optional[dict[str, Any]]
    super_synth_outputs: Optional[dict[str, Any]]
    super_synth_volumes: Optional[dict[str, Any]]
    super_synth_volumes_csv: Optional[Path]
    freesurfer_dir: Optional[Path]
    analysis_cfg: Optional[dict[str, Any]]
    analysis_modality: Optional[str]
    spgr_exclude_indices: Optional[set[int]]
    ssfp_exclude_indices: Optional[set[int]]
    segmentation: Any
    segmentations: Optional[dict[str, Any]]
    segmentation_stats: Optional[list[Any]]
    roi_stats: Optional[dict[str, Any]]
    roi_stats_files: Optional[dict[str, Path]]
    roi_stats_combined_csv: Optional[Path]
    qc_metrics: Optional[dict[str, Any]]
    qc_registry: Optional[dict[str, Any]]
    qc_report: Optional[Path]
    processing_steps: Optional[list[str]]
    processing_steps_detail: Optional[list[Any]]
    preprocessing_steps: Optional[list[str]]
    preprocessing_skipped: Optional[bool]
    anat_preprocessing_skipped: Optional[bool]
    errors: Optional[list[str]]
    reference_image: Any
    extra: MutableMapping[str, Any]

    def __init__(
        self,
        initial: Optional[Mapping[str, Any]] = None,
        **values: Any,
    ) -> None:
        dict.__init__(self)
        if initial is not None:
            dict.update(self, initial)
        if values:
            dict.update(self, values)

    def __getattribute__(self, name: str) -> Any:
        if name in _KNOWN_CONTEXT_KEYS:
            return dict.get(self, name)
        return dict.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _KNOWN_CONTEXT_KEYS:
            dict.__setitem__(self, name, value)
            return
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name in _KNOWN_CONTEXT_KEYS:
            try:
                dict.__delitem__(self, name)
            except KeyError as exc:
                raise AttributeError(name) from exc
            return
        object.__delattr__(self, name)

    @property
    def extra(self) -> MutableMapping[str, Any]:
        return _ExtraContextView(self)

    @classmethod
    def ensure(
        cls,
        context: Optional[Mapping[str, Any]] = None,
        **values: Any,
    ) -> "PipelineContext":
        """Return an existing typed context or wrap a plain mapping."""
        if isinstance(context, cls):
            if values:
                context.update(values)
            return context
        return cls(context, **values)

    def copy(self) -> "PipelineContext":
        return PipelineContext(self)

    def to_dict(self) -> dict[str, Any]:
        return dict(self)
