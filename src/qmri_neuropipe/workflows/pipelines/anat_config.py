"""Typed configuration parsing for the anatomical pipeline."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Union


@dataclass
class PreprocessingConfig:
    resample: dict[str, Any] = field(default_factory=dict)
    reorient: dict[str, Any] = field(default_factory=dict)
    denoising: dict[str, Any] = field(default_factory=dict)
    degibbs: dict[str, Any] = field(default_factory=dict)
    gibbs: dict[str, Any] = field(default_factory=dict)
    bias_correction: dict[str, Any] = field(default_factory=dict)
    sharpen: dict[str, Any] = field(default_factory=dict)
    brain_masking: dict[str, Any] = field(default_factory=dict)
    coregistration: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    recon_all: dict[str, Any] = field(default_factory=dict)
    use_freesurfer: bool = False
    force_run: bool = False
    skull_stripped_outputs: bool = False


@dataclass
class NormalizationConfig:
    enabled: bool = False
    template: Optional[str] = None
    method: str = "ants"
    options: dict[str, Any] = field(default_factory=dict)
    save_transforms: bool = True
    skull_stripped_outputs: bool = False


@dataclass
class SegmentationConfig:
    enabled: bool = False
    atlas_file: Optional[str] = None
    atlas_labels: Optional[Union[list[str], str]] = None
    metrics: Optional[list[str]] = None
    atlas_threshold: Optional[float] = None


@dataclass
class QCConfig:
    enabled: bool = False
    modalities: Optional[list[str]] = None


@dataclass
class FreeSurferConfig:
    enabled: bool = False


@dataclass
class SuperSynthConfig:
    enabled: bool = False
    mode: str = "invivo"
    sharpen_synths: bool = False
    device: Optional[str] = None
    compute_volumes: bool = False


@dataclass
class AnatomicalConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    freesurfer: FreeSurferConfig = field(default_factory=FreeSurferConfig)
    super_synth: SuperSynthConfig = field(default_factory=SuperSynthConfig)


def parse_anatomical_config(config: Mapping[str, Any]) -> AnatomicalConfig:
    """Parse legacy and current anatomical settings into one typed model."""
    anat_cfg = config.get("anat", {}) or {}
    preprocessing_cfg = anat_cfg.get("preprocessing", {}) or {}
    normalization_cfg = preprocessing_cfg.get("normalization", {}) or {}
    segmentation_cfg = anat_cfg.get("segmentation", {}) or {}
    qc_cfg = (config.get("qc", {}) or {}).get("mriqc", {}) or {}
    super_synth_cfg = anat_cfg.get("super_synth", {}) or {}
    brain_masking_cfg = preprocessing_cfg.get("brain_masking", {}) or {}
    recon_all_cfg = preprocessing_cfg.get("recon_all", {}) or {}
    use_freesurfer = (
        preprocessing_cfg.get("use_freesurfer", False)
        or anat_cfg.get("use_freesurfer", False)
    )
    skull_stripped_outputs = preprocessing_cfg.get(
        "skull_stripped_outputs",
        brain_masking_cfg.get("output_skull_stripped", False),
    )

    return AnatomicalConfig(
        preprocessing=PreprocessingConfig(
            resample=preprocessing_cfg.get("resample", {}),
            reorient=preprocessing_cfg.get("reorient", {}),
            denoising=preprocessing_cfg.get("denoising", {}),
            degibbs=preprocessing_cfg.get("degibbs", {}),
            gibbs=preprocessing_cfg.get("gibbs", {}),
            bias_correction=preprocessing_cfg.get("bias_correction", {}),
            sharpen=preprocessing_cfg.get("sharpen", {}),
            brain_masking=brain_masking_cfg,
            coregistration=preprocessing_cfg.get("coregistration", {}),
            normalization=normalization_cfg,
            recon_all=recon_all_cfg,
            use_freesurfer=use_freesurfer,
            force_run=preprocessing_cfg.get("force_run", False),
            skull_stripped_outputs=skull_stripped_outputs,
        ),
        normalization=NormalizationConfig(
            enabled=normalization_cfg.get("enabled", False),
            template=normalization_cfg.get("template"),
            method=normalization_cfg.get("method", "ants"),
            options=normalization_cfg.get("options", {}),
            save_transforms=normalization_cfg.get(
                "save_transforms",
                normalization_cfg.get("save_transform", True),
            ),
            skull_stripped_outputs=normalization_cfg.get(
                "skull_stripped_outputs",
                skull_stripped_outputs,
            ),
        ),
        segmentation=SegmentationConfig(
            enabled=segmentation_cfg.get("enabled", False),
            atlas_file=segmentation_cfg.get("atlas_file"),
            atlas_labels=segmentation_cfg.get("atlas_labels"),
            metrics=segmentation_cfg.get("metrics"),
            atlas_threshold=segmentation_cfg.get("atlas_threshold"),
        ),
        qc=QCConfig(
            enabled=qc_cfg.get("enabled", False),
            modalities=qc_cfg.get("modalities"),
        ),
        freesurfer=FreeSurferConfig(
            enabled=recon_all_cfg.get("enabled", False) or use_freesurfer,
        ),
        super_synth=SuperSynthConfig(
            enabled=super_synth_cfg.get("enabled", False),
            mode=super_synth_cfg.get("mode", "invivo"),
            sharpen_synths=super_synth_cfg.get("sharpen_synths", False),
            device=super_synth_cfg.get("device"),
            compute_volumes=super_synth_cfg.get("compute_volumes", False),
        ),
    )
