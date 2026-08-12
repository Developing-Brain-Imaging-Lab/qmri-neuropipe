"""Typed configuration and compatibility parsing for relaxometry workflows."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


def canonicalize_model_options(options: Optional[dict]) -> dict:
    """Return model options with legacy thread spelling pinned to nthreads."""
    canonical = dict(options or {})
    if "nthreads" not in canonical and "threads" in canonical:
        canonical["nthreads"] = canonical["threads"]
    canonical.pop("threads", None)
    return canonical


def canonicalize_normalization_options(options: Optional[dict]) -> dict:
    """Pin supported normalization aliases while preserving precedence."""
    canonical = dict(options or {})

    fallback_space = canonical.get("space", "MNI")
    canonical["space_name"] = canonical.get(
        "space_name",
        canonical.get("space_entity", fallback_space),
    )
    canonical["space_entity"] = canonical.get(
        "space_entity",
        canonical.get("space_name", fallback_space),
    )

    if "save_transforms" not in canonical:
        canonical["save_transforms"] = canonical.get("save_transform", True)
    if "skull_strip" not in canonical and "skull_strip_registration" in canonical:
        canonical["skull_strip"] = canonical["skull_strip_registration"]
    if "skull_strip_method" not in canonical and "brain_extraction_method" in canonical:
        canonical["skull_strip_method"] = canonical["brain_extraction_method"]
    if "skull_strip_use_gpu" not in canonical and "use_gpu" in canonical:
        canonical["skull_strip_use_gpu"] = canonical["use_gpu"]

    for alias in (
        "space",
        "save_transform",
        "skull_strip_registration",
        "brain_extraction_method",
        "use_gpu",
    ):
        canonical.pop(alias, None)
    return canonical


@dataclass
class RelaxometryPreprocConfig:
    """Preprocessing options for the relaxometry pipeline."""

    reorient: dict = field(default_factory=lambda: {"enabled": False})
    denoising: dict = field(
        default_factory=lambda: {"enabled": False, "method": "mrtrix"}
    )
    degibbs: dict = field(
        default_factory=lambda: {"enabled": False, "method": "mrtrix"}
    )
    motion_correction: dict = field(
        default_factory=lambda: {"enabled": False, "method": "ants"}
    )
    b1: dict = field(default_factory=lambda: {"method": "afi", "smoothing_fwhm": 0.0})
    spgr_reference: dict = field(default_factory=lambda: {"mode": "max_flip"})
    brain_masking: dict = field(default_factory=dict)
    exclude_indices: dict = field(default_factory=dict)


@dataclass
class RelaxometryModelingConfig:
    """DESPOT-family model-fitting options."""

    despot1: dict = field(
        default_factory=lambda: {"enabled": False, "use_hifi": False}
    )
    despot2: dict = field(default_factory=lambda: {"enabled": False})
    despot2fm: dict = field(default_factory=lambda: {"enabled": False})
    mcdespot: dict = field(
        default_factory=lambda: {"enabled": False, "cuda": False}
    )

    def __post_init__(self) -> None:
        self.despot1 = canonicalize_model_options(self.despot1)
        self.despot2 = canonicalize_model_options(self.despot2)
        self.despot2fm = canonicalize_model_options(self.despot2fm)
        self.mcdespot = canonicalize_model_options(self.mcdespot)


@dataclass
class RelaxometryQCConfig:
    enabled: bool = False


@dataclass
class RelaxometryConfig:
    """Top-level configuration for relaxometry preprocessing and modeling."""

    preprocessing: RelaxometryPreprocConfig = field(
        default_factory=RelaxometryPreprocConfig
    )
    modeling: RelaxometryModelingConfig = field(
        default_factory=RelaxometryModelingConfig
    )
    qc: RelaxometryQCConfig = field(default_factory=RelaxometryQCConfig)
    masking: dict = field(default_factory=dict)
    normalization: dict = field(default_factory=lambda: {"enabled": False})
    analysis: dict = field(default_factory=lambda: {"enabled": False})

    def __post_init__(self) -> None:
        self.normalization = canonicalize_normalization_options(self.normalization)


def parse_relaxometry_config(config: Mapping[str, Any]) -> RelaxometryConfig:
    """Build typed relaxometry settings from a pipeline configuration."""
    relaxometry = config.get("relaxometry", {}) or {}
    return RelaxometryConfig(
        preprocessing=RelaxometryPreprocConfig(
            **(relaxometry.get("preprocessing", {}) or {})
        ),
        modeling=RelaxometryModelingConfig(
            **(relaxometry.get("modeling", {}) or {})
        ),
        qc=RelaxometryQCConfig(**(relaxometry.get("qc", {}) or {})),
        masking=relaxometry.get("masking", {}) or {},
        normalization=relaxometry.get("normalization", {}) or {},
        analysis=relaxometry.get("analysis", {}) or {},
    )
