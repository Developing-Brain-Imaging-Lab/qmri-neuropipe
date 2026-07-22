"""Numerical model for MRI-derived aggregate g-ratio measurements.

The functions in this module deliberately have no workflow or NIfTI I/O
dependencies.  This keeps the scientific equations easy to test and makes the
same implementation usable by standalone and integrated workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Calibration:
    """Conversion from a myelin-sensitive marker to myelin volume fraction."""

    mode: str = "identity"
    slope: Optional[float] = None
    intercept: Optional[float] = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"identity", "linear"}:
            raise ValueError("Myelin calibration mode must be 'identity' or 'linear'.")
        object.__setattr__(self, "mode", mode)
        if mode == "linear" and (self.slope is None or self.intercept is None):
            raise ValueError("Linear myelin calibration requires slope and intercept.")

    @property
    def calibrated(self) -> bool:
        return self.mode == "linear"

    def apply(self, marker: np.ndarray) -> np.ndarray:
        marker = np.asarray(marker, dtype=np.float64)
        if self.mode == "identity":
            return marker.copy()
        return float(self.slope) * marker + float(self.intercept)


@dataclass(frozen=True)
class AggregateGRatioResult:
    """Voxelwise component maps and mathematical validity mask."""

    mvf: np.ndarray
    avf: np.ndarray
    fvf: np.ndarray
    gratio: np.ndarray
    valid: np.ndarray
    clipped: np.ndarray


def _clip_tolerance(values: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip only round-off-sized excursions beyond the unit interval."""
    values = np.asarray(values, dtype=np.float64).copy()
    clipped = np.zeros(values.shape, dtype=bool)
    low = np.isfinite(values) & (values < 0) & (values >= -tolerance)
    high = np.isfinite(values) & (values > 1) & (values <= 1 + tolerance)
    values[low] = 0.0
    values[high] = 1.0
    clipped[low | high] = True
    return values, clipped


def compute_aggregate_gratio(
    myelin_marker: np.ndarray,
    intracellular_fraction: np.ndarray,
    isotropic_fraction: Optional[np.ndarray] = None,
    *,
    calibration: Optional[Calibration] = None,
    axonal_input_is_avf: bool = False,
    epsilon: float = 1e-6,
    clipping_tolerance: float = 1e-6,
) -> AggregateGRatioResult:
    """Compute MVF, AVF, FVF, and aggregate g-ratio.

    NODDI intracellular fraction is conditional on the non-isotropic tissue
    compartment, so its default conversion is ``AVF = (1 - FISO) * ICVF``.
    Set ``axonal_input_is_avf`` for a generic map that already represents AVF.
    Invalid voxels are represented by NaN and reported through ``valid``.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if clipping_tolerance < 0:
        raise ValueError("clipping_tolerance cannot be negative.")

    marker = np.asarray(myelin_marker, dtype=np.float64)
    axonal = np.asarray(intracellular_fraction, dtype=np.float64)
    if marker.shape != axonal.shape:
        raise ValueError("Myelin and axonal maps must have the same shape.")

    calibration = calibration or Calibration()
    mvf, mvf_clipped = _clip_tolerance(calibration.apply(marker), clipping_tolerance)
    axonal, axonal_source_clipped = _clip_tolerance(axonal, clipping_tolerance)
    mvf_clipped |= axonal_source_clipped

    source_valid = np.isfinite(axonal) & (axonal >= 0) & (axonal <= 1)
    if axonal_input_is_avf:
        avf_raw = axonal
    else:
        if isotropic_fraction is None:
            raise ValueError("NODDI conversion requires an isotropic fraction map.")
        fiso = np.asarray(isotropic_fraction, dtype=np.float64)
        if fiso.shape != marker.shape:
            raise ValueError("FISO and myelin maps must have the same shape.")
        fiso, fiso_clipped = _clip_tolerance(fiso, clipping_tolerance)
        source_valid &= np.isfinite(fiso) & (fiso >= 0) & (fiso <= 1)
        avf_raw = (1.0 - fiso) * axonal
        mvf_clipped |= fiso_clipped

    avf, avf_clipped = _clip_tolerance(avf_raw, clipping_tolerance)
    clipped = mvf_clipped | avf_clipped
    fvf = mvf + avf

    finite = np.isfinite(mvf) & np.isfinite(avf) & np.isfinite(fvf)
    valid = (
        finite
        & source_valid
        & (mvf >= 0)
        & (mvf <= 1)
        & (avf >= 0)
        & (avf <= 1)
        & (fvf > epsilon)
        & (fvf <= 1 + clipping_tolerance)
    )

    gratio = np.full(marker.shape, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.divide(mvf, fvf, out=np.full_like(fvf, np.nan), where=fvf > epsilon)
    valid &= np.isfinite(ratio) & (ratio >= 0) & (ratio <= 1)
    gratio[valid] = np.sqrt(1.0 - ratio[valid])

    for values in (mvf, avf, fvf):
        values[~valid] = np.nan

    return AggregateGRatioResult(
        mvf=mvf,
        avf=avf,
        fvf=fvf,
        gratio=gratio,
        valid=valid,
        clipped=clipped & valid,
    )


def compute_myelin_thickness(
    inner_diameter: np.ndarray,
    gratio: np.ndarray,
) -> np.ndarray:
    """Compute sheath thickness from inner diameter and aggregate g-ratio."""
    diameter = np.asarray(inner_diameter, dtype=np.float64)
    g = np.asarray(gratio, dtype=np.float64)
    if diameter.shape != g.shape:
        raise ValueError("Axon diameter and aggregate g-ratio maps must have the same shape.")
    valid = np.isfinite(diameter) & (diameter >= 0) & np.isfinite(g) & (g > 0) & (g <= 1)
    thickness = np.full(g.shape, np.nan, dtype=np.float64)
    thickness[valid] = 0.5 * diameter[valid] * (1.0 / g[valid] - 1.0)
    return thickness


def compute_conduction_measures(
    gratio: np.ndarray,
    inner_diameter: Optional[np.ndarray] = None,
    *,
    rushton_coefficient: Optional[float] = None,
    waxman_bennett_coefficient: Optional[float] = None,
) -> dict[str, np.ndarray]:
    """Compute conduction factor, diameter-dependent indices, and optional CV.

    The coefficient-free outputs are morphology indices, not velocities.  A
    calibrated ``CV`` map is emitted only when its model-specific coefficient
    is supplied by the user.
    """
    g = np.asarray(gratio, dtype=np.float64)
    valid_g = np.isfinite(g) & (g > 0) & (g <= 1)
    factor = np.full(g.shape, np.nan, dtype=np.float64)
    factor[valid_g] = np.sqrt(-np.log(g[valid_g]))
    outputs = {"ConductionFactor": factor}

    if inner_diameter is None:
        if rushton_coefficient is not None or waxman_bennett_coefficient is not None:
            raise ValueError("Absolute conduction velocity requires an inner axon diameter map.")
        return outputs

    diameter = np.asarray(inner_diameter, dtype=np.float64)
    if diameter.shape != g.shape:
        raise ValueError("Axon diameter and aggregate g-ratio maps must have the same shape.")
    valid = valid_g & np.isfinite(diameter) & (diameter >= 0)

    rushton = np.full(g.shape, np.nan, dtype=np.float64)
    waxman_bennett = np.full(g.shape, np.nan, dtype=np.float64)
    rushton[valid] = diameter[valid] * factor[valid]
    waxman_bennett[valid] = diameter[valid] / g[valid]
    outputs["RushtonCVIndex"] = rushton
    outputs["WaxmanBennettCVIndex"] = waxman_bennett
    if rushton_coefficient is not None:
        outputs["RushtonCV"] = float(rushton_coefficient) * rushton
    if waxman_bennett_coefficient is not None:
        outputs["WaxmanBennettCV"] = float(waxman_bennett_coefficient) * waxman_bennett
    return outputs
