"""Reference b=0 selection for diffusion motion-correction workflows.

The native selector follows the strategy used by TORTOISEV4: candidate b=0
volumes are aligned to a common space, the most mutually similar pair is
identified, and one member of that pair is selected as the registration
reference.  The pair average is also written as a robust target image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, shift, uniform_filter

from ...core import ValidationError
from ...core.types import DWIFile


@dataclass(frozen=True)
class B0ReferenceSelection:
    """Files and metrics produced by optimal b=0 selection."""

    index: int
    paired_index: int
    candidate_indices: tuple[int, ...]
    score: float
    reference_image: Path
    pair_average_image: Path
    metrics_file: Path


def load_bvals(path: Path) -> np.ndarray:
    values = np.asarray(np.loadtxt(path), dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValidationError(f"Invalid b-value table: {path}")
    return values


def b0_candidate_indices(bvals: Sequence[float], threshold: float = 50.0) -> np.ndarray:
    """Return b=0 candidates, falling back to TORTOISE's min-b rule."""
    values = np.asarray(bvals, dtype=float).reshape(-1)
    candidates = np.flatnonzero(values <= float(threshold))
    if candidates.size == 0:
        minimum = float(np.min(values))
        candidates = np.flatnonzero(np.abs(values - minimum) < 10.0)
    return candidates.astype(int)


def _foreground_mask(volume: np.ndarray) -> np.ndarray:
    finite = np.isfinite(volume)
    positive = volume[finite & (volume > 0)]
    if positive.size == 0:
        return finite
    # A low robust threshold excludes air while retaining low-intensity brain.
    threshold = float(np.percentile(positive, 20.0))
    return finite & (volume > threshold)


def _translation_align(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Fast native initialization used when ANTs is not required/available."""
    ref_mask = _foreground_mask(reference)
    mov_mask = _foreground_mask(moving)
    if not np.any(ref_mask) or not np.any(mov_mask):
        return moving
    ref_com = np.asarray(center_of_mass(ref_mask), dtype=float)
    mov_com = np.asarray(center_of_mass(mov_mask), dtype=float)
    return shift(moving, ref_com - mov_com, order=1, mode="constant", cval=0.0)


def local_squared_correlation(
    first: np.ndarray,
    second: np.ndarray,
    radius: int = 3,
) -> float:
    """Mean local squared normalized cross-correlation (TORTOISE-like)."""
    first = np.nan_to_num(np.asarray(first, dtype=np.float32), copy=True)
    second = np.nan_to_num(np.asarray(second, dtype=np.float32), copy=True)
    size = 2 * max(1, int(radius)) + 1
    mean_a = uniform_filter(first, size=size, mode="constant")
    mean_b = uniform_filter(second, size=size, mode="constant")
    var_a = uniform_filter(first * first, size=size, mode="constant") - mean_a * mean_a
    var_b = uniform_filter(second * second, size=size, mode="constant") - mean_b * mean_b
    cov = uniform_filter(first * second, size=size, mode="constant") - mean_a * mean_b
    denom = np.maximum(var_a * var_b, np.finfo(np.float32).eps)
    correlation = (cov * cov) / denom
    mask = _foreground_mask(first) & _foreground_mask(second) & (var_a > 0) & (var_b > 0)
    return float(np.mean(correlation[mask])) if np.any(mask) else 0.0


def select_best_b0_pair(
    data: np.ndarray,
    candidate_indices: Sequence[int],
    *,
    local_radius: int = 3,
) -> tuple[int, int, float, list[np.ndarray]]:
    """Select the most locally correlated aligned pair of b=0 volumes."""
    if data.ndim != 4:
        raise ValidationError(f"Expected a 4D DWI array, got shape {data.shape}")
    indices = [int(index) for index in candidate_indices]
    if not indices:
        raise ValidationError("No b=0 candidates were found")
    if any(index < 0 or index >= data.shape[3] for index in indices):
        raise ValidationError("A b=0 candidate index is outside the DWI volume range")

    reference = np.asarray(data[..., indices[0]], dtype=np.float32)
    aligned = [reference]
    aligned.extend(
        _translation_align(reference, np.asarray(data[..., index], dtype=np.float32))
        for index in indices[1:]
    )
    if len(indices) == 1:
        return indices[0], indices[0], 1.0, aligned

    best = (indices[0], indices[1], -np.inf)
    for row in range(len(indices)):
        for column in range(row + 1, len(indices)):
            score = local_squared_correlation(aligned[row], aligned[column], local_radius)
            if score > best[2]:
                best = (indices[row], indices[column], score)
    return best[0], best[1], float(best[2]), aligned


def select_optimal_b0(
    dwi: DWIFile,
    output_dir: Path,
    *,
    threshold: float = 50.0,
    local_radius: int = 3,
    preferred_index: int | None = None,
    force: bool = False,
) -> B0ReferenceSelection:
    """Select and persist a robust b=0 motion-correction target."""
    if not dwi.bval or not Path(dwi.bval).exists():
        raise ValidationError("Optimal b0 selection requires a b-value sidecar")
    image = nib.load(str(dwi.img))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 4:
        raise ValidationError(f"Optimal b0 selection requires 4D DWI data, got {data.shape}")
    bvals = load_bvals(Path(dwi.bval))
    if bvals.size != data.shape[3]:
        raise ValidationError(
            f"DWI has {data.shape[3]} volumes but {dwi.bval} has {bvals.size} b-values"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "optimal_b0.nii.gz"
    average_path = output_dir / "optimal_b0_pair_average.nii.gz"
    metrics_path = output_dir / "optimal_b0_selection.json"
    if not force and reference_path.exists() and average_path.exists() and metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
        return B0ReferenceSelection(
            index=int(payload["index"]),
            paired_index=int(payload["paired_index"]),
            candidate_indices=tuple(int(v) for v in payload["candidate_indices"]),
            score=float(payload["score"]),
            reference_image=reference_path,
            pair_average_image=average_path,
            metrics_file=metrics_path,
        )

    candidates = b0_candidate_indices(bvals, threshold)
    if preferred_index is not None:
        preferred_index = int(preferred_index)
        if preferred_index not in candidates:
            raise ValidationError(
                f"Requested reference volume {preferred_index} is not a b=0 candidate: "
                f"{candidates.tolist()}"
            )
        candidates = np.asarray([preferred_index], dtype=int)
    index, paired_index, score, aligned = select_best_b0_pair(
        data, candidates, local_radius=local_radius
    )
    index_position = list(candidates).index(index)
    pair_position = list(candidates).index(paired_index)
    output_header = image.header.copy()
    output_header.set_data_dtype(np.float32)
    nib.save(
        nib.Nifti1Image(np.asarray(data[..., index], dtype=np.float32), image.affine, output_header),
        reference_path,
    )
    pair_average = 0.5 * (aligned[index_position] + aligned[pair_position])
    nib.save(nib.Nifti1Image(pair_average.astype(np.float32), image.affine, output_header), average_path)

    selection = B0ReferenceSelection(
        index=index,
        paired_index=paired_index,
        candidate_indices=tuple(int(v) for v in candidates),
        score=score,
        reference_image=reference_path,
        pair_average_image=average_path,
        metrics_file=metrics_path,
    )
    payload = asdict(selection)
    payload.update(
        {
            "reference_image": str(reference_path),
            "pair_average_image": str(average_path),
            "metrics_file": str(metrics_path),
            "algorithm": "TORTOISE-like aligned maximum local squared correlation pair",
            "b0_threshold": float(threshold),
            "local_radius": int(local_radius),
        }
    )
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    return selection


__all__ = [
    "B0ReferenceSelection",
    "b0_candidate_indices",
    "load_bvals",
    "local_squared_correlation",
    "select_best_b0_pair",
    "select_optimal_b0",
]
