"""Input discovery for aggregate g-ratio analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from ...io.bids import get_entities_from_path


IDENTITY_ENTITIES = ("sub", "ses", "acq", "rec", "run", "dir", "task")


@dataclass(frozen=True)
class GRatioInputs:
    myelin: Path
    spgr_reference: Path
    intracellular: Path
    isotropic: Optional[Path]
    diffusion_reference: Path
    entities: dict[str, str]
    axonal_input_is_avf: bool = False
    axon_diameter: Optional[Path] = None
    recommended_mask: Optional[Path] = None
    registration_metadata: Optional[dict[str, Any]] = None


def _entities(path: Path) -> dict[str, str]:
    try:
        return get_entities_from_path(Path(path))
    except Exception:
        return {}


def _identity(path: Path) -> tuple:
    entities = _entities(path)
    return tuple((key, entities.get(key)) for key in IDENTITY_ENTITIES if entities.get(key))


def _single(paths: list[Path], label: str) -> Path:
    unique = sorted({Path(path) for path in paths})
    if not unique:
        raise FileNotFoundError(f"No {label} was found.")
    if len(unique) > 1:
        names = ", ".join(str(path) for path in unique)
        raise ValueError(f"Ambiguous {label}; found {len(unique)} candidates: {names}")
    return unique[0]


def _metric_files(root: Path, metric: str, model: Optional[str] = None) -> list[Path]:
    matches = []
    for path in Path(root).glob("**/*.nii.gz"):
        entities = _entities(path)
        suffix = str(entities.get("suffix", ""))
        if suffix.lower() != metric.lower() and not path.name.endswith(f"_{metric}.nii.gz"):
            continue
        if model and str(entities.get("model", "")).lower() != model.lower() and f"model-{model}" not in path.name:
            continue
        matches.append(path)
    return sorted(matches)


def discover_noddi_pairs(dwi_root: Path) -> list[tuple[Path, Path, dict[str, str]]]:
    """Return exact entity-matched NODDI ICVF/FISO pairs."""
    icvf_files = _metric_files(dwi_root, "ICVF", "NODDI")
    fiso_files = _metric_files(dwi_root, "FISO", "NODDI")
    pairs = []
    for icvf in icvf_files:
        identity = _identity(icvf)
        matches = [path for path in fiso_files if _identity(path) == identity]
        fiso = _single(matches, f"NODDI FISO map matching {icvf.name}")
        pairs.append((icvf, fiso, _entities(icvf)))
    if not pairs:
        raise FileNotFoundError(f"No NODDI ICVF results were found under {dwi_root}.")
    return pairs


def discover_mcdespot_vfm(anat_root: Path) -> Path:
    return _single(_metric_files(anat_root, "VFm", "mcDESPOT"), "mcDESPOT VFm map")


def discover_spgr_reference(anat_root: Path) -> Path:
    candidates = []
    for path in Path(anat_root).glob("**/*.nii.gz"):
        lower = path.name.lower()
        if "spgrref" in lower or "spgr_ref" in lower or "desc-spgrref" in lower:
            if "masked" not in lower:
                candidates.append(path)
    return _single(candidates, "SPGR reference")


def expand_subject_path(value: str | Path, subject: str, session: Optional[str]) -> Path:
    return Path(str(value).format(subject=subject, session=session or ""))
