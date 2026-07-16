"""Read-only inventory of raw BIDS data and derivative datasets."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from qmri_neuropipe.io.bids import parse_bids_filename, select_participants_sessions


PRIMARY_EXTENSIONS = {
    ".nii",
    ".nii.gz",
    ".tsv",
    ".csv",
    ".gii",
    ".surf.gii",
    ".shape.gii",
    ".func.gii",
    ".label.gii",
    ".trk",
    ".tck",
}


@dataclass
class ModalityCoverage:
    """Participant and observation coverage for one BIDS datatype."""

    n_subjects: int = 0
    n_observations: int = 0
    n_files: int = 0
    suffixes: list[str] = field(default_factory=list)


@dataclass
class ModelCoverage:
    """Coverage of a fitted model across a derivative dataset."""

    n_subjects: int = 0
    n_observations: int = 0


@dataclass
class DerivativeProductCoverage:
    """Scientific-product coverage for a derivative processing family."""

    n_subjects: int = 0
    n_observations: int = 0
    models: list[str] = field(default_factory=list)
    model_coverage: dict[str, ModelCoverage] = field(default_factory=dict)
    roi_stats_observations: int = 0
    normalized_observations: int = 0
    template_spaces: list[str] = field(default_factory=list)


@dataclass
class DataInventory:
    """Counts for a raw or derivative data tree."""

    n_files: int = 0
    datatypes: dict[str, int] = field(default_factory=dict)
    suffixes: dict[str, int] = field(default_factory=dict)
    entities: dict[str, dict[str, int]] = field(default_factory=dict)
    modality_coverage: dict[str, ModalityCoverage] = field(default_factory=dict)


@dataclass
class DerivativeInventory:
    """Summary of one BIDS derivative dataset."""

    name: str
    path: str
    generated_by: list[dict[str, Any]] = field(default_factory=list)
    bids_version: Optional[str] = None
    n_subjects: int = 0
    n_observations: int = 0
    data: DataInventory = field(default_factory=DataInventory)
    products: dict[str, DerivativeProductCoverage] = field(default_factory=dict)


@dataclass
class BIDSDatasetInventory:
    """Serializable summary of a BIDS dataset."""

    path: str
    name: Optional[str]
    bids_version: Optional[str]
    n_subjects: int
    n_sessions: int
    n_observations: int
    sessionless_subjects: list[str]
    participants: list[str]
    sessions: list[str]
    raw_data: DataInventory
    derivatives: list[DerivativeInventory] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> tuple[dict[str, Any], Optional[str]]:
    if not path.is_file():
        return {}, f"Missing {path.name}: {path}"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Could not read {path}: {exc}"
    if not isinstance(value, dict):
        return {}, f"Expected a JSON object in {path}"
    return value, None


def _full_extension(path: Path) -> str:
    return "".join(path.suffixes)


def _is_primary_data_file(path: Path) -> bool:
    extension = _full_extension(path)
    return extension in PRIMARY_EXTENSIONS or path.suffix in PRIMARY_EXTENSIONS


def _datatype(path: Path, root: Path) -> str:
    """Return the nearest BIDS datatype directory, or ``other``."""
    standard = {"anat", "dwi", "func", "fmap", "perf", "pet", "meg", "eeg", "ieeg", "beh", "micr"}
    try:
        parents = path.relative_to(root).parts[:-1]
    except ValueError:
        return "other"
    for part in reversed(parents):
        if part in standard:
            return part
    return "other"


def _iter_primary_files(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in root.rglob("*"):
        if path.is_file() and _is_primary_data_file(path):
            yield path


def _inventory_data(root: Path) -> DataInventory:
    datatypes: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    entity_counts: dict[str, Counter[str]] = {
        "task": Counter(),
        "acq": Counter(),
        "dir": Counter(),
        "space": Counter(),
        "model": Counter(),
        "desc": Counter(),
    }
    n_files = 0
    for path in _iter_primary_files(root):
        n_files += 1
        parsed = parse_bids_filename(path)
        datatypes[_datatype(path, root)] += 1
        suffix = parsed.get("suffix")
        if suffix:
            suffixes[str(suffix)] += 1
        for entity, counts in entity_counts.items():
            value = parsed.get(entity)
            if value:
                counts[str(value)] += 1
    return DataInventory(
        n_files=n_files,
        datatypes=dict(sorted(datatypes.items())),
        suffixes=dict(sorted(suffixes.items())),
        entities={key: dict(sorted(value.items())) for key, value in entity_counts.items() if value},
    )


def _filtered_pairs(
    root: Path,
    participants: Optional[list[str]],
    sessions: Optional[list[str]],
) -> list[tuple[str, Optional[str]]]:
    pairs = select_participants_sessions(root, participants, sessions)
    if sessions:
        normalized_sessions = {str(value).removeprefix("ses-") for value in sessions}
        pairs = [(subject, session) for subject, session in pairs if session in normalized_sessions]
    return pairs


def _observation_roots(root: Path, pairs: Iterable[tuple[str, Optional[str]]]) -> list[Path]:
    paths = []
    for subject, session in pairs:
        subject_root = root / f"sub-{subject}"
        paths.append(subject_root / f"ses-{session}" if session else subject_root)
    return paths


def _modality_coverage(
    root: Path,
    pairs: Iterable[tuple[str, Optional[str]]],
) -> dict[str, ModalityCoverage]:
    subjects: dict[str, set[str]] = {}
    observations: dict[str, set[tuple[str, Optional[str]]]] = {}
    files: Counter[str] = Counter()
    suffixes: dict[str, set[str]] = {}

    for pair in pairs:
        subject, session = pair
        observation_root = root / f"sub-{subject}"
        if session:
            observation_root /= f"ses-{session}"
        for path in _iter_primary_files(observation_root):
            datatype = _datatype(path, root)
            parsed = parse_bids_filename(path)
            subjects.setdefault(datatype, set()).add(subject)
            observations.setdefault(datatype, set()).add(pair)
            files[datatype] += 1
            if parsed.get("suffix"):
                suffixes.setdefault(datatype, set()).add(str(parsed["suffix"]))

    return {
        datatype: ModalityCoverage(
            n_subjects=len(subjects.get(datatype, set())),
            n_observations=len(observations.get(datatype, set())),
            n_files=files[datatype],
            suffixes=sorted(suffixes.get(datatype, set())),
        )
        for datatype in sorted(files)
    }


_MODEL_ALIASES = {
    "dti": "DTI",
    "dki": "DKI",
    "noddi": "NODDI",
    "sandi": "SANDI",
    "nexi": "NEXI",
    "mapmri": "MAPMRI",
    "fwedti": "FWE_DTI",
    "fwe_dti": "FWE_DTI",
    "csd": "CSD",
    "microglia": "microglia",
    "despot1": "DESPOT1",
    "despot1hifi": "DESPOT1HIFI",
    "despot1_hifi": "DESPOT1HIFI",
    "despot2": "DESPOT2",
    "despot2fm": "DESPOT2FM",
    "despot2_fm": "DESPOT2FM",
    "mcdespot": "mcDESPOT",
}
_DWI_MODELS = {"DTI", "DKI", "NODDI", "SANDI", "NEXI", "MAPMRI", "FWE_DTI", "CSD", "microglia"}
_RELAX_MODELS = {"DESPOT1", "DESPOT1HIFI", "DESPOT2", "DESPOT2FM", "mcDESPOT"}
_NATIVE_SPACES = {"native", "dwi", "anat", "t1w", "scanner", "subject"}


def _canonical_model(value: object) -> Optional[str]:
    token = str(value or "").strip()
    if not token:
        return None
    normalized = token.lower().replace("-", "_")
    compact = normalized.replace("_", "")
    return _MODEL_ALIASES.get(normalized) or _MODEL_ALIASES.get(compact) or token


def _model_from_path(path: Path, parsed: dict[str, Any]) -> Optional[str]:
    model = _canonical_model(parsed.get("model"))
    if model:
        return model
    for part in reversed(path.parts[:-1]):
        model = _canonical_model(part)
        if model in _DWI_MODELS or model in _RELAX_MODELS:
            return model
    return None


def _processing_family(datatype: str, model: Optional[str]) -> str:
    if model in _DWI_MODELS:
        return "dwi"
    if model in _RELAX_MODELS:
        return "relaxometry"
    return datatype


def _derivative_product_coverage(
    root: Path,
    pairs: Iterable[tuple[str, Optional[str]]],
) -> dict[str, DerivativeProductCoverage]:
    family_observations: dict[str, set[tuple[str, Optional[str]]]] = {}
    family_subjects: dict[str, set[str]] = {}
    model_observations: dict[str, dict[str, set[tuple[str, Optional[str]]]]] = {}
    model_subjects: dict[str, dict[str, set[str]]] = {}
    roi_observations: dict[str, set[tuple[str, Optional[str]]]] = {}
    normalized_observations: dict[str, set[tuple[str, Optional[str]]]] = {}
    spaces: dict[str, set[str]] = {}

    for pair in pairs:
        subject, session = pair
        observation_root = root / f"sub-{subject}"
        if session:
            observation_root /= f"ses-{session}"
        records: list[tuple[Path, str, Optional[str], Optional[str]]] = []
        model_families: set[str] = set()
        for path in _iter_primary_files(observation_root):
            parsed = parse_bids_filename(path)
            datatype = _datatype(path, root)
            model = _model_from_path(path, parsed)
            family = _processing_family(datatype, model)
            space = str(parsed.get("space")) if parsed.get("space") else None
            records.append((path, family, model, space))
            family_observations.setdefault(family, set()).add(pair)
            family_subjects.setdefault(family, set()).add(subject)
            if model:
                model_families.add(family)
                model_observations.setdefault(family, {}).setdefault(model, set()).add(pair)
                model_subjects.setdefault(family, {}).setdefault(model, set()).add(subject)

        for path, family, model, space in records:
            lower_name = path.name.lower()
            if "roi_stats" in lower_name or "desc-roi" in lower_name:
                if model or family in model_families:
                    targets = {family}
                else:
                    targets = model_families or {family}
                for target in targets:
                    roi_observations.setdefault(target, set()).add(pair)
            if space and space.lower() not in _NATIVE_SPACES:
                normalized_observations.setdefault(family, set()).add(pair)
                spaces.setdefault(family, set()).add(space)

    products: dict[str, DerivativeProductCoverage] = {}
    for family in sorted(family_observations):
        models = sorted(model_observations.get(family, {}))
        products[family] = DerivativeProductCoverage(
            n_subjects=len(family_subjects.get(family, set())),
            n_observations=len(family_observations[family]),
            models=models,
            model_coverage={
                model: ModelCoverage(
                    n_subjects=len(model_subjects[family][model]),
                    n_observations=len(model_observations[family][model]),
                )
                for model in models
            },
            roi_stats_observations=len(roi_observations.get(family, set())),
            normalized_observations=len(normalized_observations.get(family, set())),
            template_spaces=sorted(spaces.get(family, set())),
        )
    return products


def _derivative_roots(derivatives_dir: Path) -> list[Path]:
    if not derivatives_dir.is_dir():
        return []
    roots = [path.parent for path in derivatives_dir.rglob("dataset_description.json")]
    if (derivatives_dir / "dataset_description.json").is_file():
        roots.append(derivatives_dir)
    # Do not count nested datasets twice if rglob happens to encounter duplicates.
    return sorted(set(roots))


def inspect_bids_dataset(
    bids_dir: str | Path,
    *,
    participants: Optional[list[str]] = None,
    sessions: Optional[list[str]] = None,
    include_derivatives: bool = False,
) -> BIDSDatasetInventory:
    """Inspect a BIDS dataset without modifying it."""
    root = Path(bids_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS directory not found: {root}")

    metadata, warning = _read_json(root / "dataset_description.json")
    warnings = [warning] if warning else []
    pairs = _filtered_pairs(root, participants, sessions)
    participant_values = sorted({subject for subject, _ in pairs})
    session_values = sorted({session for _, session in pairs if session is not None})
    sessionless = sorted({subject for subject, session in pairs if session is None})

    # Scan only selected subject/session roots and merge their counts.
    raw_parts = _observation_roots(root, pairs)
    raw_data = _merge_data_inventories(_inventory_data(path) for path in raw_parts)
    raw_data.modality_coverage = _modality_coverage(root, pairs)

    derivatives: list[DerivativeInventory] = []
    if include_derivatives:
        for derivative_root in _derivative_roots(root / "derivatives"):
            desc, desc_warning = _read_json(derivative_root / "dataset_description.json")
            if desc_warning:
                warnings.append(desc_warning)
            derivative_pairs = _filtered_pairs(derivative_root, participants, sessions)
            derivative_subjects = {subject for subject, _ in derivative_pairs}
            derivative_data = _merge_data_inventories(
                _inventory_data(path)
                for path in _observation_roots(derivative_root, derivative_pairs)
            )
            derivative_data.modality_coverage = _modality_coverage(
                derivative_root, derivative_pairs
            )
            derivatives.append(
                DerivativeInventory(
                    name=str(desc.get("Name") or derivative_root.name),
                    path=str(derivative_root),
                    generated_by=list(desc.get("GeneratedBy") or []),
                    bids_version=desc.get("BIDSVersion"),
                    n_subjects=len(derivative_subjects),
                    n_observations=len(derivative_pairs),
                    data=derivative_data,
                    products=_derivative_product_coverage(
                        derivative_root, derivative_pairs
                    ),
                )
            )

    return BIDSDatasetInventory(
        path=str(root),
        name=metadata.get("Name"),
        bids_version=metadata.get("BIDSVersion"),
        n_subjects=len(participant_values),
        n_sessions=len(session_values),
        n_observations=len(pairs),
        sessionless_subjects=sessionless,
        participants=participant_values,
        sessions=session_values,
        raw_data=raw_data,
        derivatives=derivatives,
        warnings=warnings,
    )


def _merge_data_inventories(inventories: Iterable[DataInventory]) -> DataInventory:
    datatypes: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    entities: dict[str, Counter[str]] = {}
    n_files = 0
    for inventory in inventories:
        n_files += inventory.n_files
        datatypes.update(inventory.datatypes)
        suffixes.update(inventory.suffixes)
        for entity, values in inventory.entities.items():
            entities.setdefault(entity, Counter()).update(values)
    return DataInventory(
        n_files=n_files,
        datatypes=dict(sorted(datatypes.items())),
        suffixes=dict(sorted(suffixes.items())),
        entities={key: dict(sorted(value.items())) for key, value in sorted(entities.items())},
    )
