"""Shared, behavior-neutral policies and probes for reusable derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import nibabel as nib

from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.io.bids import build_bids_name


def force_requested(
    config: Any,
    *,
    explicit: bool = False,
    keys: tuple[str, ...] = ("force", "force_run"),
) -> bool:
    """Resolve the common force flags used to bypass cached outputs."""
    if explicit:
        return True
    if not hasattr(config, "get"):
        return False
    return any(bool(config.get(key, False)) for key in keys)


def reuse_enabled(
    config: Any,
    *,
    explicit_force: bool = False,
    default: bool = False,
    force_keys: tuple[str, ...] = ("force", "force_run"),
) -> bool:
    """Return whether existing outputs may be reused for this invocation."""
    if not hasattr(config, "get"):
        return False
    return bool(config.get("skip_existing", default)) and not force_requested(
        config,
        explicit=explicit_force,
        keys=force_keys,
    )


def all_outputs_exist(paths: Iterable[Path]) -> bool:
    """Return true when every required output path exists."""
    required = tuple(Path(path) for path in paths)
    return bool(required) and all(path.exists() for path in required)


def reuse_path_if_exists(
    path: Path,
    entities: Mapping[str, str],
    *,
    force: bool = False,
    readable: bool = False,
    json_path: Optional[Path] = None,
) -> Optional[ImageFile]:
    """Return an existing derivative as an ``ImageFile`` when reusable."""
    path = Path(path)
    if force or not path.exists():
        return None

    if readable:
        try:
            nib.load(str(path))
        except Exception:
            return None

    return ImageFile(
        entities=dict(entities),
        img=path,
        json=Path(json_path) if json_path is not None else None,
    )


def reuse_if_exists(
    entities: Mapping[str, str],
    out_dir: Path,
    *,
    suffix: Optional[str] = None,
    force: bool = False,
    readable: bool = False,
    json_path: Optional[Path] = None,
) -> Optional[ImageFile]:
    """Build the canonical BIDS name and reuse it when present and valid."""
    path = Path(out_dir) / build_bids_name(dict(entities), suffix=suffix)
    return reuse_path_if_exists(
        path,
        entities,
        force=force,
        readable=readable,
        json_path=json_path,
    )
