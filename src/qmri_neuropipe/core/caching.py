"""Shared, behavior-neutral probes for reusable NIfTI derivatives."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import nibabel as nib

from qmri_neuropipe.core.types import ImageFile
from qmri_neuropipe.io.bids import build_bids_name


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
