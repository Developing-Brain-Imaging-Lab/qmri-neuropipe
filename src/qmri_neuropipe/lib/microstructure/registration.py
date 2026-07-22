"""SPGR-to-b0 registration for aggregate g-ratio analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

from ...interfaces import ants
from ..common.registration import prepare_registration_images


def grids_match(first: Path, second: Path) -> bool:
    a = nib.load(str(first))
    b = nib.load(str(second))
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-5, rtol=1e-5)


def register_myelin_to_b0(
    config,
    myelin_map: Path,
    spgr_reference: Path,
    b0_reference: Path,
    output_dir: Path,
    registration_cfg: dict,
    logger: logging.Logger,
) -> tuple[Path, dict]:
    """Estimate SPGR→b0 transform and apply it to the original myelin map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = str(registration_cfg.get("transform", "rigid")).lower()
    if transform not in {"rigid", "affine"}:
        raise ValueError("Aggregate g-ratio registration transform must be rigid or affine.")
    tool = str(registration_cfg.get("tool", "ants")).lower()
    if tool != "ants":
        raise ValueError("Initial aggregate g-ratio registration supports tool: ants.")

    output = output_dir / "myelin_space-dwi.nii.gz"
    prefix = output_dir / "spgr_to_b0_"
    transform_file = Path(str(prefix) + "0GenericAffine.mat")
    force = bool(registration_cfg.get("force", False) or config.get("force", False))
    if output.exists() and transform_file.exists() and not force:
        return output, {"Tool": "ANTs", "Transform": transform.title(), "TransformFile": str(transform_file), "Reused": True}

    skull_cfg = registration_cfg.get("skull_strip", {"enabled": True, "method": "fsl"})
    options = {"skull_strip": skull_cfg}
    moving_reg, fixed_reg, stripped = prepare_registration_images(
        config,
        logger,
        Path(spgr_reference),
        Path(b0_reference),
        output_dir,
        options,
        int(getattr(config, "n_cpus", 1) or 1),
        force=force,
    )
    _, transforms = ants.registration(
        fixed_file=fixed_reg,
        moving_file=moving_reg,
        out_prefix=prefix,
        transform_type=transform.title(),
        nthreads=int(getattr(config, "n_cpus", 1) or 1),
    )
    ants.apply_transforms(
        fixed_file=b0_reference,
        moving_file=myelin_map,
        out_file=output,
        transforms=transforms,
        interpolator=registration_cfg.get("interpolation", "linear"),
        nthreads=int(getattr(config, "n_cpus", 1) or 1),
    )
    if not output.exists():
        raise RuntimeError(f"SPGR-to-b0 registration did not produce {output}")
    return output, {
        "Tool": "ANTs",
        "Transform": transform.title(),
        "TransformFile": str(transform_file),
        "SkullStrippedForEstimation": stripped,
        "MovingReference": str(spgr_reference),
        "FixedReference": str(b0_reference),
        "Reused": False,
    }
