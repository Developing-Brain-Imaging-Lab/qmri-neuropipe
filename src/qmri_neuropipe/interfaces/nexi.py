from pathlib import Path
from typing import Optional, Dict, Union

from ..core import ProcessingError
from ..core.utils import ensure_dir, extract_image_path
from ..core.types import ImageLike, DWIFile


def fit_nexi(
    in_file: Union[Path, ImageLike],
    out_dir: Path,
    bval_file: Optional[Path] = None,
    td_file: Optional[Path] = None,
    lowb_noisemap_file: Optional[Path] = None,
    mask_file: Optional[Path] = None,
    debug: bool = False,
) -> Dict[str, Path]:
    """
    Fit the NEXI model using the nexi package.

    Parameters
    ----------
    in_file : Path | ImageLike
        Preprocessed DWI NIfTI or DWIFile.
    out_dir : Path
        Output directory for intermediate and NEXI files.
    bval_file : Path, optional
        B-values file. If not provided and in_file is DWIFile, uses in_file.bval.
    td_file : Path
        Diffusion time file (ms). Required by NEXI.
    lowb_noisemap_file : Path
        Noise map from low-b data (b < 2 ms/um^2). Required by NEXI.
    mask_file : Path, optional
        Brain mask file.
    debug : bool
        Enable debug mode in NEXI.
    """
    try:
        from nexi.estimate_nexi import estimate_nexi
    except ImportError as exc:
        raise ProcessingError("NEXI required but not installed. Install with `pip install nexi`. ") from exc

    in_path = extract_image_path(in_file)
    out_dir = ensure_dir(out_dir)

    if bval_file is None and isinstance(in_file, DWIFile):
        bval_file = in_file.bval

    if not bval_file:
        raise ValueError("NEXI requires a b-values file.")
    if not td_file:
        raise ValueError("NEXI requires a diffusion time file (td_file).")
    if not lowb_noisemap_file:
        raise ValueError("NEXI requires a low-b noise map (lowb_noisemap_file).")

    bval_path = Path(bval_file)
    td_path = Path(td_file)
    lowb_path = Path(lowb_noisemap_file)

    if not bval_path.exists():
        raise FileNotFoundError(f"B-values file not found: {bval_path}")
    if not td_path.exists():
        raise FileNotFoundError(f"Diffusion time file not found: {td_path}")
    if not lowb_path.exists():
        raise FileNotFoundError(f"Low-b noise map not found: {lowb_path}")

    mask_path = Path(mask_file) if mask_file else None
    if mask_path and not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    estimate_nexi(
        str(in_path),
        str(bval_path),
        str(td_path),
        str(lowb_path),
        str(out_dir),
        mask_path=str(mask_path) if mask_path else None,
        debug=debug,
    )

    outputs: Dict[str, Path] = {}
    for path in out_dir.glob("nexi_rice_mean_*.nii.gz"):
        key = path.name.replace("nexi_rice_mean_", "").replace(".nii.gz", "")
        outputs[key] = path

    return outputs
