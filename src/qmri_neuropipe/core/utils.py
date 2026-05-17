from pathlib import Path
from typing import Optional, Union
from .types import ImageLike, DWIFile, ImageFile

def ensure_path(path_like: Union[str, Path, None]) -> Union[Path, None]:
    """
    Ensure the input is a Path object or None.
    
    Args:
        path_like: String or Path object.
        
    Returns:
        Path object or None.
    """
    if path_like is None:
        return None
    return Path(path_like)

def ensure_dir(path_like: Union[str, Path], is_file: Optional[bool] = None) -> Path:
    """
    Ensure the directory for the given file path exists.

    Args:
        path_like: File or directory path.
        is_file: If True, creates the parent directory. If False, creates the path itself
                 as a directory. If None (default), infers from the presence of a suffix.

    Returns:
        The Path object.
    """
    p = Path(path_like)
    if is_file is None:
        is_file = bool(p.suffix)
    if is_file:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p

def extract_image_path(curr: ImageLike) -> Path:
    """
    Robustly extract the image path from various types.
    
    args:
        curr: DWIFile, ImageFile, Path, or str.
        
    Returns:
        Path to the image file.
    """
    if hasattr(curr, "img"):
        return Path(curr.img)
    elif hasattr(curr, "path"):
         # Some BIDS entities might use 'path'
         return Path(curr.path)
    return Path(curr)

def check_nifti_integrity(file_path: Union[str, Path]) -> bool:
    """
    Check if a NIfTI file is valid and not corrupt.
    
    Checks:
    1. File exists and is not empty.
    2. If .gz, checks GZIP integrity (using gzip -t).
    3. If .nii/.gz, checks nibabel header parsing.
    
    Args:
        file_path: Path to file.
        
    Returns:
        True if valid, False if corrupt/missing.
    """
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return False

    # 1. Gzip Check (most robust for truncated files)
    if p.suffix == '.gz':
        import subprocess
        try:
            subprocess.run(["gzip", "-t", "-q", str(p)], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return False  # gzip ran but file failed integrity check
        except FileNotFoundError:
            pass  # gzip not installed; fall through to nibabel check

    # 2. Nibabel Header Check
    try:
        import nibabel as nib
        img = nib.load(p)
        _ = img.shape # Force header read
    except Exception:
        return False
        
    return True

def get_nifti_stem(file_path: Union[str, Path]) -> str:
    """
    Get the stem of a NIfTI file, handling both .nii and .nii.gz.
    
    Example:
        'sub-01_T1w.nii.gz' -> 'sub-01_T1w'
        'sub-01_T1w.nii' -> 'sub-01_T1w'
        
    Args:
        file_path: Path to NIfTI file.
        
    Returns:
        Clean stem without .nii or .nii.gz.
    """
    p = Path(file_path)
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem
