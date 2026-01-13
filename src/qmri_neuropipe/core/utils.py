from pathlib import Path
from typing import Union
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

def ensure_dir(path_like: Union[str, Path]) -> Path:
    """
    Ensure the directory for the given file path exists.
    If path_like is a directory (no suffix), creates it.
    If path_like is a file (has suffix), creates its parent.
    
    Args:
        path_like: File or directory path.
        
    Returns:
        The Path object.
    """
    p = Path(path_like)
    # Heuristic: if it has a suffix, it's a file, so create parent.
    # If no suffix, assume directory and create it.
    # Note: This heuristic might fail for files without extensions, 
    # but in neuroimaging standard extensions are pervasive.
    # Better approach might be explicit 'is_file' arg, but for "out_file" use cases:
    if p.suffix:
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
        # Check if gzip is available
        try:
            # -t = test integrity, -q = quiet
            subprocess.run(["gzip", "-t", "-q", str(p)], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If gzip command fails (integrity error) or not found
            # If not found, we skip this check (fallback to nibabel)
            # But usually it means corrupt.
            # We assume gzip exists in this env (standard linux/mac)
            if subprocess.call(["which", "gzip"], stdout=subprocess.DEVNULL) == 0:
                 return False
            # If gzip tool missing, proceed to nibabel check
            pass

    # 2. Nibabel Header Check
    try:
        import nibabel as nib
        img = nib.load(p)
        _ = img.shape # Force header read
    except Exception:
        return False
        
    return True
