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
    
    Args:
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
