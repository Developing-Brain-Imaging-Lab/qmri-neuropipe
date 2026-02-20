
from pathlib import Path
from typing import Optional, Union, Literal
from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.utils import extract_image_path

def hd_bet(
    in_file: Union[ImageLike, Path], 
    out_file: Path, 
    device: str = 'cuda', 
    disable_tta: bool = False,
    verbose: bool = False
):
    """
    Run HD-BET brain extraction.
    
    Args:
        in_file: Input image.
        out_file: Output brain image.
        device: 'cuda', 'cpu', 'mps', or a GPU index like '0'.
        disable_tta: If True, disables test time augmentation (faster).
        verbose: If True, enables verbose output.
    """
    import shlex
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Correct command construction: arguments should be separate
    cmd = ["hd-bet", "-i", str(in_p), "-o", str(out_p)]
    
    if device:
        cmd.extend(["-device", str(device)])
    
    # Always save the mask as the pipeline expects it
    cmd.append("--save_bet_mask")
    
    if disable_tta:
        cmd.append("--disable_tta")
    
    if verbose:
        cmd.append("--verbose")
    
    run_cmd(shlex.join(cmd), label="hd-bet")
    return out_p

