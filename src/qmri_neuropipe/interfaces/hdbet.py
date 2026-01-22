
from pathlib import Path
from typing import Optional, Union, Literal
from ..core.run import run_cmd
from ..core.types import ImageLike
from ..core.utils import extract_image_path

def hd_bet(
    in_file: Union[ImageLike, Path], 
    out_file: Path, 
    device: str = 'cpu', 
    mode: Literal['fast', 'accurate'] = 'fast', 
    tta: bool = False
):
    """
    Run HD-BET brain extraction.
    
    Args:
        in_file: Input image.
        out_file: Output brain image.
        device: 'cpu' or '0' (for GPU 0) etc.
        mode: 'fast' or 'accurate'
        tta: Test time augmentation (can improve accuracy but slower).
    """
    in_p = extract_image_path(in_file)
    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # hd-bet -i <input> -o <output> [-device DEVICE] [-mode MODE] [-tta TTA]
    tta_val = 1 if tta else 0
    cmd = f"hd-bet -i {in_p} -o {out_p} -device {device} -mode {mode} -tta {tta_val}"
    
    run_cmd(cmd, label="hd-bet")
    return out_p
