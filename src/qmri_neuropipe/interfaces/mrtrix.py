from pathlib import Path
from typing import Optional, Tuple
from ..core.run import run_cmd


def dwidenoise(in_img: Path, out: Path, nthreads: int=2, mask: Optional[Path]=None, noise_map: Optional[Path]=None, force: bool=False) -> Tuple[Path, Optional[Path]]:
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    nm_p = Path(noise_map) if noise_map else None
    if nm_p:
        nm_p.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done (unless force)
    if not force and out_p.exists() and (not nm_p or nm_p.exists()):
        return str(out_p), (str(nm_p) if nm_p else None)
    
    mask_arg  = f"-mask {mask}" if mask else ""
    noise_arg = f"-noise {noise_map}" if noise_map else ""
    force_arg = f"-force" if force else ""
    cmd = f"dwidenoise {in_img} {out} {mask_arg} {noise_arg} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="dwidenoise")
   
    return out_p, (nm_p if nm_p else None)
    
def dwibiascorrect(in_img: Path, in_bvec: Path, in_bval: Path, out: Path, method: str = "ants", mask: Optional[Path]=None, nthreads: int = 2, force: bool = False):
    
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    diff_arg  = f"-fslgrad {in_bvec} {in_bval}"
    mask_arg  = f"-mask {mask}" if mask else ""
    force_arg = f"-force" if force else ""
    cmd = f"dwibiascorrect {method} {in_img} {diff_arg} {out} {mask_arg} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="dwibiascorrect")
    
    return out_p

def mrdegibbs(in_img: Path, out: Path, nthreads: int = 2, force: bool = False):
    
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done (unless force)
    if not force and out_p.exists():
        return out_p

    force_arg = f"-force" if force else ""
    cmd = f"mrdegibbs {in_img} {out} -nthreads {nthreads} {force_arg} -quiet"
    run_cmd(cmd, label="mrdegibbs")

    return out_p