from pathlib import Path
from typing import Optional, Tuple
import os, ants

def n4bias(in_img: Path, out: Path, nthreads: Optional[int] = 2, shrink: Optional[int] = 2, iters=[50,50,30,20], mask: Optional[Path]=None, bias_field: Optional[Path]=None):
    
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    bf_p = Path(bias_field) if bias_field else None
    if bf_p:
        bf_p.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done (unless force)
    if out_p.exists() and (not bf_p or not bf_p.exists()):
        return str(out_p), (str(bf_p) if bf_p else None)

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    image  = ants.image_read( str(in_img) )
    n4_img = ants.n4_bias_field_correction(image, 
                                           shrink_factor=shrink, 
                                           convergence={'iters': iters, 'tol':1e-7}, 
                                           mask=ants.image_read( str(mask) ) if mask else None,
                                           return_bias_field=bool(bias_field))
    
    ants.image_write( n4_img['corrected_image'], str(out_p) )
    if bias_field:
        ants.image_write( n4_img['bias_field'], str(bf_p) )
    
    return str(out_p), (str(bf_p) if bf_p else None)


def denoise_image(in_img: Path, out: Path, noise_model: str = "Rician", nthreads: Optional[int] = 2, mask: Optional[Path]=None, noise_map: Optional[Path]=None):
    
    out_p = Path(out); out_p.parent.mkdir(parents=True, exist_ok=True)
    nm_p  = Path(noise_map) if noise_map else None
    if nm_p:
        nm_p.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already done (unless force)
    if out_p.exists() and (not nm_p or not nm_p.exists()):
        return out_p, (nm_p if nm_p else None)
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    image   = ants.image_read( str(in_img) )
    den_img = ants.denoise_image(image, 
                                 noise_model=noise_model, 
                                 mask=ants.image_read( mask ) if mask else None )
    
    ants.image_write( den_img, str(out_p) )

    return out_p, (nm_p if nm_p else None)


