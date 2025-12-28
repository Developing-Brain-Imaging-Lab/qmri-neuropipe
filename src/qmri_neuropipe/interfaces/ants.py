from pathlib import Path
from typing import Optional, Tuple
from ..core.types import ImageLike
from ..core.utils import ensure_path, ensure_dir, extract_image_path
import os, ants

# Or better, replicate _as_path locally to avoid cross-interface dependency if preferred, 
# but fsl.py exports it? It's private.
# Let's import Path and ImageLike

def n4bias(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, shrink: Optional[int] = 2, iters=[50,50,30,20], mask: Optional[Path]=None, bias_field: Optional[Path]=None):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    bf_p = ensure_dir(bias_field) if bias_field else None

    # Removed strict existence check to allow re-runs. 
    # Logic should be handled by caller (Step classes).

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    image  = ants.image_read( str(in_p) )
    n4_img = ants.n4_bias_field_correction(image, 
                                           shrink_factor=shrink, 
                                           convergence={'iters': iters, 'tol':1e-7}, 
                                           mask=ants.image_read( str(mask) ) if mask else None)
    
    ants.image_write( n4_img, str(out_p) )
    if bias_field:
        n4_img = ants.n4_bias_field_correction(image, 
                                                shrink_factor=shrink, 
                                                convergence={'iters': iters, 'tol':1e-7}, 
                                                mask=ants.image_read( str(mask) ) if mask else None,
                                                return_bias_field=True)
        ants.image_write( n4_img, str(bf_p) )
    
    return str(out_p), (str(bf_p) if bf_p else None)


def denoise_image(in_file: ImageLike | Path, out_file: Path, noise_model: str = "Rician", nthreads: int = 1, mask: Optional[Path]=None, noise_map: Optional[Path]=None):
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    nm_p  = ensure_dir(noise_map) if noise_map else None

    # Removed strict existence check.

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    image   = ants.image_read( str(in_p) )
    den_img = ants.denoise_image(image, 
                                 noise_model=noise_model, 
                                 mask=ants.image_read( mask ) if mask else None )
    
    ants.image_write( den_img, str(out_p) )

    return out_p, (nm_p if nm_p else None)

def ants_brain_extraction(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, mask_out: Optional[Path] = None):
    """
    Wrapper for antsBrainExtraction.sh 
    """
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)

    # Placeholder logic - no strict check removed as it's a placeholder
    if out.exists() and (not mask_out or Path(mask_out).exists()):
        return out, mask_out
    
    pass
    return out, mask_out

def apply_transforms(fixed_file: ImageLike | Path, moving_file: ImageLike | Path, out_file: Path, transforms: list[Path], invert_transforms: list[bool] = None, interpolator: str = "linear", nthreads: int = 1, **kwargs):
    """
    Apply ANTs transforms.
    """
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Removed strict existence check.

    fixed_p = extract_image_path(fixed_file)
    moving_p = extract_image_path(moving_file)
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    
    fixed_img = ants.image_read(str(fixed_p))
    moving_img = ants.image_read(str(moving_p))
    
    # ANTsPy expects transforms as a list of strings
    transform_list = [str(t) for t in transforms]
    
    # Merge kwargs
    call_kwargs = kwargs.copy()
    if invert_transforms:
        call_kwargs['whichtoinvert'] = invert_transforms 
        
    warped_img = ants.apply_transforms(fixed=fixed_img, moving=moving_img, transformlist=transform_list, interpolator=interpolator, **call_kwargs)
    
    ants.image_write(warped_img, str(out))
    return out

def registration(fixed_file: ImageLike | Path, moving_file: ImageLike | Path, out_prefix: Path, transform_type: str = "Rigid", interpolator: str = "linear", nthreads: int = 1, **kwargs):
    """
    Run ANTs registration.
    """
    fixed_p = extract_image_path(fixed_file)
    moving_p = extract_image_path(moving_file)
    out_prefix = ensure_dir(out_prefix)
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)

    warped_out = out_prefix.with_suffix("").parent / (out_prefix.name + "Warped.nii.gz")
    
    fixed_img = ants.image_read(str(fixed_p))
    moving_img = ants.image_read(str(moving_p))
    
    # Remove interpolator from kwargs if present to avoid passing it to optimization routine
    if 'interpolator' in kwargs:
        kwargs.pop('interpolator')

    mytx = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform=transform_type, outprefix=str(out_prefix), **kwargs)
    
    #Apply transforms to moving image (force implementation)
    apply_transforms(fixed_file=fixed_p, 
                     moving_file=moving_p, 
                     out_file=warped_out, 
                     transforms=mytx['fwdtransforms'], 
                     interpolator=interpolator,
                     **kwargs)
    
    return warped_out, mytx['fwdtransforms']

def iMath(in_file: ImageLike | Path, out_file: Path, operation: str, *args, **kwargs):
    """
    Wrapper for ANTs iMath.
    """
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    # Removed strict existence check.
        
    image = ants.image_read(str(in_p))
    result = ants.iMath(image, operation, *args, **kwargs)
    
    ants.image_write(result, str(out_p))
    return out_p

def resample_to_image(source_file: ImageLike | Path, reference_file: ImageLike | Path, out_file: Path, interpolator: str = "linear", nthreads: int = 1) -> Path:
    """
    Resample source_file to the resolution/grid of reference_file.
    """
    src_p = extract_image_path(source_file)
    ref_p = extract_image_path(reference_file)
    out_p = ensure_dir(out_file)
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    
    # We can use ants.resample_image_to_target which handles everything
    src_img = ants.image_read(str(src_p))
    ref_img = ants.image_read(str(ref_p))
    
    # INTERPOLATION MAP
    # generic 'linear' -> ants 'linear'
    # generic 'nearest' -> ants 'nearestNeighbor'
    # generic 'cubic', 'spline' -> ants 'bspline' or 'cubic' (ants supports: linear, nearestNeighbor, multiLabel, gaussian, bspline, cosineWindowedSinc, welchWindowedSinc, hammingWindowedSinc, lanczosWindowedSinc, blackmanWindowedSinc, cosineWindowedSinc, welchWindowedSinc, hammingWindowedSinc, lanczosWindowedSinc, blackmanWindowedSinc)
    
    ants_interp = interpolator
    if interpolator == 'nearest': ants_interp = 'nearestNeighbor'
    elif interpolator == 'cubic': ants_interp = 'bspline' # approximate default for cubic-like

    resampled = ants.resample_image_to_target(
        image=src_img,
        target=ref_img,
        interp_type=ants_interp
    )
    
    ants.image_write(resampled, str(out_p))
    return out_p
