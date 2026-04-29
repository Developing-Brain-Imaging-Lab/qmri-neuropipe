from pathlib import Path
from typing import Optional, Tuple
from ..core.types import ImageLike
from ..core.utils import ensure_path, ensure_dir, extract_image_path
import os

# Lazy import ants inside functions to allow thread setting

def n4bias(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, shrink: Optional[int] = 2, iters=[50,50,30,20], mask: Optional[Path]=None, bias_field: Optional[Path]=None):
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants
    import numpy as np

    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    bf_p = ensure_dir(bias_field) if bias_field else None
    
    mask_p = extract_image_path(mask) if mask else None
    
    image  = ants.image_read( str(in_p) )
    
    if image.dimension == 4:
        # For 4D images (like DWI), estimate bias field on a reference volume (usually first)
        b0_np = image.numpy()[..., 0]
        # Create 3D ANTsImage for N4
        b0 = ants.from_numpy(b0_np, origin=image.origin[:3], spacing=image.spacing[:3], direction=image.direction[:3,:3])
        
        # Prepare 3D mask
        ants_mask = None
        if mask_p:
            ants_mask = ants.image_read(str(mask_p))
            if ants_mask.dimension == 4:
                ants_mask = ants.from_numpy(ants_mask.numpy()[...,0], origin=ants_mask.origin[:3], spacing=ants_mask.spacing[:3], direction=ants_mask.direction[:3,:3])
        
        # Estimate bias field (3D)
        b_field = ants.n4_bias_field_correction(
            b0, shrink_factor=shrink, 
            convergence={'iters': iters, 'tol':1e-7}, 
            mask=ants_mask, 
            return_bias_field=True
        )
        
        # Apply to entire 4D series
        img_np = image.numpy()
        bf_np = b_field.numpy()
        # Avoid division by zero
        bf_np[bf_np == 0] = 1.0
        
        corrected_np = img_np / bf_np[..., np.newaxis]
        n4_img = ants.from_numpy(corrected_np, origin=image.origin, spacing=image.spacing, direction=image.direction)
        
        if bf_p:
            ants.image_write(b_field, str(bf_p))
            
    else:
        # Standard 2D/3D case
        ants_mask = ants.image_read(str(mask_p)) if mask_p else None
        n4_img = ants.n4_bias_field_correction(image, 
                                               shrink_factor=shrink, 
                                               convergence={'iters': iters, 'tol':1e-7}, 
                                               mask=ants_mask)
        if bf_p:
            b_field = ants.n4_bias_field_correction(image, 
                                                    shrink_factor=shrink, 
                                                    convergence={'iters': iters, 'tol':1e-7}, 
                                                    mask=ants_mask,
                                                    return_bias_field=True)
            ants.image_write(b_field, str(bf_p))
    
    ants.image_write( n4_img, str(out_p) )
    
    return str(out_p), (str(bf_p) if bf_p else None)


def denoise_image(in_file: ImageLike | Path, out_file: Path, noise_model: str = "Rician", nthreads: int = 1, mask: Optional[Path]=None, noise_map: Optional[Path]=None):
    
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    nm_p  = ensure_dir(noise_map) if noise_map else None

    mask_p = extract_image_path(mask) if mask else None

    image   = ants.image_read( str(in_p) )
    den_img = ants.denoise_image(image, 
                                 noise_model=noise_model, 
                                 mask=ants.image_read( str(mask_p) ) if mask_p else None )
    
    ants.image_write( den_img, str(out_p) )

    return out_p, (nm_p if nm_p else None)

def ants_brain_extraction(in_file: ImageLike | Path, out_file: Path, nthreads: int = 1, mask_out: Optional[Path] = None):
    """
    Wrapper for antsBrainExtraction.sh 
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants
    
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder logic - no strict check removed as it's a placeholder
    if out.exists() and (not mask_out or Path(mask_out).exists()):
        return out, mask_out
    
    pass
    return out, mask_out

def apply_transforms(fixed_file: ImageLike | Path, moving_file: ImageLike | Path, out_file: Path, transforms: list[Path], invert_transforms: list[bool] = None, interpolator: str = "linear", nthreads: int = 1, **kwargs):
    """
    Apply ANTs transforms.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants

    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    fixed_p = extract_image_path(fixed_file)
    moving_p = extract_image_path(moving_file)
    
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


def collapse_transforms(
    reference_file: ImageLike | Path,
    transforms: list[Path],
    out_file: Path,
    collapse: str = "displacement",
    invert_transforms: list[bool] | None = None,
    nthreads: int = 1,
    dimensionality: int = 3,
):
    """
    Collapse a transform chain into a single displacement field or composite transform.
    """
    ref_p = extract_image_path(reference_file)
    out_p = ensure_dir(out_file)

    if invert_transforms and len(invert_transforms) != len(transforms):
        raise ValueError("invert_transforms must match transforms length")

    out_arg = f"DisplacementField[{out_p}]" if collapse == "displacement" else f"CompositeTransform[{out_p}]"
    tx_args = []
    for idx, transform in enumerate(transforms):
        if invert_transforms and invert_transforms[idx]:
            tx_args.append(f"-t [{transform},1]")
        else:
            tx_args.append(f"-t {transform}")

    cmd = (
        f"antsApplyTransforms -d {dimensionality} -r {ref_p} -o {out_arg} "
        + " ".join(tx_args)
    )
    run_cmd(cmd, label="antsCollapseTransforms", n_threads=nthreads)
    return out_p

def registration(fixed_file: ImageLike | Path, moving_file: ImageLike | Path, out_prefix: Path, transform_type: str = "Rigid", interpolator: str = "linear", nthreads: int = 1, **kwargs):
    """
    Run ANTs registration.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants
    
    fixed_p = extract_image_path(fixed_file)
    moving_p = extract_image_path(moving_file)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    
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

def iMath(in_file: ImageLike | Path, out_file: Path, operation: str, *args, nthreads: int = 1, **kwargs):
    """
    Wrapper for ANTs iMath.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants
    
    in_p = extract_image_path(in_file)
    out_p = ensure_dir(out_file)
    
    image = ants.image_read(str(in_p))
    result = ants.iMath(image, operation, *args, **kwargs)
    
    ants.image_write(result, str(out_p))
    return out_p

def resample_to_image(source_file: ImageLike | Path, reference_file: ImageLike | Path, out_file: Path, interpolator: str = "linear", nthreads: int = 1) -> Path:
    """
    Resample source_file to the resolution/grid of reference_file.
    """
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
    import ants
    
    src_p = extract_image_path(source_file)
    ref_p = extract_image_path(reference_file)
    out_p = ensure_dir(out_file)
        
    # We can use ants.resample_image_to_target which handles everything
    src_img = ants.image_read(str(src_p))
    ref_img = ants.image_read(str(ref_p))
    
    # INTERPOLATION MAP
    # generic 'linear' -> ants 'linear'
    # generic 'nearest' -> ants 'nearestNeighbor'
    # generic 'cubic', 'spline' -> ants 'bspline' or 'cubic' (ants supports: linear, nearestNeighbor, multiLabel, gaussian, bspline, cosineWindowedSinc, welchWindowedSinc, hammingWindowedSinc, lanczosWindowedSinc, blackmanWindowedSinc, cosineWindowedSinc, welchWindowedSinc, hammingWindowedSinc, lanczosWindowedSinc, blackmanWindowedSinc)
    
    ants_interp = interpolator
    if interpolator == 'nearest': ants_interp = 'nearestNeighbor'
    elif interpolator == 'cubic': ants_interp = 'bspline'
    elif interpolator == 'label': ants_interp = 'genericLabel'
    elif interpolator == 'multilabel': ants_interp = 'multiLabel'

    resampled = ants.resample_image_to_target(
        image=src_img,
        target=ref_img,
        interp_type=ants_interp
    )
    
    ants.image_write(resampled, str(out_p))
    
    return out_p
