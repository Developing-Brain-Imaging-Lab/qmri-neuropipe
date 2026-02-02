"""
Segmentation utilities for anatomical processing.
"""

from pathlib import Path
from typing import Optional, Union, Literal
import nibabel as nib
import numpy as np

from ...core import ValidationError
from ...core.run import run_cmd
from ...core.types import ImageLike
from ...core.utils import extract_image_path, ensure_dir, get_nifti_stem
from ...interfaces import fsl, freesurfer

def generate_wm_segmentation(
    in_file: Union[Path, ImageLike], 
    out_dir: Path, 
    method: Literal['fast', 'synthseg'] = 'fast',
    nthreads: int = 1,
    gpu: bool = False
) -> Path:
    """
    Generate a White Matter binary segmentation mask from a T1w/Anatomical image.
    
    Args:
        in_file: Input anatomical image
        out_dir: Output directory
        method: Segmentation method ('fast' or 'synthseg')
        nthreads: Number of threads
        
    Returns:
        Path to white matter binary mask
    """
    out_dir = ensure_dir(out_dir)
    in_path = extract_image_path(in_file)
    
    # Define output path
    # We want a standard name to reuse if possible
    stem = get_nifti_stem(in_path)
    wm_mask = out_dir / f"{stem}_wmseg.nii.gz"
    
    if wm_mask.exists():
        return wm_mask
        
    if method == 'fast':
        # Run FSL FAST
        # Output base for FAST
        fast_base = out_dir / f"{stem}_fast"
        
        # Check if already run
        fast_seg = out_dir / f"{stem}_fast_seg.nii.gz"
        if not fast_seg.exists():
            fsl.fast(in_file, fast_base, img_type=1, num_classes=3)
            
        # Extract WM (Usually class 3 in T1w? Or class 2? 
        # CSF=1, GM=2, WM=3 usually in FSL visual order, but FAST output labels depend.
        # FAST default T1w: CSF=1, GM=2, WM=3.
        # But let's verify.
        # We can extract class 3.
        
        # Extract class 3 from segregation
        # fslmaths <seg> -thr 3 -uthr 3 -bin <wm_mask>
        if fast_seg.exists():
             run_cmd(f"fslmaths {fast_seg} -thr 3 -uthr 3 -bin {wm_mask}", label="extract_wm_fast")
             
    elif method == 'synthseg':
        # Run SynthSeg
        synth_seg = out_dir / f"{stem}_synthseg.nii.gz"
        if not synth_seg.exists():
             freesurfer.mri_synthseg(in_file, synth_seg, nthreads=nthreads, gpu=gpu)
             
        # Extract WM from SynthSeg labels
        # SynthSeg labels: Left WM=2, Right WM=41. 
        # We need to bin specific labels.
        # 2, 41 are standard cerebral WM. 
        # Also Cerebellum WM? (7, 46).
        # And others?
        # A simple robust mapping for coregistration usually targets deep WM. 
        # Let's target 2 and 41.
        
        # mri_binarize --i <seg> --wm --o <wm_mask> ? 
        # mri_binarize has --wm flag which ensures WM is selected? Or explicit match.
        # "mri_binarize --i synth_seg.nii.gz --match 2 41 --o wm_mask.nii.gz"
        
        if synth_seg.exists():
             run_cmd(f"mri_binarize --i {synth_seg} --match 2 41 7 46 --o {wm_mask}", label="extract_wm_synthseg")
             
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
        
    if not wm_mask.exists():
         raise RuntimeError(f"Failed to generate WM segmentation using {method}")
         
    return wm_mask
