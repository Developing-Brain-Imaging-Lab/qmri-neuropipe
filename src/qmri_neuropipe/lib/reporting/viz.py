"""
Visualization utilities for reporting.
"""
from pathlib import Path
import ants
import logging

def _calculate_smart_slices(img, overlay=None, requested_slices=None):
    """
    Calculate optimal slice indices based on mask or intensity center of mass.
    """
    sl = requested_slices
    
    if not sl:
         import numpy as np
         # Strategy 1: Mask COM (if mask provided)
         if overlay:
             try:
                 mask_data = overlay.numpy()
                 if mask_data.ndim == 4: mask_data = mask_data[..., 0]
                 z_indices = np.where(np.any(mask_data, axis=(0, 1)))[0]
                 if z_indices.size > 0:
                     z_center = int(np.mean(z_indices))
                     sl = [z_center, z_center + 5, z_center - 5]
             except Exception:
                 pass

         # Strategy 2: Intensity COM (if no mask or mask failed)
         if not sl:
             try:
                 # ants.get_center_of_mass returns physical [x, y, z]
                 com_phys = ants.get_center_of_mass(img)
                 # Convert to index
                 try:
                     com_idx = img.transform_physical_point_to_index(com_phys)
                     z_center = int(com_idx[2])
                     sl = [z_center, z_center + 5, z_center - 5]
                 except Exception:
                     # Fallback if transform fails
                     pass
             except Exception:
                 pass

    # Strategy 3: Geometric Center (Fallback)
    if not sl:
         z_dim = img.shape[2]
         sl = [int(z_dim * 0.5), int(z_dim * 0.55), int(z_dim * 0.60)]
         
    # Ensure bounds for any calculated slices
    z_dim = img.shape[2]
    # Filter valid
    valid_sl = [s for s in sl if 0 <= s < z_dim]
    
    # Fallback again if filtering removed all
    if not valid_sl: 
        valid_sl = [int(z_dim * 0.5)]
        
    return valid_sl

def create_ortho_view(nii_path: Path, output_path: Path, overlay_path: Path = None, title: str = "", slices: list = None):
    """
    Create an orthogonal view of the image (with optional overlay) and save to file.
    """
    try:
        img = ants.image_read(str(nii_path))
        if img.dimension == 4:
            img = _ensure_3d(img)
            
        overlay = ants.image_read(str(overlay_path)) if overlay_path else None
        
        sl = _calculate_smart_slices(img, overlay, slices)

        ants.plot(
            img, 
            overlay=overlay, 
            title=title,
            filename=str(output_path),
            axis=2, 
            slices=sl,
            crop=True
        )
    except Exception as e:
        logging.getLogger("ReportViz").warning(f"Failed to plot {nii_path}: {e}")

def _ensure_3d(img):
    """Ensure image is 3D (extract first volume if 4D)."""
    if img.dimension == 4:
        data = img.numpy()
        data_3d = data[..., 0]
        spacing = img.spacing[:3]
        origin = img.origin[:3] 
        direction = img.direction[:3, :3]
        return ants.from_numpy(data_3d, origin=origin, spacing=spacing, direction=direction)
    return img

def plot_comparison(
    ref_path: Path, 
    mov_path: Path, 
    output_path: Path, 
    title: str = "",
    overlay_alpha: float = 0.5,
    slices: list = None
):
    """
    Create a comparison plot (overlay) of two images.
    """
    try:
        ref = ants.image_read(str(ref_path))
        mov = ants.image_read(str(mov_path))
        
        ref = _ensure_3d(ref)
        mov = _ensure_3d(mov)
        
        # Use ref image and mov as overlay to determine slices? 
        # Or just ref? Ref is usually the target (e.g. T1w), so its COM is good.
        sl = _calculate_smart_slices(ref, overlay=None, requested_slices=slices)

        # Plot moving OVER reference
        ants.plot(
            ref,
            overlay=mov,
            overlay_alpha=overlay_alpha,
            title=title,
            filename=str(output_path),
            axis=2, 
            slices=sl,
            crop=True
        )
    except Exception as e:
        logging.getLogger("ReportViz").warning(f"Failed to plot comparison {ref_path} vs {mov_path}: {e}")
