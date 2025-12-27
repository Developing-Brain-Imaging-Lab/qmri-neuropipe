"""
Visualization utilities for reporting.
"""
from pathlib import Path
import logging
import numpy as np
try:
    import ants
except ImportError:
    ants = None
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg') # Force non-interactive backend
except ImportError:
    plt = None

def _calculate_smart_slices(img, overlay=None, requested_slices=None):
    """
    Calculate optimal slice indices based on mask or intensity center of mass.
    """
    sl = requested_slices
    
    if not sl:
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

def plot_ortho_with_colorbar(nii_path: Path, output_path: Path, title: str = "", slices: list = None, overlay_path: Path = None, overlay_cmap='Reds', overlay_alpha=0.5):
    """
    Create an orthogonal view (Sagittal, Coronal, Axial) using matplotlib with optional colorbar and overlay.
    """
    try: 
        img = ants.image_read(str(nii_path))
        if img.dimension == 4:
            img = _ensure_3d(img)
            
        data = img.numpy()
        
        overlay_data = None
        if overlay_path:
             ov = ants.image_read(str(overlay_path))
             if ov.dimension == 4: ov = _ensure_3d(ov)
             overlay_data = ov.numpy()
        
        # Calculate slices
        sl = _calculate_smart_slices(img, overlay=(ants.image_read(str(overlay_path)) if overlay_path else None), requested_slices=slices)
        
        # Calculate geometry center for default
        center = [int(s/2) for s in data.shape[:3]]
        
        z_slice = sl[0] if sl else center[2]
        y_slice = center[1]
        x_slice = center[0]
        
        # Create figure with dark background
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black', constrained_layout=True)
        
        cmap = 'gray'
        
        # Helper to plot slice with optional overlay
        def plot_slice(ax, slice_data, slice_overlay, aspect, title_text):
            # Main image
            ax.imshow(np.rot90(slice_data), cmap=cmap, aspect=aspect)
            # Overlay
            if slice_overlay is not None:
                # Mask out zeros/background in overlay
                masked_overlay = np.ma.masked_where(slice_overlay < 1e-3, slice_overlay)
                ax.imshow(np.rot90(masked_overlay), cmap=overlay_cmap, alpha=overlay_alpha, aspect=aspect)
            
            ax.set_title(title_text, color='white', fontsize=12)
            ax.axis('off')

        spacing = img.spacing
        
        # 1. Sagittal (YZ plane, slice X)
        asp_sag = spacing[2] / spacing[1] 
        plot_slice(axes[0], data[x_slice, :, :], 
                   overlay_data[x_slice, :, :] if overlay_data is not None else None, 
                   asp_sag, 'Sagittal')
        
        # 2. Coronal (XZ plane, slice Y) 
        asp_cor = spacing[2] / spacing[0]
        plot_slice(axes[1], data[:, y_slice, :], 
                   overlay_data[:, y_slice, :] if overlay_data is not None else None, 
                   asp_cor, 'Coronal')
        
        # 3. Axial (XY plane, slice Z)
        asp_ax = spacing[1] / spacing[0]
        plot_slice(axes[2], data[:, :, z_slice], 
                   overlay_data[:, :, z_slice] if overlay_data is not None else None, 
                   asp_ax, 'Axial')
        
        if overlay_path is None:
            im_dummy = axes[2].get_images()[0] # get base image
            cbar = fig.colorbar(im_dummy, ax=axes, shrink=0.7, location='right', pad=0.05)
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.ax.yaxis.label.set_color('white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            label = "Value"
            if "FA" in title: label = "FA"
            elif "MD" in title: label = "MD (mm²/s)"
            cbar.set_label(label, color='white')
        
        if title: 
            plt.suptitle(title, color='white', fontsize=16)
        
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150, facecolor='black')
        plt.close(fig)
        
    except Exception as e:
        logging.getLogger("ReportViz").warning(f"Matplotlib plotting failed for {nii_path}: {e}")
        pass

def create_ortho_view(nii_path: Path, output_path: Path, overlay_path: Path = None, title: str = "", slices: list = None, colorbar: bool = False, overlay_cmap='Reds', overlay_alpha=0.5):
    """
    Create an orthogonal view of the image (with optional overlay) and save to file.
    """
    # Always use the matplotlib 3-plane implementation
    return plot_ortho_with_colorbar(nii_path, output_path, title, slices, overlay_path=overlay_path, overlay_cmap=overlay_cmap, overlay_alpha=overlay_alpha)

def create_metric_grid(metrics_info: list, output_path: Path, title: str = ""):
    """
    Create a grid of metric maps (Rows: Metrics, Cols: Sag/Cor/Axial).
    
    metrics_info: List of dicts with 'path' and 'title' (e.g. 'FA').
    """
    try:
        
        N = len(metrics_info)
        if N == 0: return

        # Load first image to get geometry/slice indices
        first_img = ants.image_read(str(metrics_info[0]['path']))
        if first_img.dimension == 4: first_img = _ensure_3d(first_img)
        
        sl = _calculate_smart_slices(first_img)
        data_sh = first_img.numpy().shape
        spacing = first_img.spacing
        
        center = [int(s/2) for s in data_sh[:3]]
        z_slice = sl[0] if sl else center[2]
        y_slice = center[1]
        x_slice = center[0]
        
        asp_sag = spacing[2] / spacing[1]
        asp_cor = spacing[2] / spacing[0]
        asp_ax = spacing[1] / spacing[0]
        
        # Create figure: Rows = Metrics, Cols = 3
        # Create figure: Rows = Metrics, Cols = 3
        # Increase size per row to make maps larger (similar to plot_ortho_with_colorbar)
        fig, axes = plt.subplots(N, 3, figsize=(15, 4*N), facecolor='black', constrained_layout=True)
        if N == 1: axes = np.array([axes]) # Ensure 2D array
        
        for i, info in enumerate(metrics_info):
            p = info['path']
            lbl = info['title']
            
            img = ants.image_read(str(p))
            if img.dimension == 4: img = _ensure_3d(img)
            data = img.numpy()
            
            # Row i, Col 0: Sagittal
            ax = axes[i, 0]
            ax.imshow(np.rot90(data[x_slice, :, :]), cmap='gray', aspect=asp_sag)
            ax.axis('off')
            if i == 0: ax.set_title("Sagittal", color='white')
            ax.text(-0.1, 0.5, lbl, transform=ax.transAxes, rotation=90, va='center', ha='right', color='white', fontsize=14, fontweight='bold')
            
            # Row i, Col 1: Coronal
            ax = axes[i, 1]
            ax.imshow(np.rot90(data[:, y_slice, :]), cmap='gray', aspect=asp_cor)
            ax.axis('off')
            if i == 0: ax.set_title("Coronal", color='white')
            
            # Row i, Col 2: Axial
            ax = axes[i, 2]
            im = ax.imshow(np.rot90(data[:, :, z_slice]), cmap='gray', aspect=asp_ax)
            ax.axis('off')
            if i == 0: ax.set_title("Axial", color='white')
            
            # Add small colorbar per row? Or rely on grayscale. 
            # Often useful to have colorbar. Let's add one for the row.
            cbar = fig.colorbar(im, ax=axes[i, :], shrink=0.9, location='right', pad=0.02)
            cbar.ax.yaxis.set_tick_params(color='white') 
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        if title:
             plt.suptitle(title, color='white', fontsize=16)
             
        plt.savefig(str(output_path), bbox_inches='tight', dpi=150, facecolor='black')
        plt.close(fig)
        
    except Exception as e:
        logging.getLogger("ReportViz").warning(f"Grid plotting failed: {e}")

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
    overlay_cmap: str = 'hot', # Different from default if needed
    slices: list = None
):
    """
    Create a comparison plot (overlay) of two images (3-plane).
    """
    # Simply delegate to create_ortho_view which now handles 3-plane overlays
    return create_ortho_view(
        nii_path=ref_path, 
        output_path=output_path, 
        overlay_path=mov_path, 
        title=title, 
        slices=slices,
        overlay_cmap=overlay_cmap,
        overlay_alpha=overlay_alpha
    )
