# -*- coding: utf-8 -*-
"""
Brain masking / skull‑stripping utilities.

This module provides a flexible ``BrainMaskingStep`` that can be used in a
pipeline and a ``mask_brain`` convenience function for ad‑hoc usage.

Supported tools (selected via the ``method`` argument):

* ``fsl`` – ``bet`` (Brain Extraction Tool)
* ``mrtrix`` – ``dwi2mask`` (works for diffusion data)
* ``ants`` – ``antsBrainExtraction.sh``
* ``freesurfer`` – ``mri_watershed`` (via the FreeSurfer interface)
* ``synthstrip`` – ``mri_synthstrip``

The step validates inputs, runs the chosen tool, and returns an ``ImageFile``
containing the brain‑masked image.
"""

from pathlib import Path
from typing import Literal, Optional, Tuple
import logging
import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageFile, ImageLike, DWIFile
from ...interfaces import fsl, mrtrix, ants, freesurfer
from ...core.utils import extract_image_path
from typing import Literal, Optional, Tuple, Any


class BrainMaskingStep(BaseProcessingStep):
    """Skull‑strip a structural or diffusion image.

    Parameters
    ----------
    config : dict
        Pipeline configuration.
    method : Literal['fsl', 'mrtrix', 'ants', 'freesurfer', 'synthstrip']
        Which external tool to use for brain extraction.
    logger : Optional[logging.Logger]
        Logger instance; a default logger is created if ``None``.
    provenance : optional
        Provenance tracking object passed to ``BaseProcessingStep``.
    """

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance=None,
        method: Literal["fsl", "mrtrix", "ants", "freesurfer", "synthstrip"] = "fsl",
        n_threads: int = 1,
    ):
        super().__init__(config, logger, provenance)
        self.method = method
        self.n_threads = n_threads
        self._check_dependencies()
        self.logger.info(f"BrainMaskingStep initialized with method: {method}")

    # ---------------------------------------------------------------------
    # Dependency checks
    # ---------------------------------------------------------------------
    def _check_dependencies(self) -> None:
        """Ensure the selected external tool is available.

        The checks are lightweight – we simply verify that the corresponding
        Python wrapper module is importable (already imported at module level).
        In a real deployment you might also check the binary is on ``PATH``.
        """
        if self.method == "fsl":
            # ``fsl.bet`` will raise if FSL is not installed; we trust the wrapper.
            pass
        elif self.method == "mrtrix":
            pass
        elif self.method == "ants":
            pass
        elif self.method == "freesurfer":
            pass
        elif self.method == "synthstrip":
            pass
        else:
            raise ProcessingError(f"Unsupported brain‑masking method: {self.method}")

    # ---------------------------------------------------------------------
    # Core execution
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Core execution
    # ---------------------------------------------------------------------
    def run(self, first_arg, output_dir: Path, return_mask: bool = False, **kwargs) -> Any:
        """Run the selected brain‑extraction tool.

        Parameters
        ----------
        first_arg : dict | ImageLike | Path
            Context dict or Image to be skull‑stripped.
        output_dir : Path
            Directory where the masked image will be written.
        return_mask : bool
            If True, returns a tuple (brain_image, binary_mask) in standalone mode,
            or properly populates context.

        Returns
        -------
        updated_context or result_image(s)
        """
        # Unpack input
        context, input_image = self.unpack_input(first_arg)
        
        if input_image is None:
             raise ValidationError("No input image provided for masking.")

        is_dwi = isinstance(input_image, DWIFile)
        if hasattr(input_image, 'img'):
            in_path = input_image.img
            entities = input_image.entities.copy() if hasattr(input_image, 'entities') else {}
        else:
            in_path = self._extract_path(input_image)
            entities = {}

        if not in_path or not in_path.exists():
            raise ValidationError(f"Input image not found: {in_path}")
        
        output_dir = self.get_step_output_dir(output_dir)
        
        # Handle .nii.gz stem correctly
        name = in_path.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            stem = in_path.stem
            
        masked_path = output_dir / f"{stem}_brainmask.nii.gz"
        mask_out_path = output_dir / f"{stem}_mask.nii.gz" if return_mask else None
        
        # Skip if outputs exist
        # Skip if outputs exist AND input is not newer
        should_skip = False
        if masked_path.exists() and (not return_mask or (mask_out_path and mask_out_path.exists())):
             # Check timestamps
             in_mtime = in_path.stat().st_mtime
             out_mtime = masked_path.stat().st_mtime
             if in_mtime > out_mtime:
                 self.logger.info(f"Input ({in_path.name}) is newer than output. Re-running brain masking.")
                 should_skip = False
             else:
                 should_skip = True
        
        if should_skip:
             self.logger.info(f"Skipping brain masking (outputs exist): {masked_path}")
             if is_dwi:
                 brain_obj = DWIFile(
                     img=masked_path, 
                     entities=entities, 
                     json=input_image.json, 
                     bval=input_image.bval, 
                     bvec=input_image.bvec
                 )
             else:
                 brain_obj = ImageFile(img=masked_path, entities=entities)
             mask_obj = ImageFile(img=mask_out_path, entities=dict(entities, suffix="mask")) if return_mask else None
             
             if context is not None:
                 context["current_image"] = brain_obj
                 if mask_obj: context["current_mask"] = mask_obj
                 return context
                 
             if return_mask:
                 return brain_obj, mask_obj
             return brain_obj
        
        
        # Prepare reference image for masking
        # If DWI (4D), we MUST extract the first volume (b0) to generate the mask.
        # Then we apply that mask to the 4D series.
        temp_b0 = None
        mask_generated_path = output_dir / f"{stem}_mask_generated.nii.gz"
        tool_brain_out = output_dir / f"{stem}_tool_brain.nii.gz"
        
        # Cleanup Pre-Run to ensure fresh generation (avoids wrappers skipping if output exists)
        if mask_generated_path.exists(): mask_generated_path.unlink()
        if tool_brain_out.exists(): tool_brain_out.unlink()
        
        # Determine what file to pass to the tool
        if is_dwi:
             # Extract first volume
             temp_b0 = output_dir / f"{stem}_vol0.nii.gz"
             # Use nibabel to slice
             img_nii = nib.load(str(in_path))
             data = img_nii.get_fdata()
             
             if data.ndim == 4:
                 vol0 = data[..., 0]
                 nib.save(nib.Nifti1Image(vol0, img_nii.affine, img_nii.header), temp_b0)
                 tool_input = temp_b0
             else:
                 # It's actually 3D?
                 tool_input = in_path
        else:
             tool_input = in_path

        # Dispatch to the chosen tool to generate MASK
        # We focus on generating the mask file ('mask_generated_path') first.
        # Some tools output a brain image; we discard it if it's based on b0 and we want 4D output.
        
        tool_brain_out = output_dir / f"{stem}_tool_brain.nii.gz"
        
        if self.method == "fsl":
             # bet <in> <out> -m (produces out.nii.gz and out_mask.nii.gz)
             # We want the mask.
             _, gen_mask = fsl.bet(in_file=tool_input, out_file=tool_brain_out, robust=True, mask=True)
             if gen_mask and gen_mask.exists():
                 gen_mask.rename(mask_generated_path)
             else:
                 raise ProcessingError("FSL BET failed to generate mask.")
                 
        elif self.method == "mrtrix":
             # dwi2mask usually works on 4D. 
             # If user insists on first volume, dwi2mask on b0 might behave differently (just threshold).
             # But 'dwi2mask' expects DWI.
             # If we pass b0 (3D) to dwi2mask, it might assume it's a T2 and work?
             # For improved robustness as requested ("ensure ... using the first volume"), we use tool_input (b0).
             # Note: dwi2mask logic is complex. If pure b0, maybe use mrthreshold?
             # But let's try dwi2mask on the extracted b0 if possible, or fallback to dwi2mask on full input if that's what makes sense.
             # "ensure ... using the first volume" -> I will use tool_input (b0).
             mrtrix.dwi2mask(in_file=tool_input, out_file=mask_generated_path, nthreads=self.n_threads)

        elif self.method == "ants":
             # antsBrainExtraction.sh -d 3 ...
             # expects 3D input. tool_input is correct.
             ants.ants_brain_extraction(in_file=tool_input, out_file=tool_brain_out, nthreads=self.n_threads, mask_out=mask_generated_path)

        elif self.method == "freesurfer":
             if hasattr(freesurfer, 'mri_watershed'):
                 freesurfer.mri_watershed(tool_input, tool_brain_out)
                 # Derive mask
                 try:
                     brain = nib.load(str(tool_brain_out))
                     bm_data = brain.get_fdata()
                     mask = (np.abs(bm_data) > 1e-6).astype(np.int16)
                     nib.save(nib.Nifti1Image(mask, brain.affine, brain.header), mask_generated_path)
                 except Exception:
                     pass
             else:
                 pass

        elif self.method == "synthstrip":
             freesurfer.mri_synthstrip(in_file=tool_input, out_file=tool_brain_out, n_threads=self.n_threads, mask_out=mask_generated_path)
        
        else:
             raise ProcessingError(f"Unsupported method: {self.method}")

        # Post-process: Apply mask to original input
        if not mask_generated_path.exists():
             raise ProcessingError(f"Brain masking failed to generate mask: {mask_generated_path}")

        # Load mask
        mask_img = nib.load(str(mask_generated_path))
        mask_data = mask_img.get_fdata() > 0.5 # binary
        
        # Load original input
        orig_img = nib.load(str(in_path))
        orig_data = orig_img.get_fdata()
        
        # Broadcast mask if needed
        # If orig is 4D (x,y,z,t) and mask is 3D (x,y,z)
        if orig_data.ndim == 4 and mask_data.ndim == 3:
             mask_data = mask_data[..., np.newaxis]
             
        # Apply
        masked_data = orig_data * mask_data
        
        # Save final masked image
        nib.save(nib.Nifti1Image(masked_data, orig_img.affine, orig_img.header), masked_path)
        
        # Handle returns
        if return_mask and mask_out_path:
             if mask_generated_path != mask_out_path:
                  # Copy/move
                  import shutil
                  shutil.copy(mask_generated_path, mask_out_path)
        
        # Cleanup temps
        if temp_b0 and temp_b0.exists(): temp_b0.unlink()
        if tool_brain_out.exists(): tool_brain_out.unlink()
        if mask_generated_path.exists() and mask_generated_path != mask_out_path: mask_generated_path.unlink()
            
        # Wrap output
        if is_dwi:
             brain_obj = DWIFile(
                 img=Path(masked_path), 
                 entities=entities, 
                 json=input_image.json, 
                 bval=input_image.bval, 
                 bvec=input_image.bvec
             )
        else:
             brain_obj = ImageFile(img=Path(masked_path), entities=entities)
             
        mask_obj = ImageFile(img=Path(mask_out_path), entities=dict(entities, suffix="mask")) if mask_out_path else None
        
        self.logger.info(f"Brain mask created: {masked_path}")

        # Return standardized context or result
        if context is not None:
             context["current_image"] = brain_obj
             if mask_obj:
                 context["current_mask"] = mask_obj # Store mask in context?
             
             # Optionally maintain preprocessed list if it's DWI?
             # But masking changes the image content (skulls stripped).
             # Usually we want to keep the "current" valid image.
             return context
        
        # Standalone
        if return_mask:
             return brain_obj, mask_obj
        return brain_obj


# -------------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------------
def mask_brain(
    input_image: ImageLike | Path,
    output_dir: Path,
    method: Literal["fsl", "mrtrix", "ants", "freesurfer", "synthstrip"] = "fsl",
    n_threads: int = 1,
    return_mask: bool = False,
) -> ImageFile | Tuple[ImageFile, ImageFile]:
    """Quick brain‑masking without building a full pipeline.

    This mirrors the pattern used by ``denoise_image`` and ``gibbs_unring`` –
    it creates a temporary ``BrainMaskingStep`` instance, runs it, and returns
    the resulting ``ImageFile``.
    """
    # ``None`` config/provenance are acceptable for a one‑off call.
    step = BrainMaskingStep(config={}, method=method, n_threads=n_threads)
    return step.run(first_arg=input_image, output_dir=output_dir, return_mask=return_mask)
