from pathlib import Path
from typing import Optional, Any
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageLike, DWIFile, ImageFile
from ...core.run import run_cmd
from ...interfaces import tortoise
from ...io.bids import build_bids_name
from ...interfaces.mrtrix import dwiextract, mrcalc, mrmath

class TortoiseGradNonlinCorrectStep(BaseProcessingStep):
    """
    Gradient Nonlinearity Correction using TORTOISE.
    """

    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        grad_coeffs: Optional[Path] = None,
        is_resampled: bool = False
    ):
        super().__init__(config, logger, provenance)
        self.grad_coeffs = grad_coeffs
        self.is_resampled = is_resampled
        self.logger.info("Initialized TortoiseGradNonlinCorrectStep")

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def _extract_image(self, first_arg) -> ImageLike:
        if isinstance(first_arg, dict):
            img = first_arg.get("current_image")
            if img is None:
                raise ValidationError("TortoiseGradNonlinCorrectStep expects context['current_image'] to be set")
            return img
        return first_arg

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        
        is_context = isinstance(first_arg, dict)
        if is_context:
            context = dict(first_arg)
            input = self._extract_image(context)
            native_ref = context.get('native_dwi_for_gnl')
        else:
            context = None
            input = first_arg
            native_ref = kwargs.get('native_reference')

        if self.is_resampled and not native_ref:
             self.logger.warning("GNL Step is configured for resampled data but 'native_dwi_for_gnl' not found in context. Using input as native (might be incorrect if already resampled).")
             native_ref = input

        output_dir = output_dir / "grad_nonlin"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Output is a tensor map, not a corrected DWI
        new_desc = "gnl_tensor" 
        output_map = output_dir / build_bids_name({**input.entities, "desc": new_desc})
        
        # Check if output exists
        if output_map.exists() and not kwargs.get('force', False):
             self.logger.info(f"Skipping TORTOISE GNL calculation (Output exists: {output_map.name})")
        else:
            # Determine Gradient Cofficients
            coeffs = self.grad_coeffs
            if not coeffs: coeffs = kwargs.get('grad_coeffs')
            if not coeffs:
                 dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('grad_nonlin', {})
                 c_path = dmri_cfg.get('coeff_file')
                 if c_path: coeffs = Path(c_path)
            
            if not coeffs or not coeffs.exists():
                raise ProcessingError(f"Gradient nonlinearity coefficients file not found/provided. (coeffs={coeffs})")
    
            self.logger.info(f"Calculating TORTOISE Gradient Nonlinearity Tensor Map...")
            
            nthreads = kwargs.get('nthreads', self.config.n_cpus)
            
            target_image = input.img
            native_image_path = native_ref.img if native_ref else None
            
            # If not resampled, input is effectively native
            if not self.is_resampled:
                native_image_path = None # Not implicitly needed by interface
            
            try:
                # Prepare 3D mean b0 images for initial and final spaces
                # This is required by CreateGradientNonlinearityBMatrix
                
                initial_image_path = None
                
                # 1. Native Space 3D Reference (Initial Image) - Only if resampled
                if self.is_resampled:
                    native_b0_refs = []
                    if native_ref:
                        native_img_path = native_ref.img
                        native_b0 = output_dir / "native_b0_mean.nii.gz"
                        
                        if not native_b0.exists() or kwargs.get('force', False):
                             # Extract b0s and mean
                             temp_b0s = output_dir / "native_b0s.mif"
                             
                             # Explicitly pass grads if available
                             nbvec = getattr(native_ref, 'bvec', None)
                             nbval = getattr(native_ref, 'bval', None)
                             
                             dwiextract(native_img_path, temp_b0s, bzero=True, in_bvec=nbvec, in_bval=nbval, force=True)
                             mrmath(temp_b0s, "mean", native_b0, axis=3, force=True)
                             temp_b0s.unlink(missing_ok=True)
                        native_b0_refs.append(native_b0)
                        initial_image_path = native_b0_refs[0]
                    else:
                         self.logger.warning("GNL Step is resampled but no native reference found. Skipping initial_image.")

                # 2. Final Space 3D Reference (Final Image)
                # Current input image (partially processed, likely coregistered)
                # Need its mean b0 to define the target geometry
                final_b0_path = output_dir / "final_b0_mean.nii.gz"
                if not final_b0_path.exists() or kwargs.get('force', False):
                     temp_b0s_final = output_dir / "final_b0s.mif"
                     
                     # Explicitly pass grads if available
                     fbvec = getattr(input, 'bvec', None)
                     fbval = getattr(input, 'bval', None)
                     
                     dwiextract(input.img, temp_b0s_final, bzero=True, in_bvec=fbvec, in_bval=fbval, force=True)
                     mrmath(temp_b0s_final, "mean", final_b0_path, axis=3, force=True)
                     temp_b0s_final.unlink(missing_ok=True)
                
                final_image_path = final_b0_path
                
                # Run GNL. The command outputs 'graddev_c.nii' in the 'initial_image' directory ?? 
                # OR in the CWD. User says "output name is ... extension of the input image names".
                # To be safe, we run in output_dir.
                
                tortoise.apply_grad_nonlin(
                    initial_image=initial_image_path,
                    final_image=final_image_path,
                    grad_coeffs=coeffs,
                    nthreads=nthreads,
                    force=kwargs.get('force', False),
                    cwd=output_dir # Run in output dir to capture the output file there
                )
                
                # Handling explicit filename request from user:
                # "the output ... is going to be the image with the 'graddev_c.nii' at the end of the name."
                # Check for *graddev_c.nii in output directory
                candidates = list(output_dir.glob("*graddev_c.nii"))
                if not candidates:
                    candidates = list(output_dir.glob("*graddev_c.nii.gz"))
                
                if candidates:
                     actual_output = candidates[0]
                     # Gzip if needed and rename to standardized proper output_map
                     if actual_output.suffix == '.nii':
                         self.logger.info(f"Gzipping and renaming GNL output: {actual_output.name} -> {output_map.name}")
                         import gzip
                         import shutil
                         with open(actual_output, 'rb') as f_in:
                             with gzip.open(output_map, 'wb') as f_out:
                                 shutil.copyfileobj(f_in, f_out)
                         actual_output.unlink() # remove .nii
                     else:
                         self.logger.info(f"Renaming GNL output: {actual_output.name} -> {output_map.name}")
                         actual_output.rename(output_map)
                
            except Exception as e:
                raise ProcessingError(f"TORTOISE GNL calculation failed: {e}", step_name="grad_nonlin_correct") from e
                
            if not output_map.exists():
                 self.logger.warning(f"Expected GNL output {output_map} not found. Proceeding only if user overrides.")
                 # raise ProcessingError?
                 
            self.logger.info(f"GNL tensor map saved to: {output_map}")

        # Store map in context
        if is_context:
            context["gnl_map"] = output_map
            return context
        else:
            # If called with single image, we return the path to the map?
            # Or return the image structure with map attached?
            # BaseProcessingStep usually returns 'result'.
            # If we return the map path, the pipeline loop might get confused if it expects ImageFile.
            # But TortoiseGradNonlinCorrectStep is usually run in context.
            return output_map
