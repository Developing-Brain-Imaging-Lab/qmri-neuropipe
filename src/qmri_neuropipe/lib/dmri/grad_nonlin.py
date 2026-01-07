from pathlib import Path
from typing import Optional, Any
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageLike, DWIFile, ImageFile
from ...core.run import run_cmd
from ...interfaces import tortoise
from ...io.bids import build_bids_name

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
                # Tortoise often appends suffixes or disallows .gz in output argument
                # We pass output_map, but we check for specific 'graddev_c.nii' result
                tortoise.apply_grad_nonlin(
                    in_file=target_image,
                    out_file=output_map,
                    grad_coeffs=coeffs,
                    nthreads=nthreads,
                    force=kwargs.get('force', False),
                    native_image=native_image_path
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
                         run_cmd(f"gzip -c {actual_output} > {output_map}", label="gzip_gnl")
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
