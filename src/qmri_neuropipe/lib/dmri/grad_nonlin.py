from pathlib import Path
from typing import Optional, Any
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import ImageLike, DWIFile, ImageFile
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
        grad_coeffs: Optional[Path] = None
    ):
        super().__init__(config, logger, provenance)
        self.grad_coeffs = grad_coeffs
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
        else:
            context = None
            input = first_arg

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
            
            try:
                tortoise.apply_grad_nonlin(
                    in_file=input.img,
                    out_file=output_map,
                    grad_coeffs=coeffs,
                    nthreads=nthreads,
                    force=kwargs.get('force', False)
                )
            except Exception as e:
                raise ProcessingError(f"TORTOISE GNL calculation failed: {e}", step_name="grad_nonlin_correct") from e
                
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
