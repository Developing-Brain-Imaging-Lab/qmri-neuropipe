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
        new_desc = "gnlcorrected"
        output_img = output_dir / build_bids_name({**input.entities, "desc": new_desc})
        
        # Check if output exists
        if output_img.exists() and not kwargs.get('force', False):
             self.logger.info(f"Skipping TORTOISE GNL correction (Output exists: {output_img.name})")
             if isinstance(input, DWIFile):
                 result_img = DWIFile(
                    entities=input.entities,
                    img=output_img,
                    json=input.json,
                    bval=input.bval,
                    bvec=input.bvec
                 )
             else:
                 result_img = ImageFile(entities=input.entities, img=output_img, json=input.json)
                 
             if is_context:
                context["current_image"] = result_img
                return context
             else:
                return result_img
        
        # Determine Gradient Cofficients
        # 1. Constructor arg
        coeffs = self.grad_coeffs
        # 2. Kwargs
        if not coeffs:
            coeffs = kwargs.get('grad_coeffs')
        # 3. Config (dmri.preprocessing.grad_nonlin.coeff_file)
        if not coeffs:
             dmri_cfg = self.config.get('dmri', {}).get('preprocessing', {}).get('grad_nonlin', {})
             c_path = dmri_cfg.get('coeff_file')
             if c_path:
                 coeffs = Path(c_path)
        
        if not coeffs or not coeffs.exists():
            raise ProcessingError(f"Gradient nonlinearity coefficients file not found/provided. (coeffs={coeffs})")

        self.logger.info(f"Running TORTOISE Gradient Nonlinearity Correction...")
        
        nthreads = kwargs.get('nthreads', self.config.n_cpus)
        
        # Assume input image should be DWI or similar. 
        # Interface should handle it.
        
        try:
            tortoise.apply_grad_nonlin(
                in_file=input.img,
                out_file=output_img,
                grad_coeffs=coeffs,
                nthreads=nthreads,
                force=kwargs.get('force', False)
            )
        except Exception as e:
            raise ProcessingError(f"TORTOISE GNL correction failed: {e}", step_name="grad_nonlin_correct") from e
            
        self.logger.info(f"GNL corrected image saved to: {output_img}")

        # Create result object
        if isinstance(input, DWIFile):
            result_img = DWIFile(
                entities=input.entities,
                img=output_img,
                json=input.json,
                bval=input.bval,
                bvec=input.bvec # Vectors might need correction too? TORTOISE typically handles or outputs them.
                                # For now assuming bvecs/bvals stay valid or are updated in sidecar.
                                # If Diffprep outputs new bvecs, we should look for them.
            )
        else:
             result_img = ImageFile(entities=input.entities, img=output_img, json=input.json)

        if is_context:
            context["current_image"] = result_img
            # Optional: maintain preprocessed list
            if isinstance(result_img, DWIFile):
                 pre_list = context.setdefault("preprocessed_dwis", [])
                 if result_img not in pre_list:
                     pre_list.append(result_img)
            return context
        else:
            return result_img
