"""
Bias field correction module.

Supports ANTs N4BiasFieldCorrection and MRtrix3 dwibiascorrect.
"""

from pathlib import Path
from typing import Optional, Literal, Any
import logging

from ...core import BaseProcessingStep, ProcessingError
from ...core.types import DWIFile, ImageFile
from ...interfaces import ants, mrtrix
from ...io.bids import build_bids_name


class BiasCorrectionStep(BaseProcessingStep):
    """
    Bias field correction step.
    
    Methods:
    - 'ants': Uses ANTs N4BiasFieldCorrection (via simple wrapper or dipy-like logic if needed).
    - 'mrtrix': Uses MRtrix3 dwibiascorrect.
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
        method: Literal['ants', 'mrtrix', 'dipy'] = 'ants'
    ):
        super().__init__(config, logger, provenance)
        self.method = method
        self.logger.info(f"Initialized BiasCorrectionStep with method: {method}")

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    # _extract_image logic replaced by self.unpack_input in base class

    def run(self, first_arg, output_dir: Path, mask: Optional[Path]=None, **kwargs) -> Any:
        context, input_img = self.unpack_input(first_arg)
        if input_img is None:
             raise ProcessingError("No input image provided")

        # Reuse mask from denoising step if available and not provided explicitly
        if mask is None and context:
             if 'current_mask' in context:
                  mask = context['current_mask']
                  self.logger.info(f"Using current_mask from context: {mask}")
             elif 'temp_denoise_mask' in context:
                  mask = context['temp_denoise_mask']
                  self.logger.info(f"Reusing mask from DenoisingStep (fallback): {mask}")
             elif 'brain_mask' in context:
                  # Some workflows might have mask earlier (though rare for Bias)
                  # mask = context['brain_mask'].img # usually ImageFile
                  pass


        output_dir = self.get_step_output_dir(output_dir)
        # Suffix handling
        new_desc = "biascorrected"
        output_img = output_dir / build_bids_name({**input_img.entities, "desc": new_desc})
        bias_field = output_dir / build_bids_name({**input_img.entities, "desc": "biasfield", "suffix": "bfield"})

        # Check if outputs exist
        # Check if outputs exist
        if output_img.exists() and bias_field.exists() and not kwargs.get('force', False):
             # Check timestamps
             in_mtime = input_img.img.stat().st_mtime
             out_mtime = output_img.stat().st_mtime
             
             if in_mtime > out_mtime:
                  self.logger.info(f"Bias input ({input_img.img.name}) is newer than output. Re-running.")
                  self.logger.debug(f"Debug: Input mtime={in_mtime}, Output mtime={out_mtime}, Diff={in_mtime-out_mtime:.2f}s")
             else:
                 self.logger.info(f"Skipping {self.method} bias correction (Outputs exist and up-to-date: {output_img.name})")
                 # If input was DWIFile, preserve it.
                 if isinstance(input_img, DWIFile):
                     result_img = DWIFile(
                        entities=input_img.entities,
                        img=output_img,
                        json=input_img.json,
                        bval=input_img.bval,
                        bvec=input_img.bvec,
                        Delta=getattr(input_img, "Delta", None),
                        delta=getattr(input_img, "delta", None),
                     )
                 else:
                     result_img = ImageFile(entities=input_img.entities, img=output_img, json=input_img.json)

                 if context is not None:
                    context["current_image"] = result_img
                    return context
                 else:
                    return result_img
        elif output_img.exists() and not bias_field.exists() and not kwargs.get('force', False):
            self.logger.info(f"Bias correction output exists but bias field missing ({bias_field.name}). Re-running.")
        else:
            self.logger.info(f"Bias correction output missing ({output_img.name}). Running.")

        self.logger.info(f"Running {self.method} bias correction...")
        
        nthreads = kwargs.get('nthreads', self.config.n_cpus)

        try:
            if self.method == 'ants':
                # Assuming ants.n4bias signature matches expectation
                ants.n4bias(
                    in_file=input_img.img,
                    out_file=output_img,
                    mask=mask,
                    bias_field=bias_field,
                    nthreads=nthreads
                    # Do not pass **kwargs to n4bias as it does not accept them
                )
            elif self.method == 'mrtrix':
                # MRtrix requires bvec/bval for DWI usually if using dwibiascorrect -fslgrad
                # Check if input is DWIFile
                if isinstance(input_img, DWIFile) and input_img.bvec and input_img.bval:
                    mrtrix.dwibiascorrect(
                        in_file=input_img.img,
                        out_file=output_img,
                        in_bvec=input_img.bvec,
                        in_bval=input_img.bval,
                        mask=mask,
                        bias_field=bias_field,
                        nthreads=nthreads,
                        **kwargs
                    )
                else:
                    # Fallback or error if not DWI? 
                    # dwibiascorrect is specific to DWI. 
                    # If this is structural, maybe just use N4 via ants?
                    # For now, let's assume this is mostly for DWI pipeline based on usage.
                    raise ProcessingError("MRtrix dwibiascorrect requires DWI input with bvec/bval.")
            else:
                 raise ValueError(f"Unknown bias correction method: {self.method}")

        except Exception as e:
             raise ProcessingError(f"Bias correction failed: {e}", step_name="bias_correction") from e

        self.logger.info(f"Bias corrected image saved to: {output_img}")

        # Create result object
        if isinstance(input_img, DWIFile):
            result_img = DWIFile(
                entities=input_img.entities,
                img=output_img,
                json=input_img.json,
                bval=input_img.bval,
                bvec=input_img.bvec,
                Delta=getattr(input_img, "Delta", None),
                delta=getattr(input_img, "delta", None),
            )
        else:
             result_img = ImageFile(entities=input_img.entities, img=output_img, json=input_img.json)

        if context is not None:
            context["current_image"] = result_img
            # Optional: maintain preprocessed list
            if isinstance(result_img, DWIFile):
                 pre_list = context.setdefault("preprocessed_dwis", [])
                 if result_img not in pre_list:
                     pre_list.append(result_img)
            return context
        else:
            return result_img
