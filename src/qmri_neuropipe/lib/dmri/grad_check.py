"""
Gradient Check Module.

This module provides a step to check and correct gradient tables using MRtrix3 dwigradcheck.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

from ...core import BaseProcessingStep, ProcessingError
from ...core.types import DWIFile
from ...interfaces import mrtrix
from ...io.bids import build_bids_name

class GradientCheckStep(BaseProcessingStep):
    """
    Step to check and correct gradient tables.
    
    Uses MRtrix3's dwigradcheck to verify and potentially flip gradients if they are incorrect.
    This should be run early in the pipeline.
    """
    
    def __init__(
        self,
        config,
        logger: Optional[logging.Logger] = None,
        provenance = None,
    ):
        super().__init__(config, logger, provenance)
    
    def validate_inputs(self, first_arg, **kwargs) -> None:
        """
        Validate inputs.
        """
        # Minimal validation or delegate to run
        pass

    def validate_outputs(self, result) -> None:
        pass

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        """
        Run gradient check.
        
        Args:
            first_arg: Context dict or DWIFile.
            output_dir: Output directory.
            
        Returns:
            Updated context or new DWIFile.
        """
        # Support context or direct input
        context, _ = self.unpack_input(first_arg)
            
        if context:
            # Global execution mode: process all files in context
            dwi_files = context.get("dwi_files", [])
            processed_files = []
            
            grad_check_dir = self.get_step_output_dir(output_dir)
            
            for input_img in dwi_files:
                # We only create new bvec/bval
                # Standard names based on input, but in grad_check dir
                base_name = grad_check_dir / input_img.img.name.replace("".join(input_img.img.suffixes), "")
                out_bvec = base_name.with_suffix(".bvec")
                out_bval = base_name.with_suffix(".bval")
                
                # Check existing
                if out_bvec.exists() and out_bval.exists():
                    self.logger.info(f"Skipping GradientCheck for {input_img.img.name} (Outputs exist)")
                    result_img = DWIFile(
                        entities=input_img.entities, img=input_img.img, json=input_img.json, bval=out_bval, bvec=out_bvec
                    )
                else:
                    self.logger.info(f"Running dwigradcheck on {input_img.img.name}...")
                    nthreads = kwargs.get('nthreads', self.config.n_cpus)
                    try:
                        mrtrix.dwigradcheck(
                            in_file=input_img,
                            in_bvec=input_img.bvec,
                            in_bval=input_img.bval,
                            export_grad_fsl=(out_bvec, out_bval),
                            nthreads=nthreads,
                            force=bool(kwargs.get('force', False))
                        )
                    except Exception as e:
                        raise ProcessingError(f"Gradient check failed for {input_img.img.name}: {e}") from e
                        
                    result_img = DWIFile(
                        entities=input_img.entities, img=input_img.img, json=input_img.json, bval=out_bval, bvec=out_bvec
                    )
                
                processed_files.append(result_img)
            
            # Update context
            context["dwi_files"] = processed_files
            return context
        
        else:
            # Standalone/Single-file mode
            input_img = first_arg
            if not isinstance(input_img, DWIFile):
                self.logger.warning("GradientCheckStep: Input is not a DWIFile. Skipping.")
                return input_img

            grad_check_dir = self.get_step_output_dir(output_dir)
            
            base_name = grad_check_dir / input_img.img.name.replace("".join(input_img.img.suffixes), "")
            out_bvec = base_name.with_suffix(".bvec")
            out_bval = base_name.with_suffix(".bval")

            if out_bvec.exists() and out_bval.exists():
                self.logger.info(f"Skipping GradientCheckStep (Outputs exist)")
                return DWIFile(
                    entities=input_img.entities, img=input_img.img, json=input_img.json, bval=out_bval, bvec=out_bvec
                )

            self.logger.info(f"Running dwigradcheck on {input_img.img.name}...")
            nthreads = kwargs.get('nthreads', self.config.n_cpus)
            try:
                mrtrix.dwigradcheck(
                    in_file=input_img,
                    in_bvec=input_img.bvec,
                    in_bval=input_img.bval,
                    export_grad_fsl=(out_bvec, out_bval),
                    nthreads=nthreads,
                    force=bool(kwargs.get('force', False))
                )
            except Exception as e:
                raise ProcessingError(f"Gradient check failed: {e}") from e
            
            return DWIFile(
                entities=input_img.entities, img=input_img.img, json=input_img.json, bval=out_bval, bvec=out_bvec
            )
