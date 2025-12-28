from pathlib import Path
from typing import Dict, Any, Optional

from ...core.base import BaseProcessingStep
from ...core.types import ImageFile
from ...interfaces.nifreeze import nifreeze_shoreline
import logging

class NiiFreezeStep(BaseProcessingStep):
    """
    Motion correction using NiiFreeze (ShoreLine).
    Alternative to FSL Eddy.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.b0_thresh = config.get("b0_thresh", 5)
        
    def run(self, context: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """
        Run NiiFreeze on the input DWI.
        """
        dwi_files = context.get('dwi_files')
        if not dwi_files:
            self.logger.warning("No DWI files found for NiiFreeze.")
            return context

        # NiiFreeze runs on single 4D file usually.
        # Assuming concatenated or single run.
        # If multiple, loop? 
        # Standard pipeline usually has one 'dwi_files' list, often just one 4D file after concatenation (if implemented) 
        # or we process each run independently.
        
        # NOTE: PreprocessingWorkflow processes 'files' individually in current loop structure?
        # Re-checking dmri.py: process_subject iterates self.pipeline.loader.load_subject...
        # which returns a Subject object with .sessions -> .scans
        # But `process_subject` loops 'preprocessed_dwis'.
        # Actually `PreprocessingWorkflow.run` takes `context` which has `dwi_files`.
        # `dwi_files` is a list of ImageFile.
        
        corrected_files = []
        
        for idx, dwi in enumerate(dwi_files):
            stem = dwi.img.stem.replace('.nii', '').replace('.gz', '')
            out_prefix = f"{stem}_desc-niifreeze"
            
            # Subdir for this run
            step_dir = output_dir / "motion_correction" / stem
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Check skip
            final_dwi = step_dir / f"{out_prefix}_corrected.nii.gz"
            final_bvec = step_dir / f"{out_prefix}_corrected.bvec"
            final_bval = step_dir / f"{out_prefix}_corrected.bval"
            
            if final_dwi.exists() and self.config.get("skip_existing", False):
                self.logger.info(f"Skipping NiiFreeze (Output exists): {final_dwi}")
                
                # Create ImageFile wrapper
                corrected = ImageFile(
                    path=final_dwi, 
                    bids_dir=dwi.bids_dir,
                    entities=dwi.entities.copy()
                )
                corrected.bvec = final_bvec
                corrected.bval = final_bval
                # Update entities
                corrected.entities['desc'] = 'motion' # or 'niifreeze'
                corrected_files.append(corrected)
                continue

            self.logger.info(f"Running NiiFreeze on {dwi.img.name}")
            
            if not dwi.bval or not dwi.bvec:
                 self.logger.error(f"Missing bval/bvec for {dwi.img.name}. Skipping NiiFreeze.")
                 corrected_files.append(dwi) # Fallback to input?
                 continue

            # Fetch nthreads
            nthreads = kwargs.get('nthreads', self.config.get("n_cpus", 1))

            try:
                out_d, out_v, out_l = nifreeze_shoreline(
                    in_dwi=dwi.img,
                    in_bval=dwi.bval,
                    in_bvec=dwi.bvec,
                    out_dir=step_dir,
                    out_prefix=out_prefix,
                    b0_thresh=self.b0_thresh,
                    nthreads=nthreads,
                    verbose=True
                )
                
                # Wrap
                corrected = ImageFile(
                    path=out_d,
                    bids_dir=dwi.bids_dir,
                    entities=dwi.entities.copy()
                )
                corrected.bvec = out_v
                corrected.bval = out_l
                corrected.entities['desc'] = 'motion'
                
                # Propagate Sidecar JSON
                if dwi.json:
                     shutil.copy(dwi.json, out_d.with_suffix("").with_suffix(".json"))
                     corrected.json = out_d.with_suffix("").with_suffix(".json")

                corrected_files.append(corrected)
                
            except Exception as e:
                self.logger.error(f"NiiFreeze failed for {dwi.img.name}: {e}")
                corrected_files.append(dwi) # Fallback
                
        # Update Context
        context['dwi_files'] = corrected_files
        
        # Cleanup if needed?
        return context
