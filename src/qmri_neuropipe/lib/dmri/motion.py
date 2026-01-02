from pathlib import Path
from typing import Dict, Any, Optional

from ...core.base import BaseProcessingStep
from ...core.types import ImageFile
from ...interfaces.nifreeze import run_nifreeze
import logging

class NiiFreezeStep(BaseProcessingStep):
    """
    Motion correction using NiiFreeze (ShoreLine).
    Alternative to FSL Eddy.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.b0_thresh = config.get("b0_thresh", 5)
        self.model = config.get("model", "b0")
        self.strategy = config.get("strategy", "random")
        self.seed = config.get("seed", 2021)
        
    def run(self, context: Dict[str, Any], output_dir: Path, **kwargs) -> Dict[str, Any]:
        """
        Run NiiFreeze on the input DWI.
        """
        dwi_files = context.get('dwi_files')
        if not dwi_files:
            self.logger.warning("No DWI files found for NiiFreeze.")
            return context

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
            
            if final_dwi.exists() and self.config.get("skip_existing", False) and not kwargs.get('force', False):
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
            # run() signature in base.py usually allows **kwargs, but here run is fixed.
            # We fetch from config.
            nthreads = self.config.get("n_cpus", 1)

            try:
                out_d, out_v, out_l = run_nifreeze(
                    in_dwi=dwi.img,
                    in_bval=dwi.bval,
                    in_bvec=dwi.bvec,
                    out_dir=step_dir,
                    out_prefix=out_prefix,
                    b0_thresh=self.b0_thresh,
                    nthreads=nthreads,
                    verbose=True,
                    model=self.model,
                    strategy=self.strategy,
                    seed=self.seed
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
                     dest_json = out_d.parent / (out_d.name.replace('.nii.gz', '.json').replace('.nii', '.json'))
                     shutil.copy(dwi.json, dest_json)
                     corrected.json = dest_json

                corrected_files.append(corrected)
                
            except ImportError:
                 self.logger.error("NiiFreeze is not installed. Skipping motion correction.")
                 corrected_files.append(dwi)
            except Exception as e:
                self.logger.error(f"NiiFreeze failed for {dwi.img.name}: {e}")
                corrected_files.append(dwi) # Fallback
                
        # Update Context
        context['dwi_files'] = corrected_files
        
        # Cleanup if needed?
        return context
