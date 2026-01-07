"""
Tractography and Bundle Segmentation Module.

This module provides processing steps for tractography (TractSeg, PyAFQ).
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging
import shutil

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...interfaces import mrtrix, tractseg
from ...io.bids import build_bids_name, get_entities_from_path

class TractSegStep(BaseProcessingStep):
    """
    Step to run TractSeg bundle segmentation.
    
    Can run on:
    1. Peaks (generated via CSD -> sh2peaks)
    2. Input DWI (TractSeg handles preprocessing)
    
    If 'peaks' are not found in context, we generate them using MRtrix.
    """
    def __init__(self, config, logger, provenance, method='tractseg', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method # 'tractseg'
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        tractseg_out = output_dir / "TractSeg"
        tractseg_out.mkdir(parents=True, exist_ok=True)
        
        # Check logic for skip existing
        skip = hasattr(self.config, 'skip_existing') and self.config.skip_existing
        bundle_out_dir = tractseg_out / "bundle_masks"
        
        if skip and bundle_out_dir.exists() and any(bundle_out_dir.glob("*.nii.gz")):
             self.logger.info(f"Skipping TractSeg for {dwi.img.name} (Found existing bundles)")
             # Populate context
             # TractSeg outputs many files (one per bundle).
             # We store the directory path in context, or a list?
             # Probably directory path is cleaner for downstream Analysis step.
             
             context.setdefault('segmentations', {})['TractSeg'] = bundle_out_dir
             return context

        self.logger.info(f"Running TractSeg on {dwi.img.name}")
        
        # 1. Input Preparation: Peaks vs DWI
        # TractSeg works best with Peaks.
        # Check context for CSD peaks?
        input_file = None
        input_type = "peaks" 
        
        # Check if we have CSD output in context (e.g. from CSDFittingStep)
        # We need the SH coefficients (FOD) to convert to peaks.
        # usually stored in context['modeling_results']['CSD']['wmFOD'] or 'FOD'
        
        csd_results = context.get('modeling_results', {}).get('CSD', {})
        fod_path = csd_results.get('wmFOD') or csd_results.get('FOD')
        
        peaks_path = tractseg_out / "peaks.nii.gz"
        
        if fod_path and Path(fod_path).exists():
             self.logger.info(f"Using existing CSD FOD for peak generation: {fod_path}")
             # Generate peaks
             if not peaks_path.exists() or not skip:
                 mrtrix.sh2peaks(fod_path, peaks_path, nthreads=self.nthreads, force=True)
             input_file = peaks_path
        else:
             # Fallback: Let TractSeg handle DWI? Or generate CSD here?
             # TractSeg from DWI requires simplified preprocessing (DWI -> CSD -> Peaks).
             # It might be safer to let TractSeg do it if we trust its pipeline, 
             # OR we enforce our pipeline's CSD.
             # If we haven't run CSD step, we probably should pass DWI to TractSeg and let it run 'auto'.
             
             self.logger.info("No CSD FOD found in context. Passing DWI to TractSeg (will perform internal CSD Preprocessing).")
             input_file = dwi.img
             input_type = "dmri"
        
        # Prepare Mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask
             
        # Run TractSeg
        # Output type: bundle_masks (default). We might also want TOM/endings later?
        # For now, just bundles.
        
        # Extract extra args
        ts_kwargs = self.kwargs.copy()
        output_type = ts_kwargs.pop('output_type', 'bundle_masks')
        
        tractseg.run_tractseg(
            input_file=input_file,
            output_dir=tractseg_out,
            input_type=input_type,
            output_type=output_type,
            brain_mask=mask_path,
            raw_diffusion=dwi.img if input_type == 'dmri' else None,
            bvals=dwi.bval,
            bvecs=dwi.bvec,
            nr_cpus=self.nthreads,
            extra_args=ts_kwargs
        )
        
        # TractSeg puts output in a subdirectory typically?
        # CLI usage: -o output_dir. Files go into output_dir/bundle_masks/
        # if output_type=bundle_masks
        
        if (tractseg_out / "bundle_masks").exists():
             context.setdefault('segmentations', {})['TractSeg'] = tractseg_out / "bundle_masks"
        else:
             self.logger.warning("TractSeg completed but 'bundle_masks' directory not found.")
             
        # Optional: Tractometry
        # If requested via config?
        # For now, we just do segmentation.
        
        return context


class PyAFQStep(BaseProcessingStep):
    """
    Step to run PyAFQ / BabyAFQ fiber quantification.
    """
    def __init__(self, config, logger, provenance, method='pyafq', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method 
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        from ...interfaces import pyafq
        
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        afq_out = output_dir / "PyAFQ"
        afq_out.mkdir(parents=True, exist_ok=True)
        
        # Output directory for AFQ (PyAFQ usually creates subfolders)
        # We can pass afq_out as the base.
        
        # Check logic for skip existing
        skip = hasattr(self.config, 'skip_existing') and self.config.skip_existing
        # AFQ usually creates 'bundles' folder
        bundles_dir = afq_out / "sub-01" / "ses-01" / "dwi" / "bundles" # Standard AFQ structure is strict?
        # Actually PyAFQ's structure depends on customization but usually follows BIDS derivatives.
        
        # Let's rely on interface wrapper returning output path or checking existence
        # For simple check:
        if skip and any(afq_out.glob("**/*.trk")): # Check for any track file
             self.logger.info(f"Skipping PyAFQ for {dwi.img.name} (Found existing outputs)")
             # Populate context
             context.setdefault('segmentations', {})['PyAFQ'] = afq_out
             return context

        self.logger.info(f"Running PyAFQ on {dwi.img.name}")
        
        # Prepare Mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask
             
        # Extract Config
        afq_kwargs = self.kwargs.copy()
        profile = afq_kwargs.pop('profile', 'default')
        
        # Clean kwargs of known non-AFQ args?
        
        try:
            out_p = pyafq.run_afq(
                dwi_file=dwi.img,
                bval_file=dwi.bval,
                bvec_file=dwi.bvec,
                output_dir=afq_out,
                brain_mask=mask_path,
                profile=profile,
                n_cpus=self.nthreads,
                **afq_kwargs
            )
            
            context.setdefault('segmentations', {})['PyAFQ'] = out_p
            
        except ImportError:
            self.logger.error("PyAFQ not installed. Skipping.")
        except Exception as e:
            self.logger.error(f"PyAFQ failed: {e}")
            if self.config.stop_on_error: raise e
            
        return context
