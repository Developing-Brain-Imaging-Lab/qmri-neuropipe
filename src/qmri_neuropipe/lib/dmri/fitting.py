"""
Diffusion Model Fitting Module.

This module provides classes for fitting various diffusion models:
- DTI (Diffusion Tensor Imaging)
- DKI (Diffusion Kurtosis Imaging)
- NODDI (Neurite Orientation Dispersion and Density Imaging)
- SANDI (Soma and Neurite Density Imaging)
- MAPMRI (Mean Apparent Propagator MRI)

Supports multiple backends (DIPY, FSL, MRtrix3, AMICO, Dmipy).
Refactored to delegate logic to interface modules.
"""

from pathlib import Path
from typing import Optional, Any
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import DWIFile
from ...interfaces import dipy, fsl, mrtrix, amico, dmipy
from ...io.bids import build_bids_name


class DTIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='dipy', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    # Removed separate should_skip to ensure context is populated during skip logic in run()


    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        # Resolve inputs
        dwi = context.get('current_image') if isinstance(context, dict) else context
        
        # Output directory for this model
        # Output directory for this model
        model_out = output_dir / "DTI"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Check for existing outputs
        # Logic: Skip if output exists unless force is True
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'DTI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        # Resolve metrics early for smart skip
        requested_metrics = kwargs.get('metrics')
        if not requested_metrics:
            if isinstance(self.config, dict):
                requested_metrics = self.config.get('metrics', [])
            elif hasattr(self.config, 'metrics'):
                requested_metrics = self.config.metrics
            else:
                requested_metrics = [] # will trigger default set
        
        # Replicate defaults from interface if empty
        if not requested_metrics:
             requested_metrics = ["fa", "md", "ad", "rd", "color_fa"]
             
        # Ensure tensors are checked (legacy behavior always adds them in interface)
        if "tensor" not in requested_metrics: requested_metrics.append("tensor")
        if "tensor_mrtrix" not in requested_metrics: requested_metrics.append("tensor_mrtrix")

        # Check for key output (FA)
        fa_path = model_out / build_bids_name(ents, suffix='FA')
        
        # Smart Skip Logic
        should_run = True
        existing_results = {}
        
        if fa_path.exists() and not force:
             # Check if ALL requested metrics exist
             missing_metrics = []
             all_found = True
             
             for m in requested_metrics:
                 if m == 'tensor': suffix = 'tensor'
                 elif m == 'tensor_fsl': suffix = 'tensorFSL'
                 elif m == 'tensor_mrtrix': suffix = 'tensorMRTRIX'
                 elif m == 'color_fa': suffix = 'DECFA'
                 else: suffix = m.upper()
                 
                 fpath = model_out / build_bids_name(ents, suffix=suffix)
                 
                 if fpath.exists():
                     existing_results[suffix] = fpath
                 else:
                     # dynamic glob fallback check
                     found = list(model_out.glob(f"*_{suffix}.nii.gz"))
                     if found:
                         existing_results[suffix] = found[0]
                     else:
                         all_found = False
                         missing_metrics.append(m)
             
             # Also assume 'FA' is critical, if specific FA file missing but glob found it? 
             # (handled by check above)
             
             if all_found:
                  # PARAMETER CHECK: Validate 'FittingMethod' if using DIPY
                  # We check the sidecar of the FA map (assuming it represents the set)
                  param_mismatch = False
                  sidecar_path = None
                  
                  # Find a valid sidecar path (prefer FA)
                  candidates = ['FA', 'fa', list(existing_results.keys())[0]]
                  for c in candidates:
                       if c in existing_results:
                            p = existing_results[c]
                            # Try replacing extension
                            s = p.parent / (p.name.split('.')[0] + '.json')
                            if not s.exists():
                                 # Try simplistic extension swap
                                 s = Path(str(p).replace('.nii.gz', '.json'))
                            
                            if s.exists():
                                 sidecar_path = s
                                 break
                  
                  if sidecar_path and self.method == 'dipy':
                       try:
                            import json
                            with open(sidecar_path, 'r') as f:
                                 meta = json.load(f)
                            
                            prev_method = meta.get('FittingMethod')
                            
                            # Resolve current requested method (matching execution logic)
                            current_method = None
                            if isinstance(self.config, dict):
                                 current_method = self.config.get('fit_method')
                            elif hasattr(self.config, 'fit_method'):
                                 current_method = self.config.fit_method
                                 
                            if not current_method and 'fit_method' in self.kwargs:
                                 current_method = self.kwargs['fit_method']
                            if not current_method and 'sub_method' in self.kwargs:
                                 # Fallback
                                 current_method = self.kwargs['sub_method']
                            
                            # Normalize for comparison? (Usually uppercase/exact match required by DIPY)
                            
                            if prev_method and current_method and prev_method != current_method:
                                 param_mismatch = True
                                 self.logger.info(f"Re-running DTI fit. Parameter mismatch: Stored='{prev_method}', Config='{current_method}'")
                                 
                       except Exception as e:
                            self.logger.warning(f"Failed to check DTI parameters from sidecar: {e}")
                  
                  if not param_mismatch:
                       should_run = False
                       self.logger.info(f"Skipping DTI fit (Found all {len(requested_metrics)} requested metrics).")
                  else:
                       # Mismatch detected, intentionally leaving should_run = True
                       pass
             else:
                  self.logger.info(f"Re-running DTI fit. Existing outputs found but missing metrics: {missing_metrics}")
        
        if not should_run:
             context.setdefault('modeling_results', {})['DTI'] = existing_results
             return context

        self.logger.info(f"Running DTI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        # Check for Gradient Nonlinearity Map in context
        gnl_map = context.get('gnl_map')
        if gnl_map and not gnl_map.exists():
            self.logger.warning(f"GNL map path found in context but file missing: {gnl_map}")
            gnl_map = None

        if self.method == 'dipy':
            from ...interfaces.dipy import fit_dti
            # Map parameters
            dipy_kwargs = self.kwargs.copy()
            
            # Additional config extraction
            if isinstance(self.config, dict):
                 for k in ['smoothing_fwhm', 'fit_method', 'weights_method', 'return_S0_hat', 'sigma']:
                      if k in self.config and k not in dipy_kwargs:
                          dipy_kwargs[k] = self.config[k]

            if 'sub_method' in self.kwargs:
                # Only fallback to sub_method if fit_method NOT explicitly provided in config
                # (Prevents flattening of 'parameters' overwriting top-level 'fit_method')
                has_fit_method = False
                if isinstance(self.config, dict) and 'fit_method' in self.config: has_fit_method=True
                elif hasattr(self.config, 'fit_method'): has_fit_method=True
                
                if not has_fit_method:
                    dipy_kwargs['fit_method'] = self.kwargs['sub_method']
            
            # Default metrics if none provided
            if 'metrics' not in dipy_kwargs:
                dipy_kwargs['metrics'] = ["fa", "md", "ad", "rd", "color_fa", "evals", "evecs"]
            
            # Ensure "tensor" outputs are included (legacy behavior)
            curr_metrics = dipy_kwargs['metrics']
            if "tensor" not in curr_metrics: curr_metrics.append("tensor")
            if "tensor_mrtrix" not in curr_metrics: curr_metrics.append("tensor_mrtrix")
            
            if gnl_map:
                self.logger.info(f"Using Gradient Nonlinearity Tensor Map for DIPY: {gnl_map}")
                dipy_kwargs['grad_nonlin'] = gnl_map
            
            if hasattr(dwi, 'Delta') and dwi.Delta: dipy_kwargs['Delta_file'] = dwi.Delta
            if hasattr(dwi, 'delta') and dwi.delta: dipy_kwargs['delta_file'] = dwi.delta
                
            fit_dti(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **dipy_kwargs)
            
        elif self.method == 'fsl':
            from ...interfaces.fsl import fit_dti
            # fsl parameters
            fsl_kwargs = {}
            if 'save_tensor' in self.kwargs:
                fsl_kwargs['save_tensor'] = self.kwargs['save_tensor']
                
            if gnl_map:
                self.logger.info(f"Using Gradient Nonlinearity Tensor Map: {gnl_map}")
                fsl_kwargs['grad_nonlin'] = gnl_map
                
            fit_dti(dwi, model_out, mask_file=mask_path, **fsl_kwargs)
            
        elif self.method == 'mrtrix':
            from ...interfaces.mrtrix import fit_dti
            # mrtrix parameters
            mrtrix_kwargs = {}
            if 'metrics' in self.kwargs:
                mrtrix_kwargs['metrics'] = self.kwargs['metrics']
            
            if gnl_map:
                self.logger.warning("Gradient nonlinearity tensor map found but not currently supported by MRtrix backend (in this pipeline). GNL correction will be ignored.")
                
            fit_dti(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **mrtrix_kwargs)
        else:
            raise ValueError(f"Unknown DTI method: {self.method}")
            
        # Track Outputs for Normalization
        # Collect paths matching pattern in model_out
        # Track Outputs for Normalization
        # Collect paths matching pattern in model_out, FILTERED by requested_metrics
        results = {}
        req_norm = [m.strip().lower() for m in requested_metrics]
        
        # Ensure tensors are included in check list (legacy)
        req_norm.append('tensor')
        req_norm.append('tensormrtrix')
        req_norm.append('tensorfsl')
        
        for p in model_out.glob("*.nii.gz"):
             # Assuming BIDS: sub-XX_desc-XX_model-DTI_SUFFIX.nii.gz
             name_part = p.name.replace('.nii.gz', '')
             if '_' in name_part:
                 suffix = name_part.split('_')[-1]
                 
                 check_suffix = suffix.lower()
                 if check_suffix == 'decfa': check_suffix = 'color_fa'
                 
                 if check_suffix in req_norm:
                     results[suffix] = p
             else:
                 # Fallback?
                 pass
        
        context.setdefault('modeling_results', {})['DTI'] = results
        return context


class DKIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='dipy', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    # Removed separate should_skip to ensure context is populated during skip logic in run()
    
    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        # Resolve inputs
        dwi = context.get('current_image') if isinstance(context, dict) else context
        
        # Output directory for this model
        model_out = output_dir / "DKI"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Check for existing outputs
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'DKI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        # Resolve metrics early for smart skip
        requested_metrics = kwargs.get('metrics')
        if not requested_metrics:
            if isinstance(self.config, dict):
                requested_metrics = self.config.get('metrics')
            elif hasattr(self.config, 'metrics'):
                requested_metrics = getattr(self.config, 'metrics')
        
        if not requested_metrics:
             requested_metrics = ["mk", "ak", "rk", "fa", "md", "ad", "rd"] # Default (7 metrics)
             
        # Check for key output (MK or FA from DKI)
        # DKI outputs typically: MK, AK, RK, FA, MD
        mk_path = model_out / build_bids_name(ents, suffix='MK')
        
        # Smart Skip Logic
        should_run = True
        existing_results = {}
        
        if (mk_path.exists() or (model_out / build_bids_name(ents, suffix='FA')).exists()) and not force:
             # Check if ALL requested metrics exist
             missing_metrics = []
             all_found = True
             
             for m in requested_metrics:
                 suffix = m.upper()
                 # Predict filename
                 # Handle special naming if needed (e.g. color_fa -> DECFA)
                 # But for DKI standard metrics, suffix usually matches metric name
                 fpath = model_out / build_bids_name(ents, suffix=suffix)
                 
                 if fpath.exists():
                     existing_results[suffix] = fpath
                 else:
                     # dynamic glob fallback?
                     found = list(model_out.glob(f"*_{suffix}.nii.gz"))
                     if found:
                         existing_results[suffix] = found[0]
                     else:
                         all_found = False
                         missing_metrics.append(m)
             
             if all_found:
                  # PARAMETER CHECK: Validate 'FittingMethod' (Smart Skip)
                  param_mismatch = False
                  sidecar_path = None
                  
                  # Find a valid sidecar path (prefer MK)
                  candidates = ['MK', 'mk', list(existing_results.keys())[0]]
                  for c in candidates:
                       if c in existing_results:
                            p = existing_results[c]
                            s = p.parent / (p.name.split('.')[0] + '.json')
                            if not s.exists():
                                 s = Path(str(p).replace('.nii.gz', '.json'))
                            
                            if s.exists():
                                 sidecar_path = s
                                 break
                  
                  if sidecar_path:
                       try:
                            import json
                            with open(sidecar_path, 'r') as f:
                                 meta = json.load(f)
                            
                            prev_method = meta.get('FittingMethod')
                            current_method = None
                            
                            if isinstance(self.config, dict):
                                 current_method = self.config.get('fit_method')
                            elif hasattr(self.config, 'fit_method'):
                                 current_method = self.config.fit_method
                                 
                            if not current_method and 'fit_method' in self.kwargs:
                                 current_method = self.kwargs['fit_method']
                                 
                            # If fit method changed, force re-run
                            # Note: Earlier we fixed fit_dki to actually save this method!
                            if prev_method and current_method and prev_method != current_method:
                                 param_mismatch = True
                                 self.logger.info(f"Re-running DKI fit. Parameter mismatch: Stored='{prev_method}', Config='{current_method}'")
                                 
                       except Exception as e:
                            self.logger.warning(f"Failed to check DKI parameters from sidecar: {e}")

                  if not param_mismatch:
                       should_run = False
                       self.logger.info(f"Skipping DKI fit (Found all {len(requested_metrics)} requested metrics).")
                  else:
                       pass
             else:
                  self.logger.info(f"Re-running DKI fit. Existing outputs found but missing metrics: {missing_metrics}")
        
        if not should_run:
             context.setdefault('modeling_results', {})['DKI'] = existing_results
             return context

        self.logger.info(f"Running DKI fit ({self.method}) on {dwi.img.name}")
        

        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        if self.method == 'dipy':
             from ...interfaces.dipy import fit_dki
             
             fit_kwargs = self.kwargs.copy()
             
             # Robustly checking config for new options (dict or object)
             opts = ['smoothing_fwhm', 'mean_signal', 'fit_method', 'weights_method', 'return_S0_hat', 'metrics']
             
             if isinstance(self.config, dict):
                 for k in opts:
                      if k in self.config and k not in fit_kwargs:
                          fit_kwargs[k] = self.config[k]
                          
             for k in opts:
                  if k not in fit_kwargs and hasattr(self.config, k):
                       fit_kwargs[k] = getattr(self.config, k)

             # Check for GNL Map
             gnl_map = context.get('gnl_map') if isinstance(context, dict) else None
             if gnl_map:
                if not gnl_map.exists():
                     self.logger.warning(f"GNL map path missing: {gnl_map}")
                else:
                     self.logger.info(f"Using Gradient Nonlinearity Tensor Map for DIPY DKI: {gnl_map}")
                     fit_kwargs['grad_nonlin'] = gnl_map
            
             if hasattr(dwi, 'Delta') and dwi.Delta: fit_kwargs['Delta_file'] = dwi.Delta
             if hasattr(dwi, 'delta') and dwi.delta: fit_kwargs['delta_file'] = dwi.delta
             
             fit_dki(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **fit_kwargs)
        else:
             raise ValueError(f"Unknown DKI method: {self.method}")
             
        # Track Outputs (for normalization)
        results = {}
        req_norm = [m.strip().lower() for m in requested_metrics]
        
        for p in model_out.glob("*.nii.gz"):
             name_part = p.name.replace('.nii.gz', '')
             suffix = name_part.split('_')[-1]
             
             if suffix.lower() in req_norm:
                  results[suffix] = p
             elif suffix.upper() in [m.upper() for m in requested_metrics]: # Fallback check?
                  results[suffix] = p
                  
        context.setdefault('modeling_results', {})['DKI'] = results

        return context


class NODDIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='dmipy', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    # Removed separate should_skip to ensure context is populated during skip logic in run()


    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        from ...io.bids import build_bids_name, get_entities_from_path
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        model_out = output_dir / "NODDI"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Check for existing outputs
        # Logic: Skip if output exists unless force is True
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'NODDI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        # AMICO/DMIPY use odi, vic, icvf
        odi_path = model_out / build_bids_name(ents, suffix='ODI')
        icvf_path = model_out / build_bids_name(ents, suffix='ICVF')
        
        if (odi_path.exists() or icvf_path.exists()) and not force:
             self.logger.info(f"Skipping NODDI fit for {dwi.img.name} (Found NODDI outputs)")
             # Populate Context
             results = {}
             for p in model_out.glob("*.nii.gz"):
                   name_part = p.name.replace('.nii.gz', '')
                   suffix = name_part.split('_')[-1]
                   results[suffix] = p
             context.setdefault('modeling_results', {})['NODDI'] = results
             return context
        
        self.logger.info(f"Running NODDI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        outputs = {}
        outputs = {}
        if self.method == 'dmipy':
             from ...interfaces.dmipy import fit_noddi
             
             # Extract config options
             # Priority: kwargs > config > default
             noddi_kwargs = self.kwargs.copy()
             
             # Helper to get from config or kwargs
             def get_cfg(key, default=None):
                 if key in noddi_kwargs: return noddi_kwargs.pop(key)
                 if isinstance(self.config, dict):
                     return self.config.get(key, default)
                 return getattr(self.config, key, default)

             # Explicitly extract known parameters to pass them cleanly
             # (fit_noddi now has explicit args for these)
             distribution = get_cfg('distribution', 'Watson')
             parallel_diff = get_cfg('parallel_diffusivity', 1.7e-9)
             iso_diff = get_cfg('iso_diffusivity', 3.0e-9)
             model_type = get_cfg('model_type', 'standard')
             fiso_map = get_cfg('fiso_map', None)
             
             # Pass to fit_noddi
             outputs = fit_noddi(
                 dwi, 
                 model_out, 
                 mask_file=mask_path, 
                 nthreads=self.nthreads,
                 distribution=distribution,
                 parallel_diffusivity=parallel_diff,
                 iso_diffusivity=iso_diff,
                 model_type=model_type,
                 fiso_file=fiso_map,
                 **noddi_kwargs
             )
        elif self.method == 'amico':
             from ...interfaces.amico import fit_noddi
             outputs = fit_noddi(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **self.kwargs)
        else:
             raise ValueError(f"Unknown NODDI method: {self.method}")
             
        # Rename/BIDSify outputs
        # Keys might be 'odi', 'icvf', 'fiso', 'f_intra', 'f_extra'
        
        ent_base = get_entities_from_path(dwi.img)
        if 'desc' in ent_base: del ent_base['desc']
        ent_base['model'] = 'NODDI'
        
        suffix_map = {
            'odi': 'ODI',
            'vf_intra': 'ICVF',
            'vf_extra': 'EXVF',
            'fiso': 'FISO',
            'vic': 'ICVF', # AMICO output
            'isovf': 'FISO'
        }
        
        for key, path in outputs.items():
            if key in suffix_map:
                suffix = suffix_map[key]
            else:
                suffix = key # Fallback
                
            new_name = build_bids_name({**ent_base, 'suffix': suffix})
            new_path = model_out / new_name
            
            if path.exists():
                path.rename(new_path)
                
                # Sidecar
                sidecar = {
                    "ModelName": "NODDI",
                    "FittingSoftware": "Dmipy" if self.method == 'dmipy' else "AMICO",
                    "InputData": dwi.img.name,
                    "Metric": suffix
                }
                import json
                with open(str(new_path).replace('.nii.gz', '.json'), 'w') as f:
                    json.dump(sidecar, f, indent=4)
             
        # Update Context outputs
        results = {}
        # Scan dir for renamed files (since we just renamed them)
        # Or easier: we know new_path from loop
        pass # loop above handles renaming, let's scan dir or capture in loop. 
        # Actually capturing in loop is hard since it replaces implementation.
        # Let's just scan dir. 
        for p in model_out.glob("*.nii.gz"):
             # Parse suffix from filename? or just store filename as key?
             # We want standard keys like 'ODI', 'ICVF'
             # Filename: sub-XX_model-NODDI_ODI.nii.gz
             # Extract suffix after last _ and before .nii.gz
             name_part = p.name.replace('.nii.gz', '')
             suffix = name_part.split('_')[-1] # e.g. ODI
             results[suffix] = p
             
        context.setdefault('modeling_results', {})['NODDI'] = results
        return context


class SANDIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='amico', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    # Removed separate should_skip to ensure context is populated during skip logic in run()


    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        model_out = output_dir / "sandi"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Check for existing outputs
        # Logic: Skip if output exists unless force is True
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'SANDI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        fsoma_path = model_out / build_bids_name(ents, suffix='fsoma')
        
        if fsoma_path.exists() and not force:
             self.logger.info(f"Skipping SANDI fit for {dwi.img.name} (Found SANDI outputs)")
             # Populate Context
             results = {}
             for p in model_out.glob("*.nii.gz"):
                   name_part = p.name.replace('.nii.gz', '')
                   suffix = name_part.split('_')[-1]
                   results[suffix] = p
             context.setdefault('modeling_results', {})['sandi'] = results
             return context
        
        self.logger.info(f"Running SANDI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        if self.method == 'amico':
             from ...interfaces.amico import fit_sandi
             fit_sandi(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **self.kwargs)
        else:
             raise ValueError(f"Unknown SANDI method: {self.method}")

        # Track Outputs for Normalization
        results = {}
        for p in model_out.glob("*.nii.gz"):
              name_part = p.name.replace('.nii.gz', '')
              suffix = name_part.split('_')[-1]
              results[suffix] = p
        context.setdefault('modeling_results', {})['sandi'] = results
        return context


class MAPMRIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='dipy', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        model_out = output_dir / "mapmri"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Check for existing outputs
        # Logic: Skip if output exists unless force is True
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'MAPMRI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        rtop_path = model_out / build_bids_name(ents, suffix='rtop')
        
        if rtop_path.exists() and not force:
             self.logger.info(f"Skipping MAPMRI fit for {dwi.img.name} (Found MAPMRI outputs)")
             # Populate Context
             results = {}
             for p in model_out.glob("*.nii.gz"):
                   name_part = p.name.replace('.nii.gz', '')
                   suffix = name_part.split('_')[-1]
                   results[suffix] = p
             context.setdefault('modeling_results', {})['mapmri'] = results
             return context
        
        self.logger.info(f"Running MAPMRI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        if self.method == 'dipy':
             from ...interfaces.dipy import fit_mapmri
             
             map_kwargs = self.kwargs.copy()
             
             # Robust extraction for flexible options (smoothing, constraints)
             opts = ['smoothing_fwhm', 'radial_order', 'laplacian_regularization', 
                     'positivity_constraint', 'cvxpy_solver', 'static_diffusivity']
             
             if isinstance(self.config, dict):
                  for k in opts:
                      if k in self.config and k not in map_kwargs:
                          map_kwargs[k] = self.config[k]
             
             for k in opts:
                  if k not in map_kwargs and hasattr(self.config, k):
                       map_kwargs[k] = getattr(self.config, k)
             
             # Check for GNL map
             gnl_map = context.get('gnl_map') if isinstance(context, dict) else None
             if gnl_map:
                if not gnl_map.exists():
                     self.logger.warning(f"GNL map path missing: {gnl_map}")
                else:
                     self.logger.info(f"Using Gradient Nonlinearity Tensor Map for DIPY MAPMRI: {gnl_map}")
                     map_kwargs['grad_nonlin'] = gnl_map
            
             if hasattr(dwi, 'Delta') and dwi.Delta: map_kwargs['Delta_file'] = dwi.Delta
             if hasattr(dwi, 'delta') and dwi.delta: map_kwargs['delta_file'] = dwi.delta
                     
             fit_mapmri(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **map_kwargs)
        else:
             raise ValueError(f"Unknown MAPMRI method: {self.method}")
             
        # Track Outputs for Normalization
        results = {}
        for p in model_out.glob("*.nii.gz"):
              name_part = p.name.replace('.nii.gz', '')
              suffix = name_part.split('_')[-1]
              results[suffix] = p
        context.setdefault('modeling_results', {})['mapmri'] = results
        return context


class CSDFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='msmt_csd', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method # e.g. 'msmt_csd' or 'csd' (used as algo for dwi2fod usually)
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs # e.g. lmax, response_algo ('dhollander')

    # Removed separate should_skip to ensure context is populated during skip logic in run()


    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        dwi = context if not isinstance(context, dict) else context.get('current_image')
        model_out = output_dir / "CSD"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Determine algorithm for response and fod
        fod_algo = self.method # e.g. 'msmt_csd' or 'csd'
        resp_algo = self.kwargs.get('response_algorithm', 'dhollander')
        lmax = self.kwargs.get('lmax', None)

        from ...io.bids import build_bids_name, get_entities_from_path

        # Check for existing outputs (using final BIDS names)
        # Logic: Skip if output exists unless force is True
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        # We need to construct the expected filenames to check existence
        ent_base = get_entities_from_path(dwi.img)
        if 'desc' in ent_base: del ent_base['desc']
        ent_base['model'] = 'CSD'
        
        # Approximate check: check if likely outputs exist
        check_suffixes = ['wmFOD', 'FOD'] 
        existing_found = False
        
        for s in check_suffixes:
             p = model_out / build_bids_name({**ent_base, 'suffix': s})
             if p.exists():
                 existing_found = True
                 break
        
        if existing_found and not force:
             self.logger.info(f"Skipping CSD fit for {dwi.img.name} (Found CSD outputs)")
             # Populate Context with existing results
             results = {}
             for p in model_out.glob("*FOD.nii.gz"):
                  name_part = p.name.replace('.nii.gz', '')
                  suffix = name_part.split('_')[-1]
                  results[suffix] = p
             context.setdefault('modeling_results', {})['CSD'] = results
             return context
        
        self.logger.info(f"Running CSD fit ({fod_algo}) on {dwi.img.name}")
        
        # IMPORTS
        from ...interfaces.mrtrix import dwi2response, dwi2fod
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask
             
        # 1. Estimate Response Function
        # Output directory for response functions
        resp_out = model_out / "response"
        resp_out.mkdir(exist_ok=True)
        
        self.logger.info(f"Estimating response function (algo={resp_algo})...")
        responses = dwi2response(
            dwi, 
            resp_out, 
            in_bvec=dwi.bvec, 
            in_bval=dwi.bval, 
            mask_file=mask_path, 
            algorithm=resp_algo, 
            nthreads=self.nthreads
        )
        
        # 2. Estimate FOD
        self.logger.info(f"Estimating FODs (algo={fod_algo})...")
        fods = dwi2fod(
            dwi,
            responses,
            model_out,
            in_bvec=dwi.bvec,
            in_bval=dwi.bval,
            mask_file=mask_path,
            algorithm=fod_algo,
            lmax=lmax,
            nthreads=self.nthreads
        )
        
        # 3. Rename/BIDSify outputs
        for key, path in fods.items():
            # key is 'wm', 'gm', 'csf', 'fod'
            # Suffix: wmFOD, gmFOD, csfFOD, FOD
            suffix = key + "FOD" 
            if key == 'fod': suffix = 'FOD'
            
            new_name = build_bids_name({**ent_base, 'suffix': suffix})
            new_path = model_out / new_name
            
            if path.exists():
                path.rename(new_path)
                
                # Sidecar
                sidecar = {
                    "ModelName": "Constrained Spherical Deconvolution",
                    "FittingSoftware": "MRtrix3",
                    "InputData": dwi.img.name,
                    "Algorithm": fod_algo,
                    "ResponseAlgorithm": resp_algo
                }
                import json
                with open(str(new_path).replace('.nii.gz', '.json'), 'w') as f:
                    json.dump(sidecar, f, indent=4)
                    
        # Track Outputs
        results = {}
        for p in model_out.glob("*FOD.nii.gz"):
             # e.g. sub-01_wmFOD.nii.gz -> wmFOD
             name_part = p.name.replace('.nii.gz', '')
             suffix = name_part.split('_')[-1]
             results[suffix] = p
             
        context.setdefault('modeling_results', {})['CSD'] = results
        return context


class FWDTIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='dipy', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
             self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
             self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def should_skip(self, context, output_dir):
        dwi = context.get('current_image') if isinstance(context, dict) else context
        if not dwi: return False
        
        force = self.config.get('force', False) or self.config.get('dmri', {}).get('force_run', False)
        if force: return False
        
        model_out = output_dir / "FWE_DTI"
        ents = dwi.entities.copy()
        ents['model'] = 'FWDTI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        fa_path = model_out / build_bids_name(ents, suffix='FA')
        return fa_path.exists()

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        # Resolve inputs
        dwi = context.get('current_image') if isinstance(context, dict) else context
        
        # Output directory for this model
        model_out = output_dir / "FWE_DTI"
        model_out.mkdir(parents=True, exist_ok=True)
        
        # Check for existing outputs
        # Logic: Skip if output exists unless force is True
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'FWDTI'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        # Check for key output (FA)
        fa_path = model_out / build_bids_name(ents, suffix='FA')
        
        if fa_path.exists() and not force:
             self.logger.info(f"Skipping FWE-DTI fit for {dwi.img.name} (Found existing outputs)")
             # Collect existing
             existing_results = {}
             for p in model_out.glob("*_FA.nii.gz"): existing_results['FA'] = p
             for p in model_out.glob("*_FW.nii.gz"): existing_results['FW'] = p
             for p in model_out.glob("*_MD.nii.gz"): existing_results['MD'] = p
             
             context.setdefault('modeling_results', {})['FWE_DTI'] = existing_results
             return context

        self.logger.info(f"Running FWE-DTI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        if self.method == 'dipy':
            from ...interfaces.dipy import fit_fwe_dti
            
            # Extract config options
            step_kwargs = self.kwargs.copy()
            
            # Check GNL
            gnl_map = context.get('gnl_map') if isinstance(context, dict) else None
            if gnl_map:
                if not gnl_map.exists():
                     self.logger.warning(f"GNL map path missing: {gnl_map}")
                else:
                     self.logger.info(f"Using Gradient Nonlinearity Tensor Map for DIPY FWE-DTI: {gnl_map}")
                     step_kwargs['grad_nonlin'] = gnl_map
            
            fit_fwe_dti(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **step_kwargs)
            
        else:
            raise ValueError(f"Unknown FWE-DTI method: {self.method} (only 'dipy' supported)")
            
        # Track Outputs
        results = {}
        for p in model_out.glob("*_FA.nii.gz"): results['FA'] = p
        for p in model_out.glob("*_FW.nii.gz"): results['FW'] = p
        for p in model_out.glob("*_MD.nii.gz"): results['MD'] = p
        for p in model_out.glob("*_AD.nii.gz"): results['AD'] = p
        for p in model_out.glob("*_RD.nii.gz"): results['RD'] = p
        
        context.setdefault('modeling_results', {})['FWE_DTI'] = results
        return context
