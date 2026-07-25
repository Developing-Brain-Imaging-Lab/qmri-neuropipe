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
import json
import logging

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...core.types import DWIFile
from ...interfaces import dipy, fsl, mrtrix, amico, dmipy
from ...io.bids import build_bids_name


def _resolve_context_gnl_map(context: dict | object, dwi: object | None = None) -> Optional[Path]:
    """Resolve the best GNL tensor for the current DWI, preferring image-specific mappings."""
    if not isinstance(context, dict):
        return None

    candidates: list[object] = []
    gnl_map_by_image = context.get("gnl_map_by_image", {}) or {}

    for image_obj in (dwi, context.get("current_image")):
        image_path = getattr(image_obj, "img", None)
        if image_path is None:
            continue
        candidates.append(gnl_map_by_image.get(image_path))
        candidates.append(gnl_map_by_image.get(str(image_path)))

        image_path = Path(image_path)
        image_entities = (getattr(image_obj, "entities", {}) or {}).copy()
        if image_entities:
            image_entities["desc"] = "gnl_tensor"
            image_entities["suffix"] = "dwi"
            candidates.append(image_path.parent / build_bids_name(image_entities))

        sibling_candidates = sorted(image_path.parent.glob("*desc-gnl_tensor*_dwi.nii.gz"))
        if len(sibling_candidates) == 1:
            candidates.append(sibling_candidates[0])

    candidates.append(context.get("gnl_map"))

    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

    return None


def _warn_unsupported_gnl(logger, context: dict | object, dwi: object | None, model_name: str) -> None:
    """Warn when a GNL tensor is available but the selected model/backend cannot use it."""
    gnl_map = _resolve_context_gnl_map(context, dwi)
    if gnl_map:
        logger.warning(
            f"Gradient nonlinearity tensor map found for {model_name}, "
            "but this model/backend does not currently support GNL-aware fitting. "
            "The GNL tensor will be ignored."
        )


_MAPMRI_METRIC_ALIASES = {
    "rtop": "rtop",
    "rtap": "rtap",
    "rtpp": "rtpp",
    "qiv": "qiv",
    "msd": "msd",
    "ng": "ng",
    "peak": "peaks",
    "peaks": "peaks",
    "ng_par": "ng_par",
    "ng_parallel": "ng_par",
    "parng": "ng_par",
    "ng_perp": "ng_perp",
    "ng_perpendicular": "ng_perp",
    "perng": "ng_perp",
}


def _normalize_mapmri_metrics(metrics: list[str] | None) -> list[str]:
    normalized = []
    seen = set()
    for metric in metrics or []:
        canonical = _MAPMRI_METRIC_ALIASES.get(str(metric).strip().lower())
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def _mapmri_metric_suffix(metric: str) -> str:
    canonical = _MAPMRI_METRIC_ALIASES.get(str(metric).strip().lower(), str(metric).strip().lower())
    if canonical == "peaks":
        return "PEAKS"
    if canonical == "ng_par":
        return "NG_PAR"
    if canonical == "ng_perp":
        return "NG_PERP"
    return canonical.upper()


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

    def _resolve_requested_metrics(self, override_metrics: Optional[list[str]] = None) -> list[str]:
        requested_metrics = override_metrics
        if not requested_metrics:
            if isinstance(self.config, dict):
                requested_metrics = self.config.get('metrics', [])
            elif hasattr(self.config, 'metrics'):
                requested_metrics = self.config.metrics
            else:
                requested_metrics = []

        if not requested_metrics:
            requested_metrics = ["fa", "md", "ad", "rd", "color_fa"]

        resolved = []
        seen = set()
        for metric in requested_metrics:
            norm = str(metric).strip().lower()
            if norm and norm not in seen:
                resolved.append(norm)
                seen.add(norm)

        if "tensor" not in seen:
            resolved.append("tensor")
            seen.add("tensor")

        if self.method in {"dipy", "mrtrix"} and "tensor_mrtrix" not in seen:
            resolved.append("tensor_mrtrix")

        return resolved

    @staticmethod
    def _metric_to_suffix(metric: str) -> str:
        mapping = {
            'color_fa': 'DECFA',
            'tensor': 'tensor',
            'tensor_fsl': 'tensorFSL',
            'tensor_mrtrix': 'tensorMRTRIX',
        }
        return mapping.get(metric, metric.upper())


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
        requested_metrics = self._resolve_requested_metrics(kwargs.get('metrics'))

        # Check for key output (FA)
        fa_path = model_out / build_bids_name(ents, suffix='FA')
        
        # Smart Skip Logic
        should_run = True
        existing_results = {}
        
        if self.check_output_validity(fa_path) and not force:
             # Check if ALL requested metrics exist and are valid (non-empty)
             missing_metrics = []
             all_found = True
             
             for m in requested_metrics:
                 suffix = self._metric_to_suffix(m)
                 
                 fpath = model_out / build_bids_name(ents, suffix=suffix)
                 
                 if self.check_output_validity(fpath):
                     existing_results[suffix] = fpath
                 else:
                     # dynamic glob fallback check but must be valid
                     found = [p for p in model_out.glob(f"*_{suffix}.nii.gz") if self.check_output_validity(p)]
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
        gnl_map = _resolve_context_gnl_map(context, dwi)
        if not gnl_map and isinstance(context, dict):
            stale_gnl = context.get('gnl_map')
            if stale_gnl:
                self.logger.warning(f"GNL map path found in context but file missing: {stale_gnl}")

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
            dipy_kwargs['metrics'] = self._resolve_requested_metrics(dipy_kwargs['metrics'])
            
            if gnl_map:
                self.logger.info(f"Using Gradient Nonlinearity Tensor Map for DIPY: {gnl_map}")
                dipy_kwargs['grad_nonlin'] = gnl_map
            
            if hasattr(dwi, 'Delta') and dwi.Delta: dipy_kwargs['Delta_file'] = dwi.Delta
            if hasattr(dwi, 'delta') and dwi.delta: dipy_kwargs['delta_file'] = dwi.delta
                
            # Standardized threading resolution
            nthreads = kwargs.get('nthreads') or getattr(self, 'nthreads', None) or self.config.get('n_cpus', 1)
            fit_dti(dwi, model_out, mask_file=mask_path, nthreads=nthreads, **dipy_kwargs)
            
        elif self.method == 'fsl':
            from ...interfaces.fsl import fit_dti
            # fsl parameters
            fsl_kwargs = {}
            if 'save_tensor' in self.kwargs:
                fsl_kwargs['save_tensor'] = self.kwargs['save_tensor']
            fsl_kwargs['metrics'] = requested_metrics
                
            if gnl_map:
                self.logger.info(f"Using Gradient Nonlinearity Tensor Map: {gnl_map}")
                fsl_kwargs['grad_nonlin'] = gnl_map
                
            fit_dti(dwi, model_out, mask_file=mask_path, **fsl_kwargs)
            
        elif self.method == 'mrtrix':
            from ...interfaces.mrtrix import fit_dti
            # mrtrix parameters
            mrtrix_kwargs = {}
            mrtrix_kwargs['metrics'] = self._resolve_requested_metrics(self.kwargs.get('metrics'))
            
            if gnl_map:
                self.logger.warning("Gradient nonlinearity tensor map found but not currently supported by MRtrix backend (in this pipeline). GNL correction will be ignored.")
                
            # Standardized threading resolution
            nthreads = kwargs.get('nthreads') or getattr(self, 'nthreads', None) or self.config.get('n_cpus', 1)
            fit_dti(dwi, model_out, mask_file=mask_path, nthreads=nthreads, **mrtrix_kwargs)
        else:
            raise ValueError(f"Unknown DTI method: {self.method}")
            
        # Track Outputs for Normalization
        # Collect paths matching pattern in model_out
        # Track Outputs for Normalization
        # Collect paths matching pattern in model_out, FILTERED by requested_metrics
        results = {}
        req_norm = set(requested_metrics)
        
        for p in model_out.glob("*.nii.gz"):
             # Assuming BIDS: sub-XX_desc-XX_model-DTI_SUFFIX.nii.gz
             name_part = p.name.replace('.nii.gz', '')
             if '_' in name_part:
                 suffix = name_part.split('_')[-1]
                 
                 check_suffix = suffix.lower()
                 if check_suffix == 'decfa':
                     check_suffix = 'color_fa'
                 elif check_suffix == 'tensorfsl':
                     check_suffix = 'tensor_fsl'
                 elif check_suffix == 'tensormrtrix':
                     check_suffix = 'tensor_mrtrix'
                 
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
        
        if (self.check_output_validity(mk_path) or self.check_output_validity(model_out / build_bids_name(ents, suffix='FA'))) and not force:
             # Check if ALL requested metrics exist and are valid
             missing_metrics = []
             all_found = True
             
             for m in requested_metrics:
                 suffix = m.upper()
                 # Predict filename
                 # Handle special naming if needed (e.g. color_fa -> DECFA)
                 # But for DKI standard metrics, suffix usually matches metric name
                 fpath = model_out / build_bids_name(ents, suffix=suffix)
                 
                 if self.check_output_validity(fpath):
                     existing_results[suffix] = fpath
                 else:
                     # dynamic glob fallback but must be valid
                     found = [p for p in model_out.glob(f"*_{suffix}.nii.gz") if self.check_output_validity(p)]
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
             gnl_map = _resolve_context_gnl_map(context, dwi)
             if gnl_map:
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
        
        # Required AMICO/DMIPY metrics for a successful NODDI fit
        required_metrics = ['ODI', 'ICVF', 'FISO']
        
        should_run = True
        existing_results = {}
        all_found = True
        
        for m in required_metrics:
             fpath = model_out / build_bids_name(ents, suffix=m)
             if self.check_output_validity(fpath):
                 existing_results[m] = fpath
             else:
                 # Check alternative naming (ISOVF vs FISO)
                 if m == 'FISO':
                      alt_path = model_out / build_bids_name(ents, suffix='ISOVF')
                      if self.check_output_validity(alt_path):
                           existing_results[m] = alt_path
                           continue
                 all_found = False
                 break
                 
        if all_found and not force:
             self.logger.info(f"Skipping NODDI fit for {dwi.img.name} (Found all required NODDI outputs)")
             should_run = False
             context.setdefault('modeling_results', {})['NODDI'] = existing_results
             
        if not should_run:
             return context
        
        self.logger.info(f"Running NODDI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        outputs = {}
        outputs = {}
        dmipy_runtime_metadata = {}
        gnl_map = _resolve_context_gnl_map(context, dwi)
        if self.method == 'dmipy':
             from ...interfaces.dmipy import fit_noddi
             from ...interfaces.dmipy_backend import DmipyRuntime
             
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
             solver_name = noddi_kwargs.get('solver', 'brute2fine')
             device_name = noddi_kwargs.get('device', 'auto')
             gpu_device = noddi_kwargs.get('gpu_device')
             dmipy_runtime_metadata = DmipyRuntime.resolve(
                 solver=solver_name,
                 device=device_name,
                 gpu_device=gpu_device,
                 jax_cache_dir=noddi_kwargs.get('jax_cache_dir'),
                 jax_log_compiles=noddi_kwargs.get('jax_log_compiles', False),
             ).provenance()
             
             # Pass to fit_noddi
             outputs = fit_noddi(
                 dwi, 
                 model_out, 
                 mask_file=mask_path, 
                 nthreads=self.nthreads,
                 grad_nonlin=gnl_map,
                 distribution=distribution,
                 parallel_diffusivity=parallel_diff,
                 iso_diffusivity=iso_diff,
                 model_type=model_type,
                 fiso_file=fiso_map,
                 **noddi_kwargs
             )
        elif self.method == 'amico':
             _warn_unsupported_gnl(self.logger, context, dwi, "NODDI")
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
                if path != new_path:
                    path.rename(new_path)
                
                # Sidecar
                sidecar_path = Path(str(new_path).replace('.nii.gz', '.json'))
                existing_sidecar = {}
                if sidecar_path.exists():
                    try:
                        with sidecar_path.open() as f:
                            existing_sidecar = json.load(f)
                    except (OSError, ValueError):
                        existing_sidecar = {}
                sidecar = {
                    "ModelName": "NODDI",
                    "FittingSoftware": "Dmipy" if self.method == 'dmipy' else "AMICO",
                    "InputData": dwi.img.name,
                    "Metric": suffix,
                    **existing_sidecar,
                    **dmipy_runtime_metadata,
                }
                with sidecar_path.open('w') as f:
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
        
        # Required AMICO SANDI metrics
        required_metrics = ['fsoma', 'fneurite', 'fextra']
        
        should_run = True
        existing_results = {}
        all_found = True
        
        for m in required_metrics:
             fpath = model_out / build_bids_name(ents, suffix=m)
             if self.check_output_validity(fpath):
                 existing_results[m] = fpath
             else:
                 all_found = False
                 break
                 
        if all_found and not force:
             self.logger.info(f"Skipping SANDI fit for {dwi.img.name} (Found all required SANDI outputs)")
             should_run = False
             context.setdefault('modeling_results', {})['sandi'] = existing_results
             
        if not should_run:
             return context
        
        self.logger.info(f"Running SANDI fit ({self.method}) on {dwi.img.name}")
        
        # Prepare mask
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        gnl_map = _resolve_context_gnl_map(context, dwi)
        if self.method == 'dmipy':
             from ...interfaces.dmipy import fit_sandi
             fit_kwargs = self.kwargs.copy()
             d_soma = kwargs.get('soma_diffusivity') or fit_kwargs.pop(
                 'soma_diffusivity', None
             )
             if d_soma is None:
                 # Accept the old ambiguous name in existing configurations.
                 d_soma = fit_kwargs.pop('iso_diffusivity', None)
             fit_kwargs.pop('parallel_diffusivity', None)
             if d_soma is None:
                 d_soma = self.config.get(
                     'soma_diffusivity', self.config.get('iso_diffusivity', 3.0e-9)
                 )
             fit_sandi(
                 dwi,
                 model_out,
                 mask_file=mask_path,
                 nthreads=self.nthreads,
                 grad_nonlin=gnl_map,
                 soma_diffusivity=d_soma,
                 **fit_kwargs,
             )
        elif self.method == 'amico':
             _warn_unsupported_gnl(self.logger, context, dwi, "SANDI")
             from ...interfaces.amico import fit_sandi
             fit_kwargs = self.kwargs.copy()
             if hasattr(dwi, 'Delta') and dwi.Delta:
                 fit_kwargs['Delta_file'] = dwi.Delta
             if hasattr(dwi, 'delta') and dwi.delta:
                 fit_kwargs['delta_file'] = dwi.delta
             fit_sandi(dwi, model_out, mask_file=mask_path, nthreads=self.nthreads, **fit_kwargs)
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


class MicrogliaFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='dmipy', nthreads=1, **kwargs):
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
        model_out = output_dir / "microglia"
        model_out.mkdir(parents=True, exist_ok=True)
        
        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)
        
        ents = dwi.entities.copy()
        ents['model'] = 'Microglia'
        if 'desc' in ents: del ents['desc']
        if 'suffix' in ents: del ents['suffix']
        
        from ...io.bids import build_bids_name
        test_path = model_out / build_bids_name(ents, suffix='f_small_sphere')
        
        if self.check_output_validity(test_path) and not force:
             self.logger.info(f"Skipping Microglia fit for {dwi.img.name} (Found existing outputs)")
             results = {}
             for p in [x for x in model_out.glob("*.nii.gz") if self.check_output_validity(x)]:
                   name_part = p.name.replace('.nii.gz', '')
                   suffix = name_part.split('_', 2)[-1] if '_model-Microglia_' in name_part else name_part.split('_')[-1]
                   results[suffix] = p
             context.setdefault('modeling_results', {})['microglia'] = results
             return context
        
        self.logger.info(f"Running Microglia fit ({self.method}) on {dwi.img.name}")
        
        if mask and hasattr(mask, 'img'):
             mask_path = mask.img
        else:
             mask_path = mask

        gnl_map = _resolve_context_gnl_map(context, dwi)
        if self.method == 'dmipy':
             from ...interfaces.dmipy_microglia import fit_microglia
             
             # Extract config from kwargs or self.config
             fit_kwargs = self.kwargs.copy()
             
             # Extract metrics and hyperparams
             configured_d_par = fit_kwargs.pop(
                 'parallel_diffusivity', self.config.get('parallel_diffusivity', 1.0e-9)
             )
             configured_d_iso = fit_kwargs.pop(
                 'iso_diffusivity', self.config.get('iso_diffusivity', 3.0e-9)
             )
             configured_d_small = fit_kwargs.pop(
                 'small_diameter', self.config.get('small_diameter', 8e-6)
             )
             configured_d_large = fit_kwargs.pop(
                 'large_diameter', self.config.get('large_diameter', 16e-6)
             )
             small_bounds = fit_kwargs.pop('small_diameter_bounds', (5e-6, 11e-6))
             large_bounds = fit_kwargs.pop('large_diameter_bounds', (12e-6, 18e-6))
             d_par = kwargs.get('parallel_diffusivity') or configured_d_par
             d_iso = kwargs.get('iso_diffusivity') or configured_d_iso
             d_small = kwargs.get('small_diameter') or configured_d_small
             d_large = kwargs.get('large_diameter') or configured_d_large
             
             fit_microglia(
                 dwi, 
                 model_out, 
                 mask_file=mask_path, 
                 nthreads=self.nthreads, 
                 grad_nonlin=gnl_map,
                 parallel_diffusivity=d_par,
                 iso_diffusivity=d_iso,
                 small_diameter=d_small,
                 large_diameter=d_large,
                 small_diameter_bounds=small_bounds,
                 large_diameter_bounds=large_bounds,
                 **fit_kwargs
             )
        else:
             raise ValueError(f"Unknown Microglia method: {self.method}")

        results = {}
        for p in model_out.glob("*.nii.gz"):
              name_part = p.name.replace('.nii.gz', '')
              # Extract suffix robustly
              if '_model-Microglia_' in name_part:
                  suffix = name_part.split('_model-Microglia_')[-1]
              else:
                  suffix = name_part.split('_')[-1]
              results[suffix] = p
        context.setdefault('modeling_results', {})['microglia'] = results
        return context


class NEXIFittingStep(BaseProcessingStep):
    def __init__(self, config, logger, provenance, method='nexi', nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.method = method
        self.nthreads = nthreads
        if hasattr(self.config, 'n_cpus'):
            self.nthreads = self.config.n_cpus
        elif isinstance(self.config, dict):
            self.nthreads = self.config.get('n_cpus', nthreads)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, mask=None, **kwargs) -> dict | object:
        from ...io.bids import build_bids_name, get_entities_from_path

        dwi = context if not isinstance(context, dict) else context.get('current_image')
        model_out = output_dir / "NEXI"
        model_out.mkdir(parents=True, exist_ok=True)

        force = kwargs.get('force', False) or self.config.get('force', False) or self.config.get('force_run', False)

        ents = dwi.entities.copy()
        ents['model'] = 'NEXI'
        if 'desc' in ents:
            del ents['desc']
        if 'suffix' in ents:
            del ents['suffix']

        requested_metrics = kwargs.get('metrics')
        if not requested_metrics:
            requested_metrics = self.kwargs.get('metrics')
        if not requested_metrics:
            if isinstance(self.config, dict):
                requested_metrics = self.config.get('metrics', [])
            elif hasattr(self.config, 'metrics'):
                requested_metrics = getattr(self.config, 'metrics')

        if not requested_metrics:
            requested_metrics = ["t_ex", "di", "de", "f", "sigma"]

        metric_key_map = {
            "t_ex": "t_ex",
            "tex": "t_ex",
            "t-ex": "t_ex",
            "di": "di",
            "de": "de",
            "f": "f",
            "sigma": "sigma",
        }

        suffix_map = {
            "t_ex": "TEX",
            "di": "DI",
            "de": "DE",
            "f": "F",
            "sigma": "SIGMA",
        }

        resolved_metrics = []
        for metric in requested_metrics:
            key = metric_key_map.get(str(metric).strip().lower())
            if key:
                resolved_metrics.append(key)

        if not resolved_metrics:
            resolved_metrics = ["t_ex", "di", "de", "f", "sigma"]

        should_run = True
        existing_results = {}
        if not force:
            missing = []
            for key in resolved_metrics:
                suffix = suffix_map[key]
                fpath = model_out / build_bids_name(ents, suffix=suffix)
                if self.check_output_validity(fpath):
                    existing_results[suffix] = fpath
                else:
                    found = [p for p in model_out.glob(f"*_{suffix}.nii.gz") if self.check_output_validity(p)]
                    if found:
                        existing_results[suffix] = found[0]
                    else:
                        missing.append(key)
            if not missing:
                should_run = False
                self.logger.info(
                    f"Skipping NEXI fit for {dwi.img.name} (Found all requested valid metrics)."
                )

        if not should_run:
            context.setdefault('modeling_results', {})['NEXI'] = existing_results
            return context

        self.logger.info(f"Running NEXI fit ({self.method}) on {dwi.img.name}")

        if mask and hasattr(mask, 'img'):
            mask_path = mask.img
        else:
            mask_path = mask

        _warn_unsupported_gnl(self.logger, context, dwi, "NEXI")

        if self.method != 'nexi':
            raise ValueError(f"Unknown NEXI method: {self.method}")

        step_kwargs = self.kwargs.copy()
        td_file = step_kwargs.pop('td_file', None)
        td_file = td_file or step_kwargs.pop('td_path', None)
        lowb_noisemap_file = step_kwargs.pop('lowb_noisemap', None)
        lowb_noisemap_file = lowb_noisemap_file or step_kwargs.pop('lowb_noisemap_file', None)
        step_kwargs.pop('metrics', None)
        if isinstance(self.config, dict):
            td_file = td_file or self.config.get('td_file')
            lowb_noisemap_file = lowb_noisemap_file or self.config.get('lowb_noisemap')
        else:
            td_file = td_file or getattr(self.config, 'td_file', None)
            lowb_noisemap_file = lowb_noisemap_file or getattr(self.config, 'lowb_noisemap', None)

        from ...interfaces.nexi import fit_nexi

        outputs = fit_nexi(
            dwi,
            model_out,
            td_file=td_file,
            lowb_noisemap_file=lowb_noisemap_file,
            mask_file=mask_path,
            **step_kwargs,
        )

        ent_base = get_entities_from_path(dwi.img)
        if 'desc' in ent_base:
            del ent_base['desc']
        ent_base['model'] = 'NEXI'

        results = {}
        for key, path in outputs.items():
            key_lower = key.lower()
            if key_lower not in suffix_map:
                continue
            suffix = suffix_map[key_lower]
            new_name = build_bids_name({**ent_base, 'suffix': suffix})
            new_path = model_out / new_name
            if path.exists():
                path.rename(new_path)
                sidecar = {
                    "ModelName": "NEXI_Rice_Mean",
                    "FittingSoftware": "nexi",
                    "InputData": dwi.img.name,
                    "Metric": suffix,
                    "DiffusionTimeFile": str(td_file) if td_file else None,
                    "LowBNoiseMap": str(lowb_noisemap_file) if lowb_noisemap_file else None,
                }
                import json

                with open(str(new_path).replace('.nii.gz', '.json'), 'w') as f:
                    json.dump(sidecar, f, indent=4)
                results[suffix] = new_path

        context.setdefault('modeling_results', {})['NEXI'] = results
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
        
        requested_metrics = kwargs.get('metrics')
        if not requested_metrics:
            requested_metrics = self.kwargs.get('metrics')
        if not requested_metrics:
            if isinstance(self.config, dict):
                requested_metrics = self.config.get('metrics')
            elif hasattr(self.config, 'metrics'):
                requested_metrics = getattr(self.config, 'metrics')
        requested_metrics = _normalize_mapmri_metrics(requested_metrics)
        if not requested_metrics:
            requested_metrics = ["rtop", "rtap", "rtpp", "qiv", "msd", "ng"]

        should_run = True
        existing_results = {}
        all_found = True
        
        for m in requested_metrics:
             suffix = _mapmri_metric_suffix(m)
             fpath = model_out / build_bids_name(ents, suffix=suffix)
             if self.check_output_validity(fpath):
                 existing_results[suffix] = fpath
             else:
                 found = [p for p in model_out.glob(f"*_{suffix}.nii.gz") if self.check_output_validity(p)]
                 if found:
                     existing_results[suffix] = found[0]
                 else:
                     all_found = False
                 
        if all_found and not force:
             self.logger.info(f"Skipping MAPMRI fit for {dwi.img.name} (Found all required MAPMRI outputs)")
             should_run = False
             context.setdefault('modeling_results', {})['mapmri'] = existing_results
             
        if not should_run:
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
                     'positivity_constraint', 'cvxpy_solver', 'static_diffusivity', 'metrics']
             
             if isinstance(self.config, dict):
                  for k in opts:
                      if k in self.config and k not in map_kwargs:
                          map_kwargs[k] = self.config[k]
             
             for k in opts:
                  if k not in map_kwargs and hasattr(self.config, k):
                       map_kwargs[k] = getattr(self.config, k)

             map_kwargs['metrics'] = requested_metrics
             
             # Check for GNL map
             gnl_map = _resolve_context_gnl_map(context, dwi)
             if gnl_map:
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
        
        # Approximate check: check if likely outputs exist and are valid
        check_suffixes = ['wmFOD', 'FOD'] 
        existing_found = False
        
        for s in check_suffixes:
             p = model_out / build_bids_name({**ent_base, 'suffix': s})
             if self.check_output_validity(p):
                 existing_found = True
                 break
        
        if existing_found and not force:
             self.logger.info(f"Skipping CSD fit for {dwi.img.name} (Found valid CSD outputs)")
             # Populate Context with existing results
             results = {}
             for p in [x for x in model_out.glob("*FOD.nii.gz") if self.check_output_validity(x)]:
                  name_part = p.name.replace('.nii.gz', '')
                  suffix = name_part.split('_')[-1]
                  results[suffix] = p
             context.setdefault('modeling_results', {})['CSD'] = results
             return context
        
        self.logger.info(f"Running CSD fit ({fod_algo}) on {dwi.img.name}")
        
        # IMPORTS
        from ...interfaces.mrtrix import dwi2response, dwi2fod
        _warn_unsupported_gnl(self.logger, context, dwi, "CSD")
        
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
        return self.check_output_validity(fa_path)

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
        f_path = model_out / build_bids_name(ents, suffix='F')
        
        if self.check_output_validity(fa_path) and self.check_output_validity(f_path) and not force:
             self.logger.info(f"Skipping FWE-DTI fit for {dwi.img.name} (Found valid existing outputs)")
             # Collect existing
             existing_results = {}
             for p in [x for x in model_out.glob("*_FA.nii.gz") if self.check_output_validity(x)]: existing_results['FA'] = p
             for p in [x for x in model_out.glob("*_F.nii.gz") if self.check_output_validity(x)]: existing_results['F'] = p
             for p in [x for x in model_out.glob("*_MD.nii.gz") if self.check_output_validity(x)]: existing_results['MD'] = p
             
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
            gnl_map = _resolve_context_gnl_map(context, dwi)
            if gnl_map:
                self.logger.info(f"Using Gradient Nonlinearity Tensor Map for DIPY FWE-DTI: {gnl_map}")
                step_kwargs['grad_nonlin'] = gnl_map
            
            # Standardized threading resolution
            nthreads = kwargs.get('nthreads') or getattr(self, 'nthreads', None) or self.config.get('n_cpus', 1)
            fit_fwe_dti(dwi, model_out, mask_file=mask_path, nthreads=nthreads, **step_kwargs)
            
        else:
            raise ValueError(f"Unknown FWE-DTI method: {self.method} (only 'dipy' supported)")
            
        # Track Outputs
        results = {}
        for p in model_out.glob("*_FA.nii.gz"): results['FA'] = p
        for p in model_out.glob("*_F.nii.gz"): results['F'] = p
        for p in model_out.glob("*_MD.nii.gz"): results['MD'] = p
        for p in model_out.glob("*_AD.nii.gz"): results['AD'] = p
        for p in model_out.glob("*_RD.nii.gz"): results['RD'] = p
        
        context.setdefault('modeling_results', {})['FWE_DTI'] = results
        return context
