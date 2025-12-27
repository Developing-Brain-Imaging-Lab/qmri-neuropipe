from pathlib import Path
import logging
from ...core import BaseProcessingStep, ProcessingError
from ...core.utils import ensure_dir

class NormalizationStep(BaseProcessingStep):
    """
    Step to register diffusion metrics to a standard space template.
    Usually calculates transform based on a driving metric (e.g. FA) 
    and applies it to all other available metrics.
    """
    def _ensure_3d(self, img, is_driving=False):
        """
        Ensure ANTs image is 3D.
        If 4D with 1 volume, slice it.
        If 4D with >1 volume:
           - if is_driving: Error (Registration requires 3D scalar/volume)
           - else: Return as is, flag as 4D.
        Returns: (img, is_4d_series)
        """
        import ants
        if img.dimension == 4:
            if img.shape[3] == 1:
                # Squeeze using slice
                # ants.slice_image(image, axis, idx)
                # axis 3 is time
                return ants.slice_image(img, axis=3, idx=0), False
            else:
                if is_driving:
                    raise ProcessingError(f"Driving metric is 4D ({img.shape}). Registration requires a 3D volume.")
                return img, True
        return img, False

    def __init__(self, config, logger, provenance, template=None, driving_metric='FA', space_name=None, tool='ants', **kwargs):
        super().__init__(config, logger, provenance)
        self.template = Path(template) if template else None
        self.driving_metric = driving_metric
        self.space_name = space_name or "Standard"
        self.tool = tool.lower()
        self.save_transforms = kwargs.get('save_transforms', True)
        self.kwargs = kwargs

    def run(self, context: dict | object, output_dir: Path, **kwargs) -> dict | object:
        # Context is likely the main dictionary from ModelingWorkflow
        if not isinstance(context, dict):
            # If passed an object, we can't really do normalization unless it's a specific file?
            # Normalization expects a set of maps.
            self.logger.warning("NormalizationStep expects a context dictionary with 'modeling_results'.")
            return context

        if not self.template or not self.template.exists():
            self.logger.warning(f"Normalization skipped: Template not found at {self.template}")
            return context

        modeling_results = context.get('modeling_results', {})
        if not modeling_results:
            self.logger.warning("Normalization skipped: No modeling results found.")
            return context

        # 1. Find driving metric
        ref_path = None
        
        # Flatten results to find driving metric
        # modeling_results = { 'DTI': {'FA': path, ...}, 'NODDI': {...} }
        # Flatten for driving metric search, but keep structure for final application
        flat_metrics = {}
        for model in modeling_results.values():
             flat_metrics.update(model)
        
        # Look for partial match if exact match fails? e.g. 'FA' vs 'tensor_fa'
        # Or require exact match.
        if self.driving_metric in flat_metrics:
            ref_path = flat_metrics[self.driving_metric]
        else:
            # Try caseless?
            for k, v in flat_metrics.items():
                if k.lower() == self.driving_metric.lower():
                    ref_path = v
                    break
        
        if not ref_path:
             self.logger.warning(f"Normalization skipped: Driving metric '{self.driving_metric}' not found in results.")
             return context

        # Output dir
        norm_out = output_dir.parent / "normalization" / f"space-{self.space_name}"
        if output_dir.name == 'modeling': # heuristic fix if nested
             norm_out = output_dir.parent / "normalization"
             
        norm_out.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Normalizing metrics to space-{self.space_name} (Template: {self.template.name}) using {self.tool}")
        self.logger.info(f"Driving metric: {ref_path.name}")
        
        # Check for existing outputs (Skip Logic)
        skip = self.config.get('skip_existing', False)
        # Predict outputs
        from ...io.bids import build_bids_name, get_entities_from_path
        
        # We need to predict filenames.
        # This is slightly expensive, but safer.
        all_exist = True
        predicted_outputs = []
        
        for model_name, metrics in modeling_results.items():
            for name, path in metrics.items():
                ents = get_entities_from_path(path)
                ents['space'] = self.space_name
                if 'model' not in ents or not ents['model']: ents['model'] = model_name
                out_path = norm_out / build_bids_name(ents)
                predicted_outputs.append(out_path)
                if not out_path.exists():
                    all_exist = False
                    
        # Also check transforms if saving
        if self.save_transforms:
             # Driving metric entities
             d_ents = get_entities_from_path(ref_path)
             # Affine
             d_ents.update({'space': self.space_name, 'suffix': 'xfm', 'desc': 'affine', 'extension': '.mat', 'model': None}) # .mat usually for ANTs affine
             # Simplified name: sub-X_ses-Y_space-MNI_desc-affine_xfm.mat
             # Remove other entities?
             for k in ['acq', 'dir', 'run', 'echo', 'part']: 
                 if k in d_ents: del d_ents[k]
                 
             affine_path = norm_out / build_bids_name(d_ents)
             
             d_ents['desc'] = 'warp'
             d_ents['extension'] = '.nii.gz'
             warp_path = norm_out / build_bids_name(d_ents)
             
             if not affine_path.exists() or not warp_path.exists():
                 all_exist = False
                 
        if skip and all_exist:
             self.logger.info("Skipping Normalization (All outputs exist).")
             # Populate context
             normalized_results = {}
             # Re-scan or use predicted?
             # Re-scan to match logic below
             # Or just trust predicted list?
             # Let's effectively reconstruct the dictionary
             idx = 0
             for model_name, metrics in modeling_results.items():
                 for name, path in metrics.items():
                     existing_path = predicted_outputs[idx]
                     normalized_results[name] = existing_path
                     idx += 1
             
             context['normalized_results'] = normalized_results
             return context
        
        # 2. Registration
        tx_forward = []
        tx_inverse = []
        
        if self.tool == 'ants':
             try:
                 import ants
                 mov_raw = ants.image_read(str(ref_path))
                 mov, _ = self._ensure_3d(mov_raw, is_driving=True)
                 
                 fix = ants.image_read(str(self.template))
                 # Fix should also be 3D
                 if fix.dimension == 4:
                     fix = ants.slice_image(fix, axis=3, idx=0)

                 # Registration
                 # Use SyN usually? 
                 # type_of_transform: 'SyN', 'SyNRA', 'Rigid', 'Affine'
                 tf_type = self.kwargs.get('transform_type', 'SyN')
                 
                 # Check for existing transform?
                 # ...
                 
                 reg = ants.registration(fixed=fix, moving=mov, type_of_transform=tf_type)
                 reg = ants.registration(fixed=fix, moving=mov, type_of_transform=tf_type)
                 tx_forward = reg['fwdtransforms']
                 
                 # Save transforms if requested
                 if self.save_transforms:
                     # Copy transform files to output dir with BIDS names
                     import shutil
                     from ...io.bids import build_bids_name, get_entities_from_path
                     
                     d_ents = get_entities_from_path(ref_path)
                     # Clean ents for transform
                     for k in ['acq', 'dir', 'run', 'echo', 'part', 'model']: 
                         if k in d_ents: del d_ents[k]
                     
                     d_ents['space'] = self.space_name
                     d_ents['suffix'] = 'xfm'
                     
                     # ANTs returns [Warp, Affine] usually for SyN
                     # Check what tx_forward contains. usually paths to tmp files.
                     # 0GenericAffine.mat, 1Warp.nii.gz
                     
                     for tf_file in tx_forward:
                         tf_path = Path(tf_file)
                         if tf_path.suffix == '.mat':
                             d_ents['desc'] = 'affine'
                             # Explicitly pass extension as the function kwarg overrides it or defaults to .nii.gz
                             out_name = build_bids_name(d_ents, extension='.mat')
                             shutil.copy(tf_path, norm_out / out_name)
                         elif 'Warp' in tf_path.name and tf_path.suffix == '.gz': 
                             d_ents['desc'] = 'warp'
                             out_name = build_bids_name(d_ents, extension='.nii.gz')
                             shutil.copy(tf_path, norm_out / out_name)
                 
                 # Save warped driving metric
                 # reg['warpedmovout']
                 
             except ImportError:
                 self.logger.error("ANTsPy not installed.")
                 return context
        else:
             self.logger.warning(f"Normalization tool '{self.tool}' not implemented.")
             return context
             
        # 3. Apply to all metrics
        from ...io.bids import build_bids_name, get_entities_from_path
        normalized_results = {}
        
        # 3. Apply to all metrics
        # Iterate over models to preserve 'model' entity if missing
        from ...io.bids import build_bids_name, get_entities_from_path
        normalized_results = {}
        
        for model_name, metrics in modeling_results.items():
            for name, path in metrics.items():
                 # Apply Transform
                 if self.tool == 'ants':
                      try:
                          img_raw = ants.image_read(str(path))
                          img, is_4d = self._ensure_3d(img_raw, is_driving=False)
                          
                          imagetype = 3 if is_4d else 0
                          
                          warped = ants.apply_transforms(
                              fixed=fix, 
                              moving=img, 
                              transformlist=tx_forward,
                              imagetype=imagetype
                          )
                      except Exception as e:
                          self.logger.warning(f"Failed to normalize {name}: {e}")
                          continue
                      
                      # Save
                      ents = get_entities_from_path(path)
                      ents['space'] = self.space_name
                      
                      # Enforce model entity if not present or incorrect?
                      # Usually 'model' is in ents if previous step set it.
                      # But if not, we can force it from the dictionary key 'model_name'.
                      # Check if model_name is valid BIDS model (DTI, DKI etc)
                      if 'model' not in ents or not ents['model']:
                           ents['model'] = model_name
                      
                      new_name = build_bids_name(ents)
                      out_path = norm_out / new_name
                      
                      ants.image_write(warped, str(out_path))
                      normalized_results[name] = out_path
        
        context['normalized_results'] = normalized_results
        return context
