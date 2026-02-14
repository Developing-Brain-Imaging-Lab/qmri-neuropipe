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
        tool_normalized = tool.lower()
        if tool_normalized == 'mri_synthmorph':
            tool_normalized = 'synthmorph'
        self.tool = tool_normalized
        self.save_transforms = kwargs.get('save_transforms', True)
        self.include_all_metrics = kwargs.get('include_all_metrics', True)
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
        if not modeling_results and not self.include_all_metrics:
            self.logger.warning("Normalization skipped: No modeling results found.")
            return context

        # Fetch nthreads
        nthreads = kwargs.get('nthreads', self.config.get('n_cpus', 1))
        import os
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nthreads)
        
        # Build full metric list (including any extra outputs not tracked in context)
        from ...io.bids import get_entities_from_path
        metrics_to_norm = {}

        def _add_metric(model_name: str, metric_name: str, path: Path):
            if not path:
                return
            if not isinstance(path, Path):
                path = Path(path)
            metrics_to_norm.setdefault(model_name, {})
            if metric_name not in metrics_to_norm[model_name]:
                metrics_to_norm[model_name][metric_name] = path

        if modeling_results:
            for model_name, metrics in modeling_results.items():
                if isinstance(metrics, dict):
                    for name, path in metrics.items():
                        _add_metric(model_name, name, path)

        if self.include_all_metrics:
            candidate_dirs = set()
            for model_metrics in metrics_to_norm.values():
                for path in model_metrics.values():
                    if path:
                        candidate_dirs.add(path.parent)

            if not candidate_dirs:
                potential_bases = []
                if (output_dir / "modeling").exists():
                    potential_bases.append(output_dir / "modeling")
                potential_bases.append(output_dir)

                for base in potential_bases:
                    for model_dir in [
                        "DTI", "DKI", "NODDI", "SANDI", "MAPMRI", "CSD",
                        "FWE_DTI", "FWDTI", "dti", "dki", "noddi", "sandi",
                        "mapmri", "csd", "fwe_dti", "fwdti"
                    ]:
                        p = base / model_dir
                        if p.exists():
                            candidate_dirs.add(p)

            for model_dir in candidate_dirs:
                for p in model_dir.glob("*.nii.gz"):
                    ents = get_entities_from_path(p)
                    model_name = ents.get("model") or model_dir.name
                    metric_name = ents.get("suffix")
                    if not metric_name:
                        name_part = p.name.replace(".nii.gz", "")
                        metric_name = name_part.split("_")[-1]
                    _add_metric(model_name, metric_name, p)

        if not metrics_to_norm:
            self.logger.warning("Normalization skipped: No modeling results found.")
            return context

        # 1. Find driving metric
        ref_path = None
        
        # Flatten results to find driving metric
        flat_metrics = {}
        for model_metrics in metrics_to_norm.values():
            flat_metrics.update(model_metrics)
        
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
        
        for model_name, metrics in metrics_to_norm.items():
            for name, path in metrics.items():
                ents = get_entities_from_path(path)
                ents['space'] = self.space_name
                if 'model' not in ents or not ents['model']:
                    ents['model'] = model_name
                out_path = norm_out / build_bids_name(ents)
                predicted_outputs.append((model_name, name, out_path))
                if not out_path.exists():
                    all_exist = False
                    
        # Also check transforms if saving (ANTs only)
        if self.save_transforms and self.tool == 'ants':
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
        elif self.tool == 'synthmorph':
             d_ents = get_entities_from_path(ref_path)
             for k in ['acq', 'dir', 'run', 'echo', 'part']:
                 if k in d_ents:
                     del d_ents[k]
             d_ents['space'] = self.space_name
             d_ents['suffix'] = 'xfm'
             d_ents['desc'] = 'synthmorph'
             ext = self.kwargs.get('synthmorph_transform_ext', '.lta')
             tx_path = norm_out / build_bids_name(d_ents, extension=ext)
             if not tx_path.exists():
                 all_exist = False
        elif self.save_transforms and self.tool not in ['ants', 'synthmorph']:
             self.logger.warning(f"Save transforms not supported for tool '{self.tool}'.")
                 
        if skip and all_exist:
             # Check timestamps
             # ref_path is the driving metric
             in_mtime = ref_path.stat().st_mtime
             # out_mtime is the first predicted output
             out_mtime = predicted_outputs[0][2].stat().st_mtime
             
             if in_mtime > out_mtime:
                  self.logger.info(f"Normalization driving metric ({ref_path.name}) is newer than output. Re-running.")
             else:
                  self.logger.info("Skipping Normalization (All outputs exist and are up-to-date).")
                  # Populate context
             normalized_results = {}
             normalized_results_by_model = {}
             for model_name, name, existing_path in predicted_outputs:
                 normalized_results_by_model.setdefault(model_name, {})[name] = existing_path
                 prefixed_key = f"{model_name}_{name}"
                 normalized_results[prefixed_key] = existing_path
                 if name not in normalized_results:
                     normalized_results[name] = existing_path
             
             context['normalized_results'] = normalized_results
             context['normalized_results_by_model'] = normalized_results_by_model
             return context
        
        # 2. Registration
        tx_forward = []
        tx_inverse = []
        synthmorph_tx = None
        
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
                 
                 # ...
                 
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
        elif self.tool == 'synthmorph':
             from ...interfaces.freesurfer import mri_synthmorph_register
             d_ents = get_entities_from_path(ref_path)
             for k in ['acq', 'dir', 'run', 'echo', 'part']:
                 if k in d_ents:
                     del d_ents[k]
             d_ents['space'] = self.space_name
             d_ents['suffix'] = 'xfm'
             d_ents['desc'] = 'synthmorph'
             ext = self.kwargs.get('synthmorph_transform_ext', '.lta')
             synthmorph_tx = norm_out / build_bids_name(d_ents, extension=ext)

             try:
                 mri_synthmorph_register(
                     moving=ref_path,
                     target=self.template,
                     transform_out=synthmorph_tx,
                     output_image=None,
                     extra_args=self.kwargs.get('synthmorph_register_args', '')
                 )
             except Exception as e:
                 self.logger.warning(f"SynthMorph register failed: {e}")
                 return context
        else:
             self.logger.warning(f"Normalization tool '{self.tool}' not implemented.")
             return context
             
        # 3. Apply to all metrics
        from ...io.bids import build_bids_name, get_entities_from_path
        normalized_results = {}
        normalized_results_by_model = {}
        
        for model_name, metrics in metrics_to_norm.items():
            for name, path in metrics.items():
                 ents = get_entities_from_path(path)
                 ents['space'] = self.space_name
                 if 'model' not in ents or not ents['model']:
                      ents['model'] = model_name
                 
                 new_name = build_bids_name(ents)
                 out_path = norm_out / new_name
                 
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
                          ants.image_write(warped, str(out_path))
                      except Exception as e:
                          self.logger.warning(f"Failed to normalize {name}: {e}")
                          continue
                 elif self.tool == 'synthmorph':
                      from ...interfaces.freesurfer import mri_synthmorph_apply
                      if not synthmorph_tx or not Path(synthmorph_tx).exists():
                          self.logger.warning("SynthMorph transform missing; skipping apply.")
                          continue
                      try:
                          mri_synthmorph_apply(
                              moving=path,
                              target=self.template,
                              transform_in=synthmorph_tx,
                              out_file=out_path,
                              extra_args=self.kwargs.get('synthmorph_apply_args', '')
                          )
                      except Exception as e:
                          self.logger.warning(f"Failed to normalize {name}: {e}")
                          continue
                 
                 normalized_results_by_model.setdefault(model_name, {})[name] = out_path
                 prefixed_key = f"{model_name}_{name}"
                 normalized_results[prefixed_key] = out_path
                 if name not in normalized_results:
                      normalized_results[name] = out_path
        
        context['normalized_results'] = normalized_results
        context['normalized_results_by_model'] = normalized_results_by_model
        return context
