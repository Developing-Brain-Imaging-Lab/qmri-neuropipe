"""
Integrated Modeling Workflow with batch checking and optimizations.

This module contains the ModelingWorkflow class with integrated
batch validation and ExecutionEngine support.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil

from qmri_neuropipe.core import BaseWorkflow, PipelineContext
from qmri_neuropipe.io.bids import build_bids_name, get_entities_from_path
from qmri_neuropipe.lib.dmri.fitting import (
    DTIFittingStep,
    DKIFittingStep,
    NODDIFittingStep,
    SANDIFittingStep,
    NEXIFittingStep,
    MAPMRIFittingStep,
    CSDFittingStep,
    FWDTIFittingStep,
    MicrogliaFittingStep
)
from qmri_neuropipe.lib.dmri.tractography import (
    MRtrixAnatomicalConstraintsStep,
    MRtrixTractographyStep,
    PyAFQStep,
    TractSegStep,
    TractSpecificAnalysisStep,
)
from qmri_neuropipe.lib.dmri.grad_nonlin import create_gnl_map
from qmri_neuropipe.core.step_control import get_rerun_from_step, any_step_matches, step_force_active
from qmri_neuropipe.core.tracking import flush_tracker, update_step_status
from qmri_neuropipe.utils.data_io import DataIOManager
from qmri_neuropipe.utils.reporting import report_modeling_step


@dataclass(frozen=True)
class ModelSpec:
    """Behavior-preserving construction policy for one diffusion model step."""

    step_cls: type
    cfg_keys: tuple[str, ...]
    default_method: str
    flatten: tuple[str, ...] = ("parameters",)
    retain_flattened: bool = False
    flatten_dicts_only: bool = False
    parameters_only: bool = False
    pop_controls_before_flatten: bool = False


MODEL_REGISTRY = (
    ModelSpec(DTIFittingStep, ("dti", "tensor"), "dipy", ("parameters", "options")),
    ModelSpec(DKIFittingStep, ("dki",), "dipy"),
    ModelSpec(
        CSDFittingStep,
        ("csd",),
        "msmt_csd",
        ("parameters", "options"),
        retain_flattened=True,
    ),
    ModelSpec(
        NODDIFittingStep,
        ("noddi",),
        "dmipy",
        flatten_dicts_only=True,
        pop_controls_before_flatten=True,
    ),
    ModelSpec(SANDIFittingStep, ("sandi",), "amico", parameters_only=True),
    ModelSpec(
        MicrogliaFittingStep,
        ("microglia",),
        "dmipy",
        pop_controls_before_flatten=True,
    ),
    ModelSpec(NEXIFittingStep, ("nexi",), "nexi", ("parameters", "options")),
    ModelSpec(MAPMRIFittingStep, ("mapmri",), "dipy"),
    ModelSpec(
        FWDTIFittingStep,
        ("fwe_dti", "fwdti"),
        "dipy",
        pop_controls_before_flatten=True,
    ),
)


class ModelingWorkflow(BaseWorkflow):
    """
    Workflow for diffusion model fitting with optimizations.
    """

    def _initialize_steps(self):
        self.modality = "Diffusion"
        self.steps = []
        
    def build_pipeline(self, context: dict):
        """Build the modeling pipeline based on configuration."""
        self.steps = []
        
        modeling_cfg = self.config.get('dmri', {}).get('modeling') or {}
        self.logger.info(f"Building modeling pipeline...")
        
        # Auto-enable dependencies
        self._check_dependencies(modeling_cfg)
        
        # Add model fitting steps
        self._add_model_steps(modeling_cfg)
        
        # Add tractography steps
        self._add_tractography_steps(modeling_cfg)
        
        self.logger.info(f"Modeling pipeline built with {len(self.steps)} steps")

    def _check_dependencies(self, modeling_cfg: dict):
        """Check and auto-enable dependencies between steps."""
        tract_cfg = modeling_cfg.get('tractography', {})
        if not tract_cfg:
            tract_cfg = self.config.get('dmri', {}).get('tractography', {})
             
        mrtrix_cfg = tract_cfg.get('mrtrix', {})
        # Backward compatibility for the formerly documented flat schema.
        if tract_cfg.get('enabled', False) and not mrtrix_cfg:
            mrtrix_cfg = {
                'enabled': True,
                'algorithm': tract_cfg.get('algorithm', 'iFOD2'),
                'select': tract_cfg.get('n_streamlines', 10_000_000),
            }
            tract_cfg['mrtrix'] = mrtrix_cfg
        if tract_cfg.get('tract_specific', {}).get('enabled', False) and not mrtrix_cfg.get('enabled', False):
            self.logger.info("Tract-specific analysis enabled: Auto-enabling MRtrix tractography")
            mrtrix_cfg = tract_cfg.setdefault('mrtrix', {})
            mrtrix_cfg.setdefault('enabled', True)
            mrtrix_cfg.setdefault('algorithm', 'iFOD2')
        needs_fod = tract_cfg.get('tractseg', {}).get('enabled', False) or (
            mrtrix_cfg.get('enabled', False)
            and not str(mrtrix_cfg.get('algorithm', 'iFOD2')).lower().startswith('tensor')
        )
        if needs_fod:
            csd_cfg = modeling_cfg.get('csd', {})
            if not csd_cfg.get('enabled', False):
                self.logger.info(
                    "FOD-based tractography enabled: Auto-enabling CSD Fitting"
                )
                modeling_cfg.setdefault('csd', {})['enabled'] = True
        if mrtrix_cfg.get('enabled', False) and str(mrtrix_cfg.get('algorithm', '')).lower().startswith('tensor'):
            modeling_cfg.setdefault('dti', {}).setdefault('enabled', True)

    def _add_model_steps(self, modeling_cfg: dict) -> None:
        """Add enabled fitting steps in the stable registry order."""
        for spec in MODEL_REGISTRY:
            model_cfg = next(
                (modeling_cfg[key] for key in spec.cfg_keys if modeling_cfg.get(key)),
                None,
            )
            if not model_cfg or not model_cfg.get("enabled", False):
                continue

            method = model_cfg.get("method", spec.default_method)
            self.logger.info(f"Adding {spec.step_cls.__name__} (method={method})")

            if spec.parameters_only:
                step_kwargs = model_cfg.get("parameters", {})
            else:
                step_kwargs = dict(model_cfg)
                if spec.pop_controls_before_flatten:
                    step_kwargs.pop("enabled", None)
                    step_kwargs.pop("method", None)
                for nested_key in spec.flatten:
                    if nested_key not in step_kwargs:
                        continue
                    nested = (
                        step_kwargs[nested_key]
                        if spec.retain_flattened
                        else step_kwargs.pop(nested_key)
                    )
                    if spec.flatten_dicts_only and not isinstance(nested, dict):
                        continue
                    step_kwargs.update(nested)
                if not spec.pop_controls_before_flatten:
                    step_kwargs.pop("enabled", None)
                    step_kwargs.pop("method", None)

            self.add_step(spec.step_cls(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **step_kwargs,
            ))

    def _add_tractography_steps(self, modeling_cfg: dict):
        """Add tractography steps if enabled."""
        tract_cfg = modeling_cfg.get('tractography', {})
        dmri_cfg = self.config.get('dmri', {})
        if not tract_cfg:
            tract_cfg = dmri_cfg.get('tractography', {})

        mrtrix_cfg = dict(tract_cfg.get('mrtrix', {}))
        if tract_cfg.get('enabled', False) and not mrtrix_cfg:
            mrtrix_cfg = {
                'enabled': True,
                'algorithm': tract_cfg.get('algorithm', 'iFOD2'),
                'select': tract_cfg.get('n_streamlines', 10_000_000),
            }
        if mrtrix_cfg.get('enabled', False):
            mrtrix_cfg.pop('enabled', None)
            act_cfg = mrtrix_cfg.get('act', {})
            if isinstance(act_cfg, dict) and act_cfg.get('enabled', False):
                self.logger.info("Adding MRtrixAnatomicalConstraintsStep")
                self.add_step(MRtrixAnatomicalConstraintsStep(
                    config=self.config, logger=self.logger, provenance=self.provenance,
                    nthreads=self.config.n_cpus,
                    **{key: value for key, value in act_cfg.items() if key != 'enabled'},
                ))
            self.logger.info("Adding MRtrixTractographyStep")
            self.add_step(MRtrixTractographyStep(
                config=self.config, logger=self.logger, provenance=self.provenance,
                nthreads=self.config.n_cpus, **mrtrix_cfg,
            ))
             
        if tract_cfg.get('tractseg', {}).get('enabled', False):
            self.logger.info("Adding TractSegStep")
            self.add_step(TractSegStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method='tractseg',
                **tract_cfg.get('tractseg', {}).get('options', {})
            ))

        if tract_cfg.get('pyafq', {}).get('enabled', False):
            self.logger.info("Adding PyAFQStep")
            self.add_step(PyAFQStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method='pyafq',
                **tract_cfg.get('pyafq', {}).get('options', {})
            ))

        tract_specific = dict(tract_cfg.get('tract_specific', {}))
        if tract_specific.get('enabled', False):
            tract_specific.pop('enabled', None)
            self.logger.info("Adding TractSpecificAnalysisStep")
            self.add_step(TractSpecificAnalysisStep(
                config=self.config, logger=self.logger, provenance=self.provenance,
                nthreads=self.config.n_cpus, **tract_specific,
            ))

    def run(
        self,
        work_dir: Path,
        context: dict,
        reporter=None,
        final_output_dir: Optional[Path] = None
    ) -> PipelineContext:
        """
        Run the modeling workflow with batch validation.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for intermediate files
        context : dict
            Context containing preprocessed_dwis
        reporter : ReportGenerator, optional
            Reporter instance
        final_output_dir : Path, optional
            Final output directory for model results
            
        Returns
        -------
        dict
            Updated context
        """
        context = PipelineContext.ensure(context)
        self.logger.info("Starting Modeling Workflow")
        
        staging_dir = work_dir / "modeling"
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        final_dest = final_output_dir if final_output_dir else staging_dir
        if final_output_dir:
            final_output_dir.mkdir(parents=True, exist_ok=True)

        preprocessed_dwis = context.get('preprocessed_dwis', [])
        preprocessed_masks = context.get('preprocessed_masks', [])
        
        if not preprocessed_dwis:
            self.logger.warning("No preprocessed DWIs for modeling")
            return context
        
        # Pad masks
        if len(preprocessed_masks) < len(preprocessed_dwis):
            preprocessed_masks.extend(
                [None] * (len(preprocessed_dwis) - len(preprocessed_masks))
            )
        
        # === FAST-PATH: Batch validation ===
        rerun_from_step = get_rerun_from_step(self.config, "dmri.modeling", "modeling")
        rerun_hits_modeling = any_step_matches(self.steps, rerun_from_step)
        if self.config.skip_existing and not self.config.get('force', False) and not rerun_hits_modeling:
            all_exist = self._check_all_outputs_exist(
                preprocessed_dwis,
                final_dest
            )
            
            if all_exist:
                self.logger.info("⚡ FAST SKIP: All modeling outputs exist")
                
                # Update tracker
                self._update_tracker_for_cache(context)
                self._populate_modeling_results_from_cache(preprocessed_dwis, final_dest, context)
                
                # Report all
                if reporter:
                    for step in self.steps:
                        for dwi in preprocessed_dwis:
                            try:
                                report_modeling_step(
                                    reporter, step, dwi, final_dest
                                )
                            except Exception as e:
                                self.logger.warning(f"Reporting failed: {e}")
                
                return context

        # Published modeling outputs are also the durable cache.  Rehydrate any
        # missing staging entries so a cleaned work directory can resume from
        # the first model whose validated outputs are absent.
        if final_output_dir:
            recovered = self.recover_intermediate_tree(staging_dir, final_dest)
            if recovered:
                self.logger.info(
                    "Recovered %d modeling cache entries from %s",
                    len(recovered),
                    final_dest,
                )
        
        # === Execute modeling ===
        try:
            from qmri_neuropipe.core.ui import console
            from rich.progress import (
                Progress, SpinnerColumn, TextColumn,
                BarColumn, TimeRemainingColumn
            )
            has_rich = True
        except ImportError:
            has_rich = False
            
        total_steps = len(preprocessed_dwis) * len(self.steps)
        
        if has_rich and getattr(console, "is_terminal", True):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task("Fitting models...", total=total_steps)
                
                self._execute_modeling(
                    preprocessed_dwis,
                    preprocessed_masks,
                    context,
                    staging_dir,
                    final_output_dir,
                    reporter,
                    progress=progress,
                    task_id=task_id
                )
        else:
            self._execute_modeling(
                preprocessed_dwis,
                preprocessed_masks,
                context,
                staging_dir,
                final_output_dir,
                reporter
            )
        
        self.logger.info("Modeling Workflow complete")
        return PipelineContext.ensure(context)
    
    def _execute_modeling(
        self,
        dwis, masks, context,
        staging_dir, final_dir,
        reporter, progress=None, task_id=None
    ):
        """Execute modeling steps with progress tracking."""
        for i, (dwi, mask) in enumerate(zip(dwis, masks)):
            img_name = dwi.img.name
            context['current_image'] = dwi
            staging_changed = False
            
            # Optional GNL map usage (enabled in preprocessing or modeling config).
            modeling_gnl_cfg = (self.config.get('dmri', {}).get('modeling') or {}).get('grad_nonlin', {})
            preproc_gnl_cfg = (self.config.get('dmri', {}).get('preprocessing') or {}).get('grad_nonlin', {})
            gnl_enabled = modeling_gnl_cfg.get('enabled', False) or preproc_gnl_cfg.get('enabled', False)
            if gnl_enabled:
                gnl_map_path = (
                    modeling_gnl_cfg.get('map_path')
                    or modeling_gnl_cfg.get('map_file')
                    or preproc_gnl_cfg.get('map_path')
                    or preproc_gnl_cfg.get('map_file')
                )
                force_gnl = modeling_gnl_cfg.get('force', False) or preproc_gnl_cfg.get('force', False)

                gnl_map = None
                if gnl_map_path:
                    candidate = Path(gnl_map_path)
                    if candidate.exists():
                        gnl_map = candidate
                    else:
                        self.logger.warning(
                            f"GNL map configured but not found: {candidate}"
                        )

                image_specific_gnl = (
                    context.get('gnl_map_by_image', {}).get(dwi.img)
                    or context.get('gnl_map_by_image', {}).get(str(dwi.img))
                )
                if not gnl_map and image_specific_gnl and not force_gnl:
                    image_specific_gnl = Path(image_specific_gnl)
                    if image_specific_gnl.exists():
                        gnl_map = image_specific_gnl

                # Prefer preprocessed output location for caching and reuse.
                ents = dwi.entities.copy()
                sub = ents.get("sub") or context.get("subject", "unknown")
                ses = ents.get("ses")
                ents["sub"] = sub
                if ses:
                    ents["ses"] = ses
                ents["desc"] = "gnl_tensor"

                preproc_out_dir = self.config.output_dir / f"sub-{sub}"
                if ses:
                    preproc_out_dir /= f"ses-{ses}"
                preproc_out_dir /= "dwi"
                preproc_out_dir.mkdir(parents=True, exist_ok=True)

                output_map = preproc_out_dir / build_bids_name(ents)

                if not gnl_map and output_map.exists() and not force_gnl:
                    gnl_map = output_map

                if not gnl_map:
                    existing_gnl = context.get('gnl_map')
                    if existing_gnl and not force_gnl:
                        existing_gnl = Path(existing_gnl)
                        if existing_gnl.exists():
                            gnl_map = existing_gnl

                if not gnl_map:
                    coeff_file = modeling_gnl_cfg.get('coeff_file') or preproc_gnl_cfg.get('coeff_file')
                    if not coeff_file:
                        self.logger.warning(
                            "GNL enabled but no coeff_file provided."
                        )
                        context.pop('gnl_map', None)
                        gnl_map = None
                    else:
                        coeffs = Path(coeff_file)
                        native_ref = context.get('gnl_native_reference_map', {}).get(dwi.img)
                        spatial_transform = context.get('gnl_transform_map', {}).get(dwi.img)
                        gnl_method = modeling_gnl_cfg.get('method') or preproc_gnl_cfg.get('method') or 'tortoise'
                        try:
                            gnl_map = create_gnl_map(
                                input_image=dwi,
                                output_path=output_map,
                                grad_coeffs=coeffs,
                                native_reference=native_ref,
                                method=gnl_method,
                                spatial_transform=spatial_transform,
                                nthreads=self.config.n_cpus,
                                force=force_gnl,
                                logger=self.logger
                            )
                        except Exception as e:
                            if self.config.stop_on_error:
                                raise
                            self.logger.error(f"GNL map generation failed: {e}")
                            gnl_map = None

                if gnl_map and gnl_map.exists():
                    context['gnl_map'] = gnl_map
                    context.setdefault('gnl_map_by_image', {})[dwi.img] = gnl_map
                    gnl_maps = context.setdefault('gnl_maps', [])
                    if gnl_map not in gnl_maps:
                        gnl_maps.append(gnl_map)
                else:
                    context.pop('gnl_map', None)
            
            rerun_from_step = get_rerun_from_step(self.config, "dmri.modeling", "modeling")
            force_from_step_active = False
            for step in self.steps:
                step_name = step.__class__.__name__
                force_from_step_active = step_force_active(force_from_step_active, step, rerun_from_step)
                
                # Check skip
                skipping = (
                    not force_from_step_active and
                    hasattr(step, 'should_skip') and
                    step.should_skip(context, final_dir or staging_dir)
                )
                
                if skipping:
                    if progress:
                        progress.update(
                            task_id,
                            description=f"Skipping {step_name} (Exists)"
                        )
                        progress.advance(task_id)
                    
                    self._update_tracker_step(context, step, "completed (cached)")
                    
                    if reporter:
                        try:
                            report_modeling_step(
                                reporter, step, dwi,
                                staging_dir
                            )
                        except Exception as e:
                            self.logger.warning(f"Reporting failed: {e}")
                    continue
                
                if progress:
                    progress.update(
                        task_id,
                        description=f"Processing {img_name} - {step_name}"
                    )
                
                try:
                    if force_from_step_active:
                        self.logger.info(f"Forcing {step_name} because rerun_from_step has been reached.")
                    step(context, output_dir=staging_dir, mask=mask, force=force_from_step_active)
                    staging_changed = True
                    
                    if progress:
                        progress.advance(task_id)
                    
                except Exception as e:
                    if self.config.stop_on_error:
                        # Preserve outputs from models that completed before a
                        # later model failed. The normal synchronization below
                        # is otherwise bypassed when this exception propagates.
                        if final_dir and staging_changed:
                            shutil.copytree(
                                staging_dir,
                                final_dir,
                                dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("figures"),
                            )
                            staging_changed = False
                        raise
                    self.logger.error(f"{step_name} failed: {e}")
                
                if reporter:
                    try:
                        report_modeling_step(
                            reporter, step, dwi, staging_dir
                        )
                    except Exception as e:
                        self.logger.warning(f"Reporting failed: {e}")

            # Synchronize once after all model steps for this DWI. Copying inside
            # the step loop repeatedly traversed the complete staging tree.
            if final_dir and staging_changed:
                shutil.copytree(
                    staging_dir, final_dir,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("figures")
                )
    
    def _check_all_outputs_exist(self, dwis, output_dir):
        """Check if all modeling outputs exist."""
        for dwi in dwis:
            for step in self.steps:
                step_name = step.__class__.__name__
                
                if 'FWDTI' in step_name:
                    model_dir = output_dir / 'FWE_DTI'
                    required = ['F', 'FA']
                elif 'DTI' in step_name:
                    model_dir = output_dir / 'DTI'
                    required = ['FA', 'MD']
                elif 'DKI' in step_name:
                    model_dir = output_dir / 'DKI'
                    required = ['MK']
                elif 'NODDI' in step_name:
                    model_dir = output_dir / 'NODDI'
                    required = ['ODI']
                elif 'CSD' in step_name:
                    model_dir = output_dir / 'CSD'
                    required = ['fod']
                elif step_name == 'MRtrixAnatomicalConstraintsStep':
                    if not any((output_dir / 'MRtrix' / 'ACT').glob('*_desc-act5tt_probseg.nii.gz')):
                        return False
                    continue
                elif step_name == 'MRtrixTractographyStep':
                    tract_dir = output_dir / 'MRtrix' / 'tractography'
                    if not any(tract_dir.glob('*_desc-wholebrain*_tractography.tck')):
                        return False
                    continue
                elif step_name == 'TractSegStep':
                    if not any((output_dir / 'TractSeg').glob('**/*.nii.gz')):
                        return False
                    continue
                elif step_name == 'PyAFQStep':
                    if not any((output_dir / 'PyAFQ').glob('**/*.trk')):
                        return False
                    continue
                elif step_name == 'TractSpecificAnalysisStep':
                    if not (output_dir / 'MRtrix' / 'tract_specific').exists():
                        return False
                    continue
                else:
                    continue
                
                if not model_dir.exists():
                    return False
                
                found = any(
                    list(model_dir.glob(f"*{metric}*.nii.gz"))
                    for metric in required
                )
                if not found:
                    return False
        
        return True

    def _populate_modeling_results_from_cache(self, dwis, output_dir, context):
        """Populate modeling_results in context when outputs are already cached."""
        modeling_results = context.setdefault('modeling_results', {})

        # Use first DWI for naming; modeling results are per-subject/session here.
        dwi = dwis[0] if dwis else None
        if not dwi:
            return

        ents_base = dwi.entities.copy()
        if 'desc' in ents_base:
            del ents_base['desc']
        if 'suffix' in ents_base:
            del ents_base['suffix']

        model_specs = {
            'DTI': ['FA', 'MD', 'AD', 'RD'],
            'DKI': ['MK', 'AK', 'RK', 'FA', 'MD', 'AD', 'RD'],
            'NODDI': ['ODI', 'NDI', 'FISO'],
            'CSD': ['fod'],
            'MAPMRI': ['RTOP', 'RTAP', 'RTPP'],
            'SANDI': ['Fsoma', 'Fneurite'],
            'FWDTI': ['F', 'MD', 'FA', 'RD', 'AD']
        }

        for model_name, metrics in model_specs.items():
            dir_name = 'FWE_DTI' if model_name == 'FWDTI' else model_name
            model_dir = output_dir / dir_name
            if not model_dir.exists():
                continue

            model_results = modeling_results.setdefault(model_name, {})
            ents = ents_base.copy()
            ents['model'] = model_name

            for metric in metrics:
                suffix = metric
                fname = build_bids_name(ents, suffix=suffix)
                fpath = model_dir / fname
                if not str(fpath).endswith('.nii.gz'):
                    fpath = Path(str(fpath) + '.nii.gz')

                if fpath.exists():
                    model_results[metric] = fpath
                    continue

                # Fallback glob search
                matches = list(model_dir.glob(f"*{metric}*.nii.gz"))
                if matches:
                    model_results[metric] = matches[0]

        tract_root = output_dir / "MRtrix"
        tract_dir = tract_root / "tractography"
        if tract_dir.exists():
            tract = context.setdefault("tractography", {})
            filtered = next(tract_dir.glob("*_desc-wholebrain*SIFT_tractography.tck"), None)
            unfiltered = next((p for p in tract_dir.glob("*_desc-wholebrain*_tractography.tck") if "SIFT" not in p.name), None)
            if filtered:
                tract["whole_brain"] = filtered
                tract["unfiltered"] = unfiltered
            elif unfiltered:
                tract["whole_brain"] = unfiltered
            weights = next(tract_dir.glob("*_desc-sift2_weights.tsv"), None)
            if weights:
                tract["sift2_weights"] = weights
            act_dir = tract_root / "ACT"
            five_tt = next(act_dir.glob("*_desc-act5tt_probseg.nii.gz"), None) if act_dir.exists() else None
            gmwmi = next(act_dir.glob("*_desc-gmwmi_mask.nii.gz"), None) if act_dir.exists() else None
            if five_tt:
                tract["act_5tt"] = five_tt
            if gmwmi:
                tract["gmwmi_seed"] = gmwmi
            bundle_dir = tract_root / "tract_specific" / "bundles"
            if bundle_dir.exists():
                tract["bundles"] = {
                    get_entities_from_path(p).get("desc") or p.stem: p
                    for p in bundle_dir.glob("*_tractography.tck")
                    if "resampled" not in p.name.lower()
                }
    
    def _update_tracker_for_cache(self, context):
        """Update tracker for all cached steps."""
        updated = False
        for step in self.steps:
            updated = update_step_status(
                self.config,
                context,
                step,
                "completed (cached)",
            ) or updated
        if updated:
            flush_tracker(self.config)
    
    def _update_tracker_step(self, context, step, status):
        """Update tracker for single step."""
        update_step_status(self.config, context, step, status)
