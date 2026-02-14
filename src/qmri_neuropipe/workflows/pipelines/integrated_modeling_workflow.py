"""
Integrated Modeling Workflow with batch checking and optimizations.

This module contains the ModelingWorkflow class with integrated
batch validation and ExecutionEngine support.
"""

from pathlib import Path
from typing import Optional

from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.lib.dmri.fitting import (
    DTIFittingStep,
    DKIFittingStep,
    NODDIFittingStep,
    SANDIFittingStep,
    NEXIFittingStep,
    MAPMRIFittingStep,
    CSDFittingStep,
    FWDTIFittingStep
)
from qmri_neuropipe.lib.dmri.tractography import TractSegStep, PyAFQStep


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
        self._add_dti_step(modeling_cfg)
        self._add_dki_step(modeling_cfg)
        self._add_csd_step(modeling_cfg)
        self._add_noddi_step(modeling_cfg)
        self._add_sandi_step(modeling_cfg)
        self._add_nexi_step(modeling_cfg)
        self._add_mapmri_step(modeling_cfg)
        self._add_fwdti_step(modeling_cfg)
        
        # Add tractography steps
        self._add_tractography_steps(modeling_cfg)
        
        self.logger.info(f"Modeling pipeline built with {len(self.steps)} steps")

    def _check_dependencies(self, modeling_cfg: dict):
        """Check and auto-enable dependencies between steps."""
        tract_cfg = modeling_cfg.get('tractography', {})
        if not tract_cfg:
            tract_cfg = self.config.get('dmri', {}).get('tractography', {})
             
        if tract_cfg.get('tractseg', {}).get('enabled', False):
            csd_cfg = modeling_cfg.get('csd', {})
            if not csd_cfg.get('enabled', False):
                self.logger.info(
                    "TractSeg enabled: Auto-enabling CSD Fitting"
                )
                modeling_cfg.setdefault('csd', {})['enabled'] = True

    def _add_dti_step(self, modeling_cfg: dict):
        """Add DTI fitting step if enabled."""
        dti_cfg = modeling_cfg.get('dti', {}) or modeling_cfg.get('tensor', {})
        if dti_cfg.get('enabled', False):
            method = dti_cfg.get('method', 'dipy')
            self.logger.info(f"Adding DTIFittingStep (method={method})")
            
            dti_kwargs = dict(dti_cfg)
            if 'parameters' in dti_kwargs:
                dti_kwargs.update(dti_kwargs.pop('parameters'))
            if 'options' in dti_kwargs:
                dti_kwargs.update(dti_kwargs.pop('options'))
            dti_kwargs.pop('enabled', None)
            dti_kwargs.pop('method', None)

            self.add_step(DTIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **dti_kwargs
            ))

    def _add_dki_step(self, modeling_cfg: dict):
        """Add DKI fitting step if enabled."""
        dki_cfg = modeling_cfg.get('dki', {})
        if dki_cfg.get('enabled', False):
            method = dki_cfg.get('method', 'dipy')
            self.logger.info(f"Adding DKIFittingStep (method={method})")
            
            dki_kwargs = dict(dki_cfg)
            if 'parameters' in dki_kwargs:
                dki_kwargs.update(dki_kwargs.pop('parameters'))
            dki_kwargs.pop('enabled', None)
            dki_kwargs.pop('method', None)

            self.add_step(DKIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **dki_kwargs
            ))

    def _add_csd_step(self, modeling_cfg: dict):
        """Add CSD fitting step if enabled."""
        csd_cfg = modeling_cfg.get('csd', {})
        if csd_cfg.get('enabled', False):
            method = csd_cfg.get('method', 'msmt_csd')
            self.logger.info(f"Adding CSDFittingStep (method={method})")
            
            csd_kwargs = dict(csd_cfg)
            if 'parameters' in csd_cfg:
                csd_kwargs.update(csd_cfg['parameters'])
            if 'options' in csd_cfg:
                csd_kwargs.update(csd_cfg['options'])
            
            csd_kwargs.pop('enabled', None)
            csd_kwargs.pop('method', None)
            
            self.add_step(CSDFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **csd_kwargs
            ))

    def _add_noddi_step(self, modeling_cfg: dict):
        """Add NODDI fitting step if enabled."""
        noddi_cfg = modeling_cfg.get('noddi', {})
        if noddi_cfg.get('enabled', False):
            method = noddi_cfg.get('method', 'dmipy')
            self.logger.info(f"Adding NODDIFittingStep (method={method})")
            
            step_kwargs = noddi_cfg.copy()
            step_kwargs.pop('method', None) 
            step_kwargs.pop('enabled', None)
            
            if 'parameters' in step_kwargs:
                nested_params = step_kwargs.pop('parameters')
                if isinstance(nested_params, dict):
                    step_kwargs.update(nested_params)

            self.add_step(NODDIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **step_kwargs
            ))

    def _add_sandi_step(self, modeling_cfg: dict):
        """Add SANDI fitting step if enabled."""
        sandi_cfg = modeling_cfg.get('sandi', {})
        if sandi_cfg.get('enabled', False):
            method = sandi_cfg.get('method', 'amico')
            self.logger.info(f"Adding SANDIFittingStep (method={method})")
            self.add_step(SANDIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **sandi_cfg.get('parameters', {})
            ))

    def _add_nexi_step(self, modeling_cfg: dict):
        """Add NEXI fitting step if enabled."""
        nexi_cfg = modeling_cfg.get('nexi', {})
        if nexi_cfg.get('enabled', False):
            method = nexi_cfg.get('method', 'nexi')
            self.logger.info(f"Adding NEXIFittingStep (method={method})")

            nexi_kwargs = dict(nexi_cfg)
            if 'parameters' in nexi_kwargs:
                nexi_kwargs.update(nexi_kwargs.pop('parameters'))
            if 'options' in nexi_kwargs:
                nexi_kwargs.update(nexi_kwargs.pop('options'))
            nexi_kwargs.pop('enabled', None)
            nexi_kwargs.pop('method', None)

            self.add_step(NEXIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **nexi_kwargs
            ))

    def _add_mapmri_step(self, modeling_cfg: dict):
        """Add MAPMRI fitting step if enabled."""
        map_cfg = modeling_cfg.get('mapmri', {})
        if map_cfg.get('enabled', False):
            method = map_cfg.get('method', 'dipy')
            self.logger.info(f"Adding MAPMRIFittingStep (method={method})")
            
            map_kwargs = dict(map_cfg)
            if 'parameters' in map_kwargs:
                map_kwargs.update(map_kwargs.pop('parameters'))
            map_kwargs.pop('enabled', None)
            map_kwargs.pop('method', None)

            self.add_step(MAPMRIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **map_kwargs
            ))

    def _add_fwdti_step(self, modeling_cfg: dict):
        """Add Free-Water DTI fitting step if enabled."""
        fwe_dti_cfg = modeling_cfg.get('fwe_dti', {}) or modeling_cfg.get('fwdti', {})
        if fwe_dti_cfg.get('enabled', False):
            method = fwe_dti_cfg.get('method', 'dipy')
            self.logger.info(f"Adding FWDTIFittingStep (method={method})")
            
            step_kwargs = fwe_dti_cfg.copy()
            step_kwargs.pop('method', None)
            step_kwargs.pop('enabled', None)
            if 'parameters' in step_kwargs:
                step_kwargs.update(step_kwargs.pop('parameters'))
            
            self.add_step(FWDTIFittingStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance,
                method=method,
                n_cpus=self.config.n_cpus,
                **step_kwargs
            ))

    def _add_tractography_steps(self, modeling_cfg: dict):
        """Add tractography steps if enabled."""
        tract_cfg = modeling_cfg.get('tractography', {})
        dmri_cfg = self.config.get('dmri', {})
        if not tract_cfg:
            tract_cfg = dmri_cfg.get('tractography', {})
             
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

    def run(
        self,
        work_dir: Path,
        context: dict,
        reporter=None,
        final_output_dir: Optional[Path] = None
    ) -> dict:
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
        from ...utils.data_io import DataIOManager
        from ...utils.reporting import report_modeling_step
        
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
        if self.config.skip_existing and not self.config.get('force', False):
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
            
        import shutil
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
        return context
    
    def _execute_modeling(
        self,
        dwis, masks, context,
        staging_dir, final_dir,
        reporter, progress=None, task_id=None
    ):
        """Execute modeling steps with progress tracking."""
        import shutil
        from pathlib import Path
        from ...utils.reporting import report_modeling_step
        from qmri_neuropipe.io.bids import build_bids_name
        from qmri_neuropipe.lib.dmri.grad_nonlin import create_gnl_map
        
        for i, (dwi, mask) in enumerate(zip(dwis, masks)):
            img_name = dwi.img.name
            context['current_image'] = dwi
            
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

                if not gnl_map:
                    existing_gnl = context.get('gnl_map')
                    if (existing_gnl and isinstance(existing_gnl, Path)
                            and existing_gnl.exists() and not force_gnl):
                        gnl_map = existing_gnl

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
                    coeff_file = modeling_gnl_cfg.get('coeff_file') or preproc_gnl_cfg.get('coeff_file')
                    if not coeff_file:
                        self.logger.warning(
                            "GNL enabled but no coeff_file provided."
                        )
                        context.pop('gnl_map', None)
                        gnl_map = None
                    else:
                        coeffs = Path(coeff_file)
                        try:
                            gnl_map = create_gnl_map(
                                input_image=dwi,
                                output_path=output_map,
                                grad_coeffs=coeffs,
                                native_reference=None,
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
                    gnl_maps = context.setdefault('gnl_maps', [])
                    if gnl_map not in gnl_maps:
                        gnl_maps.append(gnl_map)
                else:
                    context.pop('gnl_map', None)
            
            for step in self.steps:
                step_name = step.__class__.__name__
                
                # Check skip
                skipping = (
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
                    step(context, output_dir=staging_dir, mask=mask)
                    
                    # Copy to final
                    if final_dir:
                        shutil.copytree(
                            staging_dir, final_dir,
                            dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("figures")
                        )
                    
                    if progress:
                        progress.advance(task_id)
                    
                except Exception as e:
                    if self.config.stop_on_error:
                        raise e
                    self.logger.error(f"{step_name} failed: {e}")
                
                if reporter:
                    try:
                        report_modeling_step(
                            reporter, step, dwi, staging_dir
                        )
                    except Exception as e:
                        self.logger.warning(f"Reporting failed: {e}")
    
    def _check_all_outputs_exist(self, dwis, output_dir):
        """Check if all modeling outputs exist."""
        for dwi in dwis:
            for step in self.steps:
                step_name = step.__class__.__name__
                
                if 'DTI' in step_name:
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
        from qmri_neuropipe.io.bids import build_bids_name

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
            'FWDTI': ['FWF', 'MD', 'FA']
        }

        for model_name, metrics in model_specs.items():
            model_dir = output_dir / model_name
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
    
    def _update_tracker_for_cache(self, context):
        """Update tracker for all cached steps."""
        tracker = self.config.tracker
        if not tracker:
            return
        
        subject = context.get('subject')
        session = context.get('session')
        study = context.get('study_name', self.config.get('study_name'))
        
        if subject and session:
            for step in self.steps:
                tracker_module = step.normalize_tracker_module(
                    step.__class__.__name__
                )
                tracker.update_status(
                    subject, session, tracker_module,
                    "completed (cached)", study,
                    modality=step.modality
                )
            tracker.save()
    
    def _update_tracker_step(self, context, step, status):
        """Update tracker for single step."""
        tracker = self.config.tracker
        if not tracker:
            return
        
        subject = context.get('subject')
        session = context.get('session')
        study = context.get('study_name', self.config.get('study_name'))
        
        if subject and session:
            tracker_module = step.normalize_tracker_module(
                step.__class__.__name__
            )
            tracker.update_status(
                subject, session, tracker_module,
                status, study, modality=step.modality
            )
            tracker.save()
