"""
Diffusion MRI Processing Pipeline - Refactored Version

This module provides the main DMRIPipeline class that orchestrates all
diffusion MRI processing workflows. Individual workflow implementations
have been refactored into separate modules for better maintainability.

Module Structure:
    - integrated_preprocessing_workflow.py: Preprocessing steps
    - integrated_modeling_workflow.py: Model fitting
    - normalization_workflow.py: Spatial normalization
    - segmentation_workflow.py: ROI analysis
"""

from pathlib import Path
from typing import Optional
import glob
import shutil
import nibabel as nib
import numpy as np

# Core framework
from qmri_neuropipe.core import BasePipeline, PipelineConfig, PipelineContext
from qmri_neuropipe.core.caching import reuse_if_exists, reuse_path_if_exists
from qmri_neuropipe.core.types import ImageFile, DWIFile

# BIDS I/O
from qmri_neuropipe.io.bids import build_bids_name, parse_bids_filename
from qmri_neuropipe.io.anat.bids import bids_find_t1w, bids_find_t2w, select_anatomical_candidates
from qmri_neuropipe.io.dmri.bids import bids_find_dwi, find_reversed_phase_groups

# External interfaces
from qmri_neuropipe.interfaces.mriqc import run_mriqc
from qmri_neuropipe.lib.reporting.report import ReportGenerator
from qmri_neuropipe.lib.common.tracking import TrackingStep
from qmri_neuropipe.utils.data_io import DataIOManager
from qmri_neuropipe.core.step_control import get_rerun_from_step, any_step_matches

# Anatomical workflow
from .anat import AnatPreprocessingWorkflow

# === REFACTORED WORKFLOWS ===
from .integrated_preprocessing_workflow import PreprocessingWorkflow
from .integrated_modeling_workflow import ModelingWorkflow
from .normalization_workflow import NormalizationWorkflow
from .segmentation_workflow import SegmentationWorkflow
# ============================


def _select_dwi_inputs(dwi_files, dmri_cfg):
    """Select DWI inputs using exact BIDS-entity matches from configuration."""
    inputs_cfg = (dmri_cfg or {}).get("inputs", {}) or {}
    select = inputs_cfg.get("select", {}) or {}
    if not select:
        return list(dwi_files)
    if not isinstance(select, dict):
        raise ValueError("dmri.inputs.select must be a mapping of BIDS entities to values.")

    normalized = {}
    for entity, values in select.items():
        if isinstance(values, str) or not isinstance(values, (list, tuple, set)):
            values = [values]
        allowed = {str(value) for value in values if value is not None and str(value)}
        if not allowed:
            raise ValueError(f"dmri.inputs.select.{entity} must contain at least one value.")
        normalized[str(entity)] = allowed

    selected = []
    for dwi in dwi_files:
        entities = dict(getattr(dwi, "entities", {}) or {})
        if all(str(entities.get(entity)) in allowed for entity, allowed in normalized.items()):
            selected.append(dwi)
    return selected


def _partition_dwi_inputs(dwi_files, dmri_cfg):
    """Partition DWI inputs into independently processed entity groups."""
    inputs_cfg = (dmri_cfg or {}).get("inputs", {}) or {}
    separate_by = inputs_cfg.get("separate_by")
    if not separate_by:
        return [(None, list(dwi_files))]
    if isinstance(separate_by, str):
        separate_by = [separate_by]
    separate_by = [str(entity).strip() for entity in separate_by if str(entity).strip()]
    if not separate_by:
        return [(None, list(dwi_files))]

    groups = {}
    for dwi in dwi_files:
        entities = dict(getattr(dwi, "entities", {}) or {})
        missing = [entity for entity in separate_by if not entities.get(entity)]
        if missing:
            raise ValueError(
                f"Cannot separate {Path(dwi.img).name}: missing BIDS entity/entities "
                f"{', '.join(missing)}."
            )
        key = tuple(str(entities[entity]) for entity in separate_by)
        groups.setdefault(key, []).append(dwi)

    preserve_as = inputs_cfg.get("preserve_group_as")
    if preserve_as is None and "desc" in separate_by:
        preserve_as = "acq"
    allowed_identity_entities = {"acq", "rec", "run", "task", "dir"}
    if preserve_as and preserve_as not in allowed_identity_entities:
        raise ValueError(
            "dmri.inputs.preserve_group_as must be one of: "
            + ", ".join(sorted(allowed_identity_entities))
        )

    partitioned = []
    for key in sorted(groups):
        label = "x".join(key)
        group = groups[key]
        if preserve_as and preserve_as not in separate_by:
            for dwi in group:
                entities = dict(dwi.entities)
                existing = entities.get(preserve_as)
                if existing and str(existing) != label:
                    raise ValueError(
                        f"Cannot preserve DWI group '{label}' as {preserve_as}: "
                        f"{Path(dwi.img).name} already has {preserve_as}-{existing}."
                    )
                entities[preserve_as] = label
                dwi.entities = entities
        partitioned.append((label, group))
    return partitioned


def _count_bval_entries(path: Path) -> int:
    values = np.loadtxt(path)
    return int(np.atleast_1d(values).size)


def _count_bvec_entries(path: Path) -> int:
    values = np.loadtxt(path)
    values = np.asarray(values)
    if values.ndim == 1:
        if values.size % 3 != 0:
            raise ValueError(
                f"BVEC file {path} has {values.size} scalar values, which is not divisible by 3."
            )
        return int(values.size // 3)
    if values.ndim != 2:
        raise ValueError(f"BVEC file {path} must be a 2D table, got shape {values.shape}.")
    if 3 in values.shape:
        return int(values.shape[1] if values.shape[0] == 3 else values.shape[0])
    raise ValueError(f"BVEC file {path} must have one dimension of size 3, got shape {values.shape}.")


def _count_dwi_volumes(path: Path) -> int:
    shape = nib.load(str(path)).shape
    return int(shape[3]) if len(shape) > 3 else 1


def _validate_dwi_gradient_tables(dwi_files) -> None:
    """Validate that each DWI image has matching bval/bvec gradient counts."""
    errors = []
    for dwi in dwi_files:
        img = Path(dwi.img)
        bval = Path(dwi.bval) if getattr(dwi, "bval", None) else None
        bvec = Path(dwi.bvec) if getattr(dwi, "bvec", None) else None
        if not bval or not bval.exists():
            errors.append(f"{img.name}: missing matching .bval sidecar")
            continue
        if not bvec or not bvec.exists():
            errors.append(f"{img.name}: missing matching .bvec sidecar")
            continue
        try:
            n_volumes = _count_dwi_volumes(img)
            n_bvals = _count_bval_entries(bval)
            n_bvecs = _count_bvec_entries(bvec)
        except Exception as exc:
            errors.append(f"{img.name}: failed to read image/gradient sidecars ({exc})")
            continue
        if n_bvals != n_volumes or n_bvecs != n_volumes:
            errors.append(
                f"{img.name}: image has {n_volumes} volume(s), "
                f"but {bval.name} has {n_bvals} b-value(s) and "
                f"{bvec.name} has {n_bvecs} b-vector(s)"
            )

    if errors:
        raise ValueError(
            "DWI gradient sidecars do not match their image volume counts. "
            "Check BIDS conversion or select the matching acquisition/run before processing:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )


class DMRIPipeline(BasePipeline):
    """
    Diffusion MRI Processing Pipeline.
    
    Orchestrates complete dMRI processing including:
    - Anatomical preprocessing (optional)
    - Diffusion preprocessing
    - Model fitting (DTI, DKI, NODDI, etc.)
    - Spatial normalization
    - ROI segmentation and statistics
    """
    
    @property
    def name(self):
        return 'dmri-pipeline'
    
    @property
    def version(self):
        return '1.0.0'
    
    def _initialize_pipeline(self):
        """Initialize all workflow components."""
        self.preprocessing = PreprocessingWorkflow(self.config, self.logger, self.provenance)
        self.modeling = ModelingWorkflow(self.config, self.logger, self.provenance)
        self.normalization = NormalizationWorkflow(self.config, self.logger, self.provenance)
        self.segmentation = SegmentationWorkflow(self.config, self.logger, self.provenance)
        self.anat_preprocessing = AnatPreprocessingWorkflow(self.config, self.logger, self.provenance)
    
    def _should_skip(self, subject: str, session: Optional[str]) -> bool:
        """
        Override default skip logic for multi-stage pipeline.
        
        Returns False to allow granular skip logic in individual workflows.
        """
        return False

    def process_subject(self, subject: str, session: Optional[str]):
        """
        Process a single subject through all enabled workflows.
        
        Parameters
        ----------
        subject : str
            Subject identifier (without 'sub-' prefix)
        session : Optional[str]
            Session identifier (without 'ses-' prefix), if applicable
        """
        # Setup directories
        ses = f"ses-{session}" if session else ""
        subj_dir = Path(self.config.get('bids_dir')) / f'sub-{subject}' / ses
        
        # Work directories
        work_root = Path(self.config.get('work_dir', self.config.output_dir / 'work'))
        if session:
            subj_work_dir = work_root / f'sub-{subject}' / f'ses-{session}' / 'dwi'
            anat_work_dir = work_root / f'sub-{subject}' / f'ses-{session}' / 'anat'
        else:
            subj_work_dir = work_root / f'sub-{subject}' / 'dwi'
            anat_work_dir = work_root / f'sub-{subject}' / 'anat'
        
        subj_work_dir.mkdir(parents=True, exist_ok=True)
        anat_work_dir.mkdir(parents=True, exist_ok=True)

        # Output directories
        output_dir = self._get_output_dir(subject, session) / 'dwi'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize reporter
        report_title = f"QMRI-Neuropipe Report: sub-{subject} {ses}"
        reporter = ReportGenerator(output_dir.parent, title=report_title)
        reporter.set_participant_summary(
            f"Participant: sub-{subject}" + (f", Session: {session}" if session else ""),
            details={
                "Subject": subject,
                "Session": session or "N/A",
                "BIDS Path": str(self.config.bids_dir),
                "Output Path": str(self.config.output_dir)
            }
        )
        
        # Find input files
        t1w_files = self._find_anat_files(subject, session, 'T1w')
        t2w_files = self._find_anat_files(subject, session, 'T2w')
        dwi_files = bids_find_dwi(subj_dir / 'dwi')
        
        if not dwi_files:
            self.logger.warning(f"No DWI files found for sub-{subject} {ses}. Skipping.")
            return

        # Run MRIQC if enabled
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        if qc_cfg.get("enabled"):
            self._run_mriqc(subject, session)

        # Run anatomical preprocessing if enabled
        anat_cfg = self.config.get("anat", {}).get("preprocessing", {})
        if anat_cfg:
            t1w_files, t2w_files = self._run_anatomical_preprocessing(
                subject, session, t1w_files, t2w_files, anat_work_dir, reporter
            )

        # Check if dMRI processing is enabled
        dmri_cfg = self.config.get("dmri")
        if not dmri_cfg:
            self.logger.info("dMRI processing not enabled. Skipping diffusion steps.")
            reporter.generate()
            try:
                reporter.generate_pdf()
            except Exception as e:
                self.logger.warning(f"PDF generation failed: {e}")
            return

        dwi_files = _select_dwi_inputs(dwi_files, dmri_cfg)
        if not dwi_files:
            select = (dmri_cfg.get("inputs", {}) or {}).get("select", {})
            raise ValueError(
                f"No DWI inputs matched dmri.inputs.select={select} for "
                f"sub-{subject}" + (f" ses-{session}." if session else ".")
            )
        _validate_dwi_gradient_tables(dwi_files)

        input_groups = _partition_dwi_inputs(dwi_files, dmri_cfg)
        for group_label, group_dwis in input_groups:
            group_work_dir = subj_work_dir
            if group_label is not None:
                group_work_dir = subj_work_dir / f"group-{group_label}"
                group_work_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(
                    f"Processing DWI group {group_label}: "
                    f"{len(group_dwis)} input file(s)"
                )

            context = self._build_initial_context(
                subject, session, group_dwis, t1w_files, t2w_files
            )
            context["dwi_group"] = group_label

            preprocessed_context = self._run_preprocessing(
                context, group_work_dir, output_dir, reporter
            )

            if self.modeling.steps or dmri_cfg.get('modeling'):
                self._run_modeling(
                    preprocessed_context, group_work_dir, output_dir, reporter
                )

            if self.config.get("gratio", {}).get("enabled", False):
                from .gratio import run_aggregate_gratio_subject
                current_dwi = preprocessed_context.get("current_image")
                gratio_results = run_aggregate_gratio_subject(
                    self.config,
                    subject,
                    session,
                    logger=self.logger,
                    entity_filter=getattr(current_dwi, "entities", None),
                )
                bucket = preprocessed_context.setdefault("modeling_results", {}).setdefault(
                    "AggregateGRatio", {}
                )
                image_metrics = {
                    "MVF", "AVF", "FVF", "gratio", "ConductionFactor",
                    "myelinThickness", "RushtonCVIndex", "WaxmanBennettCVIndex",
                    "RushtonCV", "WaxmanBennettCV",
                }
                for result_index, result in enumerate(gratio_results, start=1):
                    for metric, path in result.items():
                        if metric in image_metrics:
                            key = metric if len(gratio_results) == 1 else f"{metric}_set{result_index}"
                            bucket[key] = path

            if self.normalization.steps or dmri_cfg.get('normalization', {}).get('enabled'):
                self._run_normalization(
                    preprocessed_context, group_work_dir, output_dir, reporter
                )

            if self.segmentation.steps or dmri_cfg.get('analysis'):
                self._run_segmentation(
                    preprocessed_context, group_work_dir, output_dir, reporter
                )

            self._update_study_tracker(preprocessed_context, output_dir)
        
        # Generate final report
        reporter.generate()
        try:
            reporter.generate_pdf()
        except Exception as e:
            self.logger.warning(f"PDF generation failed: {e}")
        
        self.logger.info(f"Processing complete for sub-{subject} {ses}. Results in {output_dir}")

    def _find_anat_files(self, subject: str, session: Optional[str], modality: str = 'T1w') -> list[ImageFile]:
        """Find anatomical files using config overrides or standard BIDS search."""
        # Check for custom inputs
        anat_section = self.config.get('anat') or {}
        anat_cfg = anat_section.get('input') or self.config.get('anat_input') or {}
        selector_key = f"{modality.lower()}_match"
        
        path_key = f"{modality.lower()}_path"
        pattern_key = f"{modality.lower()}_search_pattern"
        
        custom_path = anat_cfg.get(path_key)
        custom_pattern = anat_cfg.get(pattern_key)
        selector = anat_cfg.get(selector_key)
        
        # Use custom paths if provided
        if custom_path or custom_pattern:
            files = self._find_custom_files(subject, session, modality, custom_path, custom_pattern)
            return select_anatomical_candidates(files, selector, modality, logger=self.logger)
        
        # Standard BIDS search
        search_dir = self.config.bids_dir / f"sub-{subject}"
        if session:
            search_dir = search_dir / f"ses-{session}"
        search_dir = search_dir / "anat"
        
        if modality == 'T1w':
            files = bids_find_t1w(search_dir)
        elif modality == 'T2w':
            files = bids_find_t2w(search_dir)
        else:
            files = []

        return select_anatomical_candidates(files, selector, modality, logger=self.logger)

    def _find_custom_files(self, subject: str, session: Optional[str], modality: str, 
                          custom_path: Optional[str], custom_pattern: Optional[str]) -> list[ImageFile]:
        """Find files using custom paths or patterns."""
        results = []
        
        if custom_path:
            p = Path(custom_path)
            if p.exists():
                ent = parse_bids_filename(p)
                ent["suffix"] = modality
                results.append(ImageFile(entities=ent, img=p, json=None))
                self.logger.info(f"Using custom anatomical file: {p}")
        
        elif custom_pattern:
            p_str = custom_pattern.format(subject=subject, session=session if session else "")
            pattern = Path(p_str).expanduser()
            if pattern.is_absolute():
                matches = [Path(match) for match in sorted(glob.glob(str(pattern)))]
            else:
                matches = sorted(self.config.bids_dir.glob(p_str))
            
            for m in matches:
                ent = parse_bids_filename(m)
                ent["suffix"] = modality
                json_path = m.with_suffix("").with_suffix(".json")
                results.append(ImageFile(entities=ent, img=m, json=json_path if json_path.exists() else None))
            
            if matches:
                self.logger.info(f"Found {len(matches)} custom anatomical files.")
        
        return results

    def _run_mriqc(self, subject: str, session: Optional[str]):
        """Run MRIQC quality control."""
        qc_cfg = self.config.get("qc", {}).get("mriqc", {})
        target_mods = qc_cfg.get("modalities", ['dwi', 'T1w', 'T2w'])
        
        if target_mods:
            self.logger.info(f"Running MRIQC for sub-{subject} (modalities={target_mods})")
            mriqc_out = self.config.output_dir.parent / "mriqc"
            try:
                run_mriqc(
                    bids_dir=self.config.bids_dir,
                    output_dir=mriqc_out,
                    participant_label=subject,
                    session_id=session,
                    n_procs=self.config.n_cpus,
                    modalities=target_mods,
                    verbose_reports=self.config.verbose
                )
            except Exception as e:
                self.logger.warning(f"MRIQC failed: {e}. Continuing...")
                if self.config.stop_on_error:
                    raise

    def _run_anatomical_preprocessing(self, subject: str, session: Optional[str], 
                                     t1w_files: list, t2w_files: list, 
                                     anat_work_dir: Path, reporter) -> tuple:
        """Run anatomical preprocessing workflow."""
        self.logger.info("="*60)
        self.logger.info("  RUNNING ANATOMICAL PIPELINE  ")
        self.logger.info("="*60)
        
        anat_context = PipelineContext({
            "subject": subject,
            "session": session,
            "t1w_files": t1w_files,
            "t2w_files": t2w_files
        })
        
        anat_final_dir = self._get_output_dir(subject, session) / 'anat'
        anat_results = self.anat_preprocessing.run(
            anat_work_dir, anat_context, 
            final_output_dir=anat_final_dir, 
            reporter=reporter
        )
        
        # Update file lists with preprocessed versions
        pre_t1 = anat_results.get("preprocessed_t1w")
        pre_t2 = anat_results.get("preprocessed_t2w_coreg") or anat_results.get("preprocessed_t2w")
        
        if pre_t1:
            self.logger.info("Using preprocessed T1w as structural reference.")
            t1w_files = [pre_t1]
        
        if pre_t2:
            self.logger.info("Using preprocessed T2w as additional reference.")
            t2w_files = [pre_t2]
        
        return t1w_files, t2w_files

    def _build_initial_context(self, subject: str, session: Optional[str], 
                               dwi_files: list, t1w_files: list, t2w_files: list) -> PipelineContext:
        """Build initial processing context."""
        topup_groups = find_reversed_phase_groups(dwi_files)
        
        return PipelineContext({
            "subject": subject,
            "session": session,
            "current_image": dwi_files[0],
            "dwi_files": dwi_files,
            "topup_groups": topup_groups,
            "t1w_files": t1w_files,
            "t2w_files": t2w_files,
            "study_name": self.config.get('study_name')
        })

    def _run_preprocessing(self, context: dict, work_dir: Path, output_dir: Path, reporter) -> dict:
        """Run preprocessing workflow."""
        self.logger.info("="*60)
        self.logger.info("  RUNNING DIFFUSION PREPROCESSING   ")
        self.logger.info("="*60)

        self.preprocessing.build_pipeline(context)
        
        rerun_from_step = get_rerun_from_step(self.config, "dmri.preprocessing", "preprocessing")
        rerun_hits_preprocessing = any_step_matches(self.preprocessing.steps, rerun_from_step) if self.preprocessing.steps else False
        if self.config.get("skip_existing", True) and not self.config.get("force", False) and not rerun_hits_preprocessing:
            cached = self._load_preprocessed_from_output(context, output_dir)
            if cached:
                self.logger.info(
                    f"⚡ FAST SKIP: Found {len(cached['preprocessed_dwis'])} "
                    f"preprocessed DWI output(s) in {output_dir}"
                )
                context.update(cached)
                context["preprocessing_skipped"] = True
                return context

        # Restore saved step directories before executing the ordered workflow.
        # Each step remains responsible for validating its cached outputs and
        # adopting them into the current context; execution therefore resumes
        # at the first missing, stale, or invalid step.
        self.preprocessing.recover_intermediates(work_dir, output_dir)

        result = self.preprocessing.run(work_dir, context, reporter=reporter)
        if rerun_hits_preprocessing:
            result["preprocessing_rerun_from_step"] = rerun_from_step
        return result

    def _run_modeling(self, context: dict, work_dir: Path, output_dir: Path, reporter):
        """Run modeling workflow."""
        self.logger.info("="*60)
        self.logger.info("  RUNNING MODEL FITTING        ")
        self.logger.info("="*60)
        
        models_out = output_dir / "models"
        models_out.mkdir(parents=True, exist_ok=True)
        
        self.modeling.build_pipeline(context)
        return self.modeling.run(work_dir, context, reporter=reporter, final_output_dir=models_out)

    def _run_normalization(self, context: dict, work_dir: Path, output_dir: Path, reporter):
        """Run normalization workflow."""
        self.logger.info("="*60)
        self.logger.info("  RUNNING NORMALIZATION        ")
        self.logger.info("="*60)
        
        norm_output_dir = output_dir / "normalization"
        norm_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.normalization.build_pipeline(context)
        return self.normalization.run(work_dir, context, reporter=reporter, final_output_dir=norm_output_dir)

    def _run_segmentation(self, context: dict, work_dir: Path, output_dir: Path, reporter):
        """Run segmentation workflow."""
        self.logger.info("="*60)
        self.logger.info("  RUNNING SEGMENTATION         ")
        self.logger.info("="*60)
        
        self.segmentation.build_pipeline(context)
        return self.segmentation.run(work_dir, context, reporter=reporter, final_output_dir=output_dir)

    def _update_study_tracker(self, context: dict, output_dir: Path):
        """Update study tracking system."""
        if not self.config.get('tracker.enabled', False):
            return
        try:
            context['study_name'] = self.config.get('study_name')
            tracking = TrackingStep(self.config, self.logger)
            tracking.run(context, output_dir)
        except Exception as e:
            self.logger.warning(f"Tracker update failed: {e}")

    def _load_preprocessed_from_output(self, context: dict, output_dir: Path) -> Optional[dict]:
        """Load existing preprocessed DWIs from the final output directory."""
        dwi_files = context.get("dwi_files", [])
        if not dwi_files:
            return None

        subject = context.get("subject")
        session = context.get("session")
        io_manager = DataIOManager(self.config, self.logger)

        preprocessed_dwis = []
        preprocessed_masks = []
        gnl_map_by_image = {}
        gnl_maps = []
        missing = 0
        stale_gnl = False
        expect_merged = self._should_expect_merged_preprocessed_dwi(context)
        expected_dwis = [dwi_files[0]] if expect_merged else dwi_files
        expected_entities = io_manager.derivative_entities_for_dwis(
            expected_dwis,
            context,
            desc="preproc",
            suffix="dwi",
        )

        for dwi, ents in zip(expected_dwis, expected_entities):
            ents = ents.copy()
            if expect_merged:
                ents.pop("dir", None)
            fname = build_bids_name(ents)
            if not fname.endswith(".nii.gz"):
                fname += ".nii.gz"

            target_img = output_dir / fname
            cached_dwi = reuse_path_if_exists(target_img, ents)
            if cached_dwi is None:
                missing += 1
                continue

            bval = io_manager._find_sidecar_for_image(target_img, ".bval", [output_dir])
            bvec = io_manager._find_sidecar_for_image(target_img, ".bvec", [output_dir])
            jsn = io_manager._find_sidecar_for_image(target_img, ".json", [output_dir])
            Delta = (
                io_manager._find_sidecar_for_image(target_img, ".bigdelta", [output_dir])
                or io_manager._find_sidecar_for_image(target_img, ".Delta", [output_dir])
            )
            delta = io_manager._find_sidecar_for_image(target_img, ".delta", [output_dir])

            preprocessed_dwis.append(
                DWIFile(img=cached_dwi.img, bval=bval, bvec=bvec, json=jsn, Delta=Delta, delta=delta, entities=ents)
            )

            gnl_path = self._find_saved_gnl_tensor(target_img, ents)
            if gnl_path:
                if self._gnl_matches_image_grid(gnl_path, target_img):
                    gnl_map_by_image[target_img] = gnl_path
                    gnl_map_by_image[str(target_img)] = gnl_path
                    if gnl_path not in gnl_maps:
                        gnl_maps.append(gnl_path)
                else:
                    stale_gnl = True

            mask_path = io_manager._find_mask_for_image(target_img, [output_dir])
            if mask_path:
                mask_ents = ents.copy()
                mask_ents["suffix"] = "mask"
                preprocessed_masks.append(ImageFile(img=mask_path, json=None, entities=mask_ents))
            else:
                preprocessed_masks.append(None)

        if missing and preprocessed_dwis:
            self.logger.info(
                f"Found {len(preprocessed_dwis)} preprocessed DWI outputs, "
                f"but {missing} expected file(s) were missing"
            )
            return None

        if not preprocessed_dwis:
            if expect_merged:
                return None
            candidates = sorted(output_dir.glob("*desc-preproc*_dwi.nii*"))
            for candidate in candidates:
                try:
                    ents = parse_bids_filename(candidate)
                except Exception:
                    ents = {"sub": subject, "ses": session, "suffix": "dwi", "desc": "preproc"}

                bval = io_manager._find_sidecar_for_image(candidate, ".bval", [output_dir])
                bvec = io_manager._find_sidecar_for_image(candidate, ".bvec", [output_dir])
                jsn = io_manager._find_sidecar_for_image(candidate, ".json", [output_dir])
                Delta = (
                    io_manager._find_sidecar_for_image(candidate, ".bigdelta", [output_dir])
                    or io_manager._find_sidecar_for_image(candidate, ".Delta", [output_dir])
                )
                delta = io_manager._find_sidecar_for_image(candidate, ".delta", [output_dir])
                preprocessed_dwis.append(
                    DWIFile(img=candidate, bval=bval, bvec=bvec, json=jsn, Delta=Delta, delta=delta, entities=ents)
                )

                gnl_path = self._find_saved_gnl_tensor(candidate, ents)
                if gnl_path:
                    if self._gnl_matches_image_grid(gnl_path, candidate):
                        gnl_map_by_image[candidate] = gnl_path
                        gnl_map_by_image[str(candidate)] = gnl_path
                        if gnl_path not in gnl_maps:
                            gnl_maps.append(gnl_path)
                    else:
                        stale_gnl = True

                mask_path = io_manager._find_mask_for_image(candidate, [output_dir])
                if mask_path:
                    mask_ents = ents.copy()
                    mask_ents["suffix"] = "mask"
                    preprocessed_masks.append(ImageFile(img=mask_path, json=None, entities=mask_ents))
                else:
                    preprocessed_masks.append(None)

        if not preprocessed_dwis:
            return None

        if stale_gnl:
            self.logger.warning(
                "Cached preprocessed DWI outputs were found, but one or more saved GNL tensors "
                "do not match the cached DWI grid. Re-running preprocessing to refresh the "
                "final-space GNL tensor export."
            )
            return None

        cached = {
            "preprocessed_dwis": preprocessed_dwis,
            "preprocessed_masks": preprocessed_masks,
            "current_image": preprocessed_dwis[0]
        }
        if gnl_map_by_image:
            current_gnl = (
                gnl_map_by_image.get(preprocessed_dwis[0].img)
                or gnl_map_by_image.get(str(preprocessed_dwis[0].img))
            )
            cached["gnl_map_by_image"] = gnl_map_by_image
            cached["gnl_maps"] = gnl_maps
            cached["gnl_map"] = current_gnl or gnl_maps[0]
        return cached

    def _should_expect_merged_preprocessed_dwi(self, context: dict) -> bool:
        dwi_files = context.get("dwi_files", [])
        if len(dwi_files) <= 1:
            return False

        dmri_cfg = (self.config.get("dmri") or {}).get("preprocessing", {})
        merge_cfg = dmri_cfg.get("merging", {}) or {}
        if merge_cfg.get("enabled", True):
            return True

        distcorr_cfg = dmri_cfg.get("distcorr", {}) or {}
        dist_method = distcorr_cfg.get("method", "none")
        fallback = distcorr_cfg.get("fallback", False)
        has_topup_groups = bool(context.get("topup_groups"))
        has_t1w = bool(context.get("t1w_files"))

        if dist_method in {"topup", "topup+drbuddi"}:
            return has_topup_groups or (fallback and has_t1w)
        if dist_method == "drbuddi":
            return has_topup_groups
        if dist_method == "synb0":
            return has_t1w
        return False

    def _find_saved_gnl_tensor(self, dwi_img: Path, entities: Optional[dict] = None) -> Optional[Path]:
        """Find the canonical saved GNL tensor that corresponds to a cached preprocessed DWI."""
        image_entities = (entities or {}).copy()
        if image_entities:
            image_entities["desc"] = "gnl_tensor"
            image_entities["suffix"] = "dwi"
            cached_gnl = reuse_if_exists(image_entities, dwi_img.parent)
            if cached_gnl is not None:
                return cached_gnl.img

        siblings = sorted(dwi_img.parent.glob("*desc-gnl_tensor*_dwi.nii.gz"))
        if len(siblings) == 1:
            return siblings[0]
        return None

    def _gnl_matches_image_grid(self, gnl_img: Path, dwi_img: Path) -> bool:
        """Check whether a saved GNL tensor is on the same grid as the cached DWI."""
        try:
            gnl = nib.load(str(gnl_img))
            dwi = nib.load(str(dwi_img))
            return gnl.shape[:3] == dwi.shape[:3] and np.allclose(gnl.affine, dwi.affine, atol=1e-4)
        except Exception as exc:
            self.logger.warning(f"Could not validate cached GNL tensor grid for {gnl_img.name}: {exc}")
            return False


# Example usage
if __name__ == '__main__':
    # Create configuration
    config = PipelineConfig(
        bids_dir='/data/my_study',
        output_dir='/data/derivatives/qmri-neuropipe',
        n_cpus=8,
        skip_existing=True,
        log_level='INFO'
    )
    
    # Create and run pipeline
    pipeline = DMRIPipeline(config)
    pipeline.run()  # Processes all subjects automatically
