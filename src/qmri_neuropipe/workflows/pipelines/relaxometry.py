import logging
from pathlib import Path
from typing import Optional, List, Dict
import shutil
import json
from dataclasses import dataclass, field

from ...core import BaseWorkflow, PipelineConfig
from ...core.types import ImageFile
from ...io.bids import build_bids_name, get_entities_from_path


# Import Steps
from ...lib.relax.motion import SPGRMotionCorrectionStep
from ...lib.relax.b1 import B1MappingStep
from ...lib.common.reorient import ReorientStep
from ...lib.common.denoise import DenoisingStep
from ...lib.common.mask import BrainMaskingStep
from ...lib.common.gibbs import GibbsUnringingStep
from ...lib.common.stats import ROIStatsStep
from ...lib.dmri.analysis import AtlasRegistrationStep, StatsExtractionStep
from ...interfaces.relaxometry import fit_despot1, fit_despot1_hifi, fit_despot2, fit_despot2_fm
from ...utils.relax_params import generate_acq_params


@dataclass
class RelaxometryPreprocConfig:
    reorient: Dict = field(default_factory=lambda: {"enabled": False})
    denoising: Dict = field(default_factory=lambda: {"enabled": False, "method": "mrtrix"})
    degibbs: Dict = field(default_factory=lambda: {"enabled": False, "method": "mrtrix"})
    motion_correction: Dict = field(default_factory=lambda: {"enabled": False, "method": "ants"})
    b1: Dict = field(default_factory=lambda: {"method": "afi", "smoothing_fwhm": 0.0})
    brain_masking: Dict = field(default_factory=dict)


@dataclass
class RelaxometryModelingConfig:
    despot1: Dict = field(default_factory=lambda: {"enabled": False, "use_hifi": True})
    despot2: Dict = field(default_factory=lambda: {"enabled": False})
    mcdespot: Dict = field(default_factory=lambda: {"enabled": False})


@dataclass
class RelaxometryQCConfig:
    enabled: bool = False


@dataclass
class RelaxometryConfig:
    preprocessing: RelaxometryPreprocConfig = field(default_factory=RelaxometryPreprocConfig)
    modeling: RelaxometryModelingConfig = field(default_factory=RelaxometryModelingConfig)
    qc: RelaxometryQCConfig = field(default_factory=RelaxometryQCConfig)
    masking: Dict = field(default_factory=dict)


class RelaxometryWorkflow(BaseWorkflow):
    """
    Pipeline for processing Relaxometry (DESPOT1/DESPOT2) data.
    """

    def __init__(self, config: PipelineConfig, logger: logging.Logger, provenance: dict,
                 relax_config: Optional[RelaxometryConfig] = None):
        self.relax_config = relax_config or RelaxometryConfig()
        super().__init__(config, logger, provenance)
        self.modality = "Relaxometry"

    def _initialize_steps(self):
        preproc_cfg = self.relax_config.preprocessing

        # 0. Reorientation
        if preproc_cfg.reorient.get("enabled", False):
            self.add_step(ReorientStep(self.config, self.logger, self.provenance))

        # 1. Denoising
        den_cfg = preproc_cfg.denoising
        if den_cfg.get("enabled", False):
            self.add_step(
                DenoisingStep(self.config, self.logger, self.provenance, method=den_cfg.get("method", "mrtrix"))
            )

        # 2. Gibbs Ringing
        gibbs_cfg = preproc_cfg.degibbs
        if gibbs_cfg.get("enabled", False):
            self.add_step(
                GibbsUnringingStep(self.config, self.logger, self.provenance, method=gibbs_cfg.get("method", "mrtrix"))
            )

        # 3. Motion Correction
        moco_cfg = preproc_cfg.motion_correction
        if moco_cfg.get("enabled", False):
            # Extract extra options (exclude 'enabled' and 'method')
            moco_opts = {k: v for k, v in moco_cfg.items() if k not in ['enabled', 'method']}
            self.add_step(
                SPGRMotionCorrectionStep(
                    self.config, self.logger, self.provenance, method=moco_cfg.get("method", "ants"), options=moco_opts
                )
            )

        # 4. B1 Mapping
        b1_cfg = preproc_cfg.b1
        self.add_step(
            B1MappingStep(
                self.config,
                self.logger,
                self.provenance,
                method=b1_cfg.get("method", "afi"),
                smoothing_fwhm=b1_cfg.get("smoothing_fwhm", 0.0),
            )
        )

        # 5. Post-Processing: Atlas Registration & Stats
        # These are used optionally in run(), but we can initialize them here
        self.add_step(AtlasRegistrationStep(self.config, self.logger, self.provenance))
        self.add_step(StatsExtractionStep(self.config, self.logger, self.provenance))

    def _parse_inputs(self, context: dict):
        relax_files: List[ImageFile] = context.get("relax_files", [])

        spgr_files = []
        ssfp_files = []
        irspgr_files = []
        b1_files = []  # Map or AFI source

        for f in relax_files:
            # Check entities
            acq = f.entities.get("acq", "").lower()
            desc = f.entities.get("desc", "").lower()
            suffix = f.entities.get("suffix", "").lower()

            # SPGR detection
            if "spgr" in acq or "spgr" in desc:
                if "ir" in acq or "ir" in desc:
                    irspgr_files.append(f)
                else:
                    spgr_files.append(f)
            # SSFP detection
            elif "ssfp" in acq or "ssfp" in desc:
                ssfp_files.append(f)
            # B1 Map/AFI detection
            elif "afi" in acq or "afi" in desc or "b1" in suffix:
                b1_files.append(f)

        spgr_files.sort(key=lambda x: str(x.img))
        ssfp_files.sort(key=lambda x: str(x.img))
        return spgr_files, ssfp_files, irspgr_files, b1_files

    def _setup_directories(
        self, output_dir: Path, context: dict
    ) -> Dict[str, Path]:
        """
        Setup anat_out_dir, fmap_out_dir, intermediate_dir with respect to config work_dir and subject/session.
        """
        anat_out_dir = output_dir / "anat"
        fmap_out_dir = output_dir / "fmap"
        anat_out_dir.mkdir(parents=True, exist_ok=True)
        fmap_out_dir.mkdir(parents=True, exist_ok=True)

        work_dir = self.config.work_dir
        if work_dir:
            subj = context.get("subject")
            sess = context.get("session")
            wd_subj = work_dir / f"sub-{subj}"
            if sess:
                wd_subj = wd_subj / f"ses-{sess}"
            intermediate_dir = wd_subj / "anat" / "intermediate"
        else:
            intermediate_dir = anat_out_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        return dict(
            anat_out_dir=anat_out_dir,
            fmap_out_dir=fmap_out_dir,
            intermediate_dir=intermediate_dir,
        )

    def _preprocess_images(self, img_list: List[ImageFile], modality_name: str, output_dir: Path) -> List[ImageFile]:
        """
        Apply preprocessing steps (reorient, denoise, degibbs) to a list of images.
        """
        processed_images = []
        for img in img_list:
            curr = img
            for step in self.steps:
                if isinstance(step, (DenoisingStep, GibbsUnringingStep, ReorientStep)):
                    curr = step(curr, output_dir=output_dir)
            processed_images.append(curr)
        return processed_images

    def _select_reference(self, spgr_pre: List[ImageFile]) -> ImageFile:
        """
        Select the reference image from preprocessed SPGR images (max FlipAngle).
        """
        from ...utils.relax_params import _extract_bids_param

        ref_img = spgr_pre[0]  # Default
        max_fa = -1
        for img in spgr_pre:
            fa = _extract_bids_param(img, "FlipAngle", 0.0)
            if isinstance(fa, list):
                fa = max(fa) if fa else 0.0
            if float(fa) > max_fa:
                max_fa = float(fa)
                ref_img = img
        return ref_img

    def _run_motion_correction(
        self,
        spgr_pre: List[ImageFile],
        ssfp_pre: List[ImageFile],
        ir_pre: List[ImageFile],
        anat_out_dir: Path,
        intermediate_dir: Path,
        ref_img: ImageFile,
    ):
        """
        Run motion correction step on SPGR, SSFP, and IR images.
        """
        moco_step = next((s for s in self.steps if isinstance(s, SPGRMotionCorrectionStep)), None)

        spgr_moco = spgr_pre
        ssfp_moco = ssfp_pre
        ir_moco = None

        if moco_step:
            spgr_moco = moco_step(spgr_pre, output_dir=anat_out_dir, reference_image=ref_img, modality="SPGR")
            if ssfp_pre:
                ssfp_moco = moco_step(ssfp_pre, output_dir=anat_out_dir, reference_image=ref_img, modality="SSFP")
            if ir_pre:
                ir_moco = moco_step(ir_pre, output_dir=intermediate_dir, reference_image=ref_img, modality="IR-SPGR")
        return spgr_moco, ssfp_moco, ir_moco

    def _run_brain_masking(
        self,
        ref_img: ImageFile,
        anat_out_dir: Path,
        intermediate_dir: Path,
        context: dict,
        preproc_cfg: RelaxometryPreprocConfig,
        relax_cfg: RelaxometryConfig,
    ):
        """
        Run brain masking on the reference image, managing output paths and context.
        """
        mask_cfg = preproc_cfg.brain_masking or relax_cfg.masking

        mask_method = mask_cfg.get("method", "fsl")
        mask_step = next((s for s in self.steps if isinstance(s, BrainMaskingStep)), None)

        if not mask_step:
            mask_step = BrainMaskingStep(self.config, self.logger, self.provenance, method=mask_method)

        subj = context.get("subject")
        sess = context.get("session")
        base_prefix = f"sub-{subj}"
        if sess:
            base_prefix += f"_ses-{sess}"

        target_mask_name = anat_out_dir / f"{base_prefix}_desc-brain-mask.nii.gz"

        if target_mask_name.exists():
            self.logger.info(f"Skipping Brain Masking (Exists): {target_mask_name}")
            mask_file = target_mask_name
            context["brain_mask"] = mask_file
        else:
            self.logger.info(f"Running Brain Masking on Reference: {ref_img.img.name}")
            masked_ref, mask_obj = mask_step(ref_img, output_dir=intermediate_dir, return_mask=True)
            if mask_obj.img != target_mask_name:
                shutil.move(mask_obj.img, target_mask_name)
                mask_file = target_mask_name
            else:
                mask_file = mask_obj.img
            context["brain_mask"] = mask_file
            self.logger.info(f"Generated Brain Mask: {mask_file}")

    def _generate_params(
        self,
        spgr_moco: List[ImageFile],
        ssfp_moco: List[ImageFile],
        ir_final: List[ImageFile],
        anat_out_dir: Path,
        base_prefix: str,
    ) -> Path:
        """
        Generate acquisition parameters JSON file and return the path.
        """
        params_name = f"{base_prefix}_desc-AcqParams.json"
        params_json = anat_out_dir / params_name
        generate_acq_params(spgr_moco, ssfp_moco, ir_final, output_path=params_json)
        return params_json

    def _run_b1_mapping(
        self,
        b1_files: List[ImageFile],
        b1_step: Optional[B1MappingStep],
        ref_img: ImageFile,
        fmap_out_dir: Path,
        fmap_inter_dir: Path,
    ) -> Optional[ImageFile]:
        """
        Run B1 mapping if B1 files and step exist, with resume check.
        """
        b1_map = None
        if b1_files and b1_step:
            curr_b1 = b1_files[0]
            b1_ref = None
            fmap_inter_dir.mkdir(parents=True, exist_ok=True)

            # Resume check for existing B1 map
            existing_b1 = list(fmap_out_dir.glob("*TB1map.nii.gz"))
            if existing_b1 and not self.config.get("force_rerun", False):
                self.logger.info(f"Found existing B1 Map: {existing_b1[0].name}")
                b1_map = ImageFile(img=existing_b1[0], entities={})
            else:
                b1_map_inter = b1_step(
                    curr_b1, reference_image=ref_img, output_dir=fmap_inter_dir, b1_ref_image=b1_ref
                )
                final_path = fmap_out_dir / b1_map_inter.img.name
                shutil.move(b1_map_inter.img, final_path)
                b1_map = ImageFile(img=final_path, entities=b1_map_inter.entities)
        return b1_map

    def _run_model_fitting(
        self,
        context: dict,
        spgr_moco: List[ImageFile],
        ssfp_moco: List[ImageFile],
        ir_final: List[ImageFile],
        params_json: Path,
        fit_out_dir: Path,
        mask_file: Optional[Path],
        b1_map: Optional[ImageFile],
        base_prefix: str,
    ) -> Dict[str, ImageFile]:
        """
        Run model fitting for DESPOT1, DESPOT1 HIFI, DESPOT2, mcDESPOT if enabled.

        Returns a dictionary of fitted map ImageFiles keyed by map name.
        """
        from ...interfaces.fsl import merge as fslmerge

        results = {}
        modeling_cfg = self.relax_config.modeling
        despot1_results: Dict[str, Path] = {}

        def _mcdespot_cfg() -> Dict:
            mcdespot_cfg = dict(getattr(modeling_cfg, "mcdespot", {}) or {})
            legacy_enabled = bool(getattr(modeling_cfg, "despot2", {}).get("mcdespot", False))
            if legacy_enabled:
                self.logger.warning(
                    "relaxometry.modeling.despot2.mcdespot is deprecated. "
                    "Use relaxometry.modeling.mcdespot.enabled instead."
                )
                mcdespot_cfg.setdefault("enabled", True)
            return mcdespot_cfg

        # Ensure output directory exists
        fit_out_dir.mkdir(parents=True, exist_ok=True)

        input_dir = fit_out_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        def _stack_inputs(images: List[ImageFile], stem: str) -> Optional[Path]:
            if not images:
                return None
            if len(images) == 1:
                return images[0].img
            merged_path = input_dir / f"{base_prefix}_{stem}.nii.gz"
            return fslmerge(images, merged_path, dimension="t")

        spgr_stack = _stack_inputs(spgr_moco, "desc-spgrStack_VFA")
        ssfp_stack = _stack_inputs(ssfp_moco, "desc-ssfpStack_VFA")
        irspgr_stack = _stack_inputs(ir_final, "desc-irspgrStack_VFA")
        b1_path = b1_map.img if isinstance(b1_map, ImageFile) else b1_map

        if spgr_stack is None:
            raise ValueError("DESPOT fitting requires at least one SPGR image.")

        # DESPOT1 fitting
        if modeling_cfg.despot1.get("enabled", False):
            self.logger.info("Starting DESPOT1 fitting.")
            use_hifi = modeling_cfg.despot1.get("use_hifi", True)
            if use_hifi:
                if irspgr_stack is None:
                    raise ValueError("DESPOT1-HIFI requested, but no IR-SPGR image was found.")
                despot1_results = fit_despot1_hifi(
                    spgr_file=spgr_stack,
                    irspgr_file=irspgr_stack,
                    params_file=params_json,
                    out_dir=fit_out_dir,
                    mask_file=mask_file,
                    out_base=f"{base_prefix}_despot1_hifi",
                )
            else:
                despot1_results = fit_despot1(
                    spgr_file=spgr_stack,
                    params_file=params_json,
                    out_dir=fit_out_dir,
                    b1_file=b1_path,
                    mask_file=mask_file,
                    out_base=f"{base_prefix}_despot1",
                )
            results.update(despot1_results)

        # DESPOT2 fitting
        if modeling_cfg.despot2.get("enabled", False):
            self.logger.info("Starting DESPOT2 fitting.")
            if ssfp_stack is None:
                raise ValueError("DESPOT2 requested, but no SSFP image was found.")
            t1_path = despot1_results.get("t1") if modeling_cfg.despot1.get("enabled", False) else None
            if not t1_path:
                raise ValueError("DESPOT2 requires a DESPOT1 T1 map, but none was produced.")
            despot_b1_path = despot1_results.get("b1") if modeling_cfg.despot1.get("enabled", False) else None
            if not despot_b1_path:
                despot_b1_path = b1_path
            if not despot_b1_path:
                raise ValueError("DESPOT2 requires a B1 map, but none was available from AFI/external B1 or DESPOT1-HIFI.")
            despot2_results = fit_despot2(
                ssfp_file=ssfp_stack,
                t1_file=t1_path,
                b1_file=despot_b1_path,
                params_file=params_json,
                out_dir=fit_out_dir,
                mask_file=mask_file,
                out_base=f"{base_prefix}_despot2",
            )
            results.update(despot2_results)

        mcdespot_cfg = _mcdespot_cfg()
        if mcdespot_cfg.get("enabled", False):
            self.logger.info("Starting mcDESPOT fitting.")
            if ssfp_stack is None:
                raise ValueError("mcDESPOT requested, but no SSFP image was found.")
            t1_path = despot1_results.get("t1") if modeling_cfg.despot1.get("enabled", False) else None
            if not t1_path:
                raise ValueError("mcDESPOT requires a DESPOT1 T1 map, but none was produced.")
            despot_b1_path = despot1_results.get("b1") if modeling_cfg.despot1.get("enabled", False) else None
            if not despot_b1_path:
                despot_b1_path = b1_path
            if not despot_b1_path:
                raise ValueError("mcDESPOT requires a B1 map, but none was available from AFI/external B1 or DESPOT1-HIFI.")
            mcdespot_results = fit_despot2_fm(
                ssfp_file=ssfp_stack,
                t1_file=t1_path,
                b1_file=despot_b1_path,
                params_file=params_json,
                out_dir=fit_out_dir,
                mask_file=mask_file,
                out_base=f"{base_prefix}_mcdespot",
            )
            results.update(mcdespot_results)

        context["fitted_maps"] = results
        return results

    def _run_postprocessing_and_stats(
        self,
        context: dict,
        fit_out_dir: Path,
        output_dir: Path,
        ref_img: ImageFile,
        t1w_anat: Optional[ImageFile],
        maps: Dict[str, ImageFile],
        base_prefix: str,
    ) -> Dict[str, Path]:
        """
        Run post-processing steps: coregistration, resampling, atlas registration,
        and ROI statistics extraction. Returns dictionary with paths to stats outputs.
        """
        stats_results = {}

        # Coregistration: maps to anatomical space and T1w space
        atlas_reg_step = next((s for s in self.steps if isinstance(s, AtlasRegistrationStep)), None)
        stats_extract_step = next((s for s in self.steps if isinstance(s, StatsExtractionStep)), None)

        coreg_maps = {}
        if maps:
            for map_name, map_img in maps.items():
                # Coregister fitted map to T1w native space if T1w provided
                if t1w_anat:
                    self.logger.info(f"Coregistering {map_name} to T1w space.")
                    coreg_t1w = atlas_reg_step.coregister_to_t1w(map_img, t1w_anat, output_dir=fit_out_dir)
                    coreg_maps[f"{map_name}_coreg_t1w"] = coreg_t1w
                # Coregister fitted map to anatomical reference space
                self.logger.info(f"Coregistering {map_name} to reference anatomical space.")
                coreg_anat = atlas_reg_step.coregister_to_anat(map_img, ref_img, output_dir=fit_out_dir)
                coreg_maps[f"{map_name}_coreg_anat"] = coreg_anat

        # Merge coregistered maps into context
        context["coregistered_maps"] = coreg_maps

        # Atlas registration and stats extraction on coregistered maps
        if atlas_reg_step and stats_extract_step and coreg_maps:
            self.logger.info("Running atlas registration and ROI statistics extraction.")
            atlas_reg_step.run(output_dir=fit_out_dir, reference=ref_img)
            stats_paths = stats_extract_step.run(
                input_maps=list(coreg_maps.values()),
                output_dir=fit_out_dir,
                prefix=base_prefix,
            )
            stats_results.update(stats_paths)
            context["roi_stats"] = stats_paths

        # Quality Control output if enabled
        if self.relax_config.qc.enabled:
            try:
                from ...lib.reporting.reporting_qc import generate_relaxometry_qc
                qc_path = fit_out_dir / f"{base_prefix}_qc_report.html"
                generate_relaxometry_qc(fit_out_dir, output_file=qc_path)
                context["qc_report"] = qc_path
                self.logger.info(f"Generated QC report at {qc_path}")
            except Exception as e:
                self.logger.warning(f"Failed to generate QC report: {e}")

        return stats_results

    def run(
        self,
        output_dir: Path,
        subject: str,
        session: Optional[str] = None,
        context: Optional[dict] = None,
        final_output_dir: Optional[Path] = None,
        reporter=None,
        **kwargs,
    ) -> dict:
        """
        Run the Relaxometry workflow for a given subject/session.
        """
        self.logger.info(f"Starting RelaxometryWorkflow for sub-{subject} ses-{session or 'n/a'}")

        if context is None:
            context = {}

        context.setdefault("subject", subject)
        context.setdefault("session", session)
        context.setdefault("relax_files", [])

        # Parse inputs
        spgr_files, ssfp_files, irspgr_files, b1_files = self._parse_inputs(context)

        self.logger.info(
            f"Found inputs: {len(spgr_files)} SPGR, {len(ssfp_files)} SSFP, {len(irspgr_files)} IR-SPGR, {len(b1_files)} B1."
        )
        if not spgr_files:
            raise ValueError("No SPGR images found. Cannot proceed with Relaxometry.")

        # Setup directories
        dirs = self._setup_directories(output_dir, context)
        anat_out_dir = dirs["anat_out_dir"]
        fmap_out_dir = dirs["fmap_out_dir"]
        intermediate_dir = dirs["intermediate_dir"]

        relax_cfg = self.relax_config
        preproc_cfg = relax_cfg.preprocessing

        # Preprocess images
        spgr_pre = self._preprocess_images(spgr_files, "SPGR", intermediate_dir)
        ssfp_pre = self._preprocess_images(ssfp_files, "SSFP", intermediate_dir)
        ir_pre = self._preprocess_images(irspgr_files, "IRSPGR", intermediate_dir)

        # Select reference image
        ref_img = self._select_reference(spgr_pre)
        context["relax_reference"] = ref_img
        self.logger.info(f"Selected Relaxometry Reference (Preprocessed): {ref_img.img.name}")

        # Motion Correction
        spgr_moco, ssfp_moco, ir_moco = self._run_motion_correction(
            spgr_pre, ssfp_pre, ir_pre, anat_out_dir, intermediate_dir, ref_img
        )
        context["processed_spgr"] = spgr_moco
        context["processed_ssfp"] = ssfp_moco

        # Brain Masking
        self._run_brain_masking(ref_img, anat_out_dir, intermediate_dir, context, preproc_cfg, relax_cfg)

        # Parameter Generation
        ir_final = ir_moco if ir_moco is not None else ir_pre
        subj = context.get("subject")
        sess = context.get("session")
        base_prefix = f"sub-{subj}"
        if sess:
            base_prefix += f"_ses-{sess}"
        params_json = self._generate_params(spgr_moco, ssfp_moco, ir_final, anat_out_dir, base_prefix)

        # B1 Mapping
        b1_step = next((s for s in self.steps if isinstance(s, B1MappingStep)), None)
        fmap_inter_dir = fmap_out_dir / "intermediate"
        b1_map = self._run_b1_mapping(b1_files, b1_step, ref_img, fmap_out_dir, fmap_inter_dir)
        if b1_map:
            context["b1_map"] = b1_map

        # Model fitting
        fit_out_dir = anat_out_dir / "fit"
        fit_maps = self._run_model_fitting(
            context,
            spgr_moco,
            ssfp_moco,
            ir_final,
            params_json,
            fit_out_dir,
            context.get("brain_mask", None),
            b1_map,
            base_prefix,
        )

        # Post-processing, coregistration, atlas registration, stats extraction
        t1w_anat: Optional[ImageFile] = context.get("t1w_file", None)
        stats_results = self._run_postprocessing_and_stats(
            context, fit_out_dir, output_dir, ref_img, t1w_anat, fit_maps, base_prefix
        )

        # Save intermediates if requested
        save_inter = self.config.get("save_intermediates", False)
        if save_inter:
            final_inter_dir = anat_out_dir / "intermediate"
            if intermediate_dir != final_inter_dir and intermediate_dir.exists():
                self.logger.info(f"Saving intermediate files to {final_inter_dir}")
                try:
                    if not final_inter_dir.exists():
                        final_inter_dir.mkdir(parents=True)
                    shutil.copytree(intermediate_dir, final_inter_dir, dirs_exist_ok=True)
                except Exception as e:
                    self.logger.warning(f"Failed to copy intermediates: {e}")

        # Compose final results dictionary to return
        results = {
            "context": context,
            "fitted_maps": fit_maps,
            "roi_stats": stats_results,
            "brain_mask": context.get("brain_mask", None),
            "b1_map": b1_map,
            "qc_report": context.get("qc_report", None),
            "reference_image": ref_img,
        }
        return results


from ...core import BasePipeline
from ...io.relax.bids import bids_find_relax
from ...lib.reporting.report import ReportGenerator


class RelaxometryPipeline(BasePipeline):
    """
    Relaxometry Processing Pipeline (DESPOT1/2).
    """

    @property
    def name(self):
        return "relaxometry-pipeline"

    @property
    def version(self):
        return "1.0.0"

    def _initialize_pipeline(self):
        relax_config_dict = self.config.get("relaxometry", {})
        # Create RelaxometryConfig from dict if possible
        relax_config = RelaxometryConfig(
            preprocessing=RelaxometryPreprocConfig(**relax_config_dict.get("preprocessing", {})),
            modeling=RelaxometryModelingConfig(**relax_config_dict.get("modeling", {})),
            qc=RelaxometryQCConfig(**relax_config_dict.get("qc", {})),
            masking=relax_config_dict.get("masking", {}),
        )
        self.workflow = RelaxometryWorkflow(self.config, self.logger, self.provenance, relax_config=relax_config)

    def _should_skip(self, subject: str, session: Optional[str]) -> bool:
        return False

    def process_subject(self, subject: str, session: Optional[str]):
        ses_str = f"ses-{session}" if session else ""
        subj_dir = Path(self.config.bids_dir) / f"sub-{subject}"
        if session:
            subj_dir = subj_dir / ses_str

        if not subj_dir.exists():
            self.logger.warning(f"Subject directory not found: {subj_dir}")
            return

        output_dir = self._get_output_dir(subject, session)
        output_dir.mkdir(parents=True, exist_ok=True)

        relax_files = bids_find_relax(subj_dir)

        if not relax_files:
            self.logger.warning(f"No relaxometry files found for sub-{subject} {ses_str}. Skipping.")
            return

        self.logger.info(f"Found {len(relax_files)} relaxometry files for sub-{subject}")

        t1w_files = list((subj_dir / "anat").glob("*_T1w.nii.gz"))
        t1w_file = ImageFile(img=t1w_files[0], entities={}) if t1w_files else None

        context = {
            "relax_files": relax_files,
            "t1w_file": t1w_file,
            "subject": subject,
            "session": session,
        }

        report_title = f"QMRI-Neuropipe Report: sub-{subject} {ses_str}"
        reporter = ReportGenerator(output_dir.parent, title=report_title)

        part_summ = f"Participant: sub-{subject}"
        if session:
            part_summ += f", Session: {session}"
        reporter.set_participant_summary(
            part_summ,
            details={
                "Subject": subject,
                "Session": session or "N/A",
                "BIDS Path": str(self.config.bids_dir),
                "Output Path": str(self.config.output_dir),
            },
        )

        try:
            self.workflow.run(output_dir, subject, session, context=context, final_output_dir=output_dir, reporter=reporter)
        except Exception as e:
            self.logger.error(f"Error processing sub-{subject}: {e}")
            if self.config.stop_on_error:
                raise e

def run_relaxometry_workflow(
    config: PipelineConfig,
    output_dir: Path,
    subject: str,
    session: Optional[str] = None,
    relax_config: Optional[RelaxometryConfig] = None,
    reporter=None,
    **kwargs,
) -> dict:
    """
    Convenience function to run relaxometry workflow standalone.
    """
    logger = logging.getLogger("RelaxometryWorkflow")
    provenance = {}
    workflow = RelaxometryWorkflow(config, logger, provenance, relax_config=relax_config)
    context = kwargs.pop("context", {})
    return workflow.run(output_dir, subject, session, context=context, reporter=reporter, **kwargs)
