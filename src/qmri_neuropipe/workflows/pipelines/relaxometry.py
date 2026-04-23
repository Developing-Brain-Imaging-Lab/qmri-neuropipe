import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
import json
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

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
from ...lib.dmri.normalization import NormalizationStep
from ...lib.dmri.analysis import AtlasRegistrationStep, StatsExtractionStep
from ...interfaces.relaxometry import fit_despot1, fit_despot1_hifi, fit_despot2, fit_despot2_fm, fit_mcdespot
from ...utils.relax_params import generate_acq_params


@dataclass
class RelaxometryPreprocConfig:
    reorient: Dict = field(default_factory=lambda: {"enabled": False})
    denoising: Dict = field(default_factory=lambda: {"enabled": False, "method": "mrtrix"})
    degibbs: Dict = field(default_factory=lambda: {"enabled": False, "method": "mrtrix"})
    motion_correction: Dict = field(default_factory=lambda: {"enabled": False, "method": "ants"})
    b1: Dict = field(default_factory=lambda: {"method": "afi", "smoothing_fwhm": 0.0})
    brain_masking: Dict = field(default_factory=dict)
    exclude_indices: Dict = field(default_factory=dict)


@dataclass
class RelaxometryModelingConfig:
    despot1: Dict = field(default_factory=lambda: {"enabled": False, "use_hifi": False})
    despot2: Dict = field(default_factory=lambda: {"enabled": False})
    despot2fm: Dict = field(default_factory=lambda: {"enabled": False})
    mcdespot: Dict = field(default_factory=lambda: {"enabled": False, "cuda": False})


@dataclass
class RelaxometryQCConfig:
    enabled: bool = False


@dataclass
class RelaxometryConfig:
    preprocessing: RelaxometryPreprocConfig = field(default_factory=RelaxometryPreprocConfig)
    modeling: RelaxometryModelingConfig = field(default_factory=RelaxometryModelingConfig)
    qc: RelaxometryQCConfig = field(default_factory=RelaxometryQCConfig)
    masking: Dict = field(default_factory=dict)
    normalization: Dict = field(default_factory=lambda: {"enabled": False})
    analysis: Dict = field(default_factory=lambda: {"enabled": False})


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
        norm_cfg = self.relax_config.normalization or {}
        analysis_cfg = self.relax_config.analysis or {}

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
                registration=b1_cfg.get("registration", {}),
            )
        )

        # 5. Standard-Space Normalization
        if norm_cfg.get("enabled", False):
            self.add_step(
                NormalizationStep(
                    self.config,
                    self.logger,
                    self.provenance,
                    template=norm_cfg.get("template"),
                    driving_metric=norm_cfg.get("driving_metric", "spgr_ref"),
                    space_name=norm_cfg.get("space_name", norm_cfg.get("space", "MNI")),
                    tool=norm_cfg.get("tool", "ants"),
                    save_transforms=norm_cfg.get("save_transforms", True),
                    transform_type=norm_cfg.get("transform_type", "SyN"),
                    include_all_metrics=norm_cfg.get("include_all_metrics", True),
                    synthmorph_args=norm_cfg.get("synthmorph_args", ""),
                    synthmorph_moving_flag=norm_cfg.get("synthmorph_moving_flag", "--mov"),
                    synthmorph_target_flag=norm_cfg.get("synthmorph_target_flag", "--targ"),
                    synthmorph_output_flag=norm_cfg.get("synthmorph_output_flag", "--o"),
                    synthmorph_transform_ext=norm_cfg.get("synthmorph_transform_ext", ".lta"),
                    synthmorph_register_args=norm_cfg.get("synthmorph_register_args", ""),
                    synthmorph_apply_args=norm_cfg.get("synthmorph_apply_args", ""),
                    robust_iterative=norm_cfg.get("robust_iterative", {}),
                )
            )

        # 6. Post-Processing: Atlas Registration & Stats
        if analysis_cfg.get("enabled", False):
            self.add_step(AtlasRegistrationStep(self.config, self.logger, self.provenance))
            self.add_step(StatsExtractionStep(self.config, self.logger, self.provenance))

    @staticmethod
    def _normalize_analysis_token(value: Any) -> str:
        return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

    @classmethod
    def _relax_model_specs(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "DESPOT1": {
                "aliases": {"despot1"},
                "metrics": {"t1": "T1", "m0": "M0"},
            },
            "DESPOT1HIFI": {
                "aliases": {"despot1hifi", "despot1_hifi", "hifi"},
                "metrics": {"t1": "T1", "m0": "M0", "b1": "B1"},
            },
            "DESPOT2": {
                "aliases": {"despot2"},
                "metrics": {"t2": "T2", "m0": "M0", "f0": "F0"},
            },
            "DESPOT2FM": {
                "aliases": {"despot2fm", "despot2_fm"},
                "metrics": {
                    "mwf": "MWF",
                    "t1_fast": "T1_fast",
                    "t1_slow": "T1_slow",
                    "t2_fast": "T2_fast",
                    "t2_slow": "T2_slow",
                    "tau": "Tau",
                },
            },
            "mcDESPOT": {
                "aliases": {"mcdespot"},
                "metrics": {
                    "mwf": "MWF",
                    "t1_fast": "T1_fast",
                    "t1_slow": "T1_slow",
                    "t2_fast": "T2_fast",
                    "t2_slow": "T2_slow",
                    "tau": "Tau",
                },
            },
        }

    @classmethod
    def _resolve_relax_model_name(cls, value: Any) -> Optional[str]:
        token = cls._normalize_analysis_token(value)
        if not token:
            return None
        for model_name, spec in cls._relax_model_specs().items():
            aliases = {cls._normalize_analysis_token(model_name), *(cls._normalize_analysis_token(v) for v in spec["aliases"])}
            if token in aliases:
                return model_name
        return None

    @classmethod
    def _resolve_relax_metric_name(cls, value: Any) -> Optional[str]:
        token = cls._normalize_analysis_token(value)
        if not token:
            return None

        alias_map = {
            "t1": "T1",
            "m0": "M0",
            "b1": "B1",
            "t2": "T2",
            "f0": "F0",
            "mwf": "MWF",
            "vfm": "MWF",
            "myelinwaterfraction": "MWF",
            "t1fast": "T1_fast",
            "t1slow": "T1_slow",
            "t2fast": "T2_fast",
            "t2slow": "T2_slow",
            "tau": "Tau",
        }
        if token in alias_map:
            return alias_map[token]

        for spec in cls._relax_model_specs().values():
            for metric_name in spec["metrics"].values():
                if token == cls._normalize_analysis_token(metric_name):
                    return metric_name
        return None

    def _record_model_results(
        self,
        *,
        model_name: str,
        raw_results: Dict[str, Path],
        fitted_maps: Dict[str, ImageFile],
        modeling_results: Dict[str, Dict[str, Path]],
    ) -> None:
        spec = self._relax_model_specs()[model_name]
        model_bucket = modeling_results.setdefault(model_name, {})

        for raw_metric, metric_path in (raw_results or {}).items():
            metric_file = Path(metric_path)
            if not metric_file.exists():
                self.logger.warning("Expected %s output missing for %s: %s", raw_metric, model_name, metric_file)
                continue

            metric_name = spec["metrics"].get(raw_metric, raw_metric.upper())
            model_bucket[metric_name] = metric_file

            try:
                entities = get_entities_from_path(metric_file)
            except Exception:
                entities = {}

            image_file = ImageFile(img=metric_file, entities=entities)
            fitted_maps[f"{self._normalize_analysis_token(model_name)}_{self._normalize_analysis_token(metric_name)}"] = image_file
            fitted_maps.setdefault(raw_metric.lower(), image_file)

    def _select_analysis_parameters(
        self,
        modeling_results: Dict[str, Dict[str, Path]],
        analysis_cfg: Dict[str, Any],
    ) -> Dict[str, Dict[str, Path]]:
        raw_selection = analysis_cfg.get("parameters")
        if raw_selection in (None, "", [], {}):
            raw_selection = analysis_cfg.get("metrics")

        if raw_selection in (None, "", [], {}):
            return modeling_results

        selected_by_model: Dict[str, set[str]] = {}
        global_metrics: set[str] = set()

        def _add_metric(model_name: Optional[str], metric_name: Optional[str]) -> None:
            if not metric_name:
                return
            if model_name:
                selected_by_model.setdefault(model_name, set()).add(metric_name)
            else:
                global_metrics.add(metric_name)

        if isinstance(raw_selection, dict):
            items = raw_selection.items()
        else:
            if isinstance(raw_selection, str):
                raw_selection = [part.strip() for part in raw_selection.split(",") if part.strip()]
            items = [(None, item) for item in raw_selection]

        for model_name, selection in items:
            if model_name is not None:
                canonical_model = self._resolve_relax_model_name(model_name)
                if not canonical_model:
                    self.logger.warning("Unknown relaxometry analysis model selector: %s", model_name)
                    continue
                metric_values = selection if isinstance(selection, (list, tuple, set)) else [selection]
                for metric_value in metric_values:
                    canonical_metric = self._resolve_relax_metric_name(metric_value)
                    if not canonical_metric:
                        self.logger.warning("Unknown relaxometry analysis metric selector: %s", metric_value)
                        continue
                    _add_metric(canonical_model, canonical_metric)
                continue

            value = selection
            if not isinstance(value, str):
                canonical_metric = self._resolve_relax_metric_name(value)
                if not canonical_metric:
                    self.logger.warning("Unknown relaxometry analysis metric selector: %s", value)
                    continue
                _add_metric(None, canonical_metric)
                continue

            selector = value.strip()
            if not selector:
                continue

            for delimiter in (":", ".", "/"):
                if delimiter in selector:
                    left, right = selector.split(delimiter, 1)
                    canonical_model = self._resolve_relax_model_name(left)
                    canonical_metric = self._resolve_relax_metric_name(right)
                    if not canonical_model or not canonical_metric:
                        self.logger.warning("Unknown relaxometry analysis selector: %s", selector)
                        break
                    _add_metric(canonical_model, canonical_metric)
                    break
            else:
                canonical_metric = self._resolve_relax_metric_name(selector)
                if not canonical_metric:
                    self.logger.warning("Unknown relaxometry analysis metric selector: %s", selector)
                    continue
                _add_metric(None, canonical_metric)

        filtered: Dict[str, Dict[str, Path]] = {}
        for model_name, metrics in modeling_results.items():
            allowed = set(selected_by_model.get(model_name, set()))
            for metric_name, metric_path in metrics.items():
                if allowed and metric_name in allowed:
                    filtered.setdefault(model_name, {})[metric_name] = metric_path
                elif metric_name in global_metrics:
                    filtered.setdefault(model_name, {})[metric_name] = metric_path

        if not filtered:
            self.logger.warning("No relaxometry fitted maps matched analysis selectors. Atlas registration/statistics will be skipped.")
            return {}

        return filtered

    def _default_analysis_registration_metric(self, modeling_results: Dict[str, Dict[str, Path]]) -> Optional[str]:
        preferred_metrics = ["T1", "MWF", "T2", "M0", "T1_slow", "T1_fast"]
        for preferred in preferred_metrics:
            for metrics in modeling_results.values():
                if preferred in metrics:
                    return preferred
        for metrics in modeling_results.values():
            for metric_name in metrics:
                return metric_name
        return None

    def _build_analysis_context(
        self,
        context: dict,
        ref_img: ImageFile,
        modeling_results: Dict[str, Dict[str, Path]],
    ) -> dict:
        analysis_cfg = dict(self.relax_config.analysis or {})
        selected_results = self._select_analysis_parameters(modeling_results, analysis_cfg)
        if selected_results and not (
            analysis_cfg.get("registration_metric") or analysis_cfg.get("registration_target")
        ):
            default_metric = self._default_analysis_registration_metric(selected_results)
            if default_metric:
                analysis_cfg["registration_metric"] = default_metric

        analysis_context = dict(context)
        analysis_context["analysis_modality"] = "relaxometry"
        analysis_context["analysis_cfg"] = analysis_cfg
        analysis_context["current_image"] = ref_img
        analysis_context["modeling_results"] = selected_results
        analysis_context.setdefault("segmentations", {})
        return analysis_context

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

    @staticmethod
    def _normalize_exclude_indices(raw_value: Any, label: str) -> List[int]:
        if raw_value in (None, "", []):
            return []
        if isinstance(raw_value, str):
            values = [part.strip() for part in raw_value.split(",") if part.strip()]
        elif isinstance(raw_value, (list, tuple, set)):
            values = list(raw_value)
        else:
            raise ValueError(f"{label} exclude indices must be a list or comma-separated string.")

        normalized: List[int] = []
        for value in values:
            try:
                idx = int(value)
            except Exception as exc:
                raise ValueError(f"{label} exclude indices must be integers. Invalid value: {value!r}") from exc
            if idx < 0:
                raise ValueError(f"{label} exclude indices must be non-negative. Invalid value: {idx}")
            normalized.append(idx)
        return sorted(set(normalized))

    @staticmethod
    def _load_json_payload(src_json: Any) -> dict[str, Any]:
        if not src_json:
            return {}
        if isinstance(src_json, dict):
            return dict(src_json)
        src_path = Path(src_json)
        if not src_path.exists():
            return {}
        try:
            return json.loads(src_path.read_text())
        except Exception:
            return {}

    @staticmethod
    def _volume_json_payload(payload: dict[str, Any], volume_index: int, volume_count: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, list) and len(value) == volume_count:
                out[key] = value[volume_index]
            else:
                out[key] = value
        return out

    def _expand_modality_exclusions(
        self,
        images: List[ImageFile],
        *,
        modality_label: str,
        exclude_indices: List[int],
        derived_dir: Path,
    ) -> List[ImageFile]:
        if not images or not exclude_indices:
            return images

        logical_ranges: list[tuple[ImageFile, int, int]] = []
        running = 0
        for img in images:
            shape = nib.load(str(img.img)).shape
            count = int(shape[3]) if len(shape) >= 4 and shape[3] > 1 else 1
            logical_ranges.append((img, running, count))
            running += count

        total_count = running
        invalid = [idx for idx in exclude_indices if idx >= total_count]
        if invalid:
            raise ValueError(
                f"{modality_label} exclude indices out of range: {invalid}. "
                f"Found {total_count} logical {modality_label} acquisition(s)."
            )

        exclude_set = set(exclude_indices)
        derived_dir.mkdir(parents=True, exist_ok=True)
        filtered: List[ImageFile] = []
        excluded_paths: List[str] = []

        for img, start_idx, count in logical_ranges:
            local_keep = [local_idx for local_idx in range(count) if (start_idx + local_idx) not in exclude_set]
            local_drop = [start_idx + local_idx for local_idx in range(count) if (start_idx + local_idx) in exclude_set]
            if local_drop:
                excluded_paths.append(f"{img.img.name}: {local_drop}")
            if not local_keep:
                continue
            if count == 1 or len(local_keep) == count:
                filtered.append(img)
                continue

            nii = nib.load(str(img.img))
            base_payload = self._load_json_payload(getattr(img, "json", None))
            suffix = img.entities.get("suffix", "VFA")
            for local_idx in local_keep:
                global_idx = start_idx + local_idx
                out_entities = dict(img.entities)
                chunk_token = f"{global_idx:04d}"
                if out_entities.get("chunk"):
                    chunk_token = f"{out_entities['chunk']}{chunk_token}"
                out_entities["chunk"] = chunk_token
                out_path = derived_dir / build_bids_name(out_entities, suffix=suffix)
                out_json = out_path.with_suffix("").with_suffix(".json")

                if not out_path.exists():
                    vol_data = np.asanyarray(nii.dataobj[..., local_idx])
                    nib.save(nib.Nifti1Image(vol_data, nii.affine, nii.header.copy()), str(out_path))

                if not out_json.exists():
                    vol_payload = self._volume_json_payload(base_payload, local_idx, count)
                    vol_payload["SourceImage"] = str(img.img)
                    vol_payload["SelectedVolumeIndex"] = int(local_idx)
                    with out_json.open("w") as f:
                        json.dump(vol_payload, f, indent=2)
                        f.write("\n")

                filtered.append(ImageFile(img=out_path, entities=out_entities, json=out_json))

        self.logger.info(
            "%s exclusions applied: removed indices=%s, remaining=%d/%d",
            modality_label,
            exclude_indices,
            total_count - len(exclude_set),
            total_count,
        )
        for item in excluded_paths:
            self.logger.debug("Excluded %s acquisition(s): %s", modality_label, item)
        return filtered

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
    ) -> tuple[Dict[str, ImageFile], Dict[str, Dict[str, Path]]]:
        """
        Run model fitting for DESPOT1, DESPOT1 HIFI, DESPOT2, DESPOT2FM, mcDESPOT if enabled.

        Returns flat fitted map handles plus structured modeling results.
        """
        from ...interfaces.fsl import merge as fslmerge

        fitted_maps: Dict[str, ImageFile] = {}
        modeling_results: Dict[str, Dict[str, Path]] = {}
        modeling_cfg = self.relax_config.modeling
        despot1_results: Dict[str, Path] = {}

        def _model_cli_options(cfg: Dict, exclude: set[str]) -> Dict:
            return {k: v for k, v in (cfg or {}).items() if k not in exclude}

        def _model_nthreads(cfg: Dict) -> int:
            if "nthreads" in (cfg or {}):
                return int(cfg["nthreads"])
            if "threads" in (cfg or {}):
                self.logger.warning(
                    "Relaxometry model option `threads` is deprecated. Use `nthreads` instead."
                )
                return int(cfg["threads"])
            return int(self.config.get("n_cpus", 1))

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
        created_temp_inputs = False

        def _stack_inputs(images: List[ImageFile], stem: str) -> Optional[Path]:
            nonlocal created_temp_inputs
            if not images:
                return None
            if len(images) == 1:
                return images[0].img
            if not created_temp_inputs:
                input_dir.mkdir(parents=True, exist_ok=True)
                created_temp_inputs = True
            merged_path = input_dir / f"{base_prefix}_{stem}.nii.gz"
            return fslmerge(images, merged_path, dimension="t")

        try:
            spgr_stack = _stack_inputs(spgr_moco, "desc-spgrStack_VFA")
            ssfp_stack = _stack_inputs(ssfp_moco, "desc-ssfpStack_VFA")
            irspgr_stack = _stack_inputs(ir_final, "desc-irspgrStack_VFA")
            b1_path = b1_map.img if isinstance(b1_map, ImageFile) else b1_map

            if spgr_stack is None:
                raise ValueError("DESPOT fitting requires at least one SPGR image.")

            # DESPOT1 fitting
            if modeling_cfg.despot1.get("enabled", False):
                self.logger.info("Starting DESPOT1 fitting.")
                despot1_cfg = dict(modeling_cfg.despot1 or {})
                use_hifi = despot1_cfg.get("use_hifi", False)
                despot1_algo = despot1_cfg.get("algo", "lsq")
                despot1_nthreads = _model_nthreads(despot1_cfg)
                despot1_verbose = bool(despot1_cfg.get("verbose", False))
                despot1_extra = _model_cli_options(despot1_cfg, {"enabled", "use_hifi", "algo", "nthreads", "threads", "verbose"})
                if use_hifi:
                    if irspgr_stack is None:
                        raise ValueError("DESPOT1-HIFI requested, but no IR-SPGR image was found.")
                    despot1_results = fit_despot1_hifi(
                        spgr_file=spgr_stack,
                        irspgr_file=irspgr_stack,
                        params_file=params_json,
                        out_dir=fit_out_dir,
                        mask_file=mask_file,
                        out_base=f"{base_prefix}_model-DESPOT1HIFI",
                        algo=despot1_algo,
                        nthreads=despot1_nthreads,
                        verbose=despot1_verbose,
                        extra_options=despot1_extra,
                    )
                else:
                    despot1_results = fit_despot1(
                        spgr_file=spgr_stack,
                        params_file=params_json,
                        out_dir=fit_out_dir,
                        b1_file=b1_path,
                        mask_file=mask_file,
                        out_base=f"{base_prefix}_model-DESPOT1",
                        algo=despot1_algo,
                        nthreads=despot1_nthreads,
                        verbose=despot1_verbose,
                        extra_options=despot1_extra,
                    )
                model_name = "DESPOT1HIFI" if use_hifi else "DESPOT1"
                self._record_model_results(
                    model_name=model_name,
                    raw_results=despot1_results,
                    fitted_maps=fitted_maps,
                    modeling_results=modeling_results,
                )

            # DESPOT2 fitting
            if modeling_cfg.despot2.get("enabled", False):
                self.logger.info("Starting DESPOT2 fitting.")
                despot2_cfg = dict(modeling_cfg.despot2 or {})
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
                    out_base=f"{base_prefix}_model-DESPOT2",
                    algo=despot2_cfg.get("algo", "lsq"),
                    nthreads=_model_nthreads(despot2_cfg),
                    verbose=bool(despot2_cfg.get("verbose", False)),
                    extra_options=_model_cli_options(despot2_cfg, {"enabled", "algo", "nthreads", "threads", "verbose", "mcdespot"}),
                )
                self._record_model_results(
                    model_name="DESPOT2",
                    raw_results=despot2_results,
                    fitted_maps=fitted_maps,
                    modeling_results=modeling_results,
                )

            if modeling_cfg.despot2fm.get("enabled", False):
                self.logger.info("Starting DESPOT2FM fitting.")
                despot2fm_cfg = dict(modeling_cfg.despot2fm or {})
                if ssfp_stack is None:
                    raise ValueError("DESPOT2FM requested, but no SSFP image was found.")
                t1_path = despot1_results.get("t1") if modeling_cfg.despot1.get("enabled", False) else None
                if not t1_path:
                    raise ValueError("DESPOT2FM requires a DESPOT1 T1 map, but none was produced.")
                despot_b1_path = despot1_results.get("b1") if modeling_cfg.despot1.get("enabled", False) else None
                if not despot_b1_path:
                    despot_b1_path = b1_path
                if not despot_b1_path:
                    raise ValueError("DESPOT2FM requires a B1 map, but none was available from AFI/external B1 or DESPOT1-HIFI.")
                despot2fm_results = fit_despot2_fm(
                    ssfp_file=ssfp_stack,
                    t1_file=t1_path,
                    b1_file=despot_b1_path,
                    params_file=params_json,
                    out_dir=fit_out_dir,
                    mask_file=mask_file,
                    out_base=f"{base_prefix}_model-DESPOT2FM",
                    algo=despot2fm_cfg.get("algo", "src"),
                    nthreads=_model_nthreads(despot2fm_cfg),
                    verbose=bool(despot2fm_cfg.get("verbose", False)),
                    extra_options=_model_cli_options(despot2fm_cfg, {"enabled", "algo", "nthreads", "threads", "verbose"}),
                )
                self._record_model_results(
                    model_name="DESPOT2FM",
                    raw_results=despot2fm_results,
                    fitted_maps=fitted_maps,
                    modeling_results=modeling_results,
                )

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
                mcdespot_results = fit_mcdespot(
                    spgr_file=spgr_stack,
                    ssfp_file=ssfp_stack,
                    t1_file=t1_path,
                    b1_file=despot_b1_path,
                    params_file=params_json,
                    out_dir=fit_out_dir,
                    mask_file=mask_file,
                    out_base=f"{base_prefix}_model-mcDESPOT",
                    algo=mcdespot_cfg.get("algo", "src"),
                    nthreads=_model_nthreads(mcdespot_cfg),
                    verbose=bool(mcdespot_cfg.get("verbose", False)),
                    cuda=bool(mcdespot_cfg.get("cuda", False)),
                    extra_options=_model_cli_options(mcdespot_cfg, {"enabled", "cuda", "algo", "nthreads", "threads", "verbose"}),
                )
                self._record_model_results(
                    model_name="mcDESPOT",
                    raw_results=mcdespot_results,
                    fitted_maps=fitted_maps,
                    modeling_results=modeling_results,
                )
        finally:
            if created_temp_inputs and input_dir.exists() and not self.config.get("save_intermediates", False):
                shutil.rmtree(input_dir, ignore_errors=True)

        context["fitted_maps"] = fitted_maps
        context["modeling_results"] = modeling_results
        return fitted_maps, modeling_results

    def _run_postprocessing_and_stats(
        self,
        context: dict,
        fit_out_dir: Path,
        ref_img: ImageFile,
        modeling_results: Dict[str, Dict[str, Path]],
        base_prefix: str,
    ) -> Dict[str, Path]:
        """
        Run atlas registration and ROI statistics extraction for selected relaxometry maps.
        """
        stats_results: Dict[str, Path] = {}
        analysis_cfg = self.relax_config.analysis or {}

        if analysis_cfg.get("enabled", False):
            atlas_reg_step = next((s for s in self.steps if isinstance(s, AtlasRegistrationStep)), None)
            stats_extract_step = next((s for s in self.steps if isinstance(s, StatsExtractionStep)), None)
            analysis_context = self._build_analysis_context(context, ref_img, modeling_results)

            if not analysis_context.get("modeling_results"):
                self.logger.warning("Relaxometry analysis is enabled, but no fitted maps were selected for atlas/statistics analysis.")
            elif atlas_reg_step and stats_extract_step:
                self.logger.info("Running relaxometry atlas registration and ROI statistics extraction.")
                atlas_reg_step.run(analysis_context, fit_out_dir)
                stats_extract_step.run(analysis_context, fit_out_dir)
                stats_results.update(analysis_context.get("roi_stats_files", {}))
                context["segmentations"] = analysis_context.get("segmentations", {})
                context["roi_stats_files"] = analysis_context.get("roi_stats_files", {})
                context["roi_stats"] = stats_results
            else:
                self.logger.warning("Relaxometry analysis is enabled, but atlas/statistics steps are not available. Skipping post-fit analysis.")
        else:
            self.logger.debug("Relaxometry atlas registration/statistics analysis is disabled. Skipping post-fit analysis.")

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

    def _run_normalization(
        self,
        context: dict,
        normalization_output_dir: Path,
    ) -> dict:
        """
        Run standard-space normalization on relaxometry fitted maps if enabled.
        """
        norm_cfg = self.relax_config.normalization or {}
        if not norm_cfg.get("enabled", False):
            self.logger.debug("Relaxometry normalization is disabled. Skipping standard-space normalization.")
            return context

        norm_step = next((s for s in self.steps if isinstance(s, NormalizationStep)), None)
        if not norm_step:
            self.logger.warning("Relaxometry normalization is enabled, but no normalization step is configured.")
            return context

        normalization_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Running relaxometry normalization to standard space.")
        return norm_step.run(context, normalization_output_dir)

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

        exclude_cfg = getattr(preproc_cfg, "exclude_indices", {}) or {}
        spgr_exclude = self._normalize_exclude_indices(exclude_cfg.get("spgr"), "SPGR")
        ssfp_exclude = self._normalize_exclude_indices(exclude_cfg.get("ssfp"), "SSFP")

        if spgr_exclude:
            spgr_files = self._expand_modality_exclusions(
                spgr_files,
                modality_label="SPGR",
                exclude_indices=spgr_exclude,
                derived_dir=intermediate_dir / "excluded_inputs" / "spgr",
            )
        if ssfp_exclude:
            ssfp_files = self._expand_modality_exclusions(
                ssfp_files,
                modality_label="SSFP",
                exclude_indices=ssfp_exclude,
                derived_dir=intermediate_dir / "excluded_inputs" / "ssfp",
            )
        context["spgr_exclude_indices"] = spgr_exclude
        context["ssfp_exclude_indices"] = ssfp_exclude

        self.logger.info(
            "Using relaxometry inputs after exclusions: %d SPGR, %d SSFP, %d IR-SPGR, %d B1.",
            len(spgr_files),
            len(ssfp_files),
            len(irspgr_files),
            len(b1_files),
        )

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
        fit_out_dir = anat_out_dir / "models"
        fit_maps, modeling_results = self._run_model_fitting(
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

        # Standard-space normalization
        normalization_out_dir = anat_out_dir / "normalization"
        context = self._run_normalization(context, normalization_out_dir)

        # Post-processing, atlas registration, segmentation, stats extraction
        stats_results = self._run_postprocessing_and_stats(
            context, fit_out_dir, ref_img, modeling_results, base_prefix
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
            "modeling_results": modeling_results,
            "normalized_results": context.get("normalized_results", {}),
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
            normalization=relax_config_dict.get("normalization", {}),
            analysis=relax_config_dict.get("analysis", {}),
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
