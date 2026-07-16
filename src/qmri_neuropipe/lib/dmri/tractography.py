"""
Tractography and Bundle Segmentation Module.

This module provides processing steps for tractography (TractSeg, PyAFQ).
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union
import csv
import json
import logging
import re
import shutil

import numpy as np
import nibabel as nib

from ...core import BaseProcessingStep, ProcessingError, ValidationError
from ...core.types import DWIFile
from ...interfaces import mrtrix, tractseg
from ...io.bids import build_bids_name, get_entities_from_path


def _path(value):
    """Return a filesystem path from pipeline image objects or path values."""
    if value is None:
        return None
    return Path(getattr(value, "img", value))


def _bids_label(value: Any) -> str:
    """Return a conservative alphanumeric BIDS entity label."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)) or "unknown"


def _tract_entities(context: dict, *, desc: str) -> dict:
    image = context.get("current_image")
    entities = dict(getattr(image, "entities", {}) or {})
    entities = {
        key: value for key, value in entities.items()
        if key in {"sub", "ses", "task", "acq", "dir", "run"} and value is not None
    }
    entities["space"] = "dwi"
    entities["desc"] = _bids_label(desc)
    return entities


def _bids_path(output_dir: Path, context: dict, *, desc: str, suffix: str, extension: str) -> Path:
    return output_dir / build_bids_name(
        _tract_entities(context, desc=desc), suffix=suffix, extension=extension
    )


def _write_json_sidecar(path: Path, metadata: dict) -> Path:
    sidecar = path.with_suffix(".json")
    with sidecar.open("w") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    return sidecar


class MRtrixAnatomicalConstraintsStep(BaseProcessingStep):
    """Prepare and validate ACT tissue and GMWMI seed images."""

    def __init__(self, config, logger, provenance, nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.nthreads = getattr(config, "n_cpus", nthreads)
        self.kwargs = kwargs

    def run(self, context: dict, output_dir: Path, mask=None, force=False, **kwargs):
        act_cfg = dict(self.kwargs)
        out_dir = output_dir / "MRtrix" / "ACT"
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = act_cfg.pop("five_tt", None) or context.get("5tt")
        five_tt = _path(existing)
        if five_tt and not five_tt.exists():
            raise ValidationError(f"Configured ACT 5TT image does not exist: {five_tt}")
        if five_tt:
            bids_five_tt = _bids_path(
                out_dir, context, desc="act5tt", suffix="probseg", extension=".nii.gz"
            )
            if five_tt.resolve() != bids_five_tt.resolve() and (force or not bids_five_tt.exists()):
                shutil.copyfile(five_tt, bids_five_tt)
            five_tt = bids_five_tt
        else:
            anatomical = (
                act_cfg.pop("anatomical", None)
                or context.get("preprocessed_anat_coreg")
                or context.get("preprocessed_t1w")
            )
            if not anatomical:
                raise ValidationError(
                    "ACT requires an existing five_tt image or an anatomical T1w image"
                )
            five_tt = _bids_path(out_dir, context, desc="act5tt", suffix="probseg", extension=".nii.gz")
            mrtrix.five_tt_gen(
                act_cfg.pop("algorithm", "fsl"), _path(anatomical), five_tt,
                options=act_cfg.pop("options", {}), nthreads=self.nthreads, force=force,
            )
        if act_cfg.pop("validate", True):
            mrtrix.five_tt_check(five_tt, nthreads=self.nthreads)
        reference = _path(context.get("current_image"))
        if reference and reference.exists():
            five_img, ref_img = nib.load(str(five_tt)), nib.load(str(reference))
            if five_img.shape[:3] != ref_img.shape[:3] or not np.allclose(
                five_img.affine, ref_img.affine, atol=1e-3
            ):
                raise ValidationError(
                    "ACT 5TT image is not on the diffusion grid; supply a DWI-space "
                    "five_tt image or a DWI-coregistered anatomical image"
                )
        gmwmi = _bids_path(out_dir, context, desc="gmwmi", suffix="mask", extension=".nii.gz")
        mrtrix.five_tt_to_gmwmi(five_tt, gmwmi, force=force)
        tract = context.setdefault("tractography", {})
        tract["act_5tt"] = five_tt
        tract["gmwmi_seed"] = gmwmi
        return context


class MRtrixTractographyStep(BaseProcessingStep):
    """Generate an MRtrix whole-brain tractogram and optionally apply SIFT."""

    def __init__(self, config, logger, provenance, nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.nthreads = getattr(config, "n_cpus", nthreads)
        self.kwargs = kwargs

    @staticmethod
    def _tracking_image(context, algorithm):
        if algorithm.lower().startswith("tensor"):
            dti = context.get("modeling_results", {}).get("DTI", {})
            return dti.get("tensor_mrtrix") or dti.get("tensorMRTRIX") or dti.get("tensor")
        csd = context.get("modeling_results", {}).get("CSD", {})
        return csd.get("wmFOD") or csd.get("FOD") or csd.get("fod")

    def run(self, context: dict, output_dir: Path, mask=None, force=False, **kwargs):
        cfg = dict(self.kwargs)
        out_dir = output_dir / "MRtrix" / "tractography"
        out_dir.mkdir(parents=True, exist_ok=True)
        algorithm = cfg.pop("algorithm", "iFOD2")
        tracking_image = cfg.pop("input", None) or self._tracking_image(context, algorithm)
        if not tracking_image:
            requirement = "MRtrix tensor" if algorithm.lower().startswith("tensor") else "CSD FOD"
            raise ValidationError(f"MRtrix tractography requires a {requirement} image")
        act_cfg = cfg.pop("act", {})
        act_enabled = bool(act_cfg.get("enabled", False)) if isinstance(act_cfg, dict) else bool(act_cfg)
        tract_ctx = context.setdefault("tractography", {})
        act = tract_ctx.get("act_5tt") if act_enabled else None
        if act_enabled and not act:
            raise ValidationError("ACT was enabled but no validated 5TT image is available")
        seed_gmwmi = tract_ctx.get("gmwmi_seed") if act_enabled and act_cfg.get("seed_gmwmi", True) else None
        options = dict(cfg.pop("options", {}))
        for key in ("cutoff", "minlength", "maxlength", "step", "angle", "seeds"):
            if key in cfg:
                options[key] = cfg.pop(key)
        if act_enabled:
            if act_cfg.get("backtrack", True):
                options["backtrack"] = True
            if act_cfg.get("crop_at_gmwmi", True):
                options["crop_at_gmwmi"] = True
        tracks = _bids_path(
            out_dir, context, desc=f"wholebrain{algorithm}", suffix="tractography", extension=".tck"
        )
        select = int(cfg.pop("select", cfg.pop("n_streamlines", 10_000_000)))
        mrtrix.tckgen(
            _path(tracking_image), tracks, algorithm=algorithm,
            select=select,
            act=_path(act), seed_gmwmi=_path(seed_gmwmi),
            seed_image=_path(cfg.pop("seed_image", None)), mask=_path(mask),
            include=map(Path, cfg.pop("include", [])), exclude=map(Path, cfg.pop("exclude", [])),
            options=options, nthreads=self.nthreads, force=force,
        )
        tract_ctx.update({"whole_brain": tracks, "algorithm": algorithm, "input": _path(tracking_image)})
        _write_json_sidecar(tracks, {
            "Sources": [str(_path(tracking_image))],
            "TrackingSoftware": "MRtrix3",
            "TrackingAlgorithm": algorithm,
            "NumberOfStreamlinesRequested": select,
            "AnatomicallyConstrainedTractography": act_enabled,
            "FiveTissueTypeImage": str(act) if act else None,
            "CoordinateSystem": "dwi",
        })
        filtering = cfg.pop("filtering", {}) or {}
        method = str(filtering.get("method", "none")).lower()
        if method == "sift2":
            weights = _bids_path(out_dir, context, desc="sift2", suffix="weights", extension=".tsv")
            mrtrix.tcksift2(tracks, _path(tracking_image), weights, act=_path(act),
                           options=filtering.get("options"), nthreads=self.nthreads, force=force)
            tract_ctx["sift2_weights"] = weights
        elif method == "sift":
            filtered = _bids_path(
                out_dir, context, desc=f"wholebrain{algorithm}SIFT", suffix="tractography", extension=".tck"
            )
            mrtrix.tcksift(tracks, _path(tracking_image), filtered, act=_path(act),
                          term_number=filtering.get("term_number"), options=filtering.get("options"),
                          nthreads=self.nthreads, force=force)
            tract_ctx["unfiltered"] = tracks
            tract_ctx["whole_brain"] = filtered
            _write_json_sidecar(filtered, {
                "Sources": [str(tracks)], "TrackingSoftware": "MRtrix3",
                "TrackingAlgorithm": algorithm, "FilteringMethod": "SIFT",
                "CoordinateSystem": "dwi",
            })
        elif method not in {"none", "false", ""}:
            raise ValidationError("tractography filtering method must be none, sift, or sift2")
        return context


class TractSpecificAnalysisStep(BaseProcessingStep):
    """Extract configured bundles and perform streamline-based tractometry."""

    def __init__(self, config, logger, provenance, nthreads=1, **kwargs):
        super().__init__(config, logger, provenance)
        self.nthreads = getattr(config, "n_cpus", nthreads)
        self.kwargs = kwargs

    @staticmethod
    def _metric_maps(context, requested):
        results = context.get("modeling_results", {})
        resolved = {}
        for item in requested or []:
            if isinstance(item, dict):
                name, value = next(iter(item.items()))
                resolved[name] = _path(value)
                continue
            token = str(item)
            if "." in token:
                model, metric = token.split(".", 1)
                value = results.get(model, {}).get(metric)
            else:
                value = next((maps[token] for maps in results.values() if token in maps), None)
            if value:
                resolved[token] = _path(value)
        return resolved

    @staticmethod
    def _tractseg_mask(context, name):
        source = context.get("segmentations", {}).get("TractSeg")
        if not source:
            return None
        source = Path(source)
        candidates = [source / f"{name}.nii.gz", source / f"{name}.nii"]
        return next((p for p in candidates if p.exists()), None)

    def run(self, context: dict, output_dir: Path, mask=None, force=False, **kwargs):
        cfg = dict(self.kwargs)
        tract_ctx = context.setdefault("tractography", {})
        whole_brain = _path(tract_ctx.get("whole_brain"))
        if not whole_brain:
            raise ValidationError("Tract-specific analysis requires a whole-brain tractogram")
        out_dir = output_dir / "MRtrix" / "tract_specific"
        bundle_dir, sample_dir = out_dir / "bundles", out_dir / "samples"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        bundles = {}
        for definition in cfg.pop("bundles", []):
            if isinstance(definition, str):
                definition = {"name": definition, "source": "tractseg"}
            name = definition["name"]
            include = [Path(p) for p in definition.get("include", [])]
            if definition.get("source", "roi").lower() == "tractseg":
                tractseg_mask = self._tractseg_mask(context, name)
                if not tractseg_mask:
                    raise ValidationError(f"TractSeg mask not found for requested bundle: {name}")
                include.append(tractseg_mask)
            bundle = _bids_path(
                bundle_dir, context, desc=name, suffix="tractography", extension=".tck"
            )
            mrtrix.tckedit(whole_brain, bundle, include=include,
                           exclude=map(Path, definition.get("exclude", [])),
                           ends_only=definition.get("ends_only", False),
                           options=definition.get("options"), nthreads=self.nthreads, force=force)
            bundles[name] = bundle
            _write_json_sidecar(bundle, {
                "Sources": [str(whole_brain)], "TrackingSoftware": "MRtrix3",
                "BundleName": name, "ExtractionSource": definition.get("source", "roi"),
                "CoordinateSystem": "dwi",
            })
        tract_ctx["bundles"] = bundles
        metrics = self._metric_maps(context, cfg.pop("metrics", []))
        rows = []
        samples = {}
        statistic = cfg.pop("streamline_statistic", "mean")
        profiles = cfg.pop("profiles", {}) or {}
        profile_rows = []
        for bundle_name, bundle in bundles.items():
            samples[bundle_name] = {}
            profile_bundle = bundle
            if profiles.get("enabled", False):
                profile_bundle = _bids_path(
                    bundle_dir, context, desc=f"{bundle_name}resampled",
                    suffix="tractography", extension=".tck",
                )
                mrtrix.tckresample(
                    bundle, profile_bundle, num_points=int(profiles.get("nodes", 100)),
                    nthreads=self.nthreads, force=force,
                )
            if cfg.get("track_density", {}).get("enabled", False):
                tdi = _bids_path(out_dir, context, desc=bundle_name, suffix="tdi", extension=".nii.gz")
                mrtrix.tckmap(bundle, tdi, template=_path(context.get("current_image")),
                             options=cfg["track_density"].get("options"),
                             nthreads=self.nthreads, force=force)
                tract_ctx.setdefault("track_density_maps", {})[bundle_name] = tdi
            for metric_name, metric_path in metrics.items():
                sample_file = _bids_path(
                    sample_dir, context, desc=f"{bundle_name}{metric_name}",
                    suffix="samples", extension=".tsv",
                )
                mrtrix.tcksample(bundle, metric_path, sample_file, statistic_tck=statistic,
                                 nthreads=self.nthreads, force=force)
                samples[bundle_name][metric_name] = sample_file
                values = np.loadtxt(sample_file, ndmin=1)
                finite = values[np.isfinite(values)]
                if finite.size:
                    rows.append({"tract": bundle_name, "metric": metric_name,
                                 "n_streamlines": int(finite.size), "mean": float(np.mean(finite)),
                                 "median": float(np.median(finite)), "std": float(np.std(finite))})
                if profiles.get("enabled", False):
                    profile_file = _bids_path(
                        sample_dir, context, desc=f"{bundle_name}{metric_name}profile",
                        suffix="samples", extension=".tsv",
                    )
                    mrtrix.tcksample(profile_bundle, metric_path, profile_file,
                                     nthreads=self.nthreads, force=force)
                    profile_values = np.loadtxt(profile_file, ndmin=2)
                    for node, node_values in enumerate(profile_values.T):
                        node_values = node_values[np.isfinite(node_values)]
                        if node_values.size:
                            profile_rows.append({
                                "tract": bundle_name, "metric": metric_name, "node": node,
                                "mean": float(np.mean(node_values)),
                                "median": float(np.median(node_values)),
                                "std": float(np.std(node_values)),
                            })
        tract_ctx["samples"] = samples
        if rows:
            stats_file = _bids_path(out_dir, context, desc="tractometry", suffix="stats", extension=".tsv")
            with stats_file.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            tract_ctx["statistics"] = stats_file
        if profile_rows:
            profiles_file = _bids_path(
                out_dir, context, desc="alongtract", suffix="profiles", extension=".tsv"
            )
            with profiles_file.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(profile_rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(profile_rows)
            tract_ctx["profiles"] = profiles_file
        connectome = cfg.pop("connectome", {}) or {}
        if connectome.get("enabled", False):
            nodes = _path(connectome.get("nodes"))
            if not nodes:
                raise ValidationError("Connectome generation requires a parcellation in connectome.nodes")
            matrix = _bids_path(out_dir, context, desc="connectome", suffix="connectivity", extension=".tsv")
            mrtrix.tck2connectome(
                whole_brain, nodes, matrix,
                weights=_path(tract_ctx.get("sift2_weights")) if connectome.get("use_sift2", True) else None,
                statistic=connectome.get("statistic", "sum"),
                symmetric=connectome.get("symmetric", True),
                zero_diagonal=connectome.get("zero_diagonal", True),
                options=connectome.get("options"), nthreads=self.nthreads, force=force,
            )
            tract_ctx.setdefault("connectomes", {})["default"] = matrix
        return context

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
        output_type = ts_kwargs.pop('output_type', 'tract_segmentation')
        
        # MNI Registration / Preprocess logic
        preprocess = ts_kwargs.pop('preprocess', True) # Default to True for "improved segmentation"
        bundle_names = ts_kwargs.pop('bundles', None) or ts_kwargs.pop('bundle_names', None)
        
        # --- Manual Registration Logic ---
        input_file_for_ts = input_file
        
        fwd_xfm = context.get('MNI_fwd_xfm')
        inv_xfm = context.get('MNI_inv_xfm')
        mni_ref = context.get('MNI_ref') or ts_kwargs.pop('mni_template', None)
        
        if preprocess and fwd_xfm and inv_xfm and mni_ref:
            self.logger.info("Performing manual MNI warping for TractSeg using pipeline transforms...")
            mni_peaks = tractseg_out / "peaks_mni.nii.gz"
            if not mni_peaks.exists() or not skip:
                self._apply_ants_warp(input_file, mni_peaks, fwd_xfm, mni_ref, interp='linear')
            
            input_file_for_ts = mni_peaks
            mni_out = tractseg_out / "mni"
            mni_out.mkdir(exist_ok=True)
            
            tractseg.run_tractseg(
                input_file=input_file_for_ts,
                output_dir=mni_out,
                input_type=input_type,
                output_type=output_type,
                brain_mask=None,
                nr_cpus=self.nthreads,
                preprocess=False,
                bundle_names=bundle_names,
                extra_args=ts_kwargs
            )
            
            mni_bundle_dir = mni_out / "bundle_masks"
            if mni_bundle_dir.exists():
                self.logger.info(f"Inverse warping bundles back to native space...")
                bundle_out_dir.mkdir(parents=True, exist_ok=True)
                for mni_bundle in mni_bundle_dir.glob("*.nii.gz"):
                    native_bundle = bundle_out_dir / mni_bundle.name
                    if not native_bundle.exists() or not skip:
                        self._apply_ants_warp(mni_bundle, native_bundle, inv_xfm, input_file, interp='nearestNeighbor')
                context.setdefault('segmentations', {})['TractSeg'] = bundle_out_dir
        else:
            if preprocess:
                self.logger.info("MNI transforms not found in context. Using TractSeg internal --preprocess.")
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
                preprocess=preprocess,
                bundle_names=bundle_names,
                extra_args=ts_kwargs
            )
            if (tractseg_out / output_type).exists():
                 context.setdefault('segmentations', {})['TractSeg'] = tractseg_out / output_type
            elif (tractseg_out / "bundle_masks").exists():
                 # Fallback for standard
                 context.setdefault('segmentations', {})['TractSeg'] = tractseg_out / "bundle_masks"
             
        # Optional: Tractometry
        # If requested via config?
        # For now, we just do segmentation.
        
        return context

    def _apply_ants_warp(self, in_path: Union[Path, str], out_path: Union[Path, str], transform_list: Union[list, str, Path], reference: Union[Path, str], interp: str = 'linear'):
        """Helper to apply ANTs warping."""
        try:
            import ants
            mov = ants.image_read(str(in_path))
            fix = ants.image_read(str(reference))
            
            # Determine image type
            # TractSeg peaks are 4D (9 volumes)
            # ANTs apply_transforms handles 4D if imagetype=3
            is_4d = mov.dimension == 4
            imagetype = 3 if is_4d else 0
            
            # Ensure transform_list is a list of strings
            if not isinstance(transform_list, (list, tuple)):
                transform_list = [transform_list]
            
            tx_list = [str(tx) for tx in transform_list]
            
            warped = ants.apply_transforms(
                fixed=fix,
                moving=mov,
                transformlist=tx_list,
                interpolator=interp,
                imagetype=imagetype
            )
            ants.image_write(warped, str(out_path))
        except ImportError:
            self.logger.error("ANTsPy not installed. Manual warping failed.")
            raise ProcessingError("Manual warping requires ANTsPy.")
        except Exception as e:
            self.logger.error(f"ANTs warping failed: {e}")
            raise ProcessingError(f"Warping failed: {e}")

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
