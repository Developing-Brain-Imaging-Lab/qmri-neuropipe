
from pathlib import Path
from typing import Optional, Dict, Any
import json
import shutil

import nibabel as nib
import numpy as np

from ...core import BaseProcessingStep, ValidationError, ProcessingError
from ...interfaces import dcm2niix, dcm2bids
from .gnl_metadata import GEGnlMetadataEnrichmentStep
from ...core.utils import get_nifti_stem
from ...io.bids import get_entities_from_path


def _load_json_payload(json_path: Path) -> dict[str, Any]:
    with json_path.open() as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ProcessingError(f"Expected JSON object in sidecar: {json_path}")
    return payload


def _entity_source_path(json_path: Path) -> Path:
    return _sidecar_nifti_path(json_path) or json_path


def _normalize_context_label(value: Any, prefix: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "n/a", "na"}:
        return None
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text or None


def _context_search_root(output_dir: Path, context: dict[str, Any]) -> Path:
    root = Path(output_dir)
    subject = _normalize_context_label(context.get("subject"), "sub-")
    session = _normalize_context_label(context.get("session"), "ses-")

    if subject:
        root = root / f"sub-{subject}"
        if session:
            root = root / f"ses-{session}"
        return root

    return root if root.exists() else Path(output_dir)


def _is_hidden_artifact(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _is_primary_bids_output(path: Path, output_dir: Path) -> bool:
    if _is_hidden_artifact(path):
        return False
    root = Path(output_dir)
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return "derivatives" not in rel_parts


def _import_metadata_override_cfg(config) -> dict[str, Any]:
    import_cfg = (config.get("import", {}) or {})
    direct_cfg = import_cfg.get("metadata_overrides")
    if isinstance(direct_cfg, dict):
        return direct_cfg

    gradient_cfg = import_cfg.get("gradient_overrides") or {}
    nested_cfg = gradient_cfg.get("metadata_overrides")
    if isinstance(nested_cfg, dict):
        return nested_cfg

    return {}


def _rule_match_cfg(rule: dict[str, Any]) -> dict[str, Any]:
    match_cfg = rule.get("match")
    if not isinstance(match_cfg, dict):
        match_cfg = {}
    else:
        match_cfg = dict(match_cfg)

    for key in ("bids_name", "entities", "json_fields", "modality", "resolution", "resolution_tolerance"):
        if key not in match_cfg and key in rule:
            match_cfg[key] = rule.get(key)
    return match_cfg


def _metadata_override_updates(rule: dict[str, Any], logger=None, json_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    updates = rule.get("metadata") or rule.get("updates")
    if isinstance(updates, dict) and updates:
        return updates

    match_cfg = _rule_match_cfg(rule)
    nested_updates = match_cfg.get("metadata") or match_cfg.get("updates")
    if isinstance(nested_updates, dict) and nested_updates:
        if logger:
            target = json_path.name if json_path else "metadata override rule"
            logger.warning(
                "Detected metadata override fields nested under `match` for %s. "
                "This layout is deprecated; move `metadata` to the rule top level.",
                target,
            )
        return nested_updates

    return None


def _normalized_match_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, Path):
        return str(value).strip().lower()
    if isinstance(value, list):
        return [_normalized_match_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalized_match_value(item) for item in value)
    return value


def _values_match(found: Any, expected: Any) -> bool:
    return _normalized_match_value(found) == _normalized_match_value(expected)


def _candidate_values_match(found: Any, expected: Any) -> bool:
    if isinstance(found, (list, tuple, set)):
        return any(_candidate_values_match(item, expected) for item in found)
    if isinstance(expected, (list, tuple, set)):
        return any(_candidate_values_match(found, item) for item in expected)
    return _values_match(found, expected)


def _infer_bids_modality(json_path: Path) -> list[str]:
    entity_path = _entity_source_path(json_path)
    entities = get_entities_from_path(entity_path)
    suffix = entities.get("suffix")
    candidates = []
    if suffix:
        candidates.append(str(suffix).lower())

    for part in entity_path.parts:
        lowered = str(part).lower()
        if lowered in {"anat", "dwi", "fmap", "func", "perf"}:
            candidates.append(lowered)

    # Preserve order while deduplicating.
    return list(dict.fromkeys(candidates))


def _extract_image_resolution_mm(json_path: Path) -> Optional[tuple[float, float, float]]:
    nifti_path = _sidecar_nifti_path(json_path)
    if nifti_path and nifti_path.exists():
        zooms = nib.load(str(nifti_path)).header.get_zooms()[:3]
        if len(zooms) == 3:
            return tuple(float(z) for z in zooms)

    payload = _load_json_payload(json_path)
    pixel_spacing = payload.get("PixelSpacing")
    slice_thickness = payload.get("SliceThickness")
    if isinstance(pixel_spacing, list) and len(pixel_spacing) >= 2 and slice_thickness is not None:
        try:
            return (
                float(pixel_spacing[0]),
                float(pixel_spacing[1]),
                float(slice_thickness),
            )
        except (TypeError, ValueError):
            return None
    return None


def _normalize_resolution_spec(
    spec: Any,
    default_tolerance: float = 0.05,
) -> Optional[tuple[tuple[float, float, float], float]]:
    tolerance = default_tolerance
    value = spec
    if isinstance(spec, dict):
        value = spec.get("voxel_size", spec.get("resolution", spec.get("value")))
        tolerance = float(spec.get("tolerance", default_tolerance))

    if isinstance(value, (int, float)):
        voxel_size = (float(value), float(value), float(value))
    elif isinstance(value, (list, tuple)):
        if len(value) == 1:
            scalar = float(value[0])
            voxel_size = (scalar, scalar, scalar)
        elif len(value) == 3:
            voxel_size = tuple(float(v) for v in value)
        else:
            raise ProcessingError(
                "Import rule `match.resolution` must be a scalar, a 3-value list, or a dict with `voxel_size`."
            )
    else:
        return None

    return voxel_size, tolerance


def _resolution_matches(json_path: Path, expected: Any, tolerance: float = 0.05) -> bool:
    found = _extract_image_resolution_mm(json_path)
    if found is None:
        return False

    normalized = _normalize_resolution_spec(expected, default_tolerance=tolerance)
    if normalized is None:
        return False

    target, tol = normalized
    return all(abs(found[idx] - target[idx]) <= tol for idx in range(3))


def _match_import_rule(json_path: Path, rule: dict[str, Any]) -> bool:
    match_cfg = _rule_match_cfg(rule)
    if not match_cfg:
        return False
    payload = None
    entity_path = _entity_source_path(json_path)
    matched_any = False

    bids_name = match_cfg.get("bids_name")
    if bids_name:
        if not (
            _values_match(get_nifti_stem(entity_path), bids_name)
            or _values_match(json_path.stem, bids_name)
        ):
            return False
        matched_any = True

    modality = match_cfg.get("modality")
    if modality:
        found_modalities = _infer_bids_modality(json_path)
        if not _candidate_values_match(found_modalities, modality):
            return False
        matched_any = True

    resolution = match_cfg.get("resolution")
    if resolution is not None:
        resolution_tolerance = float(match_cfg.get("resolution_tolerance", 0.05))
        if not _resolution_matches(json_path, resolution, tolerance=resolution_tolerance):
            return False
        matched_any = True

    entities_match = match_cfg.get("entities")
    if isinstance(entities_match, dict):
        found = get_entities_from_path(entity_path)
        for key, expected in entities_match.items():
            if not _values_match(found.get(key), expected):
                return False
        matched_any = True

    json_fields = match_cfg.get("json_fields")
    if isinstance(json_fields, dict):
        payload = payload or _load_json_payload(json_path)
        for key, expected in json_fields.items():
            if not _values_match(payload.get(key), expected):
                return False
        matched_any = True

    return matched_any


def _resolve_import_rule(
    json_path: Path,
    rules: list[dict[str, Any]],
    label: str,
) -> Optional[dict[str, Any]]:
    matches = [rule for rule in rules if _match_import_rule(json_path, rule)]
    if not matches:
        return None
    if len(matches) > 1:
        raise ProcessingError(f"Multiple {label} rules matched {json_path.name}")
    return matches[0]


def _sidecar_nifti_path(json_path: Path) -> Optional[Path]:
    nifti_path = json_path.with_suffix(".nii.gz")
    if nifti_path.exists():
        return nifti_path
    alt = json_path.with_suffix(".nii")
    if alt.exists():
        return alt
    return None


def _is_relaxometry_sidecar(json_path: Path) -> bool:
    entities = get_entities_from_path(_entity_source_path(json_path))
    acq = str(entities.get("acq", "")).lower()
    desc = str(entities.get("desc", "")).lower()
    suffix = str(entities.get("suffix", "")).lower()
    name = json_path.name.lower()
    tokens = (acq, desc, suffix, name)
    return any(
        marker in token
        for token in tokens
        for marker in ("spgr", "ssfp", "irspgr", "vfa", "afi", "b1")
    )

class Dcm2NiixStep(BaseProcessingStep):
    """
    Step to convert DICOMs to NIfTI using dcm2niix.
    """
    def run(self, dicom_dir: Path, output_dir: Path, **kwargs) -> Path:
        self.logger.info(f"Converting DICOMs in {dicom_dir} to NIfTI...")
        
        # Get options from config
        cfg = self.config.get("import", {}).get("dcm2niix", {})
        filename = cfg.get("filename", "%p_%s_%t")
        compress = cfg.get("compress", True)
        bids = cfg.get("bids", True)
        
        # Override with kwargs if provided
        filename = kwargs.get("filename", filename)
        
        try:
            dcm2niix.dcm2niix(
                in_dir=dicom_dir,
                out_dir=output_dir,
                filename=filename,
                compress=compress,
                bids=bids,
                verbose=self.config.debug
            )
            return output_dir
        except Exception as e:
            self.logger.error(f"dcm2niix conversion failed: {e}")
            if self.config.stop_on_error:
                raise ProcessingError(f"dcm2niix failed: {e}")
            return dicom_dir

class Dcm2BidsStep(BaseProcessingStep):
    """
    Step to convert DICOMs to BIDS structure using dcm2bids.
    """
    @staticmethod
    def _subject_output_root(bids_dir: Path, subject: str, session: Optional[str]) -> Path:
        root = Path(bids_dir) / f"sub-{str(subject).removeprefix('sub-')}"
        if session:
            root = root / f"ses-{str(session).removeprefix('ses-')}"
        return root

    @staticmethod
    def _tmp_dcm2bids_root(bids_dir: Path, subject: str, session: Optional[str]) -> Path:
        name = f"sub-{str(subject).removeprefix('sub-')}"
        if session:
            name += f"_ses-{str(session).removeprefix('ses-')}"
        return Path(bids_dir) / "tmp_dcm2bids" / name

    def run(self, dicom_dir: Path, bids_dir: Path, **kwargs) -> Path:
        self.logger.info(f"Running dcm2bids on {dicom_dir}...")
        
        # Subject and Session from context or kwargs
        sub = _normalize_context_label(kwargs.get("subject"), "sub-")
        if sub is None:
            sub = _normalize_context_label(self.config.get("import.subject"), "sub-")

        ses = _normalize_context_label(kwargs.get("session"), "ses-")
        if ses is None:
            ses = _normalize_context_label(self.config.get("import.session"), "ses-")
        
        if not sub:
            raise ValidationError("Subject ID required for dcm2bids.")

        # Config file is mandatory for dcm2bids
        import_cfg = self.config.get("import", {}).get("dcm2bids", {})
        config_file = import_cfg.get("config_file")
        if not config_file:
            # Check for a default in common locations
            default_config = self.config.bids_dir / "code" / "dcm2bids_config.json"
            if default_config.exists():
                config_file = default_config
            else:
                raise ValidationError("dcm2bids config_file not specified in configuration.")
        
        config_file = Path(config_file)
        if not config_file.exists():
            raise ValidationError(f"dcm2bids config file not found: {config_file}")

        subject_root = self._subject_output_root(bids_dir, sub, ses)
        tmp_root = self._tmp_dcm2bids_root(bids_dir, sub, ses)
        force_dcm2bids = bool(import_cfg.get("force_dcm2bids", False) or kwargs.get("force_dcm2bids", False))
        if tmp_root.exists() and not subject_root.exists():
            self.logger.warning(
                "Detected stale dcm2bids temporary output at %s without a corresponding BIDS target at %s. "
                "Forcing dcm2bids to rerun.",
                tmp_root,
                subject_root,
            )
            force_dcm2bids = True

        try:
            dcm2bids.dcm2bids(
                dicom_dir=dicom_dir,
                participant_id=sub,
                config_file=config_file,
                output_dir=bids_dir,
                session_id=ses,
                clobber=bool(import_cfg.get("clobber", False) or kwargs.get("clobber", False)),
                force_dcm2bids=force_dcm2bids,
                force_dccm=bool(import_cfg.get("force_dccm", False) or kwargs.get("force_dccm", False)),
                extra_args=str(import_cfg.get("extra_args", "") or kwargs.get("extra_args", "")),
            )
            return bids_dir
        except Exception as e:
            self.logger.error(f"dcm2bids failed: {e}")
            if self.config.stop_on_error:
                raise ProcessingError(f"dcm2bids failed: {e}")
            return bids_dir


class ImportGnlMetadataStep(GEGnlMetadataEnrichmentStep):
    """
    Alias step for import workflow readability.
    """
    pass


class ImportGradientOverrideStep(BaseProcessingStep):
    """
    Replace imported DWI bval/bvec sidecars with curated gradient tables.
    """

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def _target_dwi_sidecars(self, output_dir: Path, context: dict[str, Any]) -> list[Path]:
        imported = context.get("imported_dwi_sidecars") or []
        candidates = {
            Path(p)
            for p in imported
            if Path(p).exists() and _is_primary_bids_output(Path(p), output_dir)
        }
        search_root = _context_search_root(output_dir, context)
        candidates.update(
            path
            for path in search_root.rglob("*_dwi.json")
            if path.is_file() and _is_primary_bids_output(path, output_dir)
        )
        return sorted(candidates)

    def _match_rule(self, json_path: Path, rule: dict[str, Any]) -> bool:
        return _match_import_rule(json_path, rule)

    def _resolve_rule(self, json_path: Path, rules: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return _resolve_import_rule(json_path, rules, label="gradient override")

    def _load_bvals(self, path: Path) -> np.ndarray:
        vals = np.loadtxt(path, dtype=np.float64)
        if np.isscalar(vals):
            vals = np.array([vals], dtype=np.float64)
        return np.ravel(vals)

    def _load_bvecs(self, path: Path) -> np.ndarray:
        vecs = np.loadtxt(path, dtype=np.float64)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        if vecs.shape[0] != 3 and vecs.shape[1] == 3:
            vecs = vecs.T
        if vecs.shape[0] != 3:
            raise ProcessingError(f"Expected 3-row bvec file: {path}")
        return vecs

    def _validate_gradients(
        self,
        nifti_path: Path,
        bval_path: Optional[Path],
        bvec_path: Optional[Path],
        require_both: bool,
    ) -> None:
        shape = nib.load(str(nifti_path)).shape
        nvols = shape[3] if len(shape) >= 4 else 1
        if bval_path:
            bvals = self._load_bvals(bval_path)
            if bvals.size != nvols:
                raise ProcessingError(
                    f"bval volume count mismatch for {nifti_path.name}: expected {nvols}, found {bvals.size}"
                )
        if bvec_path:
            bvecs = self._load_bvecs(bvec_path)
            if bvecs.shape[1] != nvols:
                raise ProcessingError(
                    f"bvec volume count mismatch for {nifti_path.name}: expected {nvols}, found {bvecs.shape[1]}"
                )
        if require_both and not (bval_path and bvec_path):
            raise ProcessingError(f"Gradient override for {nifti_path.name} requires both bval and bvec")

    def _update_sidecar(self, json_path: Path, rule: dict[str, Any], bval_path: Optional[Path], bvec_path: Optional[Path]) -> None:
        payload = _load_json_payload(json_path)
        payload["GradientTableOverride"] = {
            "Applied": True,
            "Source": "import.gradient_overrides",
            "MatchingRule": rule.get("match", {}),
            "ReplacementBval": str(bval_path) if bval_path else None,
            "ReplacementBvec": str(bvec_path) if bvec_path else None,
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def run(self, first_arg, output_dir: Path, **kwargs):
        context = first_arg if isinstance(first_arg, dict) else {}
        override_cfg = (self.config.get("import", {}) or {}).get("gradient_overrides", {})
        if not override_cfg.get("enabled", False):
            return context

        rules = override_cfg.get("rules") or []
        if not rules:
            self.logger.warning("Gradient overrides enabled but no rules were configured")
            return context

        require_both = bool(override_cfg.get("require_both", True))
        stop_on_mismatch = bool(override_cfg.get("stop_on_mismatch", True))
        sidecars = self._target_dwi_sidecars(output_dir, context)
        updated = 0
        unmatched = []

        for json_path in sidecars:
            rule = self._resolve_rule(json_path, rules)
            if rule is None:
                unmatched.append(json_path.name)
                continue

            nifti_path = _sidecar_nifti_path(json_path)
            if nifti_path is None:
                raise ProcessingError(f"Could not locate NIfTI for gradient override sidecar: {json_path}")

            dst_bval = json_path.with_suffix(".bval")
            dst_bvec = json_path.with_suffix(".bvec")
            src_bval = Path(rule["bval"]) if rule.get("bval") else None
            src_bvec = Path(rule["bvec"]) if rule.get("bvec") else None

            if src_bval and not src_bval.exists():
                raise ProcessingError(f"Gradient override bval not found: {src_bval}")
            if src_bvec and not src_bvec.exists():
                raise ProcessingError(f"Gradient override bvec not found: {src_bvec}")

            self._validate_gradients(nifti_path, src_bval, src_bvec, require_both=require_both)

            if src_bval:
                shutil.copyfile(src_bval, dst_bval)
            if src_bvec:
                shutil.copyfile(src_bvec, dst_bvec)
            self._update_sidecar(json_path, rule, src_bval, src_bvec)
            updated += 1

        if unmatched:
            msg = "No gradient override rule matched sidecars: " + ", ".join(unmatched)
            if stop_on_mismatch:
                raise ProcessingError(msg)
            self.logger.warning(msg)

        self.logger.info(f"Applied gradient overrides to {updated} DWI sidecar(s)")
        context["gradient_override_sidecars_updated"] = updated
        return context


class ImportMetadataOverrideStep(BaseProcessingStep):
    """
    Update imported image sidecars with curated metadata such as variable FlipAngle
    arrays, SSFP PhaseCycling arrays, or scalar AFI fields such as TRRatio.
    """

    _LENGTH_VALIDATED_FIELDS = {
        "FlipAngle",
        "PhaseCycling",
        "RepetitionTime",
        "InversionTime",
        "EchoTrainLength",
    }

    def validate_inputs(self, first_arg, **kwargs) -> None:
        pass

    def validate_outputs(self, result) -> None:
        pass

    def _target_sidecars(self, output_dir: Path, context: dict[str, Any], rules: list[dict[str, Any]]) -> list[Path]:
        imported = context.get("imported_json_sidecars") or []
        candidates = {
            Path(p)
            for p in imported
            if (
                Path(p).exists()
                and _is_primary_bids_output(Path(p), output_dir)
                and _sidecar_nifti_path(Path(p)) is not None
            )
        }

        search_root = _context_search_root(output_dir, context)
        for path in sorted(search_root.rglob("*.json")):
            if not _is_primary_bids_output(path, output_dir):
                continue
            if _sidecar_nifti_path(path) is not None:
                candidates.add(path)

        return sorted(
            path for path in candidates
            if self._resolve_rule(path, rules) is not None
        )

    def _resolve_rule(self, json_path: Path, rules: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        return _resolve_import_rule(json_path, rules, label="metadata override")

    def _nifti_nvols(self, nifti_path: Path) -> int:
        shape = nib.load(str(nifti_path)).shape
        return shape[3] if len(shape) >= 4 else 1

    def _validate_updates(self, nifti_path: Path, updates: dict[str, Any]) -> None:
        nvols = self._nifti_nvols(nifti_path)
        for key, value in updates.items():
            if key not in self._LENGTH_VALIDATED_FIELDS:
                continue
            if not isinstance(value, list):
                continue
            if len(value) != nvols:
                raise ProcessingError(
                    f"{key} length mismatch for {nifti_path.name}: expected {nvols}, found {len(value)}"
                )

    def _extract_resample_resolution(
        self,
        rule: dict[str, Any],
        updates: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[tuple[float, float, float]]]:
        updates = dict(updates or {})
        resample_spec = (
            rule.get("resample_resolution")
            or rule.get("resample")
            or rule.get("resolution")
            or rule.get("Resolution")
            or rule.get("voxel_size")
            or rule.get("VoxelSize")
            or _rule_match_cfg(rule).get("resample_resolution")
            or _rule_match_cfg(rule).get("resample")
            or _rule_match_cfg(rule).get("voxel_size")
            or _rule_match_cfg(rule).get("VoxelSize")
            or updates.pop("resample_resolution", None)
            or updates.pop("resample", None)
            or updates.pop("Resolution", None)
            or updates.pop("resolution", None)
            or updates.pop("VoxelSize", None)
            or updates.pop("voxel_size", None)
        )
        if resample_spec is None:
            return updates, None

        normalized = _normalize_resolution_spec(resample_spec)
        if normalized is None:
            raise ProcessingError("Invalid metadata override resample resolution specification")
        return updates, normalized[0]

    def _resample_image(
        self,
        nifti_path: Path,
        json_path: Path,
        target_resolution: tuple[float, float, float],
    ) -> bool:
        found_resolution = _extract_image_resolution_mm(json_path)
        if found_resolution is not None and all(
            abs(found_resolution[idx] - target_resolution[idx]) <= 1e-4
            for idx in range(3)
        ):
            return False

        modalities = _infer_bids_modality(json_path)
        if _candidate_values_match(modalities, "dwi"):
            raise ProcessingError(
                f"Import-time resampling is not supported for DWI sidecars ({json_path.name}). "
                "Use the diffusion preprocessing resample step instead."
            )

        from ...core.run import run_cmd

        suffix = "".join(nifti_path.suffixes)
        tmp_path = nifti_path.parent / f".{get_nifti_stem(nifti_path)}_resample_tmp{suffix}"
        res_args = " ".join(str(v) for v in target_resolution)
        run_cmd(f"mri_convert {nifti_path} {tmp_path} -vs {res_args}", label="import_resample")
        shutil.move(tmp_path, nifti_path)
        return True

    def _update_sidecar(
        self,
        json_path: Path,
        rule: dict[str, Any],
        updates: dict[str, Any],
        resampled_resolution: Optional[tuple[float, float, float]] = None,
    ) -> None:
        payload = _load_json_payload(json_path)
        payload.update(updates)
        payload["MetadataOverride"] = {
            "Applied": True,
            "Source": "import.metadata_overrides",
            "MatchingRule": rule.get("match", {}),
            "UpdatedFields": sorted(updates.keys()),
        }
        if resampled_resolution is not None:
            payload["MetadataOverride"]["ResampledToResolutionMm"] = list(resampled_resolution)
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def run(self, first_arg, output_dir: Path, **kwargs):
        context = first_arg if isinstance(first_arg, dict) else {}
        override_cfg = _import_metadata_override_cfg(self.config)
        if not override_cfg.get("enabled", False):
            return context

        gradient_cfg = (self.config.get("import", {}) or {}).get("gradient_overrides") or {}
        if isinstance(gradient_cfg.get("metadata_overrides"), dict):
            self.logger.warning(
                "Detected `import.gradient_overrides.metadata_overrides` in config. "
                "This nesting is deprecated; use `import.metadata_overrides` instead."
            )

        rules = override_cfg.get("rules") or []
        if not rules:
            self.logger.warning("Metadata overrides enabled but no rules were configured")
            return context

        sidecars = self._target_sidecars(output_dir, context, rules)
        if not sidecars:
            self.logger.info(
                "Metadata overrides enabled, but no imported image sidecars matched the configured rules. "
                "Skipping metadata override step."
            )
            context["metadata_override_sidecars_updated"] = 0
            return context

        updated = 0

        for json_path in sidecars:
            rule = self._resolve_rule(json_path, rules)
            if rule is None:
                continue

            raw_updates = _metadata_override_updates(rule, logger=self.logger, json_path=json_path) or {}
            updates, resample_resolution = self._extract_resample_resolution(rule, raw_updates)
            if not updates and resample_resolution is None:
                raise ProcessingError(
                    f"Metadata override rule for {json_path.name} is missing a metadata block or resample request"
                )

            nifti_path = _sidecar_nifti_path(json_path)
            if nifti_path is None:
                raise ProcessingError(f"Could not locate NIfTI for metadata override sidecar: {json_path}")

            if updates:
                self._validate_updates(nifti_path, updates)
            resampled = False
            if resample_resolution is not None:
                resampled = self._resample_image(nifti_path, json_path, resample_resolution)
            self._update_sidecar(
                json_path,
                rule,
                updates,
                resampled_resolution=resample_resolution if resampled else None,
            )
            updated += 1

        self.logger.info(f"Applied metadata overrides to {updated} image sidecar(s)")
        context["metadata_override_sidecars_updated"] = updated
        return context
