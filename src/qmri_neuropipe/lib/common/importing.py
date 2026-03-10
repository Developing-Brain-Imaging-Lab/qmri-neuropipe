
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


def _match_import_rule(json_path: Path, rule: dict[str, Any]) -> bool:
    match_cfg = rule.get("match") or {}
    if not match_cfg:
        return False
    payload = None
    entity_path = _entity_source_path(json_path)

    bids_name = match_cfg.get("bids_name")
    if bids_name:
        if get_nifti_stem(entity_path) == bids_name or json_path.stem == bids_name:
            return True

    entities_match = match_cfg.get("entities")
    if isinstance(entities_match, dict):
        found = get_entities_from_path(entity_path)
        for key, expected in entities_match.items():
            if str(found.get(key)) != str(expected):
                return False
        return True

    json_fields = match_cfg.get("json_fields")
    if isinstance(json_fields, dict):
        payload = payload or _load_json_payload(json_path)
        for key, expected in json_fields.items():
            if str(payload.get(key)) != str(expected):
                return False
        return True

    return False


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
    def run(self, dicom_dir: Path, bids_dir: Path, **kwargs) -> Path:
        self.logger.info(f"Running dcm2bids on {dicom_dir}...")
        
        # Subject and Session from context or kwargs
        sub = kwargs.get("subject") or self.config.get("subject")
        ses = kwargs.get("session") or self.config.get("session")
        
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

        try:
            dcm2bids.dcm2bids(
                dicom_dir=dicom_dir,
                participant_id=sub,
                config_file=config_file,
                output_dir=bids_dir,
                session_id=ses,
                clobber=kwargs.get("clobber", False)
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
        if imported:
            return [Path(p) for p in imported if Path(p).exists()]
        return sorted(Path(output_dir).rglob("*_dwi.json"))

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
        nvols = nib.load(str(nifti_path)).shape[3]
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
    arrays or SSFP PhaseCycling arrays.
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

    def _target_sidecars(self, output_dir: Path, context: dict[str, Any]) -> list[Path]:
        imported = context.get("imported_json_sidecars") or []
        if imported:
            return [
                Path(p)
                for p in imported
                if Path(p).exists() and _is_relaxometry_sidecar(Path(p))
            ]

        candidates: list[Path] = []
        for path in sorted(Path(output_dir).rglob("*.json")):
            if _sidecar_nifti_path(path) is not None and _is_relaxometry_sidecar(path):
                candidates.append(path)
        return candidates

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

    def _update_sidecar(self, json_path: Path, rule: dict[str, Any], updates: dict[str, Any]) -> None:
        payload = _load_json_payload(json_path)
        payload.update(updates)
        payload["MetadataOverride"] = {
            "Applied": True,
            "Source": "import.metadata_overrides",
            "MatchingRule": rule.get("match", {}),
            "UpdatedFields": sorted(updates.keys()),
        }
        with json_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    def run(self, first_arg, output_dir: Path, **kwargs):
        context = first_arg if isinstance(first_arg, dict) else {}
        override_cfg = (self.config.get("import", {}) or {}).get("metadata_overrides", {})
        if not override_cfg.get("enabled", False):
            return context

        rules = override_cfg.get("rules") or []
        if not rules:
            self.logger.warning("Metadata overrides enabled but no rules were configured")
            return context

        stop_on_mismatch = bool(override_cfg.get("stop_on_mismatch", True))
        sidecars = self._target_sidecars(output_dir, context)
        if not sidecars:
            self.logger.warning("Metadata overrides enabled, but no relaxometry image sidecars were found to update")
            context["metadata_override_sidecars_updated"] = 0
            return context

        updated = 0
        unmatched = []

        for json_path in sidecars:
            rule = self._resolve_rule(json_path, rules)
            if rule is None:
                unmatched.append(json_path.name)
                continue

            updates = rule.get("metadata") or rule.get("updates")
            if not isinstance(updates, dict) or not updates:
                raise ProcessingError(f"Metadata override rule for {json_path.name} is missing a metadata block")

            nifti_path = _sidecar_nifti_path(json_path)
            if nifti_path is None:
                raise ProcessingError(f"Could not locate NIfTI for metadata override sidecar: {json_path}")

            self._validate_updates(nifti_path, updates)
            self._update_sidecar(json_path, rule, updates)
            updated += 1

        if unmatched:
            msg = "No metadata override rule matched sidecars: " + ", ".join(unmatched)
            if stop_on_mismatch:
                raise ProcessingError(msg)
            self.logger.warning(msg)

        self.logger.info(f"Applied metadata overrides to {updated} image sidecar(s)")
        context["metadata_override_sidecars_updated"] = updated
        return context
