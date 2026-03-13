
from pathlib import Path
from typing import Optional
import tarfile
from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.core import ProcessingError
from qmri_neuropipe.lib.common.importing import (
    Dcm2NiixStep,
    Dcm2BidsStep,
    ImportGnlMetadataStep,
    ImportGradientOverrideStep,
    ImportMetadataOverrideStep,
    _import_metadata_override_cfg,
)

class ImportWorkflow(BaseWorkflow):
    """
    Workflow for converting DICOMs to BIDS/NIfTI.
    """
    def _initialize_steps(self):
        self.modality = "Import"
        self.steps = []
        
    def build_pipeline(self, context: dict):
        self.steps = []
        import_cfg = self.config.get('import', {})
        
        method = import_cfg.get('method', 'dcm2bids')
        
        if method == 'dcm2bids':
            self.add_step(Dcm2BidsStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))
        elif method == 'dcm2niix':
            self.add_step(Dcm2NiixStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

        metadata_override_cfg = _import_metadata_override_cfg(self.config)
        if metadata_override_cfg.get('enabled', False):
            self.add_step(ImportMetadataOverrideStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

        if import_cfg.get('gradient_overrides', {}).get('enabled', False):
            self.add_step(ImportGradientOverrideStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

        if import_cfg.get('gnl_metadata', {}).get('enabled', False):
            self.add_step(ImportGnlMetadataStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

    @staticmethod
    def _archive_source_key(path: Path) -> str:
        name = path.name
        for suffix in (".tar.gz", ".tgz", ".tar"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return path.stem

    @staticmethod
    def _target_tokens(context: dict) -> list[str]:
        tokens: list[str] = []
        subject = context.get("subject")
        session = context.get("session")
        if subject:
            subj = str(subject).removeprefix("sub-")
            tokens.extend([subj.lower(), f"sub-{subj.lower()}"])
        if session:
            ses = str(session).removeprefix("ses-")
            tokens.extend([ses.lower(), f"ses-{ses.lower()}"])
        return tokens

    @staticmethod
    def _path_matches_any_token(path: Path, tokens: list[str]) -> bool:
        haystack = str(path).lower()
        return any(token in haystack for token in tokens)

    def _archives_for_context(self, archives: list[Path], dicom_dir: Path, context: dict) -> list[Path]:
        tokens = self._target_tokens(context)
        if not tokens:
            return archives

        matches = [archive for archive in archives if self._path_matches_any_token(archive, tokens)]
        if matches:
            return matches

        if self._path_matches_any_token(dicom_dir, tokens):
            return archives

        raise ProcessingError(
            f"Import source {dicom_dir} contains multiple archives, but none could be associated with "
            f"the requested subject/session tokens {tokens}."
        )

    def _resolve_import_source(self, dicom_dir: Path, context: Optional[dict] = None) -> Path:
        """
        Resolve an import source directory from either:
        - a directory containing DICOM files
        - a single .tgz/.tar.gz archive
        - a directory containing one or more .tgz/.tar.gz archives
        - a directory containing subdirectories that each contain archive files
        """
        dicom_dir = Path(dicom_dir)
        context = context or {}
        archive_suffixes = (".tgz", ".tar.gz", ".tar")

        if dicom_dir.is_file():
            archives = [dicom_dir]
            source_key = self._archive_source_key(dicom_dir)
        else:
            archives = sorted(
                p for p in dicom_dir.rglob("*")
                if p.is_file() and any(str(p).endswith(sfx) for sfx in archive_suffixes)
            )
            source_key = dicom_dir.name

        if not archives:
            return dicom_dir

        work_root = self.config.work_dir or (self.config.output_dir / "work")
        extract_root = work_root / "import_archives" / source_key
        extract_root.mkdir(parents=True, exist_ok=True)
        subject = context.get("subject")
        session = context.get("session")
        if subject:
            target_root = extract_root / f"sub-{str(subject).removeprefix('sub-')}"
            if session:
                target_root = target_root / f"ses-{str(session).removeprefix('ses-')}"
        else:
            target_root = extract_root
        target_root.mkdir(parents=True, exist_ok=True)

        archives = self._archives_for_context(archives, dicom_dir, context)

        for archive in archives:
            archive_key = self._archive_source_key(archive)
            rel_parent = Path()
            if not dicom_dir.is_file():
                try:
                    rel_parent = archive.parent.relative_to(dicom_dir)
                except ValueError:
                    rel_parent = Path()
            archive_extract_dir = target_root / rel_parent / archive_key
            archive_extract_dir.mkdir(parents=True, exist_ok=True)
            stamp = archive_extract_dir / ".extract.done"
            if stamp.exists():
                continue
            self.logger.info(f"Extracting import archive: {archive}")
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(archive_extract_dir)
            stamp.write_text("ok\n")

        return target_root

    def _snapshot_dwi_sidecars(self, output_dir: Path) -> dict[Path, float]:
        if not Path(output_dir).exists():
            return {}
        return {
            path: path.stat().st_mtime
            for path in Path(output_dir).rglob("*_dwi.json")
            if path.is_file()
        }

    def _snapshot_image_sidecars(self, output_dir: Path) -> dict[Path, float]:
        if not Path(output_dir).exists():
            return {}
        sidecars: dict[Path, float] = {}
        for path in Path(output_dir).rglob("*.json"):
            if not path.is_file():
                continue
            nii_gz = path.with_suffix(".nii.gz")
            nii = path.with_suffix(".nii")
            if nii_gz.exists() or nii.exists():
                sidecars[path] = path.stat().st_mtime
        return sidecars

    def _snapshot_import_outputs(self, output_dir: Path) -> dict[Path, float]:
        if not Path(output_dir).exists():
            return {}
        patterns = ("*.nii", "*.nii.gz", "*.json", "*.bval", "*.bvec")
        files: dict[Path, float] = {}
        for pattern in patterns:
            for path in Path(output_dir).rglob(pattern):
                if path.is_file():
                    files[path] = path.stat().st_mtime
        return files

    def _new_or_updated_sidecars(self, before: dict[Path, float], output_dir: Path) -> list[Path]:
        after = self._snapshot_dwi_sidecars(output_dir)
        changed: list[Path] = []
        for path, mtime in after.items():
            if path not in before or mtime > before[path]:
                changed.append(path)
        return sorted(changed)

    def _new_or_updated_image_sidecars(self, before: dict[Path, float], output_dir: Path) -> list[Path]:
        after = self._snapshot_image_sidecars(output_dir)
        changed: list[Path] = []
        for path, mtime in after.items():
            if path not in before or mtime > before[path]:
                changed.append(path)
        return sorted(changed)

    def _new_or_updated_outputs(self, before: dict[Path, float], output_dir: Path) -> list[Path]:
        after = self._snapshot_import_outputs(output_dir)
        changed: list[Path] = []
        for path, mtime in after.items():
            if path not in before or mtime > before[path]:
                changed.append(path)
        return sorted(changed)

    def _existing_subject_outputs(self, output_dir: Path, context: dict) -> list[Path]:
        root = Path(output_dir)
        if not root.exists():
            return []

        subject = context.get("subject")
        session = context.get("session")
        if subject:
            subj_root = root / f"sub-{subject}"
            if session:
                subj_root = subj_root / f"ses-{session}"
            if subj_root.exists():
                patterns = ("*.nii", "*.nii.gz", "*.json", "*.bval", "*.bvec")
                found: list[Path] = []
                for pattern in patterns:
                    found.extend(p for p in subj_root.rglob(pattern) if p.is_file())
                return sorted(set(found))

        return sorted(self._snapshot_import_outputs(root).keys())
            
    def run(self, dicom_dir: Path, output_dir: Path, context: dict) -> dict:
        self.logger.info("Starting Import Workflow")
        resolved_dicom_dir = self._resolve_import_source(Path(dicom_dir), context)
        context = dict(context)
        context["dicom_dir"] = resolved_dicom_dir
        step_context = {k: v for k, v in context.items() if k != "dicom_dir"}
        outputs_before = self._snapshot_import_outputs(output_dir)
        sidecars_before = self._snapshot_dwi_sidecars(output_dir)
        image_sidecars_before = self._snapshot_image_sidecars(output_dir)
        had_existing_outputs = bool(outputs_before)
        
        for step in self.steps:
            if isinstance(step, (Dcm2BidsStep, Dcm2NiixStep)):
                step.run(resolved_dicom_dir, output_dir, **step_context)
                imported_outputs = self._new_or_updated_outputs(outputs_before, output_dir)
                imported_sidecars = self._new_or_updated_sidecars(sidecars_before, output_dir)
                imported_image_sidecars = self._new_or_updated_image_sidecars(image_sidecars_before, output_dir)
                context["imported_output_files"] = [str(p) for p in imported_outputs]
                context["imported_dwi_sidecars"] = [str(p) for p in imported_sidecars]
                context["imported_json_sidecars"] = [str(p) for p in imported_image_sidecars]
                if not imported_outputs:
                    import_method = self.config.get("import.method", "dcm2bids")
                    existing_target_outputs = self._existing_subject_outputs(output_dir, context)
                    if existing_target_outputs:
                        self.logger.info(
                            "Import created no new files, but existing imported outputs were found for the requested target. "
                            "Continuing with the existing BIDS data."
                        )
                        context["imported_output_files"] = [str(p) for p in existing_target_outputs]
                        if not context.get("imported_dwi_sidecars"):
                            context["imported_dwi_sidecars"] = [
                                str(p) for p in existing_target_outputs if p.name.endswith("_dwi.json")
                            ]
                        if not context.get("imported_json_sidecars"):
                            context["imported_json_sidecars"] = [
                                str(p)
                                for p in existing_target_outputs
                                if p.suffix == ".json" and (p.with_suffix(".nii.gz").exists() or p.with_suffix(".nii").exists())
                            ]
                        continue
                    if had_existing_outputs:
                        msg = (
                            "Import step completed without creating or updating any output files. "
                            "This usually means the dcm2bids/dcm2niix mapping did not match the source series "
                            "or outputs already exist unchanged."
                        )
                    else:
                        msg = (
                            "Import step completed without creating any NIfTI/BIDS output files under "
                            f"{output_dir}. For {import_method}, this most likely means the conversion/config "
                            "did not match any source series. With dcm2bids, the next thing to check is whether "
                            "the dcm2bids config descriptions actually match the source SeriesDescription / "
                            "ProtocolName values."
                        )
                    if self.config.stop_on_error:
                        raise ProcessingError(msg)
                    self.logger.warning(msg)
                else:
                    self.logger.info(f"Import created/updated {len(imported_outputs)} file(s)")
            else:
                result = step(context, output_dir=output_dir, **context)
                if isinstance(result, dict):
                    context.update(result)
                
        return context
