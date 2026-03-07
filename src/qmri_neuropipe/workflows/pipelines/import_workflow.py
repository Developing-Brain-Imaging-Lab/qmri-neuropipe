
from pathlib import Path
from typing import Optional
import tarfile
from qmri_neuropipe.core import BaseWorkflow
from qmri_neuropipe.core import ProcessingError
from qmri_neuropipe.lib.common.importing import Dcm2NiixStep, Dcm2BidsStep, ImportGnlMetadataStep

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

        if import_cfg.get('gnl_metadata', {}).get('enabled', False):
            self.add_step(ImportGnlMetadataStep(
                config=self.config,
                logger=self.logger,
                provenance=self.provenance
            ))

    def _resolve_import_source(self, dicom_dir: Path) -> Path:
        """
        Resolve an import source directory from either:
        - a directory containing DICOM files
        - a single .tgz/.tar.gz archive
        - a directory containing one or more .tgz/.tar.gz archives
        - a directory containing subdirectories that each contain archive files
        """
        dicom_dir = Path(dicom_dir)
        archive_suffixes = (".tgz", ".tar.gz", ".tar")

        if dicom_dir.is_file():
            archives = [dicom_dir]
            source_key = dicom_dir.stem.replace(".tar", "")
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

        for archive in archives:
            stamp = extract_root / f".{archive.name}.done"
            if stamp.exists():
                continue
            self.logger.info(f"Extracting import archive: {archive}")
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(extract_root)
            stamp.write_text("ok\n")

        return extract_root

    def _snapshot_dwi_sidecars(self, output_dir: Path) -> dict[Path, float]:
        if not Path(output_dir).exists():
            return {}
        return {
            path: path.stat().st_mtime
            for path in Path(output_dir).rglob("*_dwi.json")
            if path.is_file()
        }

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

    def _new_or_updated_outputs(self, before: dict[Path, float], output_dir: Path) -> list[Path]:
        after = self._snapshot_import_outputs(output_dir)
        changed: list[Path] = []
        for path, mtime in after.items():
            if path not in before or mtime > before[path]:
                changed.append(path)
        return sorted(changed)
            
    def run(self, dicom_dir: Path, output_dir: Path, context: dict) -> dict:
        self.logger.info("Starting Import Workflow")
        resolved_dicom_dir = self._resolve_import_source(Path(dicom_dir))
        context = dict(context)
        context["dicom_dir"] = resolved_dicom_dir
        step_context = {k: v for k, v in context.items() if k != "dicom_dir"}
        outputs_before = self._snapshot_import_outputs(output_dir)
        sidecars_before = self._snapshot_dwi_sidecars(output_dir)
        
        for step in self.steps:
            if isinstance(step, (Dcm2BidsStep, Dcm2NiixStep)):
                step.run(resolved_dicom_dir, output_dir, **step_context)
                imported_outputs = self._new_or_updated_outputs(outputs_before, output_dir)
                imported_sidecars = self._new_or_updated_sidecars(sidecars_before, output_dir)
                context["imported_output_files"] = [str(p) for p in imported_outputs]
                context["imported_dwi_sidecars"] = [str(p) for p in imported_sidecars]
                if not imported_outputs:
                    msg = (
                        "Import step completed without creating or updating any output files. "
                        "This usually means the dcm2bids/dcm2niix mapping did not match the source series "
                        "or outputs already exist unchanged."
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
