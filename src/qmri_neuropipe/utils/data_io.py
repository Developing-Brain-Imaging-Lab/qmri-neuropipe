"""
Data I/O utilities for pipeline processing.

This module provides functions for file copying, validation,
recovery of intermediate results, and output management.
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import shutil
import nibabel as nib
import numpy as np

from qmri_neuropipe.core.types import ImageFile, DWIFile
from qmri_neuropipe.io.bids import build_bids_name, parse_bids_filename
from qmri_neuropipe.io.dmri.bids import ensure_dwi_timing_sidecars
from qmri_neuropipe.core.utils import get_nifti_stem


class DataIOManager:
    """
    Manager for data input/output operations in the pipeline.
    
    Handles:
    - Copying raw data to work directories
    - Saving final outputs
    - Recovering intermediate results
    - Validating file existence
    """
    
    def __init__(self, config, logger):
        """
        Initialize the data I/O manager.
        
        Parameters
        ----------
        config : PipelineConfig
            Pipeline configuration
        logger : Logger
            Logger instance
        """
        self.config = config
        self.logger = logger
    
    def copy_raw_data_to_workdir(
        self,
        dwi_files: List[DWIFile],
        work_dir: Path,
        subject: str,
        session: Optional[str] = None
    ) -> List[DWIFile]:
        """
        Copy raw DWI data to working directory.
        
        Parameters
        ----------
        dwi_files : list of DWIFile
            Source DWI files
        work_dir : Path
            Working directory root
        subject : str
            Subject ID
        session : str, optional
            Session ID
            
        Returns
        -------
        list of DWIFile
            DWI files in working directory
        """
        raw_work_dir = work_dir / "rawdata"
        raw_work_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Copying {len(dwi_files)} DWI files to {raw_work_dir}")
        
        copied_dwi_files = []
        
        for d in dwi_files:
            ensure_dwi_timing_sidecars(d, logger=self.logger)
            # Copy main image
            dest_img = raw_work_dir / d.img.name
            if not dest_img.exists():
                shutil.copy(d.img, dest_img)
            
            # Copy sidecars
            dest_bval = self._copy_sidecar(d.bval, raw_work_dir)
            dest_bvec = self._copy_sidecar(d.bvec, raw_work_dir)
            dest_json = self._copy_sidecar(d.json, raw_work_dir)
            dest_Delta = self._copy_sidecar(d.Delta, raw_work_dir)
            dest_delta = self._copy_sidecar(d.delta, raw_work_dir)
            
            # Ensure entities match the processing subject/session
            current_entities = d.entities.copy()
            current_entities['sub'] = subject
            if session:
                current_entities['ses'] = session
            
            # Create new DWIFile pointing to work dir
            new_dwi = DWIFile(
                img=dest_img,
                json=dest_json,
                bval=dest_bval,
                bvec=dest_bvec,
                Delta=dest_Delta,
                delta=dest_delta,
                entities=current_entities
            )
            copied_dwi_files.append(new_dwi)
        
        self.logger.info("Raw data copying complete")
        return copied_dwi_files
    
    def _copy_sidecar(self, src: Optional[Path], dest_dir: Path) -> Optional[Path]:
        """Copy a sidecar file if it exists."""
        if not src or not src.exists():
            return None
        
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy(src, dest)
        
        return dest

    def _find_sidecar_for_image(
        self,
        img: Path,
        extension: str,
        search_dirs: Optional[List[Path]] = None
    ) -> Optional[Path]:
        """Find a sidecar file based on an image path and extension."""
        if not img or not img.exists():
            return None

        stem = get_nifti_stem(img)
        candidate = img.parent / f"{stem}{extension}"
        if candidate.exists():
            return candidate

        try:
            img_ents = parse_bids_filename(img)
            dirs = search_dirs or [img.parent]
            for directory in dirs:
                if not directory or not directory.exists():
                    continue
                for path in directory.glob(f"*{extension}"):
                    sidecar_ents = parse_bids_filename(path)
                    if sidecar_ents.get("suffix") != "dwi":
                        continue

                    matches = True
                    for key, value in img_ents.items():
                        if key in ("suffix", "extension", "path"):
                            continue
                        if value is None:
                            continue
                        if sidecar_ents.get(key) != value:
                            matches = False
                            break

                    if matches:
                        return path
        except Exception:
            return None

        return None

    def _find_mask_for_image(
        self,
        img: Path,
        search_dirs: Optional[List[Path]] = None
    ) -> Optional[Path]:
        """Find a mask file that corresponds to an image path."""
        if not img or not img.exists():
            return None

        stem = get_nifti_stem(img)
        candidates = [
            img.parent / f"{stem}_mask.nii.gz",
            img.parent / f"{stem}_brainmask.nii.gz",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        try:
            img_ents = parse_bids_filename(img)
            dirs = search_dirs or [img.parent]
            for directory in dirs:
                if not directory or not directory.exists():
                    continue
                for path in directory.glob("*.nii*"):
                    mask_ents = parse_bids_filename(path)
                    if mask_ents.get("suffix") != "mask":
                        continue

                    matches = True
                    for key, value in img_ents.items():
                        if key in ("suffix", "extension", "path"):
                            continue
                        if value is None:
                            continue
                        if mask_ents.get(key) != value:
                            matches = False
                            break

                    if matches:
                        return path
        except Exception:
            return None

        return None
    
    def save_final_outputs(
        self,
        context: Dict,
        output_dir: Path,
        skip_existing: bool = True
    ):
        """
        Save final preprocessed DWI files to output directory.
        
        Parameters
        ----------
        context : dict
            Processing context with preprocessed_dwis
        output_dir : Path
            Output directory
        skip_existing : bool
            Skip saving if file already exists
        """
        dwis = context.get("preprocessed_dwis", [])
        masks = context.get("preprocessed_masks", [])
        
        if not dwis:
            self.logger.warning("No preprocessed DWIs to save")
            return
        
        self.logger.info(f"Saving {len(dwis)} preprocessed DWI files to {output_dir}")
        
        base_out = self.config.output_dir
        apply_mask_to_final = bool(
            self.config.get("dmri", {})
            .get("preprocessing", {})
            .get("brain_masking", {})
            .get("apply_to_final_output", False)
        )

        for dwi, mask in zip(dwis, masks or [None] * len(dwis)):
            if not dwi.img.exists():
                self.logger.warning(f"Final DWI missing: {dwi.img}")
                continue
            
            # Determine output path
            ents = dwi.entities.copy()
            sub = ents.get("sub") or context.get("subject", "unknown")
            ses = ents.get("ses")
            
            ents['sub'] = sub
            if ses:
                ents['ses'] = ses
            
            # Force valid suffix
            ents['suffix'] = 'dwi'
            ents['desc'] = 'preproc'
            
            # Construct directory
            target_dir = base_out / f"sub-{sub}"
            if ses:
                target_dir /= f"ses-{ses}"
            target_dir /= "dwi"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Construct filenames
            fname = build_bids_name(ents)
            if not fname.endswith(".nii.gz"):
                fname += ".nii.gz"
            
            target_img = target_dir / fname
            
            copy_image = True
            if skip_existing and target_img.exists():
                self.logger.debug(f"Skipping existing image file but refreshing sidecars: {target_img}")
                copy_image = False
            
            # Copy files
            if copy_image:
                self.logger.info(f"Saving: {target_img}")
            search_dirs = []
            if dwi.img.parent:
                search_dirs.append(dwi.img.parent)
            if dwi.img.parent.parent:
                search_dirs.append(dwi.img.parent.parent)

            try:
                img_ents = parse_bids_filename(dwi.img)
                sub = img_ents.get("sub")
                ses = img_ents.get("ses")
            except Exception:
                sub = None
                ses = None

            bids_root = getattr(self.config, "bids_dir", None)
            if bids_root:
                bids_root = Path(bids_root)
                if sub:
                    if ses:
                        search_dirs.append(bids_root / f"sub-{sub}" / f"ses-{ses}" / "dwi")
                    search_dirs.append(bids_root / f"sub-{sub}" / "dwi")

            work_root = None
            if hasattr(self.config, "get"):
                work_root = self.config.get("work_dir", None)
            if not work_root:
                work_root = getattr(self.config, "work_dir", None)
            if not work_root:
                work_root = self.config.output_dir / "work"
            if work_root:
                work_root = Path(work_root)
                if sub:
                    if ses:
                        search_dirs.append(work_root / f"sub-{sub}" / f"ses-{ses}" / "dwi" / "rawdata")
                    search_dirs.append(work_root / f"sub-{sub}" / "dwi" / "rawdata")

            if output_dir:
                output_dir = Path(output_dir)
                if sub:
                    if ses:
                        search_dirs.append(output_dir / f"sub-{sub}" / f"ses-{ses}" / "dwi")
                    search_dirs.append(output_dir / f"sub-{sub}" / "dwi")

            search_dirs = [p for p in search_dirs if p and p.exists()]

            # Save mask if present (fall back to step output mask)
            mask_src = None
            if mask and hasattr(mask, 'img') and mask.img.exists():
                mask_src = mask.img
            else:
                mask_src = self._find_mask_for_image(dwi.img, search_dirs)

            mask_applied_to_final = False
            if not copy_image:
                mask_applied_to_final = False
            elif apply_mask_to_final and mask_src and mask_src.exists():
                try:
                    self._apply_mask_to_dwi_and_save(dwi.img, mask_src, target_img)
                    mask_applied_to_final = True
                except Exception as exc:
                    self.logger.warning(
                        f"Failed to apply final brain mask for {dwi.img.name}: {exc}. Saving unmasked DWI."
                    )
                    shutil.copy(dwi.img, target_img)
            else:
                shutil.copy(dwi.img, target_img)

            # Copy sidecars (fall back to BIDS-matched files in search dirs)
            bval_src = getattr(dwi, "bval", None) or self._find_sidecar_for_image(dwi.img, ".bval", search_dirs)
            bvec_src = getattr(dwi, "bvec", None) or self._find_sidecar_for_image(dwi.img, ".bvec", search_dirs)
            json_src = getattr(dwi, "json", None) or self._find_sidecar_for_image(dwi.img, ".json", search_dirs)
            Delta_src = (
                getattr(dwi, "Delta", None)
                or self._find_sidecar_for_image(dwi.img, ".bigdelta", search_dirs)
                or self._find_sidecar_for_image(dwi.img, ".Delta", search_dirs)
            )
            delta_src = getattr(dwi, "delta", None) or self._find_sidecar_for_image(dwi.img, ".delta", search_dirs)

            self._copy_sidecar_with_new_name(bval_src, target_img, ".bval")
            self._copy_sidecar_with_new_name(bvec_src, target_img, ".bvec")
            self._copy_sidecar_with_new_name(Delta_src, target_img, ".bigdelta")
            self._copy_sidecar_with_new_name(delta_src, target_img, ".delta")
            target_json = self._copy_sidecar_with_new_name(json_src, target_img, ".json")

            processing_steps = context.get("processing_steps") or context.get("preprocessing_steps") or []
            processing_details = context.get("processing_steps_detail") or []
            if target_json and processing_steps:
                self._update_json_history(target_json, processing_steps, processing_details)
            if target_json and mask_applied_to_final and mask_src and mask_src.exists():
                self._mark_json_mask_applied(target_json, mask_src)

            if mask_src:
                mask_ents = ents.copy()
                mask_ents['suffix'] = 'mask'
                
                mask_name = build_bids_name(mask_ents)
                if not mask_name.endswith(".nii.gz"):
                    mask_name += ".nii.gz"
                
                target_mask = target_dir / mask_name
                
                if not (skip_existing and target_mask.exists()):
                    self.logger.info(f"Saving mask: {target_mask}")
                    shutil.copy(mask_src, target_mask)
        
        # Save GNL maps if present
        self._save_gnl_maps(context, base_out, skip_existing)

    def _apply_mask_to_dwi_and_save(
        self,
        source_dwi: Path,
        mask_path: Path,
        target_img: Path,
    ) -> None:
        """Apply a binary mask to a DWI and save the masked data to the final target."""
        dwi_img = nib.load(str(source_dwi))
        mask_img = nib.load(str(mask_path))

        dwi_shape = dwi_img.shape[:3]
        if mask_img.shape[:3] != dwi_shape:
            raise ValueError(
                f"Mask shape {mask_img.shape[:3]} does not match DWI spatial shape {dwi_shape}"
            )

        mask_data = np.asarray(mask_img.dataobj)
        if mask_data.ndim > 3:
            mask_data = np.squeeze(mask_data)
        mask_bool = mask_data > 0

        data = np.asarray(dwi_img.dataobj)
        if data.ndim == 4:
            masked_data = data * mask_bool[..., np.newaxis]
        else:
            masked_data = data * mask_bool

        header = dwi_img.header.copy()
        masked_img = nib.Nifti1Image(masked_data.astype(data.dtype, copy=False), dwi_img.affine, header)
        nib.save(masked_img, str(target_img))

    def _mark_json_mask_applied(self, json_path: Path, mask_path: Path) -> None:
        """Record final-output mask application in the saved JSON sidecar."""
        payload: Dict[str, object] = {}
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}

        payload["BrainMaskAppliedToFinalPreprocessedDWI"] = True
        payload["BrainMaskAppliedToFinalPreprocessedDWIPath"] = str(mask_path)

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=4)
    
    def _copy_sidecar_with_new_name(
        self,
        src: Optional[Path],
        target_img: Path,
        extension: str
    ) -> Optional[Path]:
        """Copy sidecar with new name matching target image."""
        if not src or not src.exists():
            return None
        
        # Determine target path
        target_path = target_img.with_suffix("").with_suffix("")
        target_path = Path(str(target_path) + extension)
        
        shutil.copy(src, target_path)
        return target_path

    def _update_json_history(
        self,
        json_path: Path,
        steps: list,
        step_details: Optional[List[Dict]] = None
    ) -> None:
        """Update JSON sidecar with processing history."""
        if not json_path:
            return

        data: Dict[str, object] = {}
        if json_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        history_msg = "Pipeline Steps: " + ", ".join(steps)
        prev_history = data.get("History", "")
        if prev_history:
            data["History"] = prev_history + "; " + history_msg
        else:
            data["History"] = history_msg

        data["ProcessingSteps"] = steps
        if step_details:
            data["ProcessingStepsDetail"] = step_details

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
    
    def _save_gnl_maps(
        self,
        context: Dict,
        base_out: Path,
        skip_existing: bool
    ):
        """Save gradient nonlinearity maps."""
        from qmri_neuropipe.io.bids import get_entities_from_path

        ordered_sources = []
        seen_sources = set()
        selected_targets = {}

        for dwi in context.get("preprocessed_dwis", []) or []:
            img_path = getattr(dwi, "img", None)
            if img_path is None:
                continue
            gnl_map = (
                context.get("gnl_map_by_image", {}).get(img_path)
                or context.get("gnl_map_by_image", {}).get(str(img_path))
            )
            gnl_path = Path(gnl_map) if gnl_map else None
            if gnl_path and gnl_path not in seen_sources:
                ordered_sources.append(gnl_path)
                seen_sources.add(gnl_path)
            if not gnl_path or not gnl_path.exists():
                continue

            dwi_entities = dict(getattr(dwi, "entities", {}) or {})
            sub_g = dwi_entities.get("sub") or context.get("subject", "unknown")
            ses_g = dwi_entities.get("ses") or context.get("session")
            dwi_entities["sub"] = sub_g
            if ses_g:
                dwi_entities["ses"] = ses_g
            dwi_entities["desc"] = "gnl_tensor"
            dwi_entities["suffix"] = "dwi"

            target_dir = base_out / f"sub-{sub_g}"
            if ses_g:
                target_dir /= f"ses-{ses_g}"
            target_dir /= "dwi"
            target_dir.mkdir(parents=True, exist_ok=True)

            target_gnl = target_dir / build_bids_name(dwi_entities)
            selected_targets[target_gnl] = gnl_path

        for gnl_map in context.get("gnl_maps", []) or []:
            gnl_path = Path(gnl_map) if gnl_map else None
            if gnl_path and gnl_path not in seen_sources:
                ordered_sources.append(gnl_path)
                seen_sources.add(gnl_path)

        gnl_map = context.get("gnl_map")
        gnl_path = Path(gnl_map) if gnl_map else None
        if gnl_path and gnl_path not in seen_sources:
            ordered_sources.append(gnl_path)

        for gnl_map in ordered_sources:
            if not isinstance(gnl_map, Path) or not gnl_map.exists():
                continue
            if gnl_map in selected_targets.values():
                continue

            g_ents = get_entities_from_path(gnl_map)
            sub_g = g_ents.get("sub") or context.get("subject", "unknown")
            ses_g = g_ents.get("ses") or context.get("session")

            t_dir = base_out / f"sub-{sub_g}"
            if ses_g:
                t_dir /= f"ses-{ses_g}"
            t_dir /= "dwi"
            t_dir.mkdir(parents=True, exist_ok=True)

            target_gnl = t_dir / gnl_map.name
            selected_targets.setdefault(target_gnl, gnl_map)

        for target_gnl, gnl_map in selected_targets.items():
            overwrite_canonical = "desc-gnl_tensor" in target_gnl.name

            if skip_existing and target_gnl.exists() and not overwrite_canonical:
                self.logger.debug(f"Skipping existing GNL map: {target_gnl}")
                continue

            if target_gnl.exists() and target_gnl.resolve() == gnl_map.resolve():
                continue

            action = "Updating" if target_gnl.exists() else "Saving"
            self.logger.info(f"{action} GNL Tensor Map: {target_gnl}")
            shutil.copy(gnl_map, target_gnl)
    
    def recover_intermediates(
        self,
        work_dir: Path,
        output_dir: Path
    ):
        """
        Recover intermediate data from final output directory to working directory.
        
        This allows the pipeline to skip steps that were previously computed.
        
        Parameters
        ----------
        work_dir : Path
            Working directory
        output_dir : Path
            Final output directory with intermediate subdirectory
        """
        self.logger.info("Attempting to recover intermediate data...")
        
        intermediate_store = output_dir / "intermediate"
        
        if not intermediate_store.exists():
            self.logger.debug(f"No intermediate storage found at {intermediate_store}")
            return
        
        # Recover topup results
        if (intermediate_store / "topup").exists():
            target_work = work_dir / "topup"
            if not target_work.exists():
                self.logger.info(
                    f"Recovering Topup results from {intermediate_store / 'topup'}"
                )
                shutil.copytree(
                    intermediate_store / "topup",
                    target_work,
                    dirs_exist_ok=True
                )
        
        # Recover Synb0 results
        if (intermediate_store / "synb0").exists():
            target_work = work_dir / "synb0"
            if not target_work.exists():
                self.logger.info(
                    f"Recovering Synb0 results from {intermediate_store / 'synb0'}"
                )
                shutil.copytree(
                    intermediate_store / "synb0",
                    target_work,
                    dirs_exist_ok=True
                )
        
        self.logger.info("Intermediate recovery complete")
    
    def validate_outputs(
        self,
        expected_preproc_path: Path,
        config: Dict
    ) -> tuple:
        """
        Validate that all expected preprocessing outputs exist and are valid.
        
        Parameters
        ----------
        expected_preproc_path : Path
            Expected path to main preprocessed DWI
        config : dict
            Configuration dict
            
        Returns
        -------
        tuple
            (is_valid: bool, reason: str)
        """
        # Check main files
        if not expected_preproc_path.exists():
            return False, "Main DWI file missing"
        
        expected_bval_path = expected_preproc_path.with_suffix("").with_suffix(".bval")
        if not expected_bval_path.exists():
            return False, "Bval file missing"
        
        expected_bvec_path = expected_preproc_path.with_suffix("").with_suffix(".bvec")
        if not expected_bvec_path.exists():
            return False, "Bvec file missing"
        
        # Size validation (catch corrupted/truncated files)
        try:
            if expected_preproc_path.stat().st_size < 1_000_000:  # < 1MB
                return False, "DWI file too small (possibly corrupted)"
            
            if expected_bval_path.stat().st_size < 10:
                return False, "Bval file too small"
            
            if expected_bvec_path.stat().st_size < 10:
                return False, "Bvec file too small"
        except Exception as e:
            return False, f"Error checking file sizes: {e}"
        
        # Optional: Mask validation
        mask_cfg = config.get('dmri', {}).get('preprocessing', {}).get('brain_masking', {})
        if mask_cfg.get('enabled', False):
            expected_mask_path = Path(
                str(expected_preproc_path).replace('_dwi.nii.gz', '_mask.nii.gz')
            )
            
            if not expected_mask_path.exists():
                return False, "Brain mask missing (required by config)"
            
            try:
                if expected_mask_path.stat().st_size < 100_000:  # < 100KB
                    return False, "Mask file too small"
            except Exception:
                return False, "Error validating mask"
        
        return True, "All outputs valid"
