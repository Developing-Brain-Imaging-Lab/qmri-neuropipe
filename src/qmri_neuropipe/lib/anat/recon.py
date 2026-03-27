"""
FreeSurfer Recon-all step.
"""

from pathlib import Path
from typing import Optional, Any
import logging

from ...core import BaseProcessingStep, ValidationError
from ...core.types import ImageLike, ImageFile
from ...interfaces import freesurfer
from ...io.bids import build_bids_name, extract_bids_entities
import pandas as pd

class ReconAllStep(BaseProcessingStep):
    """
    Run FreeSurfer recon-all.
    """
    
    def __init__(self, config, logger=None, provenance=None):
        super().__init__(config, logger, provenance)
        # Look up config in anat.preprocessing.recon_all
        anat_cfg = config.get("anat", {}).get("preprocessing", {})
        self.recon_config = anat_cfg.get("recon_all", {})
        self.enabled = self.recon_config.get("enabled", False)
        self.method = self.recon_config.get("method", "standard") # Default to standard
        
        base_args = self.recon_config.get("args", "-all")
        extra_args = self.recon_config.get("extra_args", "")
        self.args = f"{base_args} {extra_args}".strip()
        self.subjects_dir = self.recon_config.get("subjects_dir")
        
        # Check global use_freesurfer flag (or nested in generic options)
        # Checking: anat.use_freesurfer OR anat.preprocessing.use_freesurfer
        self.use_freesurfer = config.get("anat", {}).get("use_freesurfer", False) or \
                              config.get("anat", {}).get("preprocessing", {}).get("use_freesurfer", False)

    @staticmethod
    def _is_t2w_image(image: Any) -> bool:
        if not image:
            return False
        entities = getattr(image, "entities", {}) or {}
        suffix = str(entities.get("suffix", "")).lower()
        if suffix == "t2w":
            return True
        img_path = getattr(image, "img", image)
        try:
            name = Path(img_path).name.lower()
        except Exception:
            return False
        return "_t2w" in name

    @staticmethod
    def _select_recon_input(context: Optional[dict], input_image: Any) -> Any:
        if not context:
            return input_image
        pre_t1 = context.get("preprocessed_t1w")
        if pre_t1:
            return pre_t1
        t1w_files = context.get("t1w_files") or []
        if t1w_files:
            return t1w_files[0]
        return input_image

    @staticmethod
    def _aparc_aseg_path(subj_dir: Path) -> Path:
        return subj_dir / "mri" / "aparc+aseg.mgz"

    @classmethod
    def critical_outputs(cls, subj_dir: Path) -> list[Path]:
        return [
            subj_dir / "mri" / "brain.mgz",
            subj_dir / "mri" / "aseg.mgz",
            cls._aparc_aseg_path(subj_dir),
            subj_dir / "surf" / "lh.white",
            subj_dir / "surf" / "rh.white",
            subj_dir / "scripts" / "recon-all.done",
        ]

    @classmethod
    def has_complete_recon(cls, subj_dir: Path) -> bool:
        return all(path.exists() for path in cls.critical_outputs(subj_dir))

    def run(self, first_arg, output_dir: Path, **kwargs) -> Any:
        # Check explicit enable OR use_freesurfer
        if not self.enabled and not self.use_freesurfer:
             self.logger.info("ReconAllStep disabled.")
             return first_arg

        context, input_image = self.unpack_input(first_arg)
        recon_input = self._select_recon_input(context, input_image)
        
        # Subject ID logic
        sub = context.get('subject') if context else None
        if not sub and hasattr(input_image, 'entities'):
             sub = input_image.entities.get('sub')
        
        if not sub:
             self.logger.warning("Could not determine subject ID for recon-all. Using 'subject'.")
             sub = "subject"
             
        ses = context.get('session') if context else getattr(input_image, 'entities', {}).get('ses')
        
        fs_sub_id = f"sub-{sub}"
        if ses:
             fs_sub_id += f"_ses-{ses}"

        # FS Directory
        fs_dir = Path(self.subjects_dir) if self.subjects_dir else (self.config.bids_dir / "derivatives" / "freesurfer")
        subj_dir = fs_dir / fs_sub_id
        
        # === FAST PATH: Check if we can skip everything ===
        # Define expected output files
        out_name = build_bids_name({
            'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'T1w'
        })
        t1w_nii = output_dir / out_name
        mask_nii = output_dir / out_name.replace('T1w', 'mask')
        aparc_aseg = self._aparc_aseg_path(subj_dir)
        fs_complete = self.has_complete_recon(subj_dir)
        
        # If skip_existing enabled and final outputs exist, skip entirely
        if self.config.skip_existing and not kwargs.get('force', False):
            if subj_dir.exists() and fs_complete and t1w_nii.exists() and mask_nii.exists():
                # Quick validation
                if (t1w_nii.stat().st_size > 1000 and 
                    mask_nii.stat().st_size > 1000):
                    
                    self.logger.info(f"Skipping ReconAll - outputs already exist for {fs_sub_id}")
                    
                    # Return properly structured output without doing any work
                    entities = {'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'T1w'}
                    new_img = ImageFile(img=t1w_nii, entities=entities)
                    
                    mask_entities = {'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'mask'}
                    new_mask = ImageFile(img=mask_nii, entities=mask_entities)
                    
                    if context:
                        context['current_image'] = new_img
                        context['brain_mask'] = new_mask
                        context["freesurfer_dir"] = subj_dir
                        return context
                    else:
                        return new_img
        
        # === END FAST PATH ===
        
        # Check force run
        if kwargs.get('force', False) and subj_dir.exists():
             self.logger.info(f"Recon-all force run: Removing existing subject dir {subj_dir}")
             import shutil
             shutil.rmtree(subj_dir)
        elif subj_dir.exists() and not aparc_aseg.exists():
             self.logger.warning(
                 f"FreeSurfer directory exists for {fs_sub_id} but {aparc_aseg.name} is missing. "
                 "Removing the subject directory and restarting recon-all."
             )
             import shutil
             shutil.rmtree(subj_dir)
             
        fs_dir.mkdir(parents=True, exist_ok=True)
        
        # Check integrity of key FS outputs, including aparc+aseg.mgz as a completion sentinel.
        fs_complete = self.has_complete_recon(subj_dir)
        
        if not fs_complete:
             if not recon_input:
                 raise ValidationError("FreeSurfer output missing and no input image provided to run recon-all.")
             if self._is_t2w_image(recon_input):
                 raise ValidationError(
                     "Recon-all input resolved to a T2w image. FreeSurfer recon-all requires T1w input."
                 )
                 
             n_threads = kwargs.get("nthreads") or self.config.get("n_cpus") or 8
             
             if self.method == "clinical":
                 self.logger.info(f"Running FreeSurfer recon-all-clinical for {fs_sub_id}...")
                 freesurfer.recon_all_clinical(
                    in_file=recon_input,
                    subject_id=fs_sub_id,
                    subjects_dir=fs_dir,
                    nthreads=n_threads
                 )
             else:
                 self.logger.info(f"Running FreeSurfer recon-all for {fs_sub_id}...")
                 freesurfer.recon_all(
                    in_file=recon_input,
                    subject_id=fs_sub_id,
                    subjects_dir=fs_dir,
                    openmp=n_threads,
                    extra_args=self.args
                 )
        else:
             self.logger.info(f"Using existing FreeSurfer output for {fs_sub_id}")

        if not aparc_aseg.exists():
             raise ValidationError(
                 f"FreeSurfer output for {fs_sub_id} is incomplete: missing {aparc_aseg}. "
                 "The subject directory should be removed and recon-all rerun."
             )

        # Post-Processing / Injection
        if context:
             context["freesurfer_dir"] = subj_dir
             
        if self.use_freesurfer:
             # Convert brain.mgz -> NIfTI
             fs_out_dir = output_dir
             
             out_name = build_bids_name({
                 'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'T1w'
             })
             
             t1w_nii = fs_out_dir / out_name
             mask_nii = fs_out_dir / out_name.replace('T1w', 'mask')
             
             # OPTIMIZATION: Skip conversion if outputs already exist and valid
             skip_conversion = False
             if self.config.skip_existing:
                 if t1w_nii.exists() and mask_nii.exists():
                     # Quick validation: check if files are non-zero size
                     if t1w_nii.stat().st_size > 1000 and mask_nii.stat().st_size > 1000:
                         self.logger.info(f"Skipping FS conversion - outputs already exist: {t1w_nii.name}")
                         skip_conversion = True
             
             if not skip_conversion:
                 # Convert brain.mgz
                 self.logger.info(f"Converting FS brain.mgz to {t1w_nii.name}")
                 
                 # Ensure brain_mgz is defined (from critical_files check)
                 brain_mgz = subj_dir / "mri" / "brain.mgz"
                 
                 freesurfer.mri_convert(brain_mgz, t1w_nii)
                 
                 # Create mask
                 self.logger.info(f"Creating brain mask from FS output...")
                 freesurfer.mri_binarize(brain_mgz, mask_nii, min_val=1)
             
             # Define new image objects
             entities = {'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'T1w'}
             new_img = ImageFile(img=t1w_nii, entities=entities)
             
             mask_entities = {'sub': sub, 'ses': ses, 'desc': 'preproc', 'suffix': 'mask'}
             new_mask = ImageFile(img=mask_nii, entities=mask_entities)

             # Update Context if available
             if context:
                 context['current_image'] = new_img
                 context['brain_mask'] = new_mask
                 
                 self.logger.info("Updated pipeline context to use FreeSurfer-derived structural images.")
                 return context
             else:
                 # Standalone mode: return the new image directly
                 return new_img
                  
        # Fallback if not using FS as primary but just running it
        return context if context else input_image

def parse_freesurfer_stats(stats_file: Path) -> pd.DataFrame:
    """Parse FreeSurfer .stats file into a pandas DataFrame.
    
    Optimized version with faster parsing and better error handling.
    """
    if not stats_file.exists():
        return pd.DataFrame()
    
    try:
        # Read entire file at once (faster than line-by-line)
        with open(stats_file, 'r') as f:
            content = f.read()
            
        lines = content.splitlines()
        
        # Pre-allocate lists for better performance
        column_names = []
        data_lines = []
        
        # Single pass through lines
        for line in lines:
            if line.startswith('# ColName'):
                # Extract column name efficiently
                parts = line.split()
                if len(parts) > 1:
                    column_names.append(parts[-1])
            elif line and not line.startswith('#'):
                # Only split non-empty, non-comment lines
                data_lines.append(line.split())
        
        if not column_names or not data_lines:
            return pd.DataFrame()
        
        # Create DataFrame efficiently
        # Handle variable column counts
        max_cols = max(len(row) for row in data_lines)
        min_cols = min(len(column_names), max_cols)
        
        # Pad rows if needed
        padded_data = [row[:min_cols] if len(row) >= min_cols else row + [None]*(min_cols - len(row)) 
                       for row in data_lines]
        
        df = pd.DataFrame(padded_data, columns=column_names[:min_cols])
        
        # Vectorized numeric conversion (much faster than iterating)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
        
    except Exception as e:
        # Log error but don't crash
        import logging
        logging.getLogger(__name__).warning(f"Failed to parse {stats_file}: {e}")
        return pd.DataFrame()

class FreeSurferStatsStep(BaseProcessingStep):
    """
    Step to parse FreeSurfer stats and add them to the tracker.
    """
    def run(self, context: dict, output_dir: Path, **kwargs) -> dict:
        fs_dir = context.get("freesurfer_dir")
        if not fs_dir or not Path(fs_dir).exists():
             self.logger.warning("FreeSurfer directory missing. Skipping FS stats extraction.")
             return context
             
        fs_dir = Path(fs_dir)
        stats_dir = fs_dir / "stats"
        
        # Early exit if stats dir doesn't exist
        if not stats_dir.exists():
            self.logger.warning(f"FreeSurfer stats directory missing: {stats_dir}")
            return context
        
        subject = context.get("subject")
        session = context.get("session")
        study = context.get("study_name")
        
        # Check if we should skip (outputs already exist)
        expected_outputs = [
            output_dir / f"sub-{subject}_ses-{session}_desc-aseg_stats.tsv",
            output_dir / f"sub-{subject}_ses-{session}_desc-lh_aparc_stats.tsv",
            output_dir / f"sub-{subject}_ses-{session}_desc-rh_aparc_stats.tsv"
        ]
        
        if self.config.skip_existing:
            if all(f.exists() and f.stat().st_size > 0 for f in expected_outputs):
                self.logger.info("Skipping FS stats - outputs already exist")
                # Populate context with existing files
                context.setdefault('roi_stats_files', {})
                context['roi_stats_files']['aseg'] = expected_outputs[0]
                context['roi_stats_files']['lh_aparc'] = expected_outputs[1]
                context['roi_stats_files']['rh_aparc'] = expected_outputs[2]
                return context
        
        # Initialize context dict once
        context.setdefault('roi_stats_files', {})
        
        # 1. Parse Aseg (Subcortical)
        aseg_stats = stats_dir / "aseg.stats"
        if aseg_stats.exists():
            df = parse_freesurfer_stats(aseg_stats)
            if not df.empty:
                # Vectorized processing for better performance
                # Filter valid rows first
                valid_mask = df.get("StructName").notna() & df.get("Volume_mm3").notna()
                if valid_mask.any():
                    df_valid = df[valid_mask]
                    
                    # Build stats list efficiently
                    stats_list = [
                        {
                            "roi_name": row["StructName"],
                            "model": "Structural",
                            "metric": "Volume",
                            "mean": row["Volume_mm3"],
                            "median": row["Volume_mm3"],
                            "std": 0
                        }
                        for _, row in df_valid.iterrows()
                    ]
                    
                    if stats_list:
                        out_tsv = expected_outputs[0]
                        pd.DataFrame(stats_list).to_csv(out_tsv, sep='\t', index=False)
                        context['roi_stats_files']['aseg'] = out_tsv

        # 2. Parse Aparc (Cortical) - Process both hemispheres efficiently
        for hemi_idx, hemi in enumerate(['lh', 'rh'], start=1):
            aparc_stats = stats_dir / f"{hemi}.aparc.stats"
            if aparc_stats.exists():
                df = parse_freesurfer_stats(aparc_stats)
                if not df.empty:
                    # Vectorized processing
                    valid_mask = df.get("StructName").notna()
                    if not valid_mask.any():
                        continue
                        
                    df_valid = df[valid_mask]
                    
                    # Build stats list with list comprehension (faster)
                    stats_list = []
                    for _, row in df_valid.iterrows():
                        roi = row["StructName"]
                        
                        # Process multiple metrics efficiently
                        for metric_name, col_name in [
                            ("Volume", "Volume_mm3"),
                            ("Thickness", "ThickAvg"),
                            ("Area", "SurfArea")
                        ]:
                            metric_val = row.get(col_name)
                            if pd.notna(metric_val):
                                stats_list.append({
                                    "roi_name": f"{hemi}_{roi}",
                                    "model": "Structural",
                                    "metric": metric_name,
                                    "mean": metric_val,
                                    "median": metric_val,
                                    "std": 0
                                })
                    
                    if stats_list:
                        out_tsv = expected_outputs[hemi_idx]
                        pd.DataFrame(stats_list).to_csv(out_tsv, sep='\t', index=False)
                        context['roi_stats_files'][f"{hemi}_aparc"] = out_tsv

        return context
