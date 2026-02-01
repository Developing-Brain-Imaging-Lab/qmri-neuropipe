"""
ROI Extraction Module for Cross-Modal Metric Extraction.

Extracts diffusion and relaxometry metrics from anatomical ROIs
(FreeSurfer, FSL FAST, FSL FIRST).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any
import logging
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    nib = None

logger = logging.getLogger(__name__)


# ===========================================================================
# ROI Label Mappings
# ===========================================================================

# FreeSurfer aseg labels (subcortical)
FREESURFER_ASEG_LABELS = {
    'Left_Cerebral_White_Matter': 2,
    'Left_Cerebral_Cortex': 3,
    'Left_Lateral_Ventricle': 4,
    'Left_Inf_Lat_Vent': 5,
    'Left_Cerebellum_White_Matter': 7,
    'Left_Cerebellum_Cortex': 8,
    'Left_Thalamus': 10,
    'Left_Caudate': 11,
    'Left_Putamen': 12,
    'Left_Pallidum': 13,
    'Third_Ventricle': 14,
    'Fourth_Ventricle': 15,
    'Brain_Stem': 16,
    'Left_Hippocampus': 17,
    'Left_Amygdala': 18,
    'CSF': 24,
    'Left_Accumbens_area': 26,
    'Left_VentralDC': 28,
    'Right_Cerebral_White_Matter': 41,
    'Right_Cerebral_Cortex': 42,
    'Right_Lateral_Ventricle': 43,
    'Right_Inf_Lat_Vent': 44,
    'Right_Cerebellum_White_Matter': 46,
    'Right_Cerebellum_Cortex': 47,
    'Right_Thalamus': 49,
    'Right_Caudate': 50,
    'Right_Putamen': 51,
    'Right_Pallidum': 52,
    'Right_Hippocampus': 53,
    'Right_Amygdala': 54,
    'Right_Accumbens_area': 58,
    'Right_VentralDC': 60,
}

# FSL FIRST labels (from combined segmentation)
FSL_FIRST_LABELS = {
    'Left_Thalamus': 10,
    'Left_Caudate': 11,
    'Left_Putamen': 12,
    'Left_Pallidum': 13,
    'Brain_Stem': 16,
    'Left_Hippocampus': 17,
    'Left_Amygdala': 18,
    'Left_Accumbens': 26,
    'Right_Thalamus': 49,
    'Right_Caudate': 50,
    'Right_Putamen': 51,
    'Right_Pallidum': 52,
    'Right_Hippocampus': 53,
    'Right_Amygdala': 54,
    'Right_Accumbens': 58,
}

# FSL FAST tissue labels
FSL_FAST_LABELS = {
    'CSF': 1,
    'Gray_Matter': 2,
    'White_Matter': 3,
}


class ROIExtractor:
    """
    Extract metrics from images using anatomical ROIs.
    
    Supports multiple ROI sources:
    - FreeSurfer (aseg.mgz, aparc+aseg.mgz)
    - FSL FAST (tissue segmentation)
    - FSL FIRST (subcortical structures)
    - Custom atlas (any labeled NIfTI)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self._roi_data: Optional[np.ndarray] = None
        self._roi_affine: Optional[np.ndarray] = None
        self._roi_labels: Dict[str, int] = {}
        self._roi_source: str = ""
        
    def set_roi_source(
        self,
        source: Literal['freesurfer', 'first', 'fast', 'custom'],
        segmentation_file: Optional[Path] = None,
        subjects_dir: Optional[Path] = None,
        subject_id: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        use_pve: bool = False,  # For FAST: use partial volume estimates
    ) -> None:
        """
        Set the ROI source for extraction.
        
        Args:
            source: ROI source type.
            segmentation_file: Path to segmentation file (for first, fast, custom).
            subjects_dir: FreeSurfer SUBJECTS_DIR (for freesurfer source).
            subject_id: FreeSurfer subject ID.
            label_map: Custom label mapping (structure_name -> label_value).
            use_pve: For FAST, use PVE files instead of hard segmentation.
        """
        self._roi_source = source
        
        if source == 'freesurfer':
            if not subjects_dir or not subject_id:
                raise ValueError("subjects_dir and subject_id required for FreeSurfer source")
            
            # Try aparc+aseg first, fall back to aseg
            aparc_aseg = Path(subjects_dir) / subject_id / "mri" / "aparc+aseg.mgz"
            aseg = Path(subjects_dir) / subject_id / "mri" / "aseg.mgz"
            
            seg_file = aparc_aseg if aparc_aseg.exists() else aseg
            if not seg_file.exists():
                raise FileNotFoundError(f"No FreeSurfer segmentation found for {subject_id}")
                
            self._load_segmentation(seg_file)
            self._roi_labels = label_map or FREESURFER_ASEG_LABELS
            
        elif source == 'first':
            if not segmentation_file:
                raise ValueError("segmentation_file required for FIRST source")
            self._load_segmentation(segmentation_file)
            self._roi_labels = label_map or FSL_FIRST_LABELS
            
        elif source == 'fast':
            if not segmentation_file:
                raise ValueError("segmentation_file required for FAST source")
            self._load_segmentation(segmentation_file)
            self._roi_labels = label_map or FSL_FAST_LABELS
            
        elif source == 'custom':
            if not segmentation_file:
                raise ValueError("segmentation_file required for custom source")
            if not label_map:
                raise ValueError("label_map required for custom source")
            self._load_segmentation(segmentation_file)
            self._roi_labels = label_map
            
        else:
            raise ValueError(f"Unknown source: {source}")
            
        self.logger.info(f"Loaded {source} ROIs with {len(self._roi_labels)} regions")
        
    def _load_segmentation(self, seg_file: Path) -> None:
        """Load segmentation file into memory."""
        if nib is None:
            raise ImportError("nibabel is required for ROI extraction")
            
        seg_file = Path(seg_file)
        if not seg_file.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_file}")
            
        img = nib.load(str(seg_file))
        self._roi_data = np.asarray(img.dataobj)
        self._roi_affine = img.affine
        
    def get_roi_mask(self, roi_name: str) -> Optional[np.ndarray]:
        """Get binary mask for a specific ROI."""
        if self._roi_data is None:
            raise RuntimeError("No ROI source loaded. Call set_roi_source first.")
            
        if roi_name not in self._roi_labels:
            self.logger.warning(f"ROI not found: {roi_name}")
            return None
            
        label = self._roi_labels[roi_name]
        return (self._roi_data == label).astype(np.float32)
        
    def extract_from_image(
        self,
        image_path: Path,
        metrics: List[str] = ['mean', 'std', 'median'],
        rois: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Extract metrics from an image for all ROIs.
        
        Args:
            image_path: Path to the image file.
            metrics: List of statistics to compute ('mean', 'std', 'median', 'min', 'max', 'volume').
            rois: Specific ROIs to extract. If None, extracts all.
            threshold: Optional threshold for ROI mask (for PVE maps).
            
        Returns:
            DataFrame with columns: ROI_Name, Metric, Statistic, Value
        """
        if self._roi_data is None:
            raise RuntimeError("No ROI source loaded. Call set_roi_source first.")
            
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load image
        img = nib.load(str(image_path))
        img_data = img.get_fdata()
        
        # Check dimensions match
        if img_data.shape[:3] != self._roi_data.shape[:3]:
            self.logger.warning(
                f"Image shape {img_data.shape[:3]} != ROI shape {self._roi_data.shape[:3]}. "
                "Metrics may be incorrect."
            )
            
        rois_to_extract = rois or list(self._roi_labels.keys())
        
        rows = []
        for roi_name in rois_to_extract:
            if roi_name not in self._roi_labels:
                continue
                
            mask = self.get_roi_mask(roi_name)
            if mask is None:
                continue
                
            if threshold is not None:
                mask = (mask > threshold).astype(np.float32)
                
            # Handle 4D images (extract mean across time/volumes)
            if img_data.ndim == 4:
                # Average across 4th dimension for mask application
                img_3d = np.mean(img_data, axis=3)
            else:
                img_3d = img_data
                
            # Extract values within mask
            masked_values = img_3d[mask > 0]
            
            if len(masked_values) == 0:
                continue
                
            # Compute statistics
            for stat in metrics:
                if stat == 'mean':
                    value = float(np.nanmean(masked_values))
                elif stat == 'std':
                    value = float(np.nanstd(masked_values))
                elif stat == 'median':
                    value = float(np.nanmedian(masked_values))
                elif stat == 'min':
                    value = float(np.nanmin(masked_values))
                elif stat == 'max':
                    value = float(np.nanmax(masked_values))
                elif stat == 'volume':
                    # Compute volume in mm³
                    voxel_dims = img.header.get_zooms()[:3]
                    voxel_vol = np.prod(voxel_dims)
                    value = float(np.sum(mask > 0) * voxel_vol)
                else:
                    continue
                    
                rows.append({
                    'ROI_Name': roi_name,
                    'ROI_Source': self._roi_source,
                    'Statistic': stat,
                    'Value': value,
                })
                
        return pd.DataFrame(rows)
        
    def extract_diffusion_metrics(
        self,
        fa_path: Optional[Path] = None,
        md_path: Optional[Path] = None,
        rd_path: Optional[Path] = None,
        ad_path: Optional[Path] = None,
        rois: Optional[List[str]] = None,
        statistics: List[str] = ['mean', 'std'],
    ) -> pd.DataFrame:
        """
        Extract diffusion metrics from FA, MD, RD, AD maps.
        
        Args:
            fa_path: Path to FA map.
            md_path: Path to MD map.
            rd_path: Path to RD map.
            ad_path: Path to AD map.
            rois: Specific ROIs to extract.
            statistics: Statistics to compute.
            
        Returns:
            DataFrame with columns: ROI_Name, ROI_Source, Metric, Statistic, Value
        """
        all_results = []
        
        metric_paths = {
            'FA': fa_path,
            'MD': md_path,
            'RD': rd_path,
            'AD': ad_path,
        }
        
        for metric_name, path in metric_paths.items():
            if path is None or not Path(path).exists():
                continue
                
            df = self.extract_from_image(path, metrics=statistics, rois=rois)
            df['Metric'] = metric_name
            df['Modality'] = 'Diffusion'
            all_results.append(df)
            
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
        
    def extract_relaxometry_metrics(
        self,
        t1_path: Optional[Path] = None,
        t2_path: Optional[Path] = None,
        r1_path: Optional[Path] = None,
        r2_path: Optional[Path] = None,
        pd_path: Optional[Path] = None,
        rois: Optional[List[str]] = None,
        statistics: List[str] = ['mean', 'std'],
    ) -> pd.DataFrame:
        """
        Extract relaxometry metrics from T1, T2, R1, R2, PD maps.
        
        Args:
            t1_path: Path to T1 map (ms).
            t2_path: Path to T2 map (ms).
            r1_path: Path to R1 map (1/s).
            r2_path: Path to R2 map (1/s).
            pd_path: Path to PD map.
            rois: Specific ROIs to extract.
            statistics: Statistics to compute.
            
        Returns:
            DataFrame with columns: ROI_Name, ROI_Source, Metric, Statistic, Value
        """
        all_results = []
        
        metric_paths = {
            'T1': t1_path,
            'T2': t2_path,
            'R1': r1_path,
            'R2': r2_path,
            'PD': pd_path,
        }
        
        for metric_name, path in metric_paths.items():
            if path is None or not Path(path).exists():
                continue
                
            df = self.extract_from_image(path, metrics=statistics, rois=rois)
            df['Metric'] = metric_name
            df['Modality'] = 'Relaxometry'
            all_results.append(df)
            
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()


def extract_roi_metrics(
    segmentation_file: Path,
    image_files: Dict[str, Path],
    source: Literal['freesurfer', 'first', 'fast', 'custom'] = 'first',
    label_map: Optional[Dict[str, int]] = None,
    statistics: List[str] = ['mean', 'std', 'median'],
    rois: Optional[List[str]] = None,
    subject_id: str = "",
    session: str = "",
    study: str = "",
) -> pd.DataFrame:
    """
    Convenience function to extract ROI metrics from multiple images.
    
    Args:
        segmentation_file: Path to segmentation/atlas file.
        image_files: Dictionary mapping metric_name -> image_path.
        source: ROI source type.
        label_map: Custom label mapping.
        statistics: Statistics to compute.
        rois: Specific ROIs to extract.
        subject_id: Subject ID for output.
        session: Session for output.
        study: Study name for output.
        
    Returns:
        DataFrame with ROI metrics in tidy format.
    """
    extractor = ROIExtractor()
    extractor.set_roi_source(
        source=source,
        segmentation_file=segmentation_file,
        label_map=label_map,
    )
    
    all_results = []
    for metric_name, image_path in image_files.items():
        if not Path(image_path).exists():
            continue
            
        df = extractor.extract_from_image(image_path, metrics=statistics, rois=rois)
        df['Metric'] = metric_name
        all_results.append(df)
        
    if not all_results:
        return pd.DataFrame()
        
    result = pd.concat(all_results, ignore_index=True)
    result['Subject_ID'] = subject_id
    result['Session'] = session
    result['Study'] = study
    
    return result


def save_roi_metrics(
    df: pd.DataFrame,
    out_path: Path,
    format: str = 'tsv',
) -> Path:
    """
    Save ROI metrics DataFrame to file.
    
    Args:
        df: DataFrame with ROI metrics.
        out_path: Output file path.
        format: Output format ('tsv', 'csv', 'xlsx').
        
    Returns:
        Path to saved file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'tsv':
        df.to_csv(out_path, sep='\t', index=False)
    elif format == 'csv':
        df.to_csv(out_path, index=False)
    elif format == 'xlsx':
        df.to_excel(out_path, index=False, engine='openpyxl')
    else:
        raise ValueError(f"Unknown format: {format}")
        
    return out_path
