"""
FSL Segmentation Steps for Anatomical Processing.

Provides processing steps for:
- FSL FAST (tissue segmentation)
- FSL FIRST (subcortical segmentation)
- fsl_anat (comprehensive anatomical processing)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ...core.utils import ensure_dir
from ...interfaces import fsl


logger = logging.getLogger(__name__)


class FSLFASTStep:
    """
    FSL FAST tissue segmentation step.
    
    Segments anatomical image into CSF, GM, and WM probability maps.
    """
    
    def __init__(
        self,
        img_type: int = 1,  # 1=T1, 2=T2, 3=PD
        num_classes: int = 3,
        extract_volumes: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.img_type = img_type
        self.num_classes = num_classes
        self.extract_volumes = extract_volumes
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    def run(self, context: dict) -> dict:
        """Run FAST segmentation."""
        in_file = context.get('current_anat') or context.get('t1w_file')
        if not in_file:
            raise ValueError("No input anatomical file found in context")
            
        output_dir = context.get('output_dir', Path('.'))
        output_dir = ensure_dir(output_dir / 'segmentation' / 'fast')
        
        in_path = Path(in_file)
        stem = in_path.name.replace('.nii.gz', '').replace('.nii', '')
        out_base = output_dir / f"{stem}_fast"
        
        # Check if already completed
        seg_file = output_dir / f"{stem}_fast_seg.nii.gz"
        if seg_file.exists():
            self.logger.info("FAST segmentation already complete, skipping...")
        else:
            self.logger.info(f"Running FAST segmentation on {in_path.name}")
            fsl.fast(
                in_files=in_file,
                out_base=out_base,
                img_type=self.img_type,
                num_classes=self.num_classes,
            )
        
        # Collect outputs
        pve_files = {
            'pve_0': output_dir / f"{stem}_fast_pve_0.nii.gz",  # CSF
            'pve_1': output_dir / f"{stem}_fast_pve_1.nii.gz",  # GM
            'pve_2': output_dir / f"{stem}_fast_pve_2.nii.gz",  # WM
        }
        
        context['fast_seg'] = seg_file
        context['fast_pve'] = pve_files
        context['fast_base'] = out_base
        
        # Extract volumes if requested
        if self.extract_volumes:
            volumes = fsl.extract_fast_volumes(pve_files)
            context['fast_volumes'] = volumes
            
            # Save to TSV
            volumes_file = output_dir / f"{stem}_fast_volumes.tsv"
            subject_id = context.get('subject_id', '')
            session = context.get('session', '')
            fsl.save_volumes_to_file(
                volumes, volumes_file, 
                subject_id=subject_id, session=session, format='tsv'
            )
            context['fast_volumes_file'] = volumes_file
            
        return context
        
    def _get_report_details(self, context: dict) -> Dict[str, Any]:
        """Get details for report."""
        volumes = context.get('fast_volumes', {})
        return {
            'Method': 'FSL FAST',
            'Image_Type': self.img_type,
            'Num_Classes': self.num_classes,
            'GM_Volume_mm3': volumes.get('GM_Volume_mm3', 'N/A'),
            'WM_Volume_mm3': volumes.get('WM_Volume_mm3', 'N/A'),
            'CSF_Volume_mm3': volumes.get('CSF_Volume_mm3', 'N/A'),
            'TIV_mm3': volumes.get('TIV_mm3', 'N/A'),
        }


class FSLFIRSTStep:
    """
    FSL FIRST subcortical structure segmentation step.
    
    Segments subcortical structures: hippocampus, amygdala, caudate,
    putamen, pallidum, thalamus, accumbens, brainstem.
    """
    
    def __init__(
        self,
        structures: str = "all",
        method: str = "auto",
        extract_volumes: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.structures = structures
        self.method = method
        self.extract_volumes = extract_volumes
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    def run(self, context: dict) -> dict:
        """Run FIRST segmentation."""
        in_file = context.get('current_anat') or context.get('t1w_file')
        if not in_file:
            raise ValueError("No input anatomical file found in context")
            
        output_dir = context.get('output_dir', Path('.'))
        output_dir = ensure_dir(output_dir / 'segmentation' / 'first')
        
        in_path = Path(in_file)
        stem = in_path.name.replace('.nii.gz', '').replace('.nii', '')
        
        # Check if already completed
        first_seg = output_dir / f"{stem}_all_fast_firstseg.nii.gz"
        if first_seg.exists():
            self.logger.info("FIRST segmentation already complete, skipping...")
            outputs = fsl._parse_first_outputs(output_dir, stem)
        else:
            self.logger.info(f"Running FIRST segmentation on {in_path.name}")
            outputs = fsl.first(
                in_file=in_file,
                out_dir=output_dir,
                structures=self.structures,
                method=self.method,
            )
        
        context['first_seg'] = outputs.get('combined')
        context['first_outputs'] = outputs
        
        # Extract volumes if requested
        if self.extract_volumes and outputs.get('combined'):
            volumes = fsl.extract_first_volumes(outputs['combined'])
            context['first_volumes'] = volumes
            
            # Save to TSV
            volumes_file = output_dir / f"{stem}_first_volumes.tsv"
            subject_id = context.get('subject_id', '')
            session = context.get('session', '')
            fsl.save_volumes_to_file(
                volumes, volumes_file,
                subject_id=subject_id, session=session, format='tsv'
            )
            context['first_volumes_file'] = volumes_file
            
        return context
        
    def _get_report_details(self, context: dict) -> Dict[str, Any]:
        """Get details for report."""
        volumes = context.get('first_volumes', {})
        
        details = {
            'Method': 'FSL FIRST',
            'Structures': self.structures,
            'Registration_Method': self.method,
        }
        
        # Add key volumes
        for key in ['Left_Hippocampus_Volume_mm3', 'Right_Hippocampus_Volume_mm3',
                    'Left_Thalamus_Volume_mm3', 'Right_Thalamus_Volume_mm3']:
            if key in volumes:
                details[key.replace('_Volume_mm3', '')] = f"{volumes[key]:.1f} mm³"
                
        return details


class FSLAnatStep:
    """
    Comprehensive FSL anatomical processing step.
    
    Runs the full fsl_anat pipeline including:
    - Bias field correction
    - Brain extraction
    - Tissue segmentation (FAST)
    - Subcortical segmentation (FIRST)
    - Registration to MNI
    """
    
    def __init__(
        self,
        img_type: str = "T1",
        nobias: bool = False,
        nosubcortseg: bool = False,
        noseg: bool = False,
        noreg: bool = False,
        extract_volumes: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.img_type = img_type
        self.nobias = nobias
        self.nosubcortseg = nosubcortseg
        self.noseg = noseg
        self.noreg = noreg
        self.extract_volumes = extract_volumes
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    def run(self, context: dict) -> dict:
        """Run fsl_anat pipeline."""
        in_file = context.get('current_anat') or context.get('t1w_file')
        if not in_file:
            raise ValueError("No input anatomical file found in context")
            
        output_dir = context.get('output_dir', Path('.'))
        output_dir = ensure_dir(output_dir / 'segmentation')
        
        in_path = Path(in_file)
        stem = in_path.name.replace('.nii.gz', '').replace('.nii', '')
        out_base = output_dir / f"{stem}_fsl_anat"
        
        # Check if already completed
        anat_dir = out_base.with_suffix('.anat')
        brain_file = anat_dir / "T1_biascorr_brain.nii.gz"
        
        if brain_file.exists():
            self.logger.info("fsl_anat already complete, skipping...")
            outputs = fsl._parse_fsl_anat_outputs(anat_dir)
        else:
            self.logger.info(f"Running fsl_anat on {in_path.name}")
            outputs = fsl.fsl_anat(
                in_file=in_file,
                out_dir=out_base,
                img_type=self.img_type,
                nobias=self.nobias,
                nosubcortseg=self.nosubcortseg,
                noseg=self.noseg,
                noreg=self.noreg,
            )
        
        context['fsl_anat_dir'] = anat_dir
        context['fsl_anat_outputs'] = outputs
        
        # Store key outputs in context
        if 'T1_biascorr_brain' in outputs:
            context['brain'] = outputs['T1_biascorr_brain']
        if 'T1_biascorr_brain_mask' in outputs:
            context['brain_mask'] = outputs['T1_biascorr_brain_mask']
        if 'T1_fast_seg' in outputs:
            context['fast_seg'] = outputs['T1_fast_seg']
        if 'T1_subcort_seg' in outputs:
            context['first_seg'] = outputs['T1_subcort_seg']
            
        # Extract volumes if requested
        if self.extract_volumes:
            all_volumes = {}
            
            # FAST volumes
            if not self.noseg:
                pve_files = {
                    'pve_0': outputs.get('T1_fast_pve_0'),
                    'pve_1': outputs.get('T1_fast_pve_1'),
                    'pve_2': outputs.get('T1_fast_pve_2'),
                }
                pve_files = {k: v for k, v in pve_files.items() if v and v.exists()}
                if pve_files:
                    fast_volumes = fsl.extract_fast_volumes(pve_files)
                    all_volumes.update(fast_volumes)
                    context['fast_volumes'] = fast_volumes
                    
            # FIRST volumes
            if not self.nosubcortseg and outputs.get('T1_subcort_seg'):
                first_volumes = fsl.extract_first_volumes(outputs['T1_subcort_seg'])
                all_volumes.update(first_volumes)
                context['first_volumes'] = first_volumes
                
            if all_volumes:
                context['fsl_anat_volumes'] = all_volumes
                
                # Save to TSV
                volumes_file = anat_dir / f"{stem}_fsl_anat_volumes.tsv"
                subject_id = context.get('subject_id', '')
                session = context.get('session', '')
                fsl.save_volumes_to_file(
                    all_volumes, volumes_file,
                    subject_id=subject_id, session=session, format='tsv'
                )
                context['fsl_anat_volumes_file'] = volumes_file
                
        return context
        
    def _get_report_details(self, context: dict) -> Dict[str, Any]:
        """Get details for report."""
        volumes = context.get('fsl_anat_volumes', {})
        return {
            'Method': 'fsl_anat',
            'Image_Type': self.img_type,
            'Bias_Correction': 'Disabled' if self.nobias else 'Enabled',
            'Tissue_Segmentation': 'Disabled' if self.noseg else 'Enabled',
            'Subcortical_Segmentation': 'Disabled' if self.nosubcortseg else 'Enabled',
            'MNI_Registration': 'Disabled' if self.noreg else 'Enabled',
            'TIV_mm3': volumes.get('TIV_mm3', 'N/A'),
        }


class VolumeExtractionStep:
    """
    Extract volumes from existing segmentations.
    
    Supports FreeSurfer, FSL FAST, and FSL FIRST segmentations.
    """
    
    def __init__(
        self,
        sources: List[str] = ['freesurfer', 'fast', 'first'],
        logger: Optional[logging.Logger] = None,
    ):
        self.sources = sources
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    def run(self, context: dict) -> dict:
        """Extract volumes from available segmentations."""
        output_dir = context.get('output_dir', Path('.'))
        volumes_dir = ensure_dir(output_dir / 'volumes')
        
        subject_id = context.get('subject_id', '')
        session = context.get('session', '')
        
        all_volumes = {}
        
        # FreeSurfer volumes
        if 'freesurfer' in self.sources:
            fs_subjects_dir = context.get('freesurfer_subjects_dir')
            fs_subject_id = context.get('freesurfer_subject_id')
            
            if fs_subjects_dir and fs_subject_id:
                fs_volumes = fsl.extract_freesurfer_volumes(
                    Path(fs_subjects_dir), fs_subject_id
                )
                if fs_volumes:
                    all_volumes['freesurfer'] = fs_volumes
                    
                    # Save FreeSurfer volumes
                    volumes_file = volumes_dir / f"freesurfer_volumes.tsv"
                    fsl.save_volumes_to_file(
                        fs_volumes, volumes_file,
                        subject_id=subject_id, session=session, format='tsv'
                    )
                    context['freesurfer_volumes_file'] = volumes_file
                    
        # FAST volumes
        if 'fast' in self.sources and context.get('fast_pve'):
            fast_volumes = fsl.extract_fast_volumes(context['fast_pve'])
            if fast_volumes:
                all_volumes['fast'] = fast_volumes
                
                volumes_file = volumes_dir / f"fast_volumes.tsv"
                fsl.save_volumes_to_file(
                    fast_volumes, volumes_file,
                    subject_id=subject_id, session=session, format='tsv'
                )
                context['fast_volumes_file'] = volumes_file
                
        # FIRST volumes
        if 'first' in self.sources and context.get('first_seg'):
            first_volumes = fsl.extract_first_volumes(context['first_seg'])
            if first_volumes:
                all_volumes['first'] = first_volumes
                
                volumes_file = volumes_dir / f"first_volumes.tsv"
                fsl.save_volumes_to_file(
                    first_volumes, volumes_file,
                    subject_id=subject_id, session=session, format='tsv'
                )
                context['first_volumes_file'] = volumes_file
                
        context['extracted_volumes'] = all_volumes
        
        return context
        
    def _get_report_details(self, context: dict) -> Dict[str, Any]:
        """Get details for report."""
        volumes = context.get('extracted_volumes', {})
        return {
            'Sources': ', '.join(self.sources),
            'FreeSurfer_Volumes': len(volumes.get('freesurfer', {})),
            'FAST_Volumes': len(volumes.get('fast', {})),
            'FIRST_Volumes': len(volumes.get('first', {})),
        }


class ROIExtractionStep:
    """
    Extract cross-modal metrics from anatomical ROIs.
    
    Extracts diffusion and relaxometry metrics from FreeSurfer,
    FSL FAST, or FSL FIRST ROIs.
    """
    
    def __init__(
        self,
        roi_source: str = 'first',  # 'freesurfer', 'fast', 'first'
        extract_diffusion: bool = True,
        extract_relaxometry: bool = True,
        statistics: List[str] = ['mean', 'std'],
        logger: Optional[logging.Logger] = None,
    ):
        self.roi_source = roi_source
        self.extract_diffusion = extract_diffusion
        self.extract_relaxometry = extract_relaxometry
        self.statistics = statistics
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
    def run(self, context: dict) -> dict:
        """Extract ROI metrics."""
        from ...lib.common.roi_extraction import ROIExtractor
        
        output_dir = context.get('output_dir', Path('.'))
        stats_dir = ensure_dir(output_dir / 'statistics')
        
        subject_id = context.get('subject_id', '')
        session = context.get('session', '')
        
        # Initialize ROI extractor
        extractor = ROIExtractor(logger=self.logger)
        
        # Set ROI source
        try:
            if self.roi_source == 'freesurfer':
                extractor.set_roi_source(
                    source='freesurfer',
                    subjects_dir=context.get('freesurfer_subjects_dir'),
                    subject_id=context.get('freesurfer_subject_id'),
                )
            elif self.roi_source == 'first':
                seg_file = context.get('first_seg')
                if not seg_file:
                    self.logger.warning("No FIRST segmentation found, skipping ROI extraction")
                    return context
                extractor.set_roi_source(source='first', segmentation_file=seg_file)
            elif self.roi_source == 'fast':
                seg_file = context.get('fast_seg')
                if not seg_file:
                    self.logger.warning("No FAST segmentation found, skipping ROI extraction")
                    return context
                extractor.set_roi_source(source='fast', segmentation_file=seg_file)
            else:
                self.logger.warning(f"Unknown ROI source: {self.roi_source}")
                return context
        except Exception as e:
            self.logger.warning(f"Failed to set ROI source: {e}")
            return context
            
        all_metrics = []
        
        # Extract diffusion metrics
        if self.extract_diffusion:
            dti_dir = context.get('dti_dir')
            if dti_dir:
                fa_path = context.get('fa_map') or (Path(dti_dir) / 'FA.nii.gz')
                md_path = context.get('md_map') or (Path(dti_dir) / 'MD.nii.gz')
                rd_path = context.get('rd_map') or (Path(dti_dir) / 'RD.nii.gz')
                ad_path = context.get('ad_map') or (Path(dti_dir) / 'AD.nii.gz')
                
                diff_metrics = extractor.extract_diffusion_metrics(
                    fa_path=fa_path if fa_path and Path(fa_path).exists() else None,
                    md_path=md_path if md_path and Path(md_path).exists() else None,
                    rd_path=rd_path if rd_path and Path(rd_path).exists() else None,
                    ad_path=ad_path if ad_path and Path(ad_path).exists() else None,
                    statistics=self.statistics,
                )
                if not diff_metrics.empty:
                    all_metrics.append(diff_metrics)
                    
        # Extract relaxometry metrics
        if self.extract_relaxometry:
            relax_dir = context.get('relaxometry_dir')
            if relax_dir:
                t1_path = context.get('t1_map') or (Path(relax_dir) / 'T1map.nii.gz')
                t2_path = context.get('t2_map') or (Path(relax_dir) / 'T2map.nii.gz')
                
                relax_metrics = extractor.extract_relaxometry_metrics(
                    t1_path=t1_path if t1_path and Path(t1_path).exists() else None,
                    t2_path=t2_path if t2_path and Path(t2_path).exists() else None,
                    statistics=self.statistics,
                )
                if not relax_metrics.empty:
                    all_metrics.append(relax_metrics)
                    
        # Combine and save
        if all_metrics:
            import pandas as pd
            combined = pd.concat(all_metrics, ignore_index=True)
            combined['Subject_ID'] = subject_id
            combined['Session'] = session
            
            context['roi_metrics'] = combined
            
            # Save to TSV
            metrics_file = stats_dir / f"roi_metrics_{self.roi_source}.tsv"
            from ...lib.common.roi_extraction import save_roi_metrics
            save_roi_metrics(combined, metrics_file, format='tsv')
            context['roi_metrics_file'] = metrics_file
            
        return context
        
    def _get_report_details(self, context: dict) -> Dict[str, Any]:
        """Get details for report."""
        metrics = context.get('roi_metrics')
        num_metrics = len(metrics) if metrics is not None else 0
        
        return {
            'ROI_Source': self.roi_source,
            'Diffusion_Extraction': 'Enabled' if self.extract_diffusion else 'Disabled',
            'Relaxometry_Extraction': 'Enabled' if self.extract_relaxometry else 'Disabled',
            'Statistics': ', '.join(self.statistics),
            'Total_Metrics': num_metrics,
        }
