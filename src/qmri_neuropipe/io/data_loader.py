"""
Data loader for discovering and organizing MRI data for subjects/sessions.

This module provides functionality to scan BIDS datasets and create
structured dictionaries of available data types and their paths for
given subjects and sessions.

Classes:
    SubjectData: Container for all data types available for a subject/session
    DataLoader: Main class for loading and organizing BIDS data
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DataTypeFiles:
    """
    Container for files associated with a specific data type.
    
    Attributes:
        main_file: Primary data file (e.g., .nii.gz)
        json_file: Associated JSON sidecar
        additional_files: Dictionary of additional files (e.g., bval, bvec)
    """
    main_file: Path
    json_file: Optional[Path] = None
    additional_files: Dict[str, Path] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'main_file': str(self.main_file),
            'json_file': str(self.json_file) if self.json_file else None,
            'additional_files': {k: str(v) for k, v in self.additional_files.items()}
        }
    
    def exists(self) -> bool:
        """Check if main file exists."""
        return self.main_file.exists()


@dataclass
class SubjectData:
    """
    Container for all available data types for a subject/session.
    
    Organizes data by modality (anat, dwi, func, fmap) and data type
    (T1w, T2w, dwi, bold, etc.).
    
    Attributes:
        subject: Subject identifier
        session: Session identifier (None for single-session datasets)
        bids_dir: Path to BIDS dataset root
        data: Nested dictionary of modality -> datatype -> files
    
    Example structure:
        {
            'anat': {
                'T1w': [DataTypeFiles(...)],
                'T2w': [DataTypeFiles(...)]
            },
            'dwi': {
                'dwi': [DataTypeFiles(...)]
            },
            'func': {
                'bold': [DataTypeFiles(...), DataTypeFiles(...)]
            }
        }
    """
    subject: str
    session: Optional[str] = None
    bids_dir: Optional[Path] = None
    data: Dict[str, Dict[str, List[DataTypeFiles]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    
    @property
    def subject_dir(self) -> Path:
        """Get subject directory path."""
        if not self.bids_dir:
            raise ValueError("bids_dir not set")
        
        if self.session:
            return self.bids_dir / f"sub-{self.subject}" / f"ses-{self.session}"
        else:
            return self.bids_dir / f"sub-{self.subject}"
    
    def has_modality(self, modality: str) -> bool:
        """Check if modality exists for this subject/session."""
        return modality in self.data and len(self.data[modality]) > 0
    
    def has_datatype(self, modality: str, datatype: str) -> bool:
        """Check if specific datatype exists for this subject/session."""
        return (modality in self.data and 
                datatype in self.data[modality] and 
                len(self.data[modality][datatype]) > 0)
    
    def get_files(self, modality: str, datatype: str, run: Optional[int] = None) -> Optional[DataTypeFiles]:
        """
        Get files for a specific modality and datatype.
        
        Args:
            modality: Modality name (anat, dwi, func, fmap)
            datatype: Datatype suffix (T1w, T2w, dwi, bold, etc.)
            run: Optional run number (1-indexed)
        
        Returns:
            DataTypeFiles object or None if not found
        """
        if not self.has_datatype(modality, datatype):
            return None
        
        files = self.data[modality][datatype]
        
        if run is not None:
            # Return specific run (1-indexed)
            if 0 < run <= len(files):
                return files[run - 1]
            return None
        
        # Return first available if no run specified
        return files[0] if files else None
    
    def get_all_files(self, modality: str, datatype: str) -> List[DataTypeFiles]:
        """
        Get all files for a specific modality and datatype (all runs).
        
        Args:
            modality: Modality name
            datatype: Datatype suffix
        
        Returns:
            List of DataTypeFiles objects
        """
        if not self.has_datatype(modality, datatype):
            return []
        return self.data[modality][datatype]
    
    def list_modalities(self) -> List[str]:
        """Get list of available modalities."""
        return sorted(self.data.keys())
    
    def list_datatypes(self, modality: str) -> List[str]:
        """Get list of available datatypes for a modality."""
        if modality not in self.data:
            return []
        return sorted(self.data[modality].keys())
    
    def summary(self) -> Dict[str, Any]:
        """
        Generate summary of available data.
        
        Returns:
            Dictionary with counts of each datatype
        """
        summary = {
            'subject': self.subject,
            'session': self.session,
            'modalities': {}
        }
        
        for modality in self.list_modalities():
            summary['modalities'][modality] = {}
            for datatype in self.list_datatypes(modality):
                count = len(self.data[modality][datatype])
                summary['modalities'][modality][datatype] = count
        
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of all data
        """
        result = {
            'subject': self.subject,
            'session': self.session,
            'bids_dir': str(self.bids_dir) if self.bids_dir else None,
            'data': {}
        }
        
        for modality, datatypes in self.data.items():
            result['data'][modality] = {}
            for datatype, files_list in datatypes.items():
                result['data'][modality][datatype] = [
                    f.to_dict() for f in files_list
                ]
        
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        session_str = f"/ses-{self.session}" if self.session else ""
        return f"SubjectData(sub-{self.subject}{session_str}, modalities={self.list_modalities()})"


class DataLoader:
    """
    Load and organize BIDS data for subjects/sessions.
    
    This class scans BIDS datasets and creates structured representations
    of available data, organized by modality and datatype.
    
    Supports:
    - Anatomical data (T1w, T2w, FLAIR, etc.)
    - Diffusion MRI (dwi with bval/bvec)
    - Functional MRI (BOLD)
    - Field maps (magnitude, phasediff, epi)
    - Multi-run acquisitions
    
    Example:
        >>> loader = DataLoader('/data/bids')
        >>> data = loader.load_subject('001', session='01')
        >>> 
        >>> # Check what's available
        >>> print(data.summary())
        >>> 
        >>> # Get specific data
        >>> t1w = data.get_files('anat', 'T1w')
        >>> dwi = data.get_files('dwi', 'dwi')
        >>> bold_runs = data.get_all_files('func', 'bold')
    """
    
    # Define known modalities and their typical suffixes
    MODALITY_SUFFIXES = {
        'anat': ['T1w', 'T2w', 'VFA', 'T1rho', 'T1map', 'T2map', 'T2star', 
                 'FLAIR', 'FLASH', 'PD', 'PDmap', 'PDT2', 'inplaneT1', 
                 'inplaneT2', 'angio', 'defacemask', 'SWImagandphase'],
        'dwi': ['dwi', 'sbref'],
        'func': ['bold', 'cbv', 'phase', 'sbref'],
        'fmap': ['magnitude', 'magnitude1', 'magnitude2', 
                 'phasediff', 'phase1', 'phase2', 
                 'fieldmap', 'epi'],
        'perf': ['asl', 'm0scan'],
        'pet': ['pet'],
    }
    
    # Additional files to look for based on datatype
    ADDITIONAL_FILES = {
        'dwi': ['bval', 'bvec'],
        'asl': ['aslcontext'],
    }
    
    def __init__(self, bids_dir: Union[str, Path], validate: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            bids_dir: Path to BIDS dataset root
            validate: Whether to validate BIDS structure (basic checks)
        """
        self.bids_dir = Path(bids_dir)
        
        if not self.bids_dir.exists():
            raise FileNotFoundError(f"BIDS directory not found: {self.bids_dir}")
        
        if validate:
            self._validate_bids_structure()
        
        logger.debug(f"Initialized DataLoader for {self.bids_dir}")
    
    def _validate_bids_structure(self) -> None:
        """
        Perform basic BIDS validation.
        
        Checks for:
        - dataset_description.json
        - At least one subject directory
        """
        # Check for dataset_description.json
        dataset_desc = self.bids_dir / 'dataset_description.json'
        if not dataset_desc.exists():
            logger.warning(
                f"dataset_description.json not found in {self.bids_dir}. "
                "This may not be a valid BIDS dataset."
            )
        
        # Check for subject directories
        subject_dirs = list(self.bids_dir.glob('sub-*'))
        if not subject_dirs:
            logger.warning(f"No subject directories found in {self.bids_dir}")
    
    def load_subject(
        self, 
        subject: str, 
        session: Optional[str] = None
    ) -> SubjectData:
        """
        Load all available data for a subject/session.
        
        Args:
            subject: Subject identifier (without 'sub-' prefix)
            session: Optional session identifier (without 'ses-' prefix)
        
        Returns:
            SubjectData object containing all available data
        
        Example:
            >>> loader = DataLoader('/data/bids')
            >>> data = loader.load_subject('001', session='01')
            >>> print(data.summary())
        """
        logger.info(f"Loading data for sub-{subject}" + 
                   (f"/ses-{session}" if session else ""))
        
        # Create SubjectData container
        subject_data = SubjectData(
            subject=subject,
            session=session,
            bids_dir=self.bids_dir
        )
        
        # Get subject directory
        subject_dir = subject_data.subject_dir
        if not subject_dir.exists():
            logger.warning(f"Subject directory not found: {subject_dir}")
            return subject_data
        
        # Scan each modality
        for modality in self.MODALITY_SUFFIXES.keys():
            modality_dir = subject_dir / modality
            if not modality_dir.exists():
                continue
            
            logger.debug(f"Scanning {modality} directory")
            
            # OPTIMIZATION: Batch all file operations for this modality
            # This reduces repeated directory scans
            
            # Find all files for each suffix
            for suffix in self.MODALITY_SUFFIXES[modality]:
                files = self._find_files_by_suffix(modality_dir, suffix)
                
                for main_file in files:
                    # Create DataTypeFiles object with all sidecars in one pass
                    data_files = self._create_data_files(main_file, suffix)
                    
                    # Add to subject data
                    subject_data.data[modality][suffix].append(data_files)
                    
                    logger.debug(f"Found {modality}/{suffix}: {main_file.name}")
        
        # Log summary
        summary = subject_data.summary()
        logger.info(f"Loaded data summary: {summary}")
        
        return subject_data
    
    def _create_data_files(self, main_file: Path, suffix: str) -> DataTypeFiles:
        """
        Create DataTypeFiles object with all sidecars in one operation.
        
        Optimized to reduce repeated file system operations.
        
        Args:
            main_file: Main data file
            suffix: BIDS suffix (for determining additional files)
        
        Returns:
            DataTypeFiles object with all associated files
        """
        data_files = DataTypeFiles(main_file=main_file)
        
        # Look for JSON sidecar
        json_file = self._find_json_sidecar(main_file)
        if json_file:
            data_files.json_file = json_file
        
        # Look for additional files (e.g., bval/bvec for dwi)
        if suffix in self.ADDITIONAL_FILES:
            additional = self._find_additional_files(
                main_file, 
                self.ADDITIONAL_FILES[suffix]
            )
            data_files.additional_files.update(additional)
        
        return data_files
    
    def _find_files_by_suffix(
        self, 
        directory: Path, 
        suffix: str
    ) -> List[Path]:
        """
        Find all files with a specific BIDS suffix.
        
        Optimized to search for specific extensions rather than wildcard.
        
        Args:
            directory: Directory to search
            suffix: BIDS suffix (e.g., 'T1w', 'dwi')
        
        Returns:
            Sorted list of matching files
        """
        # OPTIMIZATION: Search for specific extensions (faster than wildcard)
        files = []
        
        # Look for .nii.gz first (most common)
        pattern_gz = f"*_{suffix}.nii.gz"
        files.extend(directory.glob(pattern_gz))
        
        # Then .nii
        pattern_nii = f"*_{suffix}.nii"
        files.extend(directory.glob(pattern_nii))
        
        return sorted(set(files))  # Remove duplicates and sort
    
    def _find_json_sidecar(self, main_file: Path) -> Optional[Path]:
        """
        Find JSON sidecar for a given file.
        
        Args:
            main_file: Main data file
        
        Returns:
            Path to JSON file or None
        """
        # Replace .nii.gz or .nii with .json
        if main_file.name.endswith('.nii.gz'):
            json_file = main_file.parent / main_file.name.replace('.nii.gz', '.json')
        else:
            json_file = main_file.with_suffix('.json')
        
        return json_file if json_file.exists() else None
    
    def _find_additional_files(
        self, 
        main_file: Path, 
        extensions: List[str]
    ) -> Dict[str, Path]:
        """
        Find additional files associated with main file.
        
        Args:
            main_file: Main data file
            extensions: List of extensions to look for (e.g., ['bval', 'bvec'])
        
        Returns:
            Dictionary mapping extension to file path
        """
        additional = {}
        
        # Remove .nii.gz or .nii from filename
        if main_file.name.endswith('.nii.gz'):
            base = main_file.name.replace('.nii.gz', '')
        else:
            base = main_file.stem
        
        for ext in extensions:
            # Look for file with same base name but different extension
            candidate = main_file.parent / f"{base}.{ext}"
            if candidate.exists():
                additional[ext] = candidate
        
        return additional
    
    def load_multiple_subjects(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        pairs: Optional[List[Tuple[str, Optional[str]]]] = None
    ) -> Dict[Tuple[str, Optional[str]], SubjectData]:
        """
        Load data for multiple subjects/sessions.
        
        Args:
            subjects: List of subject IDs (None = all subjects)
            sessions: List of session IDs (None = all sessions)
            pairs: Optional list of explicit (subject, session) pairs. 
                  If provided, subjects and sessions args are ignored.
        
        Returns:
            Dictionary mapping (subject, session) to SubjectData
        """
        from qmri_neuropipe.io.bids import select_participants_sessions
        
        # Get subject/session pairs
        if pairs is None:
            pairs = select_participants_sessions(
                self.bids_dir,
                participants=subjects,
                sessions=sessions
            )
        
        # Load data for each pair
        results = {}
        for subject, session in pairs:
            key = (subject, session)
            results[key] = self.load_subject(subject, session)
        
        logger.info(f"Loaded data for {len(results)} subject/session pairs")
        
        return results

    def load_from_subjects_file(
        self,
        subjects_file: Union[str, Path]
    ) -> Dict[Tuple[str, Optional[str]], SubjectData]:
        """
        Load data for specific subject/session pairs listed in a text file.
        
        The file should contain one 'subject,session' pair per line.
        Session is optional. Lines starting with # are ignored.
        
        Args:
            subjects_file: Path to subjects file
            
        Returns:
            Dictionary mapping (subject, session) to SubjectData
        """
        subjects_file = Path(subjects_file)
        if not subjects_file.exists():
            raise FileNotFoundError(f"Subjects file not found: {subjects_file}")
            
        pairs = []
        with open(subjects_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                sub = parts[0].strip()
                if sub.startswith('sub-'):
                    sub = sub[4:]
                    
                ses = None
                if len(parts) > 1:
                    ses = parts[1].strip()
                    if ses.startswith('ses-'):
                        ses = ses[4:]
                
                pairs.append((sub, ses))
        
        # Load data for each pair
        results = {}
        for subject, session in pairs:
            # We skip validation here as we are loading specific requested pairs
            # But we check if subject dir actually exists
            key = (subject, session)
            results[key] = self.load_subject(subject, session)
            
        logger.info(f"Loaded data for {len(results)} subject/session pairs from {subjects_file.name}")
        return results
    
    def get_dataset_summary(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate summary of entire dataset.
        
        Args:
            subjects: List of subject IDs (None = all subjects)
            sessions: List of session IDs (None = all sessions)
        
        Returns:
            Dictionary with dataset-wide statistics
        """
        data_dict = self.load_multiple_subjects(subjects, sessions)
        
        summary = {
            'n_subjects': len(set(k[0] for k in data_dict.keys())),
            'n_sessions': len(data_dict),
            'modalities': defaultdict(lambda: defaultdict(int))
        }
        
        # Count datatypes across all subjects
        for subject_data in data_dict.values():
            for modality in subject_data.list_modalities():
                for datatype in subject_data.list_datatypes(modality):
                    count = len(subject_data.data[modality][datatype])
                    summary['modalities'][modality][datatype] += count
        
        # Convert defaultdicts to regular dicts
        summary['modalities'] = {
            k: dict(v) for k, v in summary['modalities'].items()
        }
        
        return summary


def load_subject_data(
    bids_dir: Union[str, Path],
    subject: str,
    session: Optional[str] = None
) -> SubjectData:
    """
    Convenience function to load subject data.
    
    Args:
        bids_dir: Path to BIDS dataset
        subject: Subject identifier
        session: Optional session identifier
    
    Returns:
        SubjectData object
    
    Example:
        >>> data = load_subject_data('/data/bids', '001', session='01')
        >>> t1w = data.get_files('anat', 'T1w')
    """
    loader = DataLoader(bids_dir)
    return loader.load_subject(subject, session)


# Module version
__version__ = '1.0.0'
