"""
Provenance tracking for qMRI neuroimaging pipelines.

This module provides comprehensive tracking of:
- Processing steps and parameters
- Input and output files
- Software versions and dependencies
- Computational environment
- Execution timing
- Data provenance (file hashes)

Classes:
    ProvenanceTracker: Main class for tracking provenance
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import hashlib
import platform
import sys
from datetime import datetime
import subprocess
import logging


logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Track processing provenance for reproducibility.
    
    Records all information needed to reproduce pipeline execution:
    - Pipeline metadata (name, version)
    - Processing steps with parameters
    - Input/output files with checksums
    - Software environment
    - Execution timing
    
    Attributes:
        output_dir: Directory for saving provenance file
        provenance: Dictionary containing all provenance data
    
    Example:
        >>> tracker = ProvenanceTracker(
        ...     output_dir=Path('/data/derivatives'),
        ...     pipeline_name='dmri-pipeline',
        ...     pipeline_version='2.0.0'
        ... )
        >>> 
        >>> tracker.log_step(
        ...     step_name='denoising',
        ...     inputs={'dwi': '/data/sub-01/dwi.nii.gz'},
        ...     outputs={'denoised': '/data/derivatives/denoised.nii.gz'},
        ...     parameters={'method': 'mppca'}
        ... )
        >>> 
        >>> tracker.save()
    """
    
    def __init__(
        self,
        output_dir: Path,
        pipeline_name: str,
        pipeline_version: str
    ):
        """
        Initialize provenance tracker.
        
        Args:
            output_dir: Directory for saving provenance file
            pipeline_name: Name of the pipeline
            pipeline_version: Version of the pipeline
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize provenance record
        self.provenance = {
            'pipeline': {
                'name': pipeline_name,
                'version': pipeline_version,
                'url': 'https://github.com/Developing-Brain-Imaging-Lab/qmri-neuropipe'
            },
            'execution': {
                'start_time': datetime.now().isoformat(),
                'end_time': None,
                'duration_seconds': None
            },
            'environment': self._get_environment(),
            'steps': []
        }
        
        logger.debug(f"Initialized provenance tracker for {pipeline_name} v{pipeline_version}")
    
    def _get_environment(self) -> Dict[str, Any]:
        """
        Capture execution environment details.
        
        Returns:
            Dictionary with environment information
        """
        env = {
            'python': {
                'version': sys.version,
                'executable': sys.executable
            },
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'node': platform.node()
            },
            'dependencies': self._get_dependencies()
        }
        
        return env
    
    def _get_dependencies(self) -> Dict[str, str]:
        """
        Get versions of key dependencies.
        
        Returns:
            Dictionary mapping package names to versions
        """
        deps = {}
        
        # Python packages
        packages = [
            'numpy', 'scipy', 'nibabel', 'dipy', 'nilearn',
            'nipype', 'bids', 'pybids'
        ]
        
        for package in packages:
            try:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                deps[package] = version
            except ImportError:
                deps[package] = 'not installed'
        
        # External tools (FSL, ANTs, FreeSurfer)
        external_tools = {
            'fsl': ['fslversion'],
            'ants': ['antsRegistration', '--version'],
            'freesurfer': ['mri_convert', '--version'],
            'mrtrix3': ['mrconvert', '--version']
        }
        
        for tool, cmd in external_tools.items():
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Extract version from output (first line usually)
                version = result.stdout.split('\n')[0] if result.stdout else 'unknown'
                deps[tool] = version
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                deps[tool] = 'not found'
        
        return deps
    
    def log_step(
        self,
        step_name: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        parameters: Dict[str, Any],
        duration: Optional[float] = None
    ) -> None:
        """
        Log a processing step with full provenance.
        
        Args:
            step_name: Name of the processing step
            inputs: Dictionary of input files/data
            outputs: Dictionary of output files/data
            parameters: Processing parameters used
            duration: Optional execution duration in seconds
        
        Example:
            >>> tracker.log_step(
            ...     step_name='brain_extraction',
            ...     inputs={'image': '/data/T1w.nii.gz'},
            ...     outputs={'mask': '/data/mask.nii.gz'},
            ...     parameters={'method': 'bet', 'threshold': 0.5},
            ...     duration=15.3
            ... )
        """
        step_record = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'inputs': self._process_io_dict(inputs),
            'outputs': self._process_io_dict(outputs),
            'parameters': parameters
        }
        
        if duration is not None:
            step_record['duration_seconds'] = duration
        
        self.provenance['steps'].append(step_record)
        
        logger.debug(f"Logged provenance for step: {step_name}")
    
    def _process_io_dict(self, io_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input/output dictionary to add file checksums.
        
        Args:
            io_dict: Dictionary of file paths or data
        
        Returns:
            Processed dictionary with checksums
        """
        processed = {}
        
        for key, value in io_dict.items():
            if isinstance(value, (str, Path)):
                # Avoid long strings that are definitely not paths (fixes OSError: File name too long)
                if isinstance(value, str) and (len(value) > 1024 or '{' in value or '\n' in value):
                    processed[key] = value
                    continue
                    
                try:
                    filepath = Path(value)
                    if filepath.exists() and filepath.is_file():
                        processed[key] = {
                            'path': str(filepath),
                            'checksum': self._compute_checksum(filepath)
                        }
                    else:
                        processed[key] = str(filepath)
                except OSError:
                    # Handle "File name too long" or other path errors
                    processed[key] = str(value)
            else:
                processed[key] = str(value)
        
        return processed
    
    def _compute_checksum(self, filepath: Path) -> str:
        """
        Compute MD5 checksum of file.
        
        Args:
            filepath: Path to file
        
        Returns:
            MD5 checksum string
        """
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to compute checksum for {filepath}: {e}")
            return 'checksum_failed'
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add custom metadata to provenance record.
        
        Args:
            key: Metadata key
            value: Metadata value
        
        Example:
            >>> tracker.add_metadata('dataset_name', 'My Study')
            >>> tracker.add_metadata('n_subjects', 50)
        """
        if 'metadata' not in self.provenance:
            self.provenance['metadata'] = {}
        
        self.provenance['metadata'][key] = value
    
    def save(self, filename: str = 'provenance.json') -> None:
        """
        Save provenance record to JSON file.
        
        Args:
            filename: Name of provenance file
        
        Example:
            >>> tracker.save()  # Saves to provenance.json
            >>> tracker.save('sub-01_provenance.json')  # Custom name
        """
        # Update end time
        self.provenance['execution']['end_time'] = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.provenance['execution']['start_time'])
        end = datetime.fromisoformat(self.provenance['execution']['end_time'])
        duration = (end - start).total_seconds()
        self.provenance['execution']['duration_seconds'] = duration
        
        # Save to file
        provenance_file = self.output_dir / filename
        with open(provenance_file, 'w') as f:
            json.dump(self.provenance, f, indent=2)
        
        logger.info(f"Saved provenance to {provenance_file}")
    
    @staticmethod
    def load(provenance_file: Path) -> Dict[str, Any]:
        """
        Load provenance record from file.
        
        Args:
            provenance_file: Path to provenance JSON file
        
        Returns:
            Provenance dictionary
        
        Example:
            >>> prov = ProvenanceTracker.load(Path('provenance.json'))
            >>> print(prov['pipeline']['name'])
        """
        with open(provenance_file) as f:
            return json.load(f)
    
    @staticmethod
    def compare_provenance(
        prov1: Dict[str, Any],
        prov2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two provenance records.
        
        Useful for checking if two datasets were processed identically.
        
        Args:
            prov1: First provenance record
            prov2: Second provenance record
        
        Returns:
            Dictionary with comparison results
        
        Example:
            >>> prov1 = ProvenanceTracker.load(Path('sub-01_provenance.json'))
            >>> prov2 = ProvenanceTracker.load(Path('sub-02_provenance.json'))
            >>> diff = ProvenanceTracker.compare_provenance(prov1, prov2)
            >>> if diff['identical']:
            ...     print("Same processing")
        """
        comparison = {
            'identical': True,
            'differences': []
        }
        
        # Compare pipeline versions
        if prov1['pipeline']['version'] != prov2['pipeline']['version']:
            comparison['identical'] = False
            comparison['differences'].append({
                'field': 'pipeline.version',
                'value1': prov1['pipeline']['version'],
                'value2': prov2['pipeline']['version']
            })
        
        # Compare number of steps
        if len(prov1['steps']) != len(prov2['steps']):
            comparison['identical'] = False
            comparison['differences'].append({
                'field': 'n_steps',
                'value1': len(prov1['steps']),
                'value2': len(prov2['steps'])
            })
        
        # Compare step parameters
        for i, (step1, step2) in enumerate(zip(prov1['steps'], prov2['steps'])):
            if step1['step'] != step2['step']:
                comparison['identical'] = False
                comparison['differences'].append({
                    'field': f'steps[{i}].name',
                    'value1': step1['step'],
                    'value2': step2['step']
                })
            
            if step1['parameters'] != step2['parameters']:
                comparison['identical'] = False
                comparison['differences'].append({
                    'field': f'steps[{i}].parameters',
                    'value1': step1['parameters'],
                    'value2': step2['parameters']
                })
        
        return comparison
    
    def generate_report(self, output_file: Path = None) -> str:
        """
        Generate human-readable provenance report.
        
        Args:
            output_file: Optional file to save report to
        
        Returns:
            Report string
        
        Example:
            >>> report = tracker.generate_report()
            >>> print(report)
            >>> 
            >>> # Save to file
            >>> tracker.generate_report(Path('provenance_report.txt'))
        """
        lines = []
        
        lines.append("=" * 70)
        lines.append(f"PROVENANCE REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Pipeline info
        lines.append("Pipeline Information:")
        lines.append(f"  Name: {self.provenance['pipeline']['name']}")
        lines.append(f"  Version: {self.provenance['pipeline']['version']}")
        lines.append("")
        
        # Execution info
        lines.append("Execution Information:")
        lines.append(f"  Start: {self.provenance['execution']['start_time']}")
        if self.provenance['execution']['end_time']:
            lines.append(f"  End: {self.provenance['execution']['end_time']}")
            duration = self.provenance['execution']['duration_seconds']
            lines.append(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        lines.append("")
        
        # Environment
        lines.append("Environment:")
        lines.append(f"  Python: {self.provenance['environment']['python']['version'].split()[0]}")
        lines.append(f"  Platform: {self.provenance['environment']['platform']['system']} "
                    f"{self.provenance['environment']['platform']['release']}")
        lines.append("")
        
        # Dependencies
        lines.append("Key Dependencies:")
        for pkg, version in self.provenance['environment']['dependencies'].items():
            lines.append(f"  {pkg}: {version}")
        lines.append("")
        
        # Processing steps
        lines.append(f"Processing Steps ({len(self.provenance['steps'])} total):")
        for i, step in enumerate(self.provenance['steps'], 1):
            lines.append(f"  {i}. {step['step']}")
            if 'duration_seconds' in step:
                lines.append(f"     Duration: {step['duration_seconds']:.1f}s")
            lines.append(f"     Parameters: {step['parameters']}")
        
        lines.append("")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Saved provenance report to {output_file}")
        
        return report


# Module version
__version__ = '2.0.0'