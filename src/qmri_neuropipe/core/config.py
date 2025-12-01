"""
Configuration management for qMRI neuroimaging pipelines.

This module provides a flexible configuration system that supports:
- YAML and JSON configuration files
- Environment variable expansion
- Configuration validation
- Hierarchical configuration merging
- Type-safe access to configuration values

Classes:
    PipelineConfig: Main configuration class
    ConfigLoader: Utility for loading and merging configurations
"""

from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import os
import yaml
import json
import logging
from dataclasses import dataclass, field, asdict


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration container for pipeline execution.
    
    This class stores all configuration parameters for a pipeline.
    It supports:
    - Loading from YAML/JSON files
    - Environment variable expansion
    - Hierarchical key access with dot notation
    - Validation of required fields
    - Merging multiple configuration sources
    
    Attributes:
        bids_dir: Path to BIDS dataset
        output_dir: Path to output directory
        work_dir: Path to working directory (temporary files)
        participant_label: Optional list of subject IDs to process
        session_label: Optional list of session IDs to process
        n_cpus: Number of CPUs to use
        memory_gb: Memory limit in GB
        skip_existing: Whether to skip already-processed subjects
        stop_on_error: Whether to stop pipeline on first error
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        debug: Enable debug mode
        config_data: Dictionary containing all configuration data
    
    Example:
        >>> # From file
        >>> config = PipelineConfig.from_file('config.yaml')
        >>> 
        >>> # Programmatic creation
        >>> config = PipelineConfig(
        ...     bids_dir='/data/bids',
        ...     output_dir='/data/derivatives',
        ...     n_cpus=8
        ... )
        >>> 
        >>> # Access nested values
        >>> denoising_method = config.get('dmri.preprocessing.denoising.method')
    """
    
    # Core paths
    bids_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    work_dir: Optional[Path] = None
    
    # Subject/session selection
    participant_label: Optional[List[str]] = None
    session_label: Optional[List[str]] = None
    
    # Computational resources
    n_cpus: int = 1
    memory_gb: float = 8.0
    use_gpu: bool = False
    
    # Execution control
    skip_existing: bool = True
    stop_on_error: bool = False
    
    # Logging
    log_level: str = 'INFO'
    debug: bool = False
    verbose: bool = False
    
    # Additional configuration
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if self.bids_dir and not isinstance(self.bids_dir, Path):
            self.bids_dir = Path(self.bids_dir)
        if self.output_dir and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if self.work_dir and not isinstance(self.work_dir, Path):
            self.work_dir = Path(self.work_dir)
        
        # Set work_dir to output_dir/work if not specified
        if self.work_dir is None and self.output_dir:
            self.work_dir = self.output_dir / 'work'
    
    @classmethod
    def from_file(
        cls,
        config_file: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None
    ) -> 'PipelineConfig':
        """
        Load configuration from YAML or JSON file.
        
        Args:
            config_file: Path to configuration file
            overrides: Optional dictionary of values to override
        
        Returns:
            PipelineConfig instance
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        
        Example:
            >>> config = PipelineConfig.from_file('config.yaml')
            >>> 
            >>> # With overrides
            >>> config = PipelineConfig.from_file(
            ...     'config.yaml',
            ...     overrides={'n_cpus': 16}
            ... )
        """
        config_file = Path(config_file)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Load file
        loader = ConfigLoader()
        config_data = loader.load(config_file)
        
        # Merge overrides
        if overrides:
            config_data = loader.merge_configs(config_data, overrides)
        
        # Extract standard fields
        standard_fields = {
            'bids_dir': config_data.get('bids_dir'),
            'output_dir': config_data.get('output_dir'),
            'work_dir': config_data.get('work_dir'),
            'participant_label': config_data.get('participant_label'),
            'session_label': config_data.get('session_label'),
            'n_cpus': config_data.get('n_cpus', 1),
            'memory_gb': config_data.get('memory_gb', 8.0),
            'use_gpu': config_data.get('use_gpu', False),
            'skip_existing': config_data.get('skip_existing', True),
            'stop_on_error': config_data.get('stop_on_error', False),
            'log_level': config_data.get('log_level', 'INFO'),
            'debug': config_data.get('debug', False),
            'verbose': config_data.get('verbose', False),
        }
        
        # Store everything in config_data
        config = cls(**standard_fields, config_data=config_data)
        
        # Validate
        config.validate()
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            PipelineConfig instance
        """
        return cls.from_file.__func__(
            cls,
            config_file=None,
            overrides=config_dict
        ) if config_dict else cls()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support.
        
        Supports nested key access using dot notation.
        First checks standard fields, then searches config_data.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config.get('n_cpus')  # Standard field
            8
            >>> config.get('dmri.preprocessing.denoising.method')  # Nested
            'mppca'
            >>> config.get('nonexistent', 'default_value')
            'default_value'
        """
        # Check standard fields first
        if '.' not in key and hasattr(self, key):
            return getattr(self, key)
        
        # Search in config_data
        return self._get_nested(self.config_data, key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value with dot notation support.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        
        Example:
            >>> config.set('n_cpus', 16)
            >>> config.set('dmri.preprocessing.denoising.enabled', True)
        """
        if '.' not in key and hasattr(self, key):
            setattr(self, key, value)
        else:
            self._set_nested(self.config_data, key, value)
    
    def _get_nested(self, d: Dict, key: str, default: Any = None) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = key.split('.')
        current = d
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _set_nested(self, d: Dict, key: str, value: Any) -> None:
        """Set nested dictionary value using dot notation."""
        keys = key.split('.')
        current = d
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def validate(self) -> None:
        """
        Validate configuration.
        
        Checks:
        - Required fields are present
        - Paths exist (if they should)
        - Values are in valid ranges
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Check required fields
        if self.bids_dir is None:
            errors.append("bids_dir is required")
        elif not self.bids_dir.exists():
            errors.append(f"bids_dir does not exist: {self.bids_dir}")
        
        if self.output_dir is None:
            errors.append("output_dir is required")
        
        # Validate resource limits
        if self.n_cpus < 1:
            errors.append(f"n_cpus must be >= 1, got {self.n_cpus}")
        
        if self.memory_gb < 1:
            errors.append(f"memory_gb must be >= 1, got {self.memory_gb}")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level not in valid_levels:
            errors.append(
                f"log_level must be one of {valid_levels}, got {self.log_level}"
            )
        
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + 
                "\n".join(f"  - {e}" for e in errors)
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        # Start with standard fields
        result = {
            'bids_dir': str(self.bids_dir) if self.bids_dir else None,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'work_dir': str(self.work_dir) if self.work_dir else None,
            'participant_label': self.participant_label,
            'session_label': self.session_label,
            'n_cpus': self.n_cpus,
            'memory_gb': self.memory_gb,
            'use_gpu': self.use_gpu,
            'skip_existing': self.skip_existing,
            'stop_on_error': self.stop_on_error,
            'log_level': self.log_level,
            'debug': self.debug,
            'verbose': self.verbose,
        }
        
        # Merge with config_data
        result.update(self.config_data)
        
        return result
    
    def save(self, output_file: Union[str, Path], format: str = 'yaml') -> None:
        """
        Save configuration to file.
        
        Args:
            output_file: Output file path
            format: File format ('yaml' or 'json')
        
        Example:
            >>> config.save('config.yaml')
            >>> config.save('config.json', format='json')
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(output_file, 'w') as f:
            if format == 'yaml':
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format == 'json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved configuration to {output_file}")


class ConfigLoader:
    """
    Utility class for loading and merging configurations.
    
    Supports:
    - Loading YAML and JSON files
    - Environment variable expansion
    - Merging multiple configurations
    - Validation
    """
    
    @staticmethod
    def load(config_file: Path) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        
        Returns:
            Configuration dictionary
        
        Raises:
            ValueError: If file format is unknown
        """
        suffix = config_file.suffix.lower()
        
        with open(config_file) as f:
            if suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(
                    f"Unknown config format: {suffix}. "
                    "Supported: .yaml, .yml, .json"
                )
        
        # Expand environment variables
        config = ConfigLoader._expand_env_vars(config)
        
        return config
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Override values take precedence over base values.
        Nested dictionaries are merged recursively.
        
        Args:
            base: Base configuration
            override: Override configuration
        
        Returns:
            Merged configuration
        
        Example:
            >>> base = {'a': 1, 'b': {'c': 2}}
            >>> override = {'b': {'d': 3}}
            >>> merge_configs(base, override)
            {'a': 1, 'b': {'c': 2, 'd': 3}}
        """
        merged = base.copy()
        
        for key, value in override.items():
            if (key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)):
                # Recursively merge nested dictionaries
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    @staticmethod
    def _expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively expand environment variables in configuration.
        
        Supports ${VAR_NAME} syntax.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Configuration with expanded variables
        """
        if isinstance(config, dict):
            return {
                k: ConfigLoader._expand_env_vars(v)
                for k, v in config.items()
            }
        elif isinstance(config, list):
            return [ConfigLoader._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            return os.path.expandvars(config)
        else:
            return config


def create_default_config(modality: str = 'dmri') -> Dict[str, Any]:
    """
    Create default configuration for a modality.
    
    Args:
        modality: Modality name ('dmri', 'fmri', 'anat')
    
    Returns:
        Default configuration dictionary
    
    Example:
        >>> config_dict = create_default_config('dmri')
        >>> config = PipelineConfig.from_dict(config_dict)
    """
    if modality == 'dmri':
        return {
            'bids_dir': None,  # Must be set by user
            'output_dir': None,  # Must be set by user
            'n_cpus': 4,
            'memory_gb': 16.0,
            'skip_existing': True,
            'log_level': 'INFO',
            'dmri': {
                'preprocessing': {
                    'denoising': {
                        'enabled': True,
                        'method': 'mppca',
                        'parameters': {
                            'patch_radius': 2
                        }
                    },
                    'degibbs': {
                        'enabled': True
                    },
                    'eddy': {
                        'enabled': True,
                        'use_gpu': False,
                        'parameters': {
                            'repol': True,
                            'data_is_shelled': True
                        }
                    },
                    'bias_correction': {
                        'enabled': True,
                        'method': 'ants'
                    }
                },
                'modeling': {
                    'tensor': {
                        'enabled': True,
                        'metrics': ['fa', 'md', 'rd', 'ad']
                    },
                    'csd': {
                        'enabled': True,
                        'method': 'msmt_csd',
                        'lmax': 8
                    },
                    'tractography': {
                        'enabled': False,
                        'algorithm': 'iFOD2',
                        'n_streamlines': 10000000
                    }
                },
                'qc': {
                    'enabled': True,
                    'generate_reports': True
                }
            }
        }
    
    elif modality == 'fmri':
        return {
            'bids_dir': None,
            'output_dir': None,
            'n_cpus': 4,
            'memory_gb': 16.0,
            'skip_existing': True,
            'log_level': 'INFO',
            'fmri': {
                'preprocessing': {
                    'slice_timing': {
                        'enabled': True
                    },
                    'motion_correction': {
                        'enabled': True,
                        'reference': 'mean'
                    },
                    'distortion_correction': {
                        'enabled': True
                    },
                    'smoothing': {
                        'enabled': True,
                        'fwhm': 6.0
                    },
                    'temporal_filtering': {
                        'enabled': True,
                        'highpass': 0.01,
                        'lowpass': 0.1
                    }
                },
                'qc': {
                    'enabled': True,
                    'compute_tsnr': True,
                    'generate_reports': True
                }
            }
        }
    
    elif modality == 'anat':
        return {
            'bids_dir': None,
            'output_dir': None,
            'n_cpus': 4,
            'memory_gb': 16.0,
            'skip_existing': True,
            'log_level': 'INFO',
            'anat': {
                'preprocessing': {
                    'brain_extraction': {
                        'enabled': True,
                        'method': 'synthstrip'
                    },
                    'bias_correction': {
                        'enabled': True
                    },
                    'segmentation': {
                        'enabled': True,
                        'method': 'fast'
                    }
                },
                'surface_reconstruction': {
                    'enabled': False,
                    'run_freesurfer': False
                },
                'qc': {
                    'enabled': True,
                    'generate_reports': True
                }
            }
        }
    
    else:
        raise ValueError(f"Unknown modality: {modality}")


# Module version
__version__ = '2.0.0'