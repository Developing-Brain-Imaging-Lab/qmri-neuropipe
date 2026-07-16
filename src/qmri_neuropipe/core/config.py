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
from dataclasses import dataclass


logger = logging.getLogger(__name__)


_UNSET = object()
_STANDARD_CONFIG_DEFAULTS = {
    "bids_dir": None,
    "output_dir": None,
    "work_dir": None,
    "participant_label": None,
    "session_label": None,
    "subjects_file": None,
    "n_cpus": 1,
    "memory_gb": 8.0,
    "use_gpu": False,
    "skip_existing": True,
    "stop_on_error": False,
    "log_level": "INFO",
    "debug": False,
    "verbose": False,
    "gpu_ids": None,
    "anat_input": None,
    "tracker": None,
}
_PATH_CONFIG_FIELDS = {"bids_dir", "output_dir", "work_dir", "subjects_file"}


class UniqueKeyYamlLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of overwriting them."""


def _construct_mapping_without_duplicate_keys(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(
                f"Duplicate YAML key '{key}' at line {key_node.start_mark.line + 1}, "
                f"column {key_node.start_mark.column + 1}. Merge repeated sections instead."
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_without_duplicate_keys,
)


@dataclass(init=False)
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
    bids_dir: Optional[Path]
    output_dir: Optional[Path]
    work_dir: Optional[Path]
    
    # Subject/session selection
    participant_label: Optional[List[str]]
    session_label: Optional[List[str]]
    subjects_file: Optional[Path]
    
    # Computational resources
    n_cpus: int
    memory_gb: float
    use_gpu: bool
    
    # Execution control
    skip_existing: bool
    stop_on_error: bool
    
    # Logging
    log_level: str
    debug: bool
    verbose: bool
    
    # GPU Configuration
    gpu_ids: Optional[List[int]]
    
    # Custom Input Configuration (e.g. non-standard Anatomical)
    anat_input: Optional[Dict[str, Any]]

    # Tracker instance (to avoid circular imports, type is Any)
    tracker: Optional[Any]

    # Additional configuration
    config_data: Dict[str, Any]

    def __init__(
        self,
        bids_dir=_UNSET,
        output_dir=_UNSET,
        work_dir=_UNSET,
        participant_label=_UNSET,
        session_label=_UNSET,
        subjects_file=_UNSET,
        n_cpus=_UNSET,
        memory_gb=_UNSET,
        use_gpu=_UNSET,
        skip_existing=_UNSET,
        stop_on_error=_UNSET,
        log_level=_UNSET,
        debug=_UNSET,
        verbose=_UNSET,
        gpu_ids=_UNSET,
        anat_input=_UNSET,
        tracker=_UNSET,
        config_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Build typed attribute views over one canonical configuration store."""
        data = dict(config_data or {})
        for key, default in _STANDARD_CONFIG_DEFAULTS.items():
            data.setdefault(key, default)

        explicit_values = {
            "bids_dir": bids_dir,
            "output_dir": output_dir,
            "work_dir": work_dir,
            "participant_label": participant_label,
            "session_label": session_label,
            "subjects_file": subjects_file,
            "n_cpus": n_cpus,
            "memory_gb": memory_gb,
            "use_gpu": use_gpu,
            "skip_existing": skip_existing,
            "stop_on_error": stop_on_error,
            "log_level": log_level,
            "debug": debug,
            "verbose": verbose,
            "gpu_ids": gpu_ids,
            "anat_input": anat_input,
            "tracker": tracker,
        }
        for key, value in explicit_values.items():
            if value is not _UNSET:
                data[key] = value

        object.__setattr__(self, "_data", data)
        self._normalize_store()

    def _get_standard_field(self, name: str) -> Any:
        """Shared getter body for every standard-field property below."""
        return self._data[name]

    def _set_standard_field(self, name: str, value: Any) -> None:
        """Shared setter body for every standard-field property below.

        Path-typed fields are coerced to ``Path`` on assignment, matching the
        coercion ``_normalize_store`` applies at construction time.
        """
        if name in _PATH_CONFIG_FIELDS and value is not None:
            value = Path(value)
        self._data[name] = value

    # Explicit properties (rather than a generic __getattribute__/__setattr__
    # override) keep attribute access on PipelineConfig instances behaving
    # exactly like normal Python attributes for every tool that introspects
    # objects generically (debuggers, copy.deepcopy, dataclasses.fields,
    # pickling, etc.), while still reading/writing the single canonical
    # ``_data`` store underneath. Each property is a thin, identical wrapper
    # around the shared get/set helpers above; the duplication here is the
    # deliberate, low-risk kind (17 one-line properties) rather than the kind
    # that hides a behavior difference.
    bids_dir = property(
        lambda self: self._get_standard_field("bids_dir"),
        lambda self, value: self._set_standard_field("bids_dir", value),
    )
    output_dir = property(
        lambda self: self._get_standard_field("output_dir"),
        lambda self, value: self._set_standard_field("output_dir", value),
    )
    work_dir = property(
        lambda self: self._get_standard_field("work_dir"),
        lambda self, value: self._set_standard_field("work_dir", value),
    )
    participant_label = property(
        lambda self: self._get_standard_field("participant_label"),
        lambda self, value: self._set_standard_field("participant_label", value),
    )
    session_label = property(
        lambda self: self._get_standard_field("session_label"),
        lambda self, value: self._set_standard_field("session_label", value),
    )
    subjects_file = property(
        lambda self: self._get_standard_field("subjects_file"),
        lambda self, value: self._set_standard_field("subjects_file", value),
    )
    n_cpus = property(
        lambda self: self._get_standard_field("n_cpus"),
        lambda self, value: self._set_standard_field("n_cpus", value),
    )
    memory_gb = property(
        lambda self: self._get_standard_field("memory_gb"),
        lambda self, value: self._set_standard_field("memory_gb", value),
    )
    use_gpu = property(
        lambda self: self._get_standard_field("use_gpu"),
        lambda self, value: self._set_standard_field("use_gpu", value),
    )
    skip_existing = property(
        lambda self: self._get_standard_field("skip_existing"),
        lambda self, value: self._set_standard_field("skip_existing", value),
    )
    stop_on_error = property(
        lambda self: self._get_standard_field("stop_on_error"),
        lambda self, value: self._set_standard_field("stop_on_error", value),
    )
    log_level = property(
        lambda self: self._get_standard_field("log_level"),
        lambda self, value: self._set_standard_field("log_level", value),
    )
    debug = property(
        lambda self: self._get_standard_field("debug"),
        lambda self, value: self._set_standard_field("debug", value),
    )
    verbose = property(
        lambda self: self._get_standard_field("verbose"),
        lambda self, value: self._set_standard_field("verbose", value),
    )
    gpu_ids = property(
        lambda self: self._get_standard_field("gpu_ids"),
        lambda self, value: self._set_standard_field("gpu_ids", value),
    )
    anat_input = property(
        lambda self: self._get_standard_field("anat_input"),
        lambda self, value: self._set_standard_field("anat_input", value),
    )
    tracker = property(
        lambda self: self._get_standard_field("tracker"),
        lambda self, value: self._set_standard_field("tracker", value),
    )

    @property
    def config_data(self) -> Dict[str, Any]:
        """Live compatibility view of the canonical configuration store."""
        return self._data

    @config_data.setter
    def config_data(self, value: Optional[Dict[str, Any]]) -> None:
        data = dict(value or {})
        for key, default in _STANDARD_CONFIG_DEFAULTS.items():
            data.setdefault(key, default)
        object.__setattr__(self, "_data", data)
        self._normalize_store()

    def _normalize_store(self) -> None:
        for key in _PATH_CONFIG_FIELDS:
            value = self._data.get(key)
            if value is not None and not isinstance(value, Path):
                self._data[key] = Path(value)
        if self._data["work_dir"] is None and self._data["output_dir"]:
            self._data["work_dir"] = self._data["output_dir"] / "work"
    
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
        config_data = {}
        loader = ConfigLoader()

        if config_file:
            config_file = Path(config_file)
            
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            # Load file
            config_data = loader.load(config_file)
        
        # Merge overrides
        if overrides:
            config_data = loader.merge_configs(config_data, overrides)
        
        config = cls(config_data=config_data)
        
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
        if key == "config_data":
            return self._data
        return self._get_nested(self._data, key, default)
    
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
        if key == "config_data":
            self.config_data = value
        elif '.' not in key and key in _STANDARD_CONFIG_DEFAULTS:
            setattr(self, key, value)
        else:
            self._set_nested(self._data, key, value)
    
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
        # Automatically set log_level to DEBUG if debug is true
        if self.debug:
            self.log_level = 'DEBUG'

        errors = []
        # Check required fields
        if self.bids_dir is None:
            errors.append("bids_dir is required")
        elif self.bids_dir.exists() and not self.bids_dir.is_dir():
            errors.append(f"bids_dir is not a directory: {self.bids_dir}")
        
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
        result = dict(self._data)
        result.pop("tracker", None)
        for key in _PATH_CONFIG_FIELDS:
            if result.get(key) is not None:
                result[key] = str(result[key])
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
                config = yaml.load(f, Loader=UniqueKeyYamlLoader)
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
                        'mrtrix': {
                            'enabled': False,
                            'algorithm': 'iFOD2',
                            'select': 10000000,
                            'act': {'enabled': False},
                            'filtering': {'method': 'none'}
                        },
                        'tractseg': {'enabled': False},
                        'pyafq': {'enabled': False},
                        'tract_specific': {'enabled': False, 'bundles': [], 'metrics': []}
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
