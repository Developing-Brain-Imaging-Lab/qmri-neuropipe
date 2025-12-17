"""
qMRI Neuropipe Core Module

This module provides the fundamental building blocks for neuroimaging pipelines:
- Abstract base classes (BasePipeline, BaseWorkflow, BaseProcessingStep)
- Configuration management (PipelineConfig)
- Provenance tracking (ProvenanceTracker)
- Exception hierarchy (PipelineError and subclasses)

Public API:
    Base classes:
        - BasePipeline: Base class for complete pipelines
        - BaseWorkflow: Base class for multi-step workflows
        - BaseProcessingStep: Base class for individual processing steps
    
    Configuration:
        - PipelineConfig: Configuration container
        - ConfigLoader: Configuration loading utilities
        - create_default_config: Create default configurations
    
    Provenance:
        - ProvenanceTracker: Provenance tracking and metadata
    
    Exceptions:
        - PipelineError: Base exception
        - ConfigurationError: Configuration errors
        - ValidationError: Input validation errors
        - ProcessingError: Processing failures
        - IOError: I/O errors
        - BIDSError: BIDS-specific errors
        - ResourceError: Resource allocation errors
        - DependencyError: Missing dependencies

Example:
    >>> from qmri_neuropipe.core import (
    ...     BasePipeline, 
    ...     PipelineConfig, 
    ...     ProvenanceTracker
    ... )
    >>> 
    >>> # Create configuration
    >>> config = PipelineConfig.from_file('config.yaml')
    >>> 
    >>> # Create custom pipeline
    >>> class MyPipeline(BasePipeline):
    ...     @property
    ...     def name(self):
    ...         return 'my-pipeline'
    ...     # ... implement abstract methods
    >>> 
    >>> pipeline = MyPipeline(config)
    >>> pipeline.run()
"""

# Import base classes
from qmri_neuropipe.core.base import (
    BaseProcessingStep,
    BaseWorkflow,
    BasePipeline
)

# Import configuration
from qmri_neuropipe.core.config import (
    PipelineConfig,
    ConfigLoader,
    create_default_config
)

# Import provenance
from qmri_neuropipe.core.provenance import ProvenanceTracker

# Import exceptions
from qmri_neuropipe.core.exceptions import (
    PipelineError,
    ConfigurationError,
    ValidationError,
    ProcessingError,
    PreprocessingError,
    ModelingError,
    QCError,
    IOError,
    BIDSError,
    FileReadError,
    FileWriteError,
    ResourceError,
    DependencyError,
    error_context
)

from qmri_neuropipe.core.utils import (
    ensure_path,
    ensure_dir,
    extract_image_path
)


# Define public API
__all__ = [
    # Base classes
    'BaseProcessingStep',
    'BaseWorkflow',
    'BasePipeline',
    
    # Configuration
    'PipelineConfig',
    'ConfigLoader',
    'create_default_config',
    
    # Provenance
    'ProvenanceTracker',
    
    # Exceptions
    'PipelineError',
    'ConfigurationError',
    'ValidationError',
    'ProcessingError',
    'PreprocessingError',
    'ModelingError',
    'QCError',
    'IOError',
    'BIDSError',
    'FileReadError',
    'FileWriteError',
    'ResourceError',
    'DependencyError',
    'DependencyError',
    'error_context',
    
    # Utils
    'ensure_path',
    'ensure_dir',
    'extract_image_path',
]


# Module metadata
__version__ = '2.0.0'
__author__ = 'Developing Brain Imaging Lab'
__license__ = 'MIT'