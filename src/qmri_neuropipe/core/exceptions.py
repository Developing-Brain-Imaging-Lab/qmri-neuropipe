"""
Custom exception classes for qMRI neuroimaging pipelines.

This module defines a hierarchy of exception classes for different
types of errors that can occur during pipeline execution.

Exception Hierarchy:
    PipelineError (base)
    ├── ConfigurationError
    ├── ValidationError
    ├── ProcessingError
    │   ├── PreprocessingError
    │   ├── ModelingError
    │   └── QCError
    ├── IOError
    │   ├── BIDSError
    │   ├── FileReadError
    │   └── FileWriteError
    └── ResourceError
"""


class PipelineError(Exception):
    """
    Base exception class for all pipeline errors.
    
    All pipeline-specific exceptions inherit from this class.
    This allows catching all pipeline errors with a single except clause.
    
    Attributes:
        message: Error message
        details: Optional detailed error information
    
    Example:
        >>> try:
        ...     pipeline.run()
        ... except PipelineError as e:
        ...     logger.error(f"Pipeline failed: {e}")
    """
    
    def __init__(self, message: str, details: str = None):
        """
        Initialize pipeline error.
        
        Args:
            message: Error message
            details: Optional detailed error information
        """
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation of error."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class ConfigurationError(PipelineError):
    """
    Exception raised for configuration errors.
    
    This includes:
    - Missing required configuration fields
    - Invalid configuration values
    - Configuration file parsing errors
    
    Example:
        >>> if not config.bids_dir:
        ...     raise ConfigurationError("bids_dir is required")
    """
    pass


class ValidationError(PipelineError):
    """
    Exception raised for input validation errors.
    
    This includes:
    - Missing required input files
    - Invalid input file formats
    - Inconsistent input data
    - Failed quality checks
    
    Example:
        >>> if not input_file.exists():
        ...     raise ValidationError(f"Input file not found: {input_file}")
    """
    pass


class ProcessingError(PipelineError):
    """
    Exception raised when a processing step fails.
    
    This is the base class for all processing-related errors.
    Subclasses exist for specific processing stages.
    
    Attributes:
        step_name: Name of the processing step that failed
        message: Error message
        details: Optional detailed error information
    
    Example:
        >>> try:
        ...     result = processing_step.run(input)
        ... except Exception as e:
        ...     raise ProcessingError("Step failed", details=str(e))
    """
    
    def __init__(self, message: str, step_name: str = None, details: str = None):
        """
        Initialize processing error.
        
        Args:
            message: Error message
            step_name: Name of the processing step that failed
            details: Optional detailed error information
        """
        self.step_name = step_name
        
        if step_name:
            full_message = f"{step_name}: {message}"
        else:
            full_message = message
        
        super().__init__(full_message, details)


class PreprocessingError(ProcessingError):
    """
    Exception raised during preprocessing steps.
    
    This includes errors in:
    - Denoising
    - Motion correction
    - Distortion correction
    - Brain extraction
    - Bias field correction
    
    Example:
        >>> if eddy_output is None:
        ...     raise PreprocessingError(
        ...         "Eddy correction failed",
        ...         step_name="eddy_correction"
        ...     )
    """
    pass


class ModelingError(ProcessingError):
    """
    Exception raised during modeling steps.
    
    This includes errors in:
    - Tensor fitting
    - CSD modeling
    - Tractography
    - Connectivity analysis
    
    Example:
        >>> if tensor is None:
        ...     raise ModelingError(
        ...         "Tensor fitting failed",
        ...         step_name="tensor_fitting",
        ...         details="Insufficient b-values"
        ...     )
    """
    pass


class QCError(ProcessingError):
    """
    Exception raised during quality control.
    
    This includes:
    - QC metric computation failures
    - Report generation errors
    - QC threshold violations (if configured to raise)
    
    Example:
        >>> if snr < threshold:
        ...     raise QCError(
        ...         f"SNR too low: {snr} < {threshold}",
        ...         step_name="qc_check"
        ...     )
    """
    pass


class IOError(PipelineError):
    """
    Exception raised for input/output errors.
    
    This is the base class for all I/O-related errors.
    
    Example:
        >>> if not file.exists():
        ...     raise IOError(f"File not found: {file}")
    """
    pass


class BIDSError(IOError):
    """
    Exception raised for BIDS-related errors.
    
    This includes:
    - Invalid BIDS dataset structure
    - Missing required BIDS files
    - BIDS validation failures
    - Incorrect BIDS naming
    
    Example:
        >>> if not dataset_description.exists():
        ...     raise BIDSError(
        ...         "Invalid BIDS dataset: missing dataset_description.json"
        ...     )
    """
    pass


class FileReadError(IOError):
    """
    Exception raised when a file cannot be read.
    
    This includes:
    - Missing files
    - Corrupted files
    - Unsupported file formats
    - Permission errors
    
    Example:
        >>> try:
        ...     img = nib.load(file)
        ... except Exception as e:
        ...     raise FileReadError(
        ...         f"Failed to read {file}",
        ...         details=str(e)
        ...     )
    """
    pass


class FileWriteError(IOError):
    """
    Exception raised when a file cannot be written.
    
    This includes:
    - Permission errors
    - Disk full errors
    - Invalid output paths
    
    Example:
        >>> try:
        ...     nib.save(img, output_file)
        ... except Exception as e:
        ...     raise FileWriteError(
        ...         f"Failed to write {output_file}",
        ...         details=str(e)
        ...     )
    """
    pass


class ResourceError(PipelineError):
    """
    Exception raised for resource-related errors.
    
    This includes:
    - Insufficient memory
    - Insufficient disk space
    - CPU/GPU allocation failures
    - Timeout errors
    
    Example:
        >>> if available_memory < required_memory:
        ...     raise ResourceError(
        ...         f"Insufficient memory: need {required_memory}GB, "
        ...         f"have {available_memory}GB"
        ...     )
    """
    pass


class DependencyError(PipelineError):
    """
    Exception raised when a required dependency is not available.
    
    This includes:
    - Missing software (FSL, ANTs, FreeSurfer, etc.)
    - Wrong software versions
    - Missing Python packages
    
    Example:
        >>> try:
        ...     subprocess.run(['fsl'], check=True, capture_output=True)
        ... except FileNotFoundError:
        ...     raise DependencyError(
        ...         "FSL not found. Please install FSL and add it to PATH."
        ...     )
    """
    pass


# Convenience function for error context
def error_context(message: str):
    """
    Context manager for adding context to errors.
    
    Args:
        message: Context message to add to any exceptions
    
    Example:
        >>> with error_context("Processing subject sub-01"):
        ...     process_data()
    """
    import contextlib
    
    @contextlib.contextmanager
    def _context():
        try:
            yield
        except PipelineError as e:
            # Re-raise with additional context
            raise type(e)(f"{message}: {e.message}", e.details) from e
        except Exception as e:
            # Wrap other exceptions
            raise PipelineError(f"{message}: {str(e)}") from e
    
    return _context()


# Module version
__version__ = '2.0.0'