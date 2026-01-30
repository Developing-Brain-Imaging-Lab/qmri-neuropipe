"""
Core abstract base classes for qMRI neuroimaging pipelines.

This module defines the fundamental building blocks that all pipelines,
workflows, and processing steps inherit from. It provides:
- Standard interfaces through abstract base classes
- Common functionality (logging, validation, provenance)
- Error handling patterns
- Execution framework

Classes:
    BaseProcessingStep: Base class for all processing steps
    BaseWorkflow: Base class for multi-step workflows
    BasePipeline: Base class for complete pipelines
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union, Type, Tuple, Dict
import logging
from datetime import datetime
from dataclasses import dataclass, field
import concurrent.futures
import os

from qmri_neuropipe.core.config import PipelineConfig
from qmri_neuropipe.core.provenance import ProvenanceTracker
from qmri_neuropipe.core.exceptions import (
    ValidationError,
    ProcessingError,
    PipelineError
)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    
try:
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
except ImportError:
    Progress = None

from qmri_neuropipe.io import DataLoader

def _worker_wrapper(pipeline_cls, config_dict, subject, session):
    """
    Worker function for parallel execution.
    Must be top-level for pickle compatibility.
    """
    try:
        # Reconstruct config from dict
        config = PipelineConfig.from_dict(config_dict)
        # Instantiate pipeline
        pipeline = pipeline_cls(config)
        # Process
        pipeline.process_subject(subject, session)
        return (subject, session, True, None)
    except Exception as e:
        return (subject, session, False, str(e))

class BaseProcessingStep(ABC):
    """
    Abstract base class for all processing steps.
    
    All processing steps (both common and modality-specific) inherit from this.
    Provides standard interface for validation, execution, and provenance tracking.
    
    The __call__ method wraps run() with automatic validation, timing, and
    provenance logging, so subclasses only need to implement run().
    
    Attributes:
        config: Pipeline configuration
        logger: Logger instance for this step
        provenance: Optional provenance tracker
        start_time: Timestamp when step started
        end_time: Timestamp when step completed
    
    Example:
        >>> class MyStep(BaseProcessingStep):
        ...     def run(self, input_file: Path, output_dir: Path) -> Path:
        ...         # Process input_file
        ...         return output_file
        ...     
        ...     def validate_inputs(self, input_file: Path, **kwargs):
        ...         if not input_file.exists():
        ...             raise ValidationError(f"Input not found: {input_file}")
        >>> 
        >>> step = MyStep(config)
        >>> result = step(input_file, output_dir)  # Automatic validation & logging
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        logger: Optional[logging.Logger] = None,
        provenance: Optional[ProvenanceTracker] = None
    ):
        """
        Initialize processing step.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance (creates new if None)
            provenance: Optional provenance tracker for metadata
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.provenance = provenance
        
        # Timing information
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Step metadata
        self.step_name = self.__class__.__name__
    
    def __call__(self, *args, **kwargs) -> Any:
        """
        Execute the processing step with automatic validation and tracking.
        
        This is the main entry point that wraps run() with:
        - Input validation
        - Timing
        - Logging
        - Provenance tracking
        - Output validation
        - Error handling
        
        Users should call this method (or use the instance as a callable)
        rather than calling run() directly.
        
        Args:
            *args: Positional arguments passed to run()
            **kwargs: Keyword arguments passed to run()
        
        Returns:
            Result from run() method
        
        Raises:
            ValidationError: If input validation fails
            ProcessingError: If processing fails
        """
        self.logger.info(f"Starting {self.step_name}")
        self.start_time = datetime.now()
        
        # Extract subject/session for tracking if available
        context, _ = self.unpack_input(args[0]) if args else (None, None)
        subject = context.get('subject') if context else None
        session = context.get('session') if context else None
        study = context.get('study_name', self.config.get('study_name')) if context else self.config.get('study_name')
        
        tracker = self.config.tracker
        if tracker and subject and session:
            tracker.update_status(subject, session, self.step_name.replace("Step", "").lower(), "running", study)

        try:
            # Validate inputs before processing
            self.validate_inputs(*args, **kwargs)
            
            # Execute the actual processing
            result = self.run(*args, **kwargs)
            
            # Validate outputs after processing
            self.validate_outputs(result)
            
            # Log provenance if tracker is available
            if self.provenance:
                self._log_provenance(*args, result=result, **kwargs)
            
            # Record completion time
            self.end_time = datetime.now()
            duration_s = (self.end_time - self.start_time).total_seconds()
            
            if tracker and subject and session:
                suffix = self.step_name.replace("Step", "").lower()
                tracker.update_status(subject, session, suffix, "completed", study)
                tracker.log_time(subject, session, suffix, duration_s / 60.0, study)
                tracker.save() # Auto-save if enabled
            
            self.logger.info(
                f"{self.step_name} completed successfully in {duration_s:.1f}s"
            )
            
            return result
            
        except ValidationError as e:
            if tracker and subject and session:
                tracker.update_status(subject, session, self.step_name.replace("Step", "").lower(), "failed", study)
                tracker.log_error(subject, session, self.step_name.replace("Step", "").lower(), str(e), study)
                tracker.save()
            self.logger.error(f"{self.step_name} validation failed: {e}")
            raise
            
        except Exception as e:
            if tracker and subject and session:
                tracker.update_status(subject, session, self.step_name.replace("Step", "").lower(), "failed", study)
                tracker.log_error(subject, session, self.step_name.replace("Step", "").lower(), str(e), study)
                tracker.save()
            self.logger.error(
                f"{self.step_name} processing failed: {e}",
                exc_info=True
            )
            raise ProcessingError(
                f"Processing step {self.step_name} failed: {str(e)}"
            ) from e
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the processing step.
        
        This is the core processing method that subclasses must implement.
        It should contain the actual processing logic without validation
        or logging (which is handled by __call__).
        
        Args:
            *args: Step-specific positional arguments
            **kwargs: Step-specific keyword arguments
        
        Returns:
            Processing results (step-specific type)
        
        Note:
            This method should focus only on processing logic.
            Input/output validation should be in validate_inputs/outputs.
        """
        pass
    
    def validate_inputs(self, *args, **kwargs) -> None:
        """
        Validate inputs before processing.
        
        Override in subclass to add specific validation logic.
        Should raise ValidationError if inputs are invalid.
        
        Default implementation does nothing (no validation).
        
        Args:
            *args: Positional arguments to validate
            **kwargs: Keyword arguments to validate
        
        Raises:
            ValidationError: If inputs are invalid
        
        Example:
            >>> def validate_inputs(self, input_file: Path, **kwargs):
            ...     if not input_file.exists():
            ...         raise ValidationError(f"Input not found: {input_file}")
            ...     if input_file.suffix not in ['.nii', '.nii.gz']:
            ...         raise ValidationError("Input must be NIfTI format")
        """
        pass
    
    def validate_outputs(self, result: Any) -> None:
        """
        Validate outputs after processing.
        
        Override in subclass to add specific validation logic.
        Should raise ProcessingError if outputs are invalid.
        
        Default implementation does nothing (no validation).
        
        Args:
            result: Output from run() method to validate
        
        Raises:
            ProcessingError: If outputs are invalid
        
        Example:
            >>> def validate_outputs(self, result: Path):
            ...     if not result.exists():
            ...         raise ProcessingError(f"Output not created: {result}")
            ...     if result.stat().st_size == 0:
            ...         raise ProcessingError("Output file is empty")
        """
        pass
    
    @staticmethod
    def check_output_validity(path: Path, min_size: int = 100) -> bool:
        """
        Check if an output file is valid (exists and has non-zero size).
        
        Args:
            path: Path to check
            min_size: Minimum size in bytes (default 100 bytes for header)
            
        Returns:
            True if valid, False otherwise
        """
        if not path.exists(): return False
        try:
            if path.stat().st_size < min_size: return False
        except OSError:
            return False
            
        return True
    
    def _log_provenance(self, *args, result: Any, **kwargs) -> None:
        """
        Log provenance information for this step.
        
        Override in subclass to log detailed provenance.
        Default implementation logs basic information.
        
        Args:
            *args: Input arguments to log
            result: Output result to log
            **kwargs: Keyword arguments to log
        """
        if self.provenance:
            self.provenance.log_step(
                step_name=self.step_name,
                inputs={f'arg_{i}': str(arg) for i, arg in enumerate(args)},
                outputs={'result': str(result)},
                parameters=kwargs,
                duration=(self.end_time - self.start_time).total_seconds()
            )

    def get_step_output_dir(self, output_dir: Path) -> Path:
        """
        Get and create the specific output directory for this step.
        
        Args:
            output_dir: Base output directory.
            
        Returns:
            Path to the step-specific output directory.
        """
        # Default strategy: use step name suffix (e.g. DenoisingStep -> denoise)
        # But some steps might want "denoising"?
        # Current pattern seems to be "denoise", "gibbs", "bias", "eddy"
        # Most are StepName - "Step" -> lower.
        suffix = self.step_name.replace("Step", "").lower()
        if suffix == "biascorrection": suffix = "bias"
        if suffix == "gibbsunringing": suffix = "gibbs"
        if suffix == "eddycorrection": suffix = "eddy"
        if suffix == "coregistration": suffix = "registration"
        
        path = output_dir / suffix
        path.mkdir(parents=True, exist_ok=True)
        return path

    def unpack_input(self, first_arg: Any) -> Tuple[Optional[Dict], Any]:
        """
        Unpack the first argument to separate context from direct input.
        
        Args:
            first_arg: The first positional argument passed to run().
            
        Returns:
            Tuple containing:
            - context (dict | None): Copy of the context dictionary if input was a dict, else None.
            - input_data (Any): The 'current_image' from context, or the first_arg itself.
        """
        if isinstance(first_arg, dict):
            context = dict(first_arg)
            input_data = context.get("current_image")
            return context, input_data
        return None, first_arg

    def _extract_path(self, item: Any) -> Path:
        """
        Helper to extract Path from input item (ImageFile, DWIFile, Path, string).
        """
        if hasattr(item, 'img'):
            return Path(item.img)
        if hasattr(item, 'path'):
            return Path(item.path)
        return Path(item)

class BaseWorkflow(ABC):
    """
    Abstract base class for processing workflows.
    
    Workflows orchestrate multiple processing steps into a coherent processing
    pipeline. Both modality-specific preprocessing workflows and general
    workflows inherit from this.
    
    Subclasses should:
    1. Implement _initialize_steps() to set up processing steps
    2. Implement run() to define workflow execution
    
    Attributes:
        config: Pipeline configuration
        logger: Logger instance for this workflow
        provenance: Provenance tracker for metadata
        steps: List of processing steps in workflow
        workflow_name: Name of the workflow
    
    Example:
        >>> class PreprocessingWorkflow(BaseWorkflow):
        ...     def _initialize_steps(self):
        ...         self.steps = [
        ...             DenoisingStep(self.config),
        ...             BrainExtractionStep(self.config)
        ...         ]
        ...     
        ...     def run(self, input_file: Path, output_dir: Path):
        ...         current = input_file
        ...         for step in self.steps:
        ...             current = step(current, output_dir)
        ...         return current
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        logger: Optional[logging.Logger] = None,
        provenance: Optional[ProvenanceTracker] = None
    ):
        """
        Initialize workflow.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
            provenance: Optional provenance tracker
        """
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.provenance = provenance
        
        self.workflow_name = self.__class__.__name__
        self.steps: List[BaseProcessingStep] = []
        
        # Initialize steps (implemented by subclass)
        self._initialize_steps()
        
        self.logger.info(
            f"Initialized {self.workflow_name} with {len(self.steps)} steps"
        )
    

    def add_step(self, step: BaseProcessingStep, position: Optional[int] = None) -> None:
        """
        Add a processing step to the workflow.
        
        Args:
            step: Instance of BaseProcessingStep (or subclass)
            position: Optional index at which to insert the step.
                      If None, appends to the end.
        """
        if not isinstance(step, BaseProcessingStep):
            raise TypeError(
                f"step must be a BaseProcessingStep, got {type(step)}"
            )
        
        if position is None:
            self.steps.append(step)
            self.logger.info(
                f"Added step {step.__class__.__name__} at end of {self.workflow_name}"
            )
        else:
            if position < 0 or position > len(self.steps):
                raise IndexError(
                    f"position {position} out of range for workflow "
                    f"with {len(self.steps)} steps"
                )
            self.steps.insert(position, step)
            self.logger.info(
                f"Inserted step {step.__class__.__name__} at position "
                f"{position} in {self.workflow_name}"
            )

    def insert_step_before(self, target: Union[Type[BaseProcessingStep], str], step: BaseProcessingStep) -> None:
        """
        Insert a step before a target step (by class or name).
        
        Args:
            target: Step class or step class name to insert before
            step: Step instance to insert
        """
        idx = self._find_step_index(target)
        self.add_step(step, position=idx)

    def insert_step_after(self, target: Union[Type[BaseProcessingStep], str], step: BaseProcessingStep) -> None:
        """
        Insert a step after a target step (by class or name).
        
        Args:
            target: Step class or step class name to insert after
            step: Step instance to insert
        """
        idx = self._find_step_index(target)
        self.add_step(step, position=idx + 1)

    def remove_step(self, target: Union[Type[BaseProcessingStep], str]) -> None:
        """
        Remove a step from the workflow (by class or name).
        """
        idx = self._find_step_index(target)
        removed = self.steps.pop(idx)
        self.logger.info(
            f"Removed step {removed.__class__.__name__} "
            f"from {self.workflow_name}"
        )

    def _find_step_index(self, target: Union[Type[BaseProcessingStep], str]) -> int:
        """
        Find index of a step by class or name.
        
        Args:
            target: Step class (e.g. DenoisingStep) or class name ('DenoisingStep')
        
        Returns:
            Index of the step in self.steps
        
        Raises:
            ValueError if no matching step is found.
        """
        if isinstance(target, str):
            for i, step in enumerate(self.steps):
                if step.__class__.__name__ == target:
                    return i
        else:  # assume it's a class
            for i, step in enumerate(self.steps):
                if isinstance(step, target):
                    return i
        
        raise ValueError(
            f"Step {target} not found in workflow {self.workflow_name}"
        )

    @abstractmethod
    def _initialize_steps(self) -> None:
        """
        Initialize the processing steps for this workflow.
        
        Subclasses should create processing step instances and append them
        to self.steps in the desired execution order.
        
        Example:
            >>> def _initialize_steps(self):
            ...     self.steps = [
            ...         DenoisingStep(self.config, self.logger, self.provenance),
            ...         BrainExtractionStep(self.config, self.logger, self.provenance),
            ...         BiasFieldCorrectionStep(self.config, self.logger, self.provenance)
            ...     ]
        """
        pass
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """
        Execute the complete workflow.
        
        Subclasses must implement this to define how steps are executed.
        Can use execute_steps() helper for simple sequential execution,
        or implement custom logic for complex workflows.
        
        Args:
            *args: Workflow-specific input arguments
            **kwargs: Workflow-specific keyword arguments
        
        Returns:
            Workflow results (workflow-specific type)
        """
        pass
    
    def execute_steps(self, *args, **kwargs) -> Any:
        """
        Execute all steps in sequence.
        
        This is a convenience method that workflows can use for simple
        sequential execution. For more complex workflows (parallel processing,
        conditional steps, etc.), implement custom logic in run().
        
        Args:
            *args: Initial input arguments
            **kwargs: Keyword arguments passed to all steps
        
        Returns:
            Result from final step
        
        Raises:
            ProcessingError: If any step fails
        """
        if not self.steps:
            raise PipelineError(f"{self.workflow_name} has no steps defined")
        
        self.logger.info(f"Executing {len(self.steps)} steps in {self.workflow_name}")
        
        result = args
        
        # Use rich progress if available, else fallback
        if Progress:
             from .ui import console
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True, # Clear bar after completion
                console=console # Use shared console to align with logging
             ) as progress:
                 
                 task_id = progress.add_task(f"[cyan]Running {self.workflow_name}...", total=len(self.steps))
                 
                 for i, step in enumerate(self.steps, 1):
                    step_name = step.__class__.__name__
                    progress.update(task_id, description=f"[cyan]Step {i}/{len(self.steps)}: {step_name}")
                    
                    self.logger.info(f"Step {i}/{len(self.steps)}: {step_name}")
                    
                    try:
                        # Execute step (result becomes input for next step)
                        if isinstance(result, tuple):
                            result = step(*result, **kwargs)
                        else:
                            result = step(result, **kwargs)
                        
                        progress.advance(task_id)
                            
                    except Exception as e:
                        self.logger.error(
                            f"{self.workflow_name} failed at step {i} ({step_name}): {e}"
                        )
                        raise ProcessingError(
                            f"Workflow failed at step {step_name}"
                        ) from e
        else:
            # Fallback for no rich
            for i, step in enumerate(self.steps, 1):
                step_name = step.__class__.__name__
                self.logger.info(f"Step {i}/{len(self.steps)}: {step_name}")
                
                try:
                    # Execute step (result becomes input for next step)
                    if isinstance(result, tuple):
                        result = step(*result, **kwargs)
                    else:
                        result = step(result, **kwargs)
                        
                except Exception as e:
                    self.logger.error(
                        f"{self.workflow_name} failed at step {i} ({step_name}): {e}"
                    )
                    raise ProcessingError(
                        f"Workflow failed at step {step_name}"
                    ) from e
        
        self.logger.info(f"{self.workflow_name} completed successfully")
        return result

    def get_progress_bar(self, iterable=None, total=None, desc=None):
        """Get a progress bar (tqdm) if available, otherwise return iterable/range."""
        if tqdm:
            return tqdm(iterable, total=total, desc=desc, leave=False)
        return iterable if iterable is not None else range(total)

class BasePipeline(ABC):
    """
    Abstract base class for complete processing pipelines.
    
    All modality-specific pipelines (DMRIPipeline, FMRIPipeline, etc.)
    inherit from this. Provides common functionality for:
    - Pipeline execution and subject iteration
    - Logging setup
    - Provenance tracking
    - Output management
    - Error handling and recovery
    
    Subclasses must implement:
    - name property: Pipeline name
    - version property: Pipeline version
    - _initialize_pipeline(): Set up workflows and steps
    - process_subject(): Process a single subject
    
    Attributes:
        config: Pipeline configuration
        logger: Logger instance for this pipeline
        provenance: Provenance tracker for metadata
    
    Example:
        >>> class DMRIPipeline(BasePipeline):
        ...     @property
        ...     def name(self) -> str:
        ...         return 'dmri-pipeline'
        ...     
        ...     @property
        ...     def version(self) -> str:
        ...         return '2.0.0'
        ...     
        ...     def _initialize_pipeline(self):
        ...         self.preprocessing = PreprocessingWorkflow(self.config)
        ...     
        ...     def process_subject(self, subject: str, session: Optional[str]):
        ...         # Process subject
        ...         pass
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Set up logging infrastructure
        self.logger = self._setup_logging()
        
        # Set up provenance tracking
        self.provenance = ProvenanceTracker(
            output_dir=Path(config.get('output_dir')),
            pipeline_name=self.name,
            pipeline_version=self.version
        )
        
        # Initialize tracker if configured
        tracker_file = config.get('tracker_file')
        if not tracker_file and config.output_dir:
            default_tracker = Path(config.output_dir) / "study_tracker.xlsx"
            if default_tracker.exists():
                tracker_file = default_tracker
        
        if tracker_file:
            from qmri_neuropipe.lib.common.tracker import NeuroimagingTracker
            self.config.tracker = NeuroimagingTracker(Path(tracker_file), logger=self.logger)

        # Initialize workflows and steps (implemented by subclass)
        self._initialize_pipeline()
        
        self.logger.info(
            f"Initialized {self.name} v{self.version} "
            f"(output: {config.get('output_dir')})"
        )
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Pipeline name.
        
        Returns:
            Pipeline name (e.g., 'dmri-pipeline')
        """
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """
        Pipeline version.
        
        Returns:
            Version string (e.g., '2.0.0')
        """
        pass
    
    @abstractmethod
    def _initialize_pipeline(self) -> None:
        """
        Initialize workflows and processing steps.
        
        Subclasses should create workflow and step instances here.
        These can be stored as instance attributes for use in process_subject().
        
        Example:
            >>> def _initialize_pipeline(self):
            ...     self.preprocessing = PreprocessingWorkflow(
            ...         self.config, self.logger, self.provenance
            ...     )
            ...     self.tensor_fitting = TensorFittingStep(
            ...         self.config, self.logger, self.provenance
            ...     )
        """
        pass
    
    @abstractmethod
    def process_subject(
        self,
        subject: str,
        session: Optional[str] = None
    ) -> None:
        """
        Process a single subject.
        
        This is the main processing method that subclasses must implement.
        It should contain the complete processing pipeline for one subject.
        
        Args:
            subject: Subject ID (e.g., 'sub-01')
            session: Optional session ID (e.g., 'ses-01')
        
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    def run(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        pairs: Optional[List[Tuple[str, Optional[str]]]] = None
    ) -> None:
        """
        Run pipeline on specified subjects.
        
        This method handles:
        - Subject iteration
        - Session handling
        - Error handling and recovery
        - Progress tracking
        - Summary reporting
        
        Subclasses typically don't need to override this method.
        
        Args:
            subjects: List of subject IDs (None = all subjects in BIDS dataset)
            sessions: List of session IDs (None = all sessions)
            pairs: Optional list of explicit (subject, session) pairs. 
                  If provided, subjects and sessions args are ignored.
        """
        self.logger.info(f"Starting {self.name} v{self.version}")
        
        # Get data to process
        loader = DataLoader(self.config.get('bids_dir'))
        
        if pairs is not None:
            self.logger.info(f"Processing {len(pairs)} specified subject/session pairs")
            data = loader.load_multiple_subjects(pairs=pairs)
        elif subjects is not None:
            # For backward compatibility and simple list-based selection
            self.logger.info(f"Processing {len(subjects)} specified subjects")
            data = loader.load_multiple_subjects(subjects=subjects, sessions=sessions)
        elif self.config.get('subjects_file'):
            self.logger.info(f"Loading subjects from file: {self.config.subjects_file}")
            data = loader.load_from_subjects_file(self.config.subjects_file)
        else:
            subjects = self._get_all_subjects()
            self.logger.info(f"Found {len(subjects)} subjects in BIDS dataset")
            data = loader.load_multiple_subjects(subjects=subjects, sessions=sessions)
        
        # Initialize counters
        n_success = 0
        n_failed = 0
        n_skipped = 0
        failed_subjects = []
        
        # Check for submit mode
        if self.config.get('submit', False):
            self._generate_htcondor_submission(data)
            return

        # Check for parallel execution
        jobs = self.config.get('jobs', 1)
        if jobs > 1:
            self._run_parallel(data, jobs)
            return

        for subject, session in data:
            try:
                # Check if already processed
                if self._should_skip(subject, session):
                    self.logger.info(
                        f"Skipping {subject} {session} "
                        "(outputs already exist)"
                    )
                    n_skipped += 1
                    continue
                
                # Process subject/session
                self.logger.info("\n" + "="*80)
                self.logger.info(f"  PROCESSING: sub-{subject}" + (f" ses-{session}" if session else ""))
                self.logger.info("="*80 + "\n")
                
                self.process_subject(subject, session)
                
                n_success += 1
                self.logger.info(
                    f"Successfully processed {subject} {session}"
                )
                
            except Exception as e:
                n_failed += 1
                failed_subjects.append(f"{subject} {session}")
                self.logger.error(
                    f"Failed to process {subject} {session}: {e}",
                    exc_info=self.config.get('debug', False)
                )
                
                # Continue with next subject unless configured to stop
                if self.config.get('stop_on_error', False):
                    self.logger.error("Stopping pipeline due to error")
                    break
        
        return {
            'n_success': n_success,
            'n_failed': n_failed,
            'n_skipped': n_skipped
        }

    def _run_parallel(self, data, jobs):
        """Execute pipeline in parallel using ProcessPoolExecutor."""
        self.logger.info(f"Running parallel execution with {jobs} jobs...")
        config_dict = self.config.to_dict()
        
        n_success = 0
        n_failed = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_worker_wrapper, self.__class__, config_dict, sub, ses): (sub, ses)
                for sub, ses in data
            }
            
            for future in concurrent.futures.as_completed(futures):
                sub, ses = futures[future]
                try:
                    res_sub, res_ses, success, error = future.result()
                    if success:
                        n_success += 1
                        self.logger.info(f"Successfully processed {res_sub} {res_ses}")
                    else:
                        n_failed += 1
                        self.logger.error(f"Failed to process {res_sub} {res_ses}: {error}")
                except Exception as e:
                    n_failed += 1
                    self.logger.error(f"Job execution failed for {sub} {ses}: {e}")
                    
        return {
            'n_success': n_success,
            'n_failed': n_failed,
            'n_skipped': 0 # Parallel runner doesn't check skip before dispatch usually, or detailed stats are lost
        }

    def _generate_htcondor_submission(self, data):
        """Generate HTCondor submit file and subjects list."""
        out_dir = self.config.output_dir
        
        # Determine submit file path
        # config.get('submit') can be:
        # - True/False (legacy bool)
        # - "DEFAULT" (flag_value from CLI)
        # - "/path/to/file.sub" (explicit path)
        submit_val = self.config.get('submit')
        submit_file = out_dir / "job.sub" # Default
        
        if isinstance(submit_val, (str, Path)):
             s_val_str = str(submit_val)
             if s_val_str and s_val_str != "DEFAULT" and s_val_str.lower() != "true":
                  submit_file = Path(s_val_str)
                  # ensure parent dir exists if custom path
                  submit_file.parent.mkdir(parents=True, exist_ok=True)

        subjects_file = out_dir / "subjects.txt"
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Save current config to be used by jobs
        config_path = out_dir / "config.yaml"
        self.config.save(config_path)
        
        # Write subjects file
        # Prepare subject list
        subjects_entries = []
        if self.config.subjects_file and self.config.subjects_file.exists():
            self.logger.info(f"Using custom subjects list from: {self.config.subjects_file}")
            with open(self.config.subjects_file, 'r') as f:
                for line in f:
                    parts = [p.strip() for p in line.strip().split(',')]
                    if parts and parts[0]:
                        # Ensure at least 2 parts (sub, ses)
                        if len(parts) == 1:
                            parts.append('')
                        subjects_entries.append(parts[:2])
        else:
            for sub, ses in data:
                subjects_entries.append([str(sub), str(ses) if ses else ''])

        # Write subjects file with optional GPU assignment
        gpu_ids = self.config.gpu_ids
        if gpu_ids is not None:
             if isinstance(gpu_ids, int):
                 gpu_ids = [gpu_ids]
             elif isinstance(gpu_ids, str):
                 try:
                     gpu_ids = [int(x.strip()) for x in gpu_ids.split(',')]
                 except ValueError:
                     self.logger.warning(f"Could not parse gpu_ids string: {gpu_ids}")
                     # Fallback to None or keep as string which might fail later? 
                     # Better to set to None or handle gracefully.
                     gpu_ids = None
        
        with open(subjects_file, "w") as f:
            for i, (sub, ses) in enumerate(subjects_entries):
                line = f"{sub},{ses}"
                if gpu_ids:
                    # Round-robin assignment
                    gpu_id = gpu_ids[i % len(gpu_ids)]
                    line += f",{gpu_id}"
                f.write(line + "\n")
        
        # Construct submit content
        req_gpus = ""
        gpu_arg = ""
        queue_line = f"queue sub,ses from {subjects_file}"
        
        if gpu_ids:
            gpu_arg = " --gpu-ids $(gpu)"
            queue_line = f"queue sub,ses,gpu from {subjects_file}"
            # Note: We do NOT set request_gpus here to avoid Condor masking devices
            # This assumes the user wants manual control over specific device IDs
        elif self.config.use_gpu:
            req_gpus = "\nrequest_gpus = 1"

        # Concurrency Limit (max_materialize)
        # Use 'jobs' parameter (CLI -j or config 'jobs')
        # Default is 1 (from CLI), which we treat as "unlimited" for submission 
        # (to avoid accidentally serializing cluster jobs).
        # Only apply limit if explicitly > 1.
        concurrency_directive = ""
        n_jobs = getattr(self.config, 'jobs', 1)
        if n_jobs > 1:
             concurrency_directive = f"\nmax_materialize = {n_jobs}"

        content = f"""
# HTCondor Submit File generated by qmri-neuropipe
executable = qmri-neuropipe
universe = vanilla

# Arguments
arguments = --config {config_path} --participant-label $(sub) --session-label $(ses){gpu_arg}

# Logging
output = {logs_dir}/job.$(Cluster).$(Process).out
error = {logs_dir}/job.$(Cluster).$(Process).err
log = {logs_dir}/job.$(Cluster).$(Process).log

# Resources
request_cpus = {self.config.n_cpus}
request_memory = {int(self.config.memory_gb)}GB{req_gpus}
getenv = True{concurrency_directive}

# Queue
{queue_line}
"""
        with open(submit_file, "w") as f:
            f.write(content.strip() + "\n")
            
        self.logger.info(f"Generated HTCondor submit file: {submit_file}")
        self.logger.info(f"To submit, run: condor_submit {submit_file}")
                

        

        

    
    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging infrastructure for the pipeline.
        
        Creates:
        - Log directory
        - File handler (detailed logs)
        - Console handler (user-friendly output)
        
        Returns:
            Configured logger instance
        """
        # Create log directory
        log_dir = Path(self.config.get('output_dir')) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)  # Capture everything
        
        # Remove existing handlers (avoid duplicates)
        logger.handlers.clear()
        
        # File handler (detailed logging)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{self.name}_{timestamp}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Create file formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(file_formatter)
        
        # Get log level from config
        log_level = self.config.get('log_level', 'INFO')
        if self.config.get('debug', False):
            log_level = 'DEBUG'
        
        # Console Handler
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        try:
            from rich.logging import RichHandler
            from .ui import console
            # Use RichHandler with cleaner output for alignment
            # omit_repeated_times=False keeps the rhythm consistent
            # show_path=False avoids alignment jumping due to filename length
            ch = RichHandler(
                rich_tracebacks=True, 
                markup=True, 
                console=console,
                show_path=False,
                omit_repeated_times=False,
                show_level=True,
                show_time=True,
                log_time_format="[%X]" # Consistent time format
            )
        except ImportError:
            ch = logging.StreamHandler()
            ch.setFormatter(console_formatter)
        
        ch.setLevel(getattr(logging, log_level))
        
        # Add handlers to pipeline logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # ALSO configure the 'qmri-neuropipe' logger used by utilities (run.py)
        # to ensure command logging is captured
        lib_logger = logging.getLogger("qmri-neuropipe")
        lib_logger.setLevel(logging.DEBUG) # Always capture, handler filters
        lib_logger.handlers.clear()
        lib_logger.addHandler(fh)
        lib_logger.addHandler(ch)
        
        logger.info(f"Logging to {log_file}")
        
        return logger
    
    def _get_all_subjects(self) -> List[str]:
        """
        Get all subjects from BIDS dataset.
        
        Returns:
            List of subject IDs
        """
        from qmri_neuropipe.io.bids import select_participants_sessions
        # Get all pairs (subject, session)
        pairs = select_participants_sessions(Path(self.config.get('bids_dir')), None, None)
        # Extract unique subjects
        subjects = sorted(list(set(p[0] for p in pairs)))
        return subjects
    
    # def _get_subject_sessions(self, subject: str) -> List[Optional[str]]:
    #     """
    #     Get all sessions for a subject.
        
    #     Args:
    #         subject: Subject ID
        
    #     Returns:
    #         List of session IDs (or [None] for single-session datasets)
    #     """
    #     from bids import BIDSLayout
        
    #     bids = BIDSLayout(self.config.get('bids_dir'))
    #     sessions = bids.get_sessions(subject)
        
    #     # Return [None] for single-session datasets
    #     return sessions if sessions else [None]
    
    def _should_skip(self, subject: str, session: Optional[str]) -> bool:
        """
        Check if subject/session should be skipped.
        
        Skips if skip_existing is enabled and outputs already exist.
        
        Args:
            subject: Subject ID
            session: Session ID
        
        Returns:
            True if should skip, False otherwise
        """
        if not self.config.get('skip_existing', False):
            return False
        
        # Check if outputs exist (subclass can override)
        output_dir = self._get_output_dir(subject, session)
        
        if not output_dir.exists():
            return False
        
        # Check if directory has any files (not just empty)
        return any(output_dir.rglob('*.*'))
    
    def _get_output_dir(
        self,
        subject: str,
        session: Optional[str] = None
    ) -> Path:
        """
        Get output directory for subject/session.
        
        Args:
            subject: Subject ID
            session: Optional session ID
        
        Returns:
            Path to output directory
        """
        output_root = Path(self.config.get('output_dir'))
        
        if session:
            return output_root / f'sub-{subject}' / f'ses-{session}'
        else:
            return output_root / f'sub-{subject}'


# Module version
__version__ = '2.0.0'