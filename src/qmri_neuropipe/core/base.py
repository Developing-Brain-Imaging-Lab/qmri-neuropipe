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
            duration = (self.end_time - self.start_time).total_seconds()
            self.logger.info(
                f"{self.step_name} completed successfully in {duration:.1f}s"
            )
            
            return result
            
        except ValidationError as e:
            self.logger.error(f"{self.step_name} validation failed: {e}")
            raise
            
        except Exception as e:
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
             with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                transient=True # Clear bar after completion
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
        sessions: Optional[List[str]] = None
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
        """
        self.logger.info(f"Starting {self.name} v{self.version}")
        
        # Get subjects to process
        if subjects is None:
            subjects = self._get_all_subjects()
            self.logger.info(f"Found {len(subjects)} subjects in BIDS dataset")
        else:
            self.logger.info(f"Processing {len(subjects)} specified subjects")
        
        # Initialize counters
        n_success = 0
        n_failed = 0
        n_skipped = 0
        failed_subjects = []

        loader = DataLoader(self.config.get('bids_dir'))

        data = loader.load_multiple_subjects(subjects=subjects,
                                             sessions=sessions)
        
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
                self.logger.info(f"Processing {subject} {session}")
                self.process_subject(subject, session)
                
                n_success += 1
                self.logger.info(
                    f"Successfully processed {subject} {session}"
                )
                
            except Exception as e:
                n_failed += 1
                failed_subjects.append(f"{subject} {session}")
                self.logger.error(
                    f"Failed to process {subject} {sessions}: {e}",
                    exc_info=self.config.get('debug', False)
                )
                
                # Continue with next subject unless configured to stop
                if self.config.get('stop_on_error', False):
                    self.logger.error("Stopping pipeline due to error")
                    break
                
        #         n_success += 1
        #         self.logger.info(
        #             f"Successfully processed {subject}{sessions}"
        #         )
                
        #     except Exception as e:
        #         n_failed += 1
        #         failed_subjects.append(f"{subject}{sessions}")
        #         self.logger.error(
        #             f"Failed to process {subject}{sessions}: {e}",
        #             exc_info=self.config.get('debug', False)
        #         )
                
        #         # Continue with next subject unless configured to stop
        #         if self.config.get('stop_on_error', False):
        #             self.logger.error("Stopping pipeline due to error")
        #             break
        
        # Generate summary
        total = n_success + n_failed + n_skipped
        self.logger.info("=" * 60)
        self.logger.info(f"Pipeline execution summary:")
        self.logger.info(f"  Total subjects/sessions: {total}")
        self.logger.info(f"  Successfully processed: {n_success}")
        self.logger.info(f"  Failed: {n_failed}")
        self.logger.info(f"  Skipped (existing): {n_skipped}")
        
        if failed_subjects:
            self.logger.warning(f"Failed subjects: {', '.join(failed_subjects)}")
        
        self.logger.info("=" * 60)
        
        # Save final provenance
        self.provenance.save()
        
        # Exit with error code if any failures
        if n_failed > 0:
            raise PipelineError(
                f"Pipeline completed with {n_failed} failed subjects/sessions"
            )
    
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
        
        # Console Handler
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        try:
            from rich.logging import RichHandler
            # Use RichHandler unless it confuses DEBUG output (optional preference)
            ch = RichHandler(rich_tracebacks=True, markup=True)
        except ImportError:
            ch = logging.StreamHandler()
            ch.setFormatter(console_formatter)
        
        ch.setLevel(getattr(logging, log_level))
        
        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        logger.info(f"Logging to {log_file}")
        
        return logger
    
    # def _get_all_subjects(self) -> List[str]:
    #     """
    #     Get all subjects from BIDS dataset.
        
    #     Returns:
    #         List of subject IDs
    #     """
    #     from qmri_neuropipe.io.bids import BIDSLayout
        
    #     bids = BIDSLayout(self.config.get('bids_dir'))
    #     return bids.get_subjects()
    
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