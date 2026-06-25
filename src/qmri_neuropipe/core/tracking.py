"""Shared tracker status updates and explicit persistence boundaries."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from qmri_neuropipe.core.config import PipelineConfig


def mark_tracker_dirty(config: PipelineConfig) -> bool:
    """Mark a configured tracker as having pending in-memory changes."""
    if config.tracker is None:
        return False
    setattr(config, "_tracker_dirty", True)
    return True


def update_step_status(
    config: PipelineConfig,
    context: Optional[Mapping[str, Any]],
    step: Any,
    status: str,
    *,
    modality: Optional[str] = None,
) -> bool:
    """Update one tracker status row without writing the workbook."""
    tracker = config.tracker
    if tracker is None or not context:
        return False

    subject = context.get("subject")
    session = context.get("session")
    if not subject or not session:
        return False

    if isinstance(step, str):
        tracker_module = step
    else:
        step_name = step.__class__.__name__
        normalize = getattr(step, "normalize_tracker_module", None)
        tracker_module = normalize(step_name) if normalize else step_name

    study = context.get("study_name", config.get("study_name"))
    tracker.update_status(
        subject,
        session,
        tracker_module,
        status,
        study,
        modality=modality or getattr(step, "modality", None),
    )
    mark_tracker_dirty(config)
    return True


def flush_tracker(config: PipelineConfig, *, force: bool = False) -> bool:
    """Persist pending tracker changes at an explicit lifecycle boundary."""
    tracker = config.tracker
    if tracker is None or not getattr(config, "_tracker_dirty", False):
        return False
    tracker.save(force=force)
    setattr(config, "_tracker_dirty", False)
    return True
