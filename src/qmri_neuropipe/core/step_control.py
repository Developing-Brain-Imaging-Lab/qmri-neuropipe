"""Helpers for controlling step-level reruns across workflows."""

from __future__ import annotations

from typing import Any, Iterable, Optional


_ALIASES = {
    "eddy": {"eddy", "eddycorrection", "eddycurrentcorrection"},
    "eddycorrection": {"eddy", "eddycorrection", "eddycurrentcorrection"},
    "denoise": {"denoise", "denoising"},
    "denoising": {"denoise", "denoising"},
    "gibbs": {"gibbs", "degibbs", "gibbsunringing"},
    "degibbs": {"gibbs", "degibbs", "gibbsunringing"},
    "merge": {"merge", "merging"},
    "topup": {"topup", "distcorr", "distortioncorrection"},
    "distcorr": {"topup", "distcorr", "distortioncorrection"},
    "bias": {"bias", "biascorrection"},
    "biascorrection": {"bias", "biascorrection"},
    "coreg": {"coreg", "coregistration", "registration"},
    "coregistration": {"coreg", "coregistration", "registration"},
    "brainmask": {"brainmask", "brainmasking", "mask", "masking", "skullstrip"},
    "brainmasking": {"brainmask", "brainmasking", "mask", "masking", "skullstrip"},
    "reorient": {"reorient", "reorientation"},
    "resample": {"resample", "resampling"},
    "motion": {"motion", "motioncorrection", "moco"},
    "motioncorrection": {"motion", "motioncorrection", "moco"},
    "b1": {"b1", "b1mapping"},
    "b1mapping": {"b1", "b1mapping"},
    "normalization": {"normalization", "normalize", "nonlinearregistration"},
    "normalize": {"normalization", "normalize", "nonlinearregistration"},
    "segmentation": {"segmentation", "segment"},
    "modeling": {"modeling", "modelfitting", "fitting"},
    "freesurfer": {"freesurfer", "reconall", "freesurferstats"},
    "reconall": {"freesurfer", "reconall", "freesurferstats"},
    "supersynth": {"supersynth", "supersynthsegmentation"},
    "atlas": {"atlas", "atlasregistration"},
    "atlasregistration": {"atlas", "atlasregistration"},
    "statistics": {"statistics", "stats", "statsextraction", "analysis"},
    "stats": {"statistics", "stats", "statsextraction", "analysis"},
    "acqparams": {"acqparams", "acquisitionparams", "acquisitionparameters"},
}


def normalize_step_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    for token in ("step", "_", "-", " "):
        text = text.replace(token, "")
    return "".join(ch for ch in text if ch.isalnum())


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            return config.get(key) or default
    if isinstance(config, dict):
        current: Any = config
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current
    return default


def get_rerun_from_step(config: Any, *scopes: str) -> Optional[str]:
    keys = ("rerun_from_step", "force_from_step", "start_at_step", "resume_from_step")
    for scope in scopes:
        if not scope:
            continue
        for key in keys:
            value = _config_get(config, f"{scope}.{key}")
            if value not in (None, "", False):
                return str(value)
    for key in keys:
        value = _config_get(config, key)
        if value not in (None, "", False):
            return str(value)
    return None


def _candidate_names(step: Any) -> set[str]:
    class_name = step if isinstance(step, str) else step.__class__.__name__
    names = {
        normalize_step_name(class_name),
        normalize_step_name(str(class_name).replace("Step", "")),
    }
    normalize_tracker_module = getattr(step, "normalize_tracker_module", None)
    if callable(normalize_tracker_module):
        try:
            names.add(normalize_step_name(normalize_tracker_module(str(class_name))))
        except Exception:
            pass
    expanded = set(names)
    for name in names:
        expanded.update(_ALIASES.get(name, set()))
    return expanded


def step_matches(step: Any, target: Any) -> bool:
    target_name = normalize_step_name(target)
    if not target_name:
        return False
    step_names = _candidate_names(step)
    target_names = {target_name, *_ALIASES.get(target_name, set())}
    if step_names & target_names:
        return True
    return any(target in name or name in target for name in step_names for target in target_names)


def step_force_active(current_active: bool, step: Any, rerun_from_step: Any) -> bool:
    return current_active or step_matches(step, rerun_from_step)


def any_step_matches(steps: Iterable[Any], rerun_from_step: Any) -> bool:
    return any(step_matches(step, rerun_from_step) for step in steps)
