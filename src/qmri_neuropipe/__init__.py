__version__='0.0.1'


def _append_pythonwarnings_filter(filter_spec: str) -> None:
    import os

    current = os.environ.get("PYTHONWARNINGS", "")
    filters = [item for item in current.split(",") if item]
    if filter_spec not in filters:
        filters.append(filter_spec)
        os.environ["PYTHONWARNINGS"] = ",".join(filters)


def configure_runtime_warnings() -> None:
    """Suppress noisy third-party/runtime warnings that obscure pipeline progress."""
    import warnings

    # These resource_tracker messages are emitted by multiprocessing/loky helper
    # processes at interpreter shutdown in some Python 3.12 environments.
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")
    warnings.filterwarnings("ignore", message=".*resource_tracker: There appear to be.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*resource_tracker: '/loky-.*", category=UserWarning)
    _append_pythonwarnings_filter("ignore:resource_tracker:UserWarning")

    # Common numerical warnings from model fitting libraries.
    warnings.filterwarnings("ignore", message=".*Solution may be inaccurate.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*overflow encountered in exp.*", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*invalid value encountered in log.*", category=RuntimeWarning)


configure_runtime_warnings()
