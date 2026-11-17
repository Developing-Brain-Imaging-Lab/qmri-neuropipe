"""Bound the voxel dimension of dmipy's exhaustive JAX grid search."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import warnings

import numpy as np


# Grid scoring allocates voxels × candidates, unlike the local optimizer.
_GRID_SEARCH_BATCH_LIMIT = 1024
_active_batch_size = ContextVar("dmipy_grid_search_batch_size", default=None)


def is_out_of_memory(exc):
    message = str(exc).lower()
    return "resource_exhausted" in message or "out of memory" in message


def initialize_in_batches(search, data, batch_size=None):
    """Reuse one signal dictionary and copy only winning parameters to the host."""
    import jax.numpy as jnp

    if len(data) == 0:
        return np.asarray(search(jnp.asarray(data)))

    if batch_size is None:
        batch_size = int(os.environ.get("DMIPY_JAX_BATCH", "8192"))
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    size = min(len(data), batch_size, _GRID_SEARCH_BATCH_LIMIT)
    print(f"JAX brute grid search batch size: {size} voxels", flush=True)

    def score(start, count, size):
        batch = data[start : start + count]
        if count < size:
            xp = np if isinstance(batch, np.ndarray) else jnp
            batch = xp.concatenate([batch, xp.repeat(batch[-1:], size - count, axis=0)])
        return np.asarray(search(jnp.asarray(batch)))[:count]

    start = 0
    result = None
    while start < len(data):
        count = min(size, len(data) - start)
        try:
            winners = score(start, count, size)
        except RuntimeError as exc:
            if not is_out_of_memory(exc) or size == 1:
                raise
            smaller = max(1, size // 2)
            warnings.warn(
                f"JAX brute grid search hit OOM at {size} voxels; retrying at {smaller}.",
                RuntimeWarning,
                stacklevel=2,
            )
            size = smaller
            continue
        if result is None:
            result = np.empty((len(data), *winners.shape[1:]), dtype=winners.dtype)
        result[start : start + count] = winners
        start += count
    return result


@contextmanager
def batch_dmipy_brute_search(batch_size):
    """Adapt dmipy-fit 2.3.0's internal builder for this fit only."""
    from dmipy_fit.jax import brute_jax

    original = brute_jax.build_jax_brute_fn
    if not getattr(original, "_qmri_batched", False):
        @wraps(original)
        def build(*args, **kwargs):
            search = original(*args, **kwargs)
            active_size = _active_batch_size.get()
            if active_size is None:
                return search
            return lambda data: initialize_in_batches(search, data, active_size)

        build._qmri_batched = True
        brute_jax.build_jax_brute_fn = build

    if batch_size is None:
        batch_size = int(os.environ.get("DMIPY_JAX_BATCH", "8192"))
    token = _active_batch_size.set(batch_size)
    try:
        yield
    finally:
        _active_batch_size.reset(token)
