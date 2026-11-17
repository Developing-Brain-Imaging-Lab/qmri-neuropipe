import numpy as np
import pytest

from qmri_neuropipe.interfaces.dmipy_jax_initialization import (
    batch_dmipy_brute_search,
    initialize_in_batches,
)


def test_grid_search_reuses_chunks_and_retries_only_unfinished_voxels():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    data = np.arange(11, dtype=np.float32)[:, None] * 2
    grid = jnp.asarray([0, 5, 10, 15, 20], dtype=jnp.float32)

    @jax.jit
    def search(batch):
        return jnp.argmin((batch - grid[None, :]) ** 2, axis=1)

    calls = []

    def memory_limited_search(batch):
        first_voxel = int(batch[0, 0]) // 2
        calls.append((first_voxel, len(batch)))
        if first_voxel == 4 and len(batch) == 4:
            raise RuntimeError("RESOURCE_EXHAUSTED: test GPU allocation failed")
        return search(batch)

    with pytest.warns(RuntimeWarning, match="retrying at 2"):
        result = initialize_in_batches(memory_limited_search, data, batch_size=4)

    np.testing.assert_array_equal(result, search(data))
    assert calls == [(0, 4), (4, 4), (4, 2), (6, 2), (8, 2), (10, 2)]


def test_dmipy_builder_batches_only_inside_qmri_fit(monkeypatch):
    jnp = pytest.importorskip("jax.numpy")
    brute_jax = pytest.importorskip("dmipy_fit.jax.brute_jax")
    calls = []

    def search(data):
        calls.append(len(data))
        return data

    monkeypatch.setattr(brute_jax, "build_jax_brute_fn", lambda: search)
    data = jnp.arange(5)
    with batch_dmipy_brute_search(2):
        result = brute_jax.build_jax_brute_fn()(data)
    np.testing.assert_array_equal(result, data)
    assert calls == [2, 2, 2]
    assert brute_jax.build_jax_brute_fn() is search
