from pathlib import Path

from qmri_neuropipe.core.caching import (
    all_outputs_exist,
    force_requested,
    reuse_enabled,
)


def test_force_requested_combines_explicit_and_config_flags():
    assert force_requested({}, explicit=True)
    assert force_requested({"force": True})
    assert force_requested({"force_run": True})
    assert not force_requested({"force": False, "force_run": False})


def test_reuse_enabled_requires_skip_without_force():
    assert reuse_enabled({"skip_existing": True})
    assert not reuse_enabled({"skip_existing": False})
    assert not reuse_enabled({"skip_existing": True, "force": True})
    assert not reuse_enabled({"skip_existing": True}, explicit_force=True)


def test_all_outputs_exist_requires_a_nonempty_complete_group(tmp_path: Path):
    first = tmp_path / "first.nii.gz"
    second = tmp_path / "second.bvec"
    first.touch()

    assert not all_outputs_exist(())
    assert not all_outputs_exist((first, second))

    second.touch()
    assert all_outputs_exist((first, second))
