"""R10 characterization tests for the shared parallel runner."""

from queue import Queue
from unittest.mock import Mock

import pytest

from qmri_neuropipe.core.config import PipelineConfig


class _PoolPipeline:
    """Pickle-safe pipeline used by the real process-pool smoke test."""

    def __init__(self, config, logger=None):
        self.config = config

    def _process_subject_with_subject_log(self, subject, session):
        assert self.config.get("jobs") == 1


class _Manager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    @staticmethod
    def Queue():
        return Queue()


class _Future:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _Executor:
    submissions = []

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def submit(self, function, *args):
        self.submissions.append((function, args))
        subject, session = args[2], args[3]
        return _Future(
            {
                "n_success": 1,
                "n_failed": 0,
                "n_skipped": 0,
                "subject": subject,
                "session": session,
            }
        )


def _config(tmp_path, **config_data):
    return PipelineConfig(
        bids_dir=tmp_path / "bids",
        output_dir=tmp_path / "out",
        config_data=config_data,
    )


def test_shared_runner_skips_before_dispatch(tmp_path, monkeypatch):
    from qmri_neuropipe.core import parallel

    _Executor.submissions = []
    monkeypatch.setattr(parallel.multiprocessing, "Manager", _Manager)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: list(futures))

    class _Pipeline:
        pass

    result = parallel.run_parallel(
        _Pipeline,
        _config(tmp_path),
        [("01", None), ("02", "A"), ("03", None)],
        2,
        should_skip=lambda subject, session: subject == "02",
        show_ui=False,
    )

    dispatched = [args[2:4] for _, args in _Executor.submissions]
    assert dispatched == [("01", None), ("03", None)]
    assert result == {
        "n_success": 2,
        "n_failed": 0,
        "n_skipped": 1,
        "failures": [],
    }


def test_skip_probe_failures_are_reported_without_dispatch(tmp_path, monkeypatch):
    from qmri_neuropipe.core import parallel

    _Executor.submissions = []
    monkeypatch.setattr(parallel.multiprocessing, "Manager", _Manager)
    monkeypatch.setattr(parallel, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(parallel, "as_completed", lambda futures: list(futures))

    def should_skip(subject, session):
        if subject == "bad":
            raise RuntimeError("skip probe failed")
        return False

    result = parallel.run_parallel(
        object,
        _config(tmp_path),
        [("bad", None), ("good", None)],
        2,
        should_skip=should_skip,
        show_ui=False,
    )

    assert [args[2] for _, args in _Executor.submissions] == ["good"]
    assert result == {
        "n_success": 1,
        "n_failed": 1,
        "n_skipped": 0,
        "failures": [
            {"subject": "bad", "session": None, "error": "skip probe failed"}
        ],
    }


def test_worker_processes_exactly_one_pair(tmp_path, monkeypatch):
    from qmri_neuropipe.core import parallel

    calls = []

    class _Pipeline:
        def __init__(self, config, logger=None):
            self.config = config
            self.logger = logger

        def _process_subject_with_subject_log(self, subject, session):
            calls.append((subject, session, self.config.get("jobs")))

    monkeypatch.delenv("QMRI_PARALLEL_WORKER", raising=False)
    result = parallel.run_pipeline_worker(
        _Pipeline,
        _config(tmp_path, jobs=4).to_dict(),
        "01",
        "A",
    )

    assert calls == [("01", "A", 1)]
    assert result == {
        "n_success": 1,
        "n_failed": 0,
        "n_skipped": 0,
        "subject": "01",
        "session": "A",
    }


def test_worker_supports_workflow_style_run_contract(tmp_path):
    from qmri_neuropipe.core import parallel

    calls = []

    class _Workflow:
        def __init__(self, config, logger=None):
            pass

        def run(self, *, pairs):
            calls.append(pairs)
            return {"n_success": 1, "n_failed": 0, "n_skipped": 0}

    result = parallel.run_pipeline_worker(
        _Workflow,
        _config(tmp_path, jobs=3).to_dict(),
        "01",
        None,
    )

    assert calls == [[("01", None)]]
    assert result == {
        "n_success": 1,
        "n_failed": 0,
        "n_skipped": 0,
        "subject": "01",
        "session": None,
    }


def test_real_process_pool_smoke(tmp_path):
    from qmri_neuropipe.core import parallel

    try:
        result = parallel.run_parallel(
            _PoolPipeline,
            _config(tmp_path, jobs=2),
            [("01", None)],
            1,
            show_ui=False,
        )
    except (EOFError, NotImplementedError, PermissionError) as exc:
        pytest.skip(f"ProcessPoolExecutor is unavailable in this environment: {exc}")

    assert result == {
        "n_success": 1,
        "n_failed": 0,
        "n_skipped": 0,
        "failures": [],
    }


def test_fd_restore_reinstates_descriptors_and_python_streams(monkeypatch):
    from qmri_neuropipe.core import parallel

    active_stdout = Mock()
    active_stderr = Mock()
    original_stdout = Mock()
    original_stderr = Mock()
    dup2 = Mock()
    close = Mock()
    monkeypatch.setattr(parallel.sys, "stdout", active_stdout)
    monkeypatch.setattr(parallel.sys, "stderr", active_stderr)
    monkeypatch.setattr(parallel.os, "dup2", dup2)
    monkeypatch.setattr(parallel.os, "close", close)

    parallel._restore_standard_streams(
        10,
        11,
        original_stdout,
        original_stderr,
    )

    assert dup2.call_args_list == [((10, 1),), ((11, 2),)]
    assert close.call_args_list == [((10,),), ((11,),)]
    assert parallel.sys.stdout is original_stdout
    assert parallel.sys.stderr is original_stderr


def test_base_pipeline_routes_to_shared_runner(tmp_path, monkeypatch):
    from qmri_neuropipe.core import base

    shared_runner = Mock(return_value={"n_success": 0})
    monkeypatch.setattr(base, "run_parallel_tasks", shared_runner)

    class _Stub:
        config = _config(tmp_path)
        logger = Mock()

        def _should_skip(self, subject, session):
            return False

    stub = _Stub()
    tasks = [("01", None)]
    result = base.BasePipeline._run_parallel(stub, tasks, 2)

    assert result == {"n_success": 0}
    shared_runner.assert_called_once_with(
        _Stub,
        stub.config,
        tasks,
        2,
        should_skip=stub._should_skip,
        logger=stub.logger,
    )
