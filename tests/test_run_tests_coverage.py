# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import io
import json
import os
import queue
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from flagos_optest import run_tests


def test_display_helpers_and_inventory(tmp_path, monkeypatch, capsys):
    run_tests.pinfo("info")
    run_tests.perror("error")
    run_tests.pwarn("warn")
    directory = tmp_path / "new"
    run_tests.ensure_dir(directory)
    assert directory.stat().st_mode & 0o777 == 0o755

    assert run_tests._progress_bar(0, 0, 3) == "   "
    assert "█" in run_tests._progress_bar(1, 4, 2, "color")
    assert "CUSTOM" in run_tests._format_status("custom", 1.0)

    monkeypatch.setattr(run_tests, "IS_TTY", False)
    display = run_tests.LiveDisplay([0], 0, op_width=8)
    display.init()
    display.log("plain")
    display.update_gpu(9, "ignored")
    display.update_gpu(0, "ready")
    display.update_progress(0)
    display.finish()
    display._draw_footer()

    monkeypatch.setattr(run_tests, "IS_TTY", True)
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    display = run_tests.LiveDisplay([0], 1, op_width=8)
    display._erase_footer()
    display.init()
    display.log("tty")
    display.update_gpu(0, "busy")
    display.update_progress(2)
    display.finish()
    assert "tty" in output.getvalue()

    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "operators.yaml").write_text("ops:\n  - id: add\n", encoding="utf-8")
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    assert run_tests.get_ops_from_inventory() == [{"id": "add"}]
    (conf / "operators.yaml").write_text("[", encoding="utf-8")
    assert run_tests.get_ops_from_inventory() == []

    shm = mock.Mock()
    shm.is_dir.return_value = True
    monkeypatch.setattr(run_tests, "SHM_PATH", shm)
    monkeypatch.setattr(run_tests.shutil, "disk_usage", mock.Mock(side_effect=OSError))
    with pytest.raises(run_tests.InfrastructureError):
        run_tests.check_shared_memory(1)


def _fake_torch(*, cuda_error=False, count=1):
    cuda = types.SimpleNamespace(
        is_available=mock.Mock(side_effect=RuntimeError if cuda_error else None),
        get_device_name=mock.Mock(side_effect=RuntimeError if cuda_error else None),
        device_count=mock.Mock(side_effect=RuntimeError if cuda_error else None),
    )
    if not cuda_error:
        cuda.is_available.return_value = True
        cuda.get_device_name.return_value = "GPU"
        cuda.device_count.return_value = count
    return types.SimpleNamespace(__version__="1.0", cuda=cuda)


def test_probe_torch_triton_frameworks_and_environment(monkeypatch):
    env = {}
    monkeypatch.setattr(run_tests, "ENV_INFO", env)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    run_tests._probe_torch()
    assert env["torch"]["device_count"] == 1

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_error=True))
    monkeypatch.setitem(
        sys.modules, "torch_txda", types.SimpleNamespace(device_count=lambda: 2)
    )
    torch_npu = types.ModuleType("torch.npu")
    torch_npu.device_count = lambda: 3
    monkeypatch.setitem(sys.modules, "torch.npu", torch_npu)
    run_tests._probe_torch()
    assert env["torch"]["device_count"] == 2
    torch_module = types.ModuleType("torch")
    torch_module.__version__ = "1.0"
    torch_module.cuda = _fake_torch(cuda_error=True).cuda
    torch_module.npu = torch_npu
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch_txda", None)
    run_tests._probe_torch()
    assert env["torch"]["device_count"] == 3

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(SystemExit):
        run_tests._probe_torch()

    triton = types.SimpleNamespace(__version__="3.0", Config=object())
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setattr(run_tests.metadata, "version", lambda _name: "1.0")
    run_tests._probe_triton()
    assert env["triton"]["has_config"] is True

    triton.__version__ = ""
    run_tests._probe_triton()

    monkeypatch.setattr(run_tests.metadata, "version", mock.Mock(side_effect=Exception))
    monkeypatch.setitem(sys.modules, "triton", None)
    with pytest.raises(SystemExit):
        run_tests._probe_triton()

    monkeypatch.setattr(run_tests.metadata, "version", lambda _name: "1.0")
    run_tests._probe_triton()

    def importer(name):
        if name == "vllm":
            return types.SimpleNamespace(__version__="0.1")
        if name == "sglang":
            raise RuntimeError("broken")
        raise ImportError

    monkeypatch.setattr(run_tests.importlib, "import_module", importer)
    run_tests._probe_frameworks()
    assert env["vllm"] == {"version": "0.1"}

    monkeypatch.setattr(
        run_tests.importlib,
        "import_module",
        mock.Mock(side_effect=ImportError),
    )
    run_tests._probe_frameworks()

    monkeypatch.setenv("ENFLAME_PT_OP_DEBUG_CONFIG", "fallback_cpu=all_ops")
    run_tests._strip_cpu_fallback()
    assert "ENFLAME_PT_OP_DEBUG_CONFIG" not in os.environ
    run_tests._strip_cpu_fallback()

    calls = []
    for name in (
        "_strip_cpu_fallback",
        "_probe_torch",
        "_probe_triton",
        "_probe_gems",
        "_probe_frameworks",
        "_probe_accelerator",
    ):
        monkeypatch.setattr(run_tests, name, lambda n=name: calls.append(n))
    run_tests.probe_env()
    assert len(calls) == 6


class _BrokenModule:
    def __getattr__(self, _name):
        raise RuntimeError("missing")


def test_probe_gems_success_and_failures(monkeypatch):
    primary = _BrokenModule()
    fallback = types.SimpleNamespace(
        __version__="2", vendor_name="nvidia", device="H800"
    )

    def importer(name):
        if name == run_tests._KNOWN_LIBS[0][0]:
            return primary
        if name == run_tests._KNOWN_LIBS[1][0]:
            return fallback
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(run_tests.importlib, "import_module", importer)
    monkeypatch.setattr(run_tests, "ENV_INFO", {})
    run_tests._probe_gems()
    assert run_tests.GEMS_ENV["vendor"] == "nvidia"
    assert run_tests.GEMS_MODULE is primary

    first = types.SimpleNamespace(__version__="1", vendor_name="a", device=None)
    second = types.SimpleNamespace(__version__="2", vendor_name="b", device="B")
    third = types.SimpleNamespace(__version__="3", vendor_name="c", device="C")
    modules = iter([first, second, third])
    monkeypatch.setattr(
        run_tests,
        "_KNOWN_LIBS",
        [("one", "One"), ("two", "Two"), ("three", "Three")],
    )
    monkeypatch.setattr(
        run_tests.importlib, "import_module", lambda _name: next(modules)
    )
    run_tests._probe_gems()
    assert run_tests.GEMS_ENV == {
        "library_name": "One",
        "version": "1",
        "vendor": "a",
        "device": "B",
    }

    for error in (ImportError("bad"), RuntimeError("bad")):
        monkeypatch.setattr(
            run_tests.importlib, "import_module", mock.Mock(side_effect=error)
        )
        with pytest.raises(SystemExit):
            run_tests._probe_gems()

    for field in ("vendor_name", "device"):
        module = types.SimpleNamespace(__version__="1", vendor_name="x", device="y")
        delattr(module, field)
        monkeypatch.setattr(run_tests, "_KNOWN_LIBS", [("library", "Library")])
        monkeypatch.setattr(
            run_tests.importlib,
            "import_module",
            lambda _name, selected=module: selected,
        )
        with pytest.raises(SystemExit):
            run_tests._probe_gems()


def test_probe_gems_uses_distribution_version(monkeypatch):
    module = types.SimpleNamespace(vendor_name="nvidia", device="H800")
    monkeypatch.setattr(run_tests, "_KNOWN_LIBS", [("flag_train", "FlagTrain")])
    monkeypatch.setattr(run_tests.importlib, "import_module", lambda _name: module)
    version = mock.Mock(return_value="0.0.0")
    monkeypatch.setattr(run_tests.metadata, "version", version)
    monkeypatch.setattr(run_tests, "ENV_INFO", {})

    run_tests._probe_gems()

    assert run_tests.ENV_INFO["flag_train"] == {
        "library_name": "FlagTrain",
        "version": "0.0.0",
        "vendor": "nvidia",
        "device": "H800",
    }
    version.assert_called_once_with("flag_train")


def test_get_env_all_vendors(monkeypatch):
    monkeypatch.setenv("KEEP", "yes")
    for vendor, expected in (
        ("ascend", "ASCEND_RT_VISIBLE_DEVICES"),
        ("hygon", "HIP_VISIBLE_DEVICES"),
        ("unknown", "CUDA_VISIBLE_DEVICES"),
    ):
        monkeypatch.setattr(run_tests, "GEMS_ENV", {"vendor": vendor})
        assert run_tests.get_env("2")[expected] == "2"


class _Process:
    def __init__(self, waits, returncode=0):
        self.waits = iter(waits)
        self.returncode = returncode
        self.pid = 42

    def wait(self, timeout=None):
        value = next(self.waits)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.mark.parametrize(
    ("waits", "kill_effects", "expected"),
    [
        ([None], [], 7),
        ([subprocess.TimeoutExpired("x", 1), None], [None], run_tests.TIMEOUT),
        (
            [subprocess.TimeoutExpired("x", 1)],
            [ProcessLookupError()],
            run_tests.TIMEOUT,
        ),
        (
            [subprocess.TimeoutExpired("x", 1), subprocess.TimeoutExpired("x", 1)],
            [None, None],
            run_tests.TIMEOUT,
        ),
        ([RuntimeError("bad")], [], -1),
    ],
)
def test_run_cmd_outcomes(tmp_path, monkeypatch, waits, kill_effects, expected):
    config = types.SimpleNamespace(dump_output=True, output_dir=tmp_path)
    (tmp_path / "add").mkdir()
    monkeypatch.setattr(run_tests, "CFG", config)
    process = _Process(waits, returncode=7)
    monkeypatch.setattr(run_tests.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(run_tests.os, "getpgid", lambda _pid: 99)
    killer = mock.Mock(side_effect=kill_effects)
    monkeypatch.setattr(run_tests.os, "killpg", killer)
    assert run_tests.run_cmd("add", "pytest -q", flavor="accuracy") == expected


def test_run_cmd_log_open_failure(monkeypatch):
    monkeypatch.setattr(
        run_tests,
        "CFG",
        types.SimpleNamespace(dump_output=True, output_dir=Path("/missing")),
    )
    monkeypatch.setattr(
        run_tests.subprocess, "Popen", lambda *_a, **_k: _Process([None])
    )
    with mock.patch("builtins.open", side_effect=OSError):
        assert run_tests.run_cmd("x", "true") == 0


def test_run_cmd_no_logs_and_second_kill_missing(monkeypatch):
    monkeypatch.setattr(run_tests, "CFG", types.SimpleNamespace(dump_output=False))
    process = _Process(
        [subprocess.TimeoutExpired("x", 1), subprocess.TimeoutExpired("x", 1)]
    )
    monkeypatch.setattr(run_tests.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(run_tests.os, "getpgid", lambda _pid: 1)
    monkeypatch.setattr(
        run_tests.os,
        "killpg",
        mock.Mock(side_effect=[None, ProcessLookupError()]),
    )
    assert run_tests.run_cmd("x", "true") == run_tests.TIMEOUT


def _accuracy_file(tmp_path, payload):
    path = tmp_path / "accuracy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_parse_accuracy_all_statuses(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert run_tests.parse_accuracy_data(invalid)["status"] == "Error"
    assert (
        run_tests.parse_accuracy_data(_accuracy_file(tmp_path, {}))["status"]
        == "NotFound"
    )
    assert (
        run_tests.parse_accuracy_data(
            _accuracy_file(
                tmp_path, {"case[x]": {"result": "passed", "params": {"x": "a b"}}}
            )
        )["status"]
        == "Passed"
    )
    failed = run_tests.parse_accuracy_data(
        _accuracy_file(tmp_path, {"case[x]": {"result": "failed", "reason": "bad"}})
    )
    assert failed["status"] == "Failed"
    for payload, status in (
        ({"case[x]": {"result": "skipped", "reason": "Issue #1"}}, "Failed"),
        ({"case[x]": {"result": "skipped"}}, "Skipped"),
        (
            {
                "case[x]": {"result": "skipped"},
                "ok[x]": {"result": "passed"},
            },
            "Passed",
        ),
    ):
        assert (
            run_tests.parse_accuracy_data(_accuracy_file(tmp_path, payload))["status"]
            == status
        )


def _queue_config(tmp_path):
    return types.SimpleNamespace(
        all_op_ids={"stack"},
        skip_cpu_tests=[],
        quick=True,
        timeout=1800,
        output_dir=tmp_path / "results",
        base_data="base data.json",
        accuracy_marks={"add"},
        benchmark_marks={"add"},
    )


def test_operator_queues_all_outcomes(tmp_path, monkeypatch):
    config = _queue_config(tmp_path)
    config.output_dir.mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "benchmark").mkdir()
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.setattr(run_tests, "CFG", config)
    monkeypatch.setattr(run_tests, "get_env", lambda _gpu: {})
    assert run_tests.op_marker("add") == "add"
    assert run_tests.op_marker("_stack") == "underscore_stack"
    assert run_tests.op_marker("__other") == "other"

    stale_accuracy = tmp_path / "tests" / "accuracy_add.json"
    stale_benchmark = tmp_path / "benchmark" / "benchmark_add.json"
    stale_accuracy.write_text("stale", encoding="utf-8")
    stale_benchmark.write_text("stale", encoding="utf-8")
    config.skip_cpu_tests = ["add"]
    monkeypatch.setattr(run_tests, "run_cmd", lambda *_a, **_k: 1)
    assert run_tests.run_accuracy_q(0, "add")["status"] == "Error"
    assert run_tests.run_benchmark_q(0, "add")["status"] == "NotFound"
    config.skip_cpu_tests = []

    for function, directory, filename, timeout_status, missing_status in (
        (run_tests.run_accuracy_q, "tests", "accuracy_add.json", "Timeout", "Error"),
        (
            run_tests.run_benchmark_q,
            "benchmark",
            "benchmark_add.json",
            "Timeout",
            "NotFound",
        ),
    ):
        monkeypatch.setattr(run_tests, "run_cmd", lambda *_a, **_k: run_tests.TIMEOUT)
        assert function(0, "add")["status"] == timeout_status
        monkeypatch.setattr(run_tests, "run_cmd", lambda *_a, **_k: 1)
        assert function(0, "add")["status"] == missing_status
        source = tmp_path / directory / filename
        payload = (
            {"case[x]": {"result": "passed"}}
            if directory == "tests"
            else {"add": {"result": "passed", "details": []}}
        )

        def write_result(*_args, result_path=source, result_payload=payload, **_kwargs):
            result_path.write_text(json.dumps(result_payload), encoding="utf-8")
            return 0

        monkeypatch.setattr(run_tests, "run_cmd", write_result)
        assert function(0, "add")["status"] == "Passed"


class _ListQueue:
    def __init__(self, values):
        self.values = list(values)
        self.sent = []

    def get_nowait(self):
        if not self.values:
            raise queue.Empty
        return self.values.pop(0)

    def put(self, value):
        self.sent.append(value)


def test_worker_success_skips_customized_and_error(tmp_path, monkeypatch):
    config = _queue_config(tmp_path)
    config.output_dir.mkdir()
    monkeypatch.setattr(run_tests, "CFG", config)
    monkeypatch.setattr(run_tests, "ensure_dir", lambda _path: None)
    monkeypatch.setattr(
        run_tests,
        "run_accuracy_q",
        lambda *_a: {"status": "Passed", "duration": 1},
    )
    monkeypatch.setattr(
        run_tests,
        "run_benchmark_q",
        lambda *_a: {"status": "Passed", "duration": 2},
    )
    backend = types.SimpleNamespace(get_customized_ops=lambda: [("add",)])
    monkeypatch.setattr(
        run_tests,
        "GEMS_MODULE",
        types.SimpleNamespace(runtime=types.SimpleNamespace(backend=backend)),
    )
    work = _ListQueue(["", "add"])
    display = _ListQueue([])
    with (
        mock.patch.object(sys, "stdout", io.StringIO()),
        mock.patch.object(sys, "stderr", io.StringIO()),
    ):
        run_tests.worker_proc(0, work, display)
    data = json.loads((config.output_dir / "summary0.json").read_text(encoding="utf-8"))
    assert data["add"]["customized"] is True
    assert display.sent[-1] == ("exit", 0)

    config.accuracy_marks = set()
    config.benchmark_marks = set()
    monkeypatch.setattr(run_tests, "GEMS_MODULE", None)
    work = _ListQueue(["skip"])
    display = _ListQueue([])
    with (
        mock.patch.object(sys, "stdout", io.StringIO()),
        mock.patch.object(sys, "stderr", io.StringIO()),
    ):
        run_tests.worker_proc(1, work, display)
    assert any(
        message[4] == "NotFound" for message in display.sent if message[0] == "done"
    )

    config.accuracy_marks = {"bad"}
    monkeypatch.setattr(
        run_tests, "run_accuracy_q", mock.Mock(side_effect=RuntimeError)
    )
    work = _ListQueue(["bad"])
    display = _ListQueue([])
    with (
        mock.patch.object(sys, "stdout", io.StringIO()),
        mock.patch.object(sys, "stderr", io.StringIO()),
    ):
        run_tests.worker_proc(2, work, display)
    assert sum(message[0] == "done" for message in display.sent) == 2


def test_count_op_statuses_includes_known_and_unexpected_statuses():
    op_data = {
        status.lower(): {"accuracy": {"status": status}}
        for status in run_tests.OP_RESULT_STATUS_KEYS
    }
    op_data["future"] = {"accuracy": {"status": "FutureStatus"}}
    op_data["missing"] = {}

    counts = run_tests.count_op_statuses(op_data, "accuracy")

    for status_key in run_tests.OP_RESULT_STATUS_KEYS.values():
        assert counts[status_key] == 1
    assert counts["futurestatus"] == 1
    assert counts["unknown"] == 1


class _Worker:
    def __init__(self, gpu_id, alive=True):
        self._gpu_id = gpu_id
        self.pid = 10
        self.exitcode = 3
        self.alive = alive
        self.killed = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False if timeout is None else self.alive

    def kill(self):
        self.killed = True


class _Display:
    def __init__(self, gpu_ids=(0,), op_count=1, op_width=5):
        self.gpu_ids = list(gpu_ids)
        self.op_count = op_count
        self.op_width = op_width
        self.footer = [""]
        self.logs = []
        self.updates = []

    def log(self, message):
        self.logs.append(message)

    def update_gpu(self, gpu, message):
        self.updates.append((gpu, message))

    def _fmt_progress(self, done):
        return str(done)


def test_display_loop_messages_and_dead_worker(monkeypatch):
    monkeypatch.setattr(run_tests, "IS_TTY", False)
    display = _Display(op_width=5)
    messages = _ListQueue([])
    messages.values = [
        ("start", 0, "accuracy", "long-operator"),
        ("done", 0, "accuracy", "add", "Passed", 1.0),
        ("done", 0, "benchmark", "long-operator", "Failed", 2.0),
        ("exit", 0),
    ]
    messages.get = lambda timeout=None: messages.values.pop(0)
    worker = _Worker(0)
    run_tests.display_loop(messages, display, [worker])
    assert len(display.logs) == 3

    monkeypatch.setattr(run_tests, "IS_TTY", True)
    display = _Display()
    messages = mock.Mock()
    messages.get.side_effect = [Exception("timeout")]
    dead = _Worker(0, alive=False)
    run_tests.display_loop(messages, display, [dead])
    assert "died without exit" in display.logs[0]

    display = _Display()
    messages = mock.Mock()
    messages.get.side_effect = [Exception("timeout"), ("exit", 0)]
    run_tests.display_loop(messages, display, [_Worker(0, alive=True)])

    display = _Display(gpu_ids=(9,))
    messages = mock.Mock()
    messages.get.return_value = ("exit", 9)
    run_tests.display_loop(messages, display, [_Worker(0)])

    monkeypatch.setattr(run_tests, "IS_TTY", True)
    display = _Display()
    messages = mock.Mock()
    messages.get.side_effect = [
        ("start", 0, "benchmark", "add"),
        ("done", 0, "accuracy", "add", "Passed", 0),
        ("unknown",),
        ("exit", 0),
    ]
    run_tests.display_loop(messages, display, [_Worker(0)])


def test_cleanup_workers_interrupt_and_selection(tmp_path, monkeypatch):
    for directory, filename in (
        ("tests", "accuracy_a.json"),
        ("benchmark", "benchmark_a.json"),
    ):
        path = tmp_path / directory
        path.mkdir()
        (path / filename).write_text("x", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    (output / "summary0.tmp").write_text("x", encoding="utf-8")
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.setattr(run_tests, "CFG", types.SimpleNamespace(output_dir=output))
    run_tests.cleanup_intermediate_files()
    assert not list(tmp_path.rglob("*.json"))

    class BrokenFile:
        def unlink(self):
            raise OSError

    class BrokenDirectory:
        def __truediv__(self, _name):
            return self

        def glob(self, _pattern):
            return [BrokenFile()]

    monkeypatch.setattr(run_tests, "ROOT", BrokenDirectory())
    monkeypatch.setattr(
        run_tests, "CFG", types.SimpleNamespace(output_dir=BrokenDirectory())
    )
    run_tests.cleanup_intermediate_files()
    monkeypatch.setattr(run_tests, "CFG", types.SimpleNamespace())
    run_tests.cleanup_intermediate_files()

    worker = _Worker(0)
    monkeypatch.setattr(run_tests, "WORKER_PROCESSES", [worker])
    monkeypatch.setattr(run_tests.os, "getpgid", lambda _pid: 1)
    monkeypatch.setattr(run_tests.os, "killpg", mock.Mock(side_effect=OSError))
    run_tests.terminate_workers()
    assert worker.killed

    stopped = _Worker(1, alive=False)
    stopped.pid = None
    monkeypatch.setattr(run_tests, "WORKER_PROCESSES", [stopped])
    run_tests.terminate_workers()

    monkeypatch.setattr(run_tests, "INTERRUPTED", True)
    run_tests.handle_interrupt(None, None)
    monkeypatch.setattr(run_tests, "INTERRUPTED", False)
    monkeypatch.setattr(run_tests, "IS_TTY", True)
    monkeypatch.setattr(run_tests, "terminate_workers", lambda: None)
    monkeypatch.setattr(run_tests, "cleanup_intermediate_files", lambda: None)
    with pytest.raises(SystemExit):
        run_tests.handle_interrupt(None, None)
    monkeypatch.setattr(run_tests, "INTERRUPTED", False)
    monkeypatch.setattr(run_tests, "IS_TTY", False)
    with pytest.raises(SystemExit):
        run_tests.handle_interrupt(None, None)

    catalog = [
        {"id": "alpha", "labels": ["NoCPU"], "stages": [{"alpha": {}}]},
        {"id": "beta", "stages": [{"beta": {}}]},
        {"id": "none", "stages": []},
    ]
    monkeypatch.setattr(run_tests, "get_ops_from_inventory", lambda: catalog)
    monkeypatch.setattr(run_tests, "CFG", types.SimpleNamespace())

    def options(**overrides):
        values = {
            "ops": None,
            "op_list_file": None,
            "exclude_ops": None,
            "exclude_op_list_file": None,
            "stages": "stable",
            "start": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    include_file = tmp_path / "include.txt"
    include_file.write_text("add\n", encoding="utf-8")
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text("other\n", encoding="utf-8")
    run_tests.OPTS = options(
        ops=" x, y, x, ,",
        op_list_file=include_file,
        exclude_ops=" y, missing, y",
        exclude_op_list_file=exclude_file,
    )
    assert run_tests.get_ops_to_test() == ["x"]
    selection = run_tests.CFG.operator_selection
    assert selection.include_count == 2
    assert selection.include_source == "--ops"
    assert selection.include_file is None
    assert selection.exclude_count == 2
    assert selection.exclude_source == "--exclude-ops"
    assert selection.exclude_file is None
    assert selection.excluded_ops == ["y", "missing"]
    assert selection.matched_count == 1
    assert selection.selected_count == 1
    assert selection.explicit is True

    missing = tmp_path / "missing"
    run_tests.OPTS = options(op_list_file=missing)
    with pytest.raises(run_tests.OperatorSelectionError):
        run_tests.get_ops_to_test()

    listing = tmp_path / "ops.txt"
    listing.write_text("# comment\nadd\n\nmul\nadd\n", encoding="utf-8")
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("# comment\nmul\nunknown\n\n", encoding="utf-8")
    run_tests.OPTS = options(op_list_file=listing, exclude_op_list_file=excluded)
    assert run_tests.get_ops_to_test() == ["add"]
    selection = run_tests.CFG.operator_selection
    assert selection.include_count == 2
    assert selection.include_file == listing
    assert selection.exclude_count == 2
    assert selection.exclude_file == excluded
    assert selection.matched_count == 1
    assert selection.selected_count == 1
    assert run_tests._selection_source_label("--ops") == "--ops"
    assert (
        run_tests._selection_source_label("--op-list-file", listing)
        == "--op-list-file: ops.txt"
    )
    assert run_tests._entry_count_label(1) == "1 entry"
    assert run_tests._entry_count_label(2) == "2 entries"

    run_tests.OPTS = options(ops="add", exclude_op_list_file=missing)
    with pytest.raises(run_tests.OperatorSelectionError):
        run_tests.get_ops_to_test()

    run_tests.OPTS = options(ops="add", exclude_ops="add")
    assert run_tests.get_ops_to_test() == []
    assert run_tests.CFG.operator_selection.matched_count == 1

    run_tests.OPTS = options(stages="bad")
    assert run_tests.get_ops_to_test() == []
    assert run_tests.CFG.operator_selection.explicit is False
    run_tests.OPTS = options(stages="all", start="beta")
    assert run_tests.get_ops_to_test() == ["beta"]
    run_tests.OPTS = options(stages="alpha,beta", exclude_ops="alpha,Alpha")
    assert run_tests.get_ops_to_test() == ["beta"]
    assert run_tests.CFG.operator_selection.include_source == "--stages alpha,beta"
    assert run_tests.CFG.operator_selection.matched_count == 1


def test_marks_parsing_and_collection(tmp_path, monkeypatch):
    marks = tmp_path / "marks.yaml"
    marks.write_text("- marks: [add, mul]\n", encoding="utf-8")
    assert run_tests._parse_marks_file(marks) == {"add", "mul"}
    marks.write_text("", encoding="utf-8")
    assert run_tests._parse_marks_file(marks) == set()
    assert run_tests._parse_marks_file(tmp_path / "missing") == set()
    assert run_tests.collect_marks(["add"])[0] == {"add"}

    ops = [f"op{i}" for i in range(11)]

    def call(command, **_kwargs):
        target = Path(
            next(
                arg.split("=", 1)[1]
                for arg in command
                if arg.startswith("--collect-marks=")
            )
        )
        if command[-1] == "tests/":
            target.write_text("- marks: [extra]\n", encoding="utf-8")
        else:
            target.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_tests.subprocess, "call", call)
    accuracy, benchmark = run_tests.collect_marks(ops)
    assert "extra" in accuracy
    assert benchmark == set(ops)

    monkeypatch.setattr(run_tests.subprocess, "call", lambda *_a, **_k: 1)
    assert run_tests.collect_marks(ops) == (set(ops), set(ops))

    def empty_accuracy_nonempty_benchmark(command, **_kwargs):
        target = Path(
            next(
                arg.split("=", 1)[1]
                for arg in command
                if arg.startswith("--collect-marks=")
            )
        )
        text = "" if command[-1] == "tests/" else "- marks: [bench]\n"
        target.write_text(text, encoding="utf-8")
        return 0

    monkeypatch.setattr(run_tests.subprocess, "call", empty_accuracy_nonempty_benchmark)
    accuracy, benchmark = run_tests.collect_marks(ops)
    assert accuracy == set(ops)
    assert "bench" in benchmark


def test_collect_marks_only_collects_tests(tmp_path, monkeypatch):
    ops = [f"op{index}" for index in range(11)]
    commands = []

    def call(command, **_kwargs):
        commands.append(command)
        target = Path(
            next(
                argument.split("=", 1)[1]
                for argument in command
                if argument.startswith("--collect-marks=")
            )
        )
        target.write_text("[]\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_tests.subprocess, "call", call)
    run_tests.collect_marks(ops)

    assert [command[-1] for command in commands] == ["tests/", "benchmark/"]
    assert all("--collect-only" in command for command in commands)


def test_main_device_selection_and_summary_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    with (
        mock.patch.object(run_tests.signal, "signal"),
        mock.patch.object(run_tests, "probe_env") as probe_env,
        mock.patch.object(
            sys,
            "argv",
            [
                "flagos-optest",
                "--ops",
                "add",
                "--exclude-op-list-file",
                "missing.txt",
            ],
        ),
    ):
        with pytest.raises(SystemExit):
            run_tests.main()
    probe_env.assert_not_called()

    common = [
        mock.patch.object(run_tests.signal, "signal"),
        mock.patch.object(run_tests, "probe_env"),
        mock.patch.object(run_tests, "get_ops_to_test", return_value=["add"]),
    ]
    with (
        common[0],
        common[1],
        common[2],
        mock.patch.object(run_tests, "ENV_INFO", {"torch": {"device_count": 0}}),
        mock.patch.object(sys, "argv", ["flagos-optest", "--gpus", "all"]),
    ):
        with pytest.raises(SystemExit):
            run_tests.main()

    with (
        mock.patch.object(run_tests.signal, "signal"),
        mock.patch.object(run_tests, "probe_env"),
        mock.patch.object(run_tests, "get_ops_to_test", return_value=["add"]),
        mock.patch.object(sys, "argv", ["flagos-optest", "--gpus", " , "]),
    ):
        with pytest.raises(SystemExit):
            run_tests.main()

    class FakeProcess(_Worker):
        def __init__(self, target=None, args=()):
            super().__init__(args[0])

        def start(self):
            pass

    output = tmp_path / "result"
    output.mkdir()
    (output / "summary0.json").write_text("{", encoding="utf-8")
    with (
        mock.patch.object(run_tests.signal, "signal"),
        mock.patch.object(run_tests, "probe_env"),
        mock.patch.object(run_tests, "get_ops_to_test", return_value=["add"]),
        mock.patch.object(run_tests, "check_shared_memory"),
        mock.patch.object(run_tests, "collect_marks", return_value=({"add"}, {"add"})),
        mock.patch.object(run_tests, "Process", FakeProcess),
        mock.patch.object(run_tests, "display_loop"),
        mock.patch.object(run_tests, "WORKER_PROCESSES", []),
        mock.patch.object(
            sys,
            "argv",
            [
                "flagos-optest",
                "--gpus",
                "0,1",
                "--output-dir",
                str(output),
                "--color",
                "always",
            ],
        ),
    ):
        run_tests.main()
    incomplete_summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert incomplete_summary["total_ops"] == 0

    default_output = tmp_path / "logs_results_20260101_0000"

    class FixedDateTime(run_tests.datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 1)

    class WritingProcess(_Worker):
        def __init__(self, target=None, args=()):
            super().__init__(args[0])

        def start(self):
            default_output.mkdir(exist_ok=True)
            (default_output / "summary0.json").write_text(
                json.dumps(
                    {
                        "add": {
                            "accuracy": {"status": "Passed"},
                            "performance": {"status": "NotFound"},
                        }
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(run_tests.datetime, "datetime", FixedDateTime)
    with (
        mock.patch.object(run_tests.signal, "signal"),
        mock.patch.object(run_tests, "probe_env"),
        mock.patch.object(run_tests, "get_ops_to_test", return_value=["add", "mul"]),
        mock.patch.object(run_tests, "check_shared_memory"),
        mock.patch.object(run_tests, "collect_marks", return_value=({"add"}, {"add"})),
        mock.patch.object(run_tests, "Process", WritingProcess),
        mock.patch.object(run_tests, "display_loop"),
        mock.patch.object(run_tests, "WORKER_PROCESSES", []),
        mock.patch.object(run_tests, "ENV_INFO", {"torch": {"device_count": 2}}),
        mock.patch.object(
            sys, "argv", ["flagos-optest", "--gpus", "all", "--color", "never"]
        ),
    ):
        run_tests.main()
    summary = json.loads((default_output / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_ops"] == 1
    assert summary["accuracy_counts"]["passed"] == 1
    assert summary["performance_counts"]["not_found"] == 1
    assert "add" in summary["result"]
    output_text = capsys.readouterr().out
    assert "[INFO] Total operators ... 1" in output_text
    assert "[INFO] Accuracy operators ... passed=1" in output_text
    assert "[INFO] Performance operators ... not_found=1" in output_text
    assert "failed=0" not in output_text
    assert "[INFO] Total elapsed time ... 0:00:00 (0.0s)" in output_text
