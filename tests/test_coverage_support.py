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

import importlib.util
import io
import json
import os
import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest

from flagos_optest import _entrypoints, add_labels, gpu_metadata, performance_report


def test_flat_script_import_fallbacks(monkeypatch):
    package_dir = Path(__file__).parents[1] / "src" / "flagos_optest"
    monkeypatch.syspath_prepend(str(package_dir))
    for filename in ("gpu_check.py", "performance_report.py", "run_tests.py"):
        runpy.run_path(
            str(package_dir / filename), run_name=f"flat_{Path(filename).stem}"
        )

    class TtyBuffer(io.StringIO):
        def isatty(self):
            return True

    with mock.patch.object(sys, "stdout", TtyBuffer()):
        runpy.run_path(str(package_dir / "run_tests.py"), run_name="flat_tty")


def test_entrypoint_wrappers_and_path_cleanup():
    calls = []

    def run_path(path, run_name):
        calls.append((Path(path).name, run_name, Path(sys.path[0]).name))

    with mock.patch.object(_entrypoints.runpy, "run_path", side_effect=run_path):
        _entrypoints.psum_html()
        _entrypoints.psum_text()
        _entrypoints.psum_web()
    assert calls == [
        ("psum_html", "__main__", "flagos_optest"),
        ("psum_text", "__main__", "flagos_optest"),
        ("psum_web", "__main__", "flagos_optest"),
    ]

    original = list(sys.path)
    with (
        mock.patch.object(_entrypoints.runpy, "run_path", side_effect=RuntimeError),
        pytest.raises(RuntimeError),
    ):
        _entrypoints._run_flat("missing")
    assert sys.path == original


def _write_perf(tmp_path, payload):
    path = tmp_path / "performance.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_performance_report_all_result_shapes(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert performance_report.parse_perf_data("add", invalid)["status"] == "Error"
    assert performance_report.parse_perf_data("add", _write_perf(tmp_path, {})) == {
        "status": "NotFound"
    }

    for status in ("failed", "skipped"):
        result = performance_report.parse_perf_data(
            "add",
            _write_perf(
                tmp_path,
                {"add": {"result": status, "reason": "why", "test_case": "case"}},
            ),
        )
        assert result == {
            "status": status.title(),
            "reason": "why",
            "test_case": "case",
        }

    native = {
        "add": {
            "result": "passed",
            "test_case": "native",
            "details": [
                {
                    "dtype": "torch.float32",
                    "result": [
                        {
                            "shape_detail": [1, 2],
                            "latency_base": 2.0,
                            "latency": 1.0,
                            "speedup": 2.0,
                            "compared_speedup": 3.0,
                            "base_data_source": "file",
                        },
                        {"speedup": 1.0},
                    ],
                },
                {"dtype": "custom", "result": []},
            ],
        }
    }
    native_file = _write_perf(tmp_path, native)
    # Default speedup_source="origin": prefer the on-machine gems-vs-torch
    # ratio, so shape1 uses speedup=2.0 (not compared_speedup=3.0). Mean over
    # shapes is (2.0 + 1.0) / 2 = 1.5.
    result = performance_report.parse_perf_data("add", native_file)
    assert result["status"] == "Passed"
    assert result["data"]["fp32"]["speedup"] == 1.5
    assert result["data"]["custom"]["result"] == "Unknown"
    # Explicit nv_baseline uses 3.0 for shape1, and the local 1.0 for shape2.
    result = performance_report.parse_perf_data(
        "add", native_file, speedup_source="nv_baseline"
    )
    fp32 = result["data"]["fp32"]
    assert fp32["speedup"] is None
    assert fp32["speedup_by_source"] == {"nv_baseline": 3.0, "origin": 1.0}
    assert "fp32=mixed" in performance_report.speedup_report_note(result["data"])
    assert (
        performance_report.speedup_report_note({"fp32": {"speedup_source": "origin"}})
        == ""
    )
    assert performance_report.speedup_report_note(
        {"fp32": {"speedup_source": "nv_baseline"}}
    ) == ("Speedup vs NVIDIA baseline")

    compared = {
        "add": {
            "result": "passed",
            "details": [
                {
                    "dtype": "torch.float16",
                    "native_baseline_skip_reason": "no native baseline",
                    "result": [
                        {"shape_detail": [1], "compared_speedup": 2.0},
                        {"shape_detail": [2], "compared_speedup": "unknown"},
                        # No speedup and no compared_speedup: _case_record keeps
                        # speedup=None and records speedup_source=None. This None
                        # is excluded from the fp16 mean below.
                        {"shape_detail": [3]},
                        {"shape_detail": [4], "speedup": 1.0, "compared_speedup": 2.0},
                    ],
                },
                {
                    "dtype": "torch.float16",
                    "native_baseline_skip_reason": "no native baseline",
                    "result": [],
                },
                {
                    "dtype": "torch.float32",
                    "native_baseline_skip_reason": "unsupported",
                    "result": [],
                },
            ],
        }
    }
    result = performance_report.parse_perf_data("add", _write_perf(tmp_path, compared))
    assert result["data"]["fp16"]["speedup_by_source"] == {
        "nv_baseline": 2.0,
        "origin": 1.0,
    }
    # The reference-less shape carries no speedup and no source.
    assert result["data"]["fp16"]["details"]["[3]"]["speedup"] is None
    assert result["data"]["fp16"]["details"]["[3]"]["speedup_source"] is None
    assert result["data"]["fp32"]["speedup"] is None
    assert result["native_baseline_skip_reason"] == ("no native baseline; unsupported")


def test_add_labels_error_and_success_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(add_labels, "ROOT", tmp_path)
    assert add_labels.load_op_labels() == {}
    assert "Failed to load" in capsys.readouterr().out

    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "operators.yaml").write_text(
        "ops:\n  - id: add\n    labels: [pointwise]\n", encoding="utf-8"
    )
    assert add_labels.load_op_labels() == {"add": ["pointwise"]}

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    for argv, message in (
        (["flagos-add-labels", str(tmp_path), "missing"], "summary.json not found"),
        (["flagos-add-labels", str(tmp_path), "results"], "No 'result' field"),
    ):
        if argv[-1] == "results":
            (result_dir / "summary.json").write_text("{}", encoding="utf-8")
        with mock.patch.object(sys, "argv", argv), pytest.raises(SystemExit):
            add_labels.main()
        assert message in capsys.readouterr().out

    (result_dir / "summary.json").write_text(
        json.dumps({"result": {"add": {}, "mul": {}}}), encoding="utf-8"
    )
    with mock.patch.object(add_labels, "load_op_labels", return_value={}):
        with (
            mock.patch.object(
                sys, "argv", ["flagos-add-labels", str(tmp_path), "results"]
            ),
            pytest.raises(SystemExit),
        ):
            add_labels.main()
    assert "No labels loaded" in capsys.readouterr().out

    with mock.patch.object(
        sys, "argv", ["flagos-add-labels", str(tmp_path), "results"]
    ):
        add_labels.main()
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["result"]["add"]["labels"] == ["pointwise"]
    assert summary["result"]["mul"]["labels"] == []
    assert "Updated 2 operators" in capsys.readouterr().out


def _load_notify_module():
    path = Path(__file__).parents[1] / ".github" / "scripts" / "notify_feishu.py"
    spec = importlib.util.spec_from_file_location("notify_feishu_coverage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _notify_environment(module, **overrides):
    env = {name: f"value-{name.lower()}" for name in module.FEISHU_VARS}
    env.update({name: f"value-{name.lower()}" for name in module.REQUIRED_VARS})
    env.update(overrides)
    return env


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_notify_validation_and_api_outcomes(capsys):
    module = _load_notify_module()
    with mock.patch.dict(os.environ, {}, clear=True):
        assert module.check_env_vars() is False
        module.main()
    assert "notification skipped" in capsys.readouterr().out

    env = _notify_environment(module, TARGET_REF="main")
    del env["CI_STATUS"]
    with mock.patch.dict(os.environ, env, clear=True):
        assert module.check_env_vars() is False
    assert "CI_STATUS" in capsys.readouterr().err

    env = _notify_environment(module, TARGET_REF="main", CI_STATUS="success")
    env.update({name: f"optional-{name.lower()}" for name in module.OPTIONAL_VARS})
    with mock.patch.dict(os.environ, env, clear=True):
        assert module.check_env_vars() is True

    outcomes = [
        ({"error": "denied"}, None),
        ({"tenant_access_token": "token"}, {"code": 9}),
        ({"tenant_access_token": "token"}, {"code": 0}),
    ]
    for index, (token, result) in enumerate(outcomes):
        responses = [_Response(token)]
        if result is not None:
            responses.append(_Response(result))
        env = _notify_environment(
            module,
            CI_STATUS=("success", "cancelled", "failed")[index],
            TARGET_REF="feature",
            PLATFORM="H800" if index == 2 else "",
            LIBRARY_NAME="FlagGems" if index == 2 else "",
            PR_NUMBER="12" if index == 2 else "",
            PR_TITLE="Coverage" if index == 2 else "",
        )
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(module.urllib.request, "urlopen", side_effect=responses),
        ):
            if index < 2:
                with pytest.raises(SystemExit):
                    module.main()
            else:
                module.main()
    captured = capsys.readouterr()
    assert "Failed to obtain" in captured.err
    assert "API returned error" in captured.err
    assert "notification sent successfully" in captured.out

    env = _notify_environment(module)
    with mock.patch.dict(os.environ, env, clear=True):
        assert module.check_env_vars() is True
    assert "TARGET_REF" in capsys.readouterr().out


def test_gpu_metadata_remaining_defensive_paths(tmp_path, monkeypatch):
    class BrokenRoot:
        def glob(self, _pattern):
            raise OSError("denied")

    monkeypatch.setattr(gpu_metadata, "AMD_SYSFS_ROOT", BrokenRoot())
    assert gpu_metadata._amd_model_from_sysfs() is None

    root = tmp_path / "pci"
    bad = root / "bad"
    bad.mkdir(parents=True)
    (bad / "vendor").write_text("0x1234", encoding="ascii")
    (bad / "device").write_text("0x1", encoding="ascii")
    (bad / "class").write_text("0x020000", encoding="ascii")
    broken = root / "broken"
    broken.mkdir()
    (broken / "vendor").write_text("0x1002", encoding="ascii")
    monkeypatch.setattr(gpu_metadata, "AMD_SYSFS_ROOT", root)
    assert gpu_metadata._amd_model_from_sysfs() is None

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert gpu_metadata._file_version(invalid, "musa") is None
    invalid.write_text("[]", encoding="utf-8")
    assert gpu_metadata._file_version(invalid, "musa") is None
    invalid.write_text('{"musa_toolkits": {"other": 1}}', encoding="utf-8")
    assert gpu_metadata._file_version(invalid, "musa") is None
    invalid.write_text('{"musa_toolkits": "1.2.3"}', encoding="utf-8")
    assert gpu_metadata._file_version(invalid, "musa") == "1.2.3"

    class BrokenPath:
        name = "version.txt"
        suffix = ".txt"

        def read_text(self, **_kwargs):
            raise UnicodeError

    assert gpu_metadata._file_version(BrokenPath(), "xre") is None

    monkeypatch.setitem(gpu_metadata.RUNTIMES, "thead", ("ppu", (), (str(tmp_path),)))
    with mock.patch.object(gpu_metadata.Path, "glob", side_effect=ValueError):
        assert (
            gpu_metadata._runtime_version(
                "thead", {}, lambda *_a, **_k: (1, "", ""), ""
            )
            is None
        )

    def run(command, env=None):
        if command[0].endswith("clang"):
            return 0, "clang (3.4.5.20260101 build)", ""
        return 1, "", ""

    monkeypatch.setitem(
        gpu_metadata.RUNTIMES, "iluvatar", ("corex", (), (str(tmp_path),))
    )
    assert gpu_metadata._runtime_version("iluvatar", {}, run, "") == "3.4.5"

    def dpkg_run(command, env=None):
        if command[0] == "dpkg-query":
            return 0, "cuda-libraries-13-0\t13.0.1-1\n", ""
        return 1, "", ""

    monkeypatch.setitem(gpu_metadata.RUNTIMES, "nvidia", ("cuda", (), ()))
    assert gpu_metadata._runtime_version("nvidia", {}, dpkg_run, "") == "13.0.1"

    def tops_run(command, env=None):
        return 0, "4.2.1", ""

    monkeypatch.setitem(gpu_metadata.RUNTIMES, "enflame", ("tops", (), ()))
    assert gpu_metadata._runtime_version("enflame", {}, tops_run, "") == "4.2.1"

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()

    def clang_run(command, env=None):
        return 0, "clang without a CoreX build", ""

    monkeypatch.setitem(
        gpu_metadata.RUNTIMES,
        "iluvatar",
        ("corex", (), (str(first), str(second))),
    )
    assert gpu_metadata._runtime_version("iluvatar", {}, clang_run, "") is None

    def ambiguous_cuda(command, env=None):
        if command[0] == "dpkg-query":
            return (
                0,
                "cuda-libraries-12-8\t12.8.1-1\ncuda-libraries-13-0\t13.0.1-1\n",
                "",
            )
        return 1, "", ""

    monkeypatch.setitem(gpu_metadata.RUNTIMES, "nvidia", ("cuda", (), ()))
    assert gpu_metadata._runtime_version("nvidia", {}, ambiguous_cuda, "") is None

    def missing_tops(command, env=None):
        return 1, "", ""

    monkeypatch.setitem(gpu_metadata.RUNTIMES, "enflame", ("tops", (), ()))
    assert (
        gpu_metadata._runtime_version("enflame", {}, missing_tops, "tops version 5.0.0")
        == "5.0.0"
    )
