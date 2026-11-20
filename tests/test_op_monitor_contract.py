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

"""Downstream op-monitor archive contract.

These tests document the files, directory layout, and structured fields that
op-monitor consumes from weekly-test archives. Do not loosen or change this
contract without coordinating the matching parser and UI changes in op-monitor.
A failure here must block CI because an apparently harmless result-format change
can otherwise break metric ingestion, shape details, raw-log display, defect
creation, or GitHub issue generation downstream.

Consumed archive data:

* exactly one ``summary.json`` at the archive root or one directory below it;
* ``env`` and per-operator accuracy/performance records in ``summary.json``;
* ``<operator>/performance_result.json`` as the preferred shape-detail source;
* accuracy/performance stdout and stderr logs for downstream log features.

``summary<N>.json``, ``result.html``, and ``result.csv`` are deliberately not
part of this contract. Op-monitor does not read them.
"""

import json
import os
import shlex
import sys
import types
from pathlib import Path
from unittest import mock

import yaml

from flagos_optest import add_labels, run_tests

_REPOSITORY_ROOT = Path(__file__).parents[1]
_ACCURACY_FIELDS = {
    "passed",
    "failed",
    "skipped",
    "errors",
    "total",
    "exit_code",
    "status",
}
_LOG_FILES = {
    "accuracy_stdout.log",
    "accuracy_stderr.log",
    "performance_stdout.log",
    "performance_stderr.log",
}


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _generate_result_directory(tmp_path, monkeypatch):
    project = tmp_path / "target"
    tests_dir = project / "tests"
    benchmark_dir = project / "benchmark"
    config_dir = project / "conf"
    tests_dir.mkdir(parents=True)
    benchmark_dir.mkdir()
    config_dir.mkdir()
    (config_dir / "operators.yaml").write_text(
        "ops:\n  - id: add\n    labels: [binary]\n",
        encoding="utf-8",
    )

    archive_root = tmp_path / "archive"
    result_dir = archive_root / "logs_results_contract"
    operator_dir = result_dir / "add"
    operator_dir.mkdir(parents=True)
    config = types.SimpleNamespace(
        all_op_ids={"add"},
        base_data=None,
        dump_output=True,
        output_dir=result_dir,
        quick=False,
        skip_cpu_tests=set(),
        speedup_source="origin",
        timeout=30,
    )
    monkeypatch.setattr(run_tests, "ROOT", project)
    monkeypatch.setattr(run_tests, "CFG", config)

    # Exercise the real log writer so this contract detects filename changes.
    for flavor in ("accuracy", "performance"):
        command = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
            "import sys; "
            f"print('{flavor} stdout'); "
            f"print('{flavor} stderr', file=sys.stderr)"
        )
        assert (
            run_tests.run_cmd(
                "add",
                command,
                cwd=project,
                env=os.environ.copy(),
                timeout=30,
                flavor=flavor,
            )
            == 0
        )

    def record_test_result(_op, _cmd, **kwargs):
        if kwargs["flavor"] == "accuracy":
            _write_json(
                tests_dir / "accuracy_add.json",
                {
                    "test_add[value]": {
                        "result": "passed",
                        "params": {"dtype": "float32"},
                    }
                },
            )
        else:
            _write_json(
                benchmark_dir / "benchmark_add.json",
                {
                    "add": {
                        "result": "passed",
                        "test_case": "test_add_perf",
                        "details": [
                            {
                                "dtype": "torch.float32",
                                "result": [
                                    {
                                        "shape_detail": [2, 2],
                                        "latency_base": 2.0,
                                        "latency": 1.0,
                                        "speedup": 2.0,
                                    }
                                ],
                            }
                        ],
                    }
                },
            )
        return 0

    monkeypatch.setattr(run_tests, "run_cmd", record_test_result)
    accuracy = run_tests.run_accuracy_q(0, "add")
    performance = run_tests.run_benchmark_q(0, "add")
    final_data = {
        "timestamp": "2026-11-20 00:00:00",
        "total_duration": "0:00:01",
        "total_ops": 1,
        "accuracy_counts": run_tests.count_op_statuses(
            {"add": {"accuracy": accuracy}}, "accuracy"
        ),
        "performance_counts": run_tests.count_op_statuses(
            {"add": {"performance": performance}}, "performance"
        ),
        "env": {"vendor": "ascend", "device": "Ascend 910B"},
        "result": {
            "add": {
                "accuracy": accuracy,
                "performance": performance,
                "customized": True,
                "labels": ["tle"],
            }
        },
    }
    run_tests._write_final_summary(final_data, [], cleanup_workers=True)

    # The workflow enriches the scheduler summary before packaging it.
    monkeypatch.setattr(add_labels, "ROOT", project)
    with mock.patch.object(
        sys,
        "argv",
        ["flagos-add-labels", str(project), str(result_dir)],
    ):
        add_labels.main()

    return archive_root


def _find_summary(archive_root):
    candidates = [
        path
        for path in archive_root.rglob("summary.json")
        if len(path.relative_to(archive_root).parts) <= 2
    ]
    assert len(candidates) == 1, (
        "op-monitor requires exactly one summary.json at the archive root or "
        "inside one top-level result directory"
    )
    return candidates[0]


def test_weekly_workflow_preserves_op_monitor_archive_inputs():
    workflow_path = _REPOSITORY_ROOT / ".github/workflows/ops-test.yaml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test-container"]["steps"]

    run_step = next(step for step in steps if step.get("name") == "Run ops tests")
    assert "--dump-output" in run_step["run"], (
        "op-monitor log views require weekly runs to retain operator stdout/stderr"
    )
    assert 'flagos-add-labels "${OUTPUT_DIR}"' in run_step["run"]

    package_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Package results"
    )
    publish_index = next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Publish results to op-monitor"
    )
    package_script = steps[package_index]["run"]
    assert 'tar -czf "${ARCHIVE}"' in package_script
    assert (
        '-C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"'
        in package_script
    )
    assert package_index < publish_index


def test_generated_archive_satisfies_op_monitor_contract(tmp_path, monkeypatch):
    archive_root = _generate_result_directory(tmp_path, monkeypatch)
    summary_path = _find_summary(archive_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert isinstance(summary.get("env"), dict)
    assert summary["env"]
    assert isinstance(summary.get("result"), dict)
    assert summary["result"]

    operator_name = "add"
    operator = summary["result"][operator_name]
    accuracy = operator["accuracy"]
    assert _ACCURACY_FIELDS <= accuracy.keys()
    assert accuracy["status"].lower() == "passed"
    count_fields = _ACCURACY_FIELDS - {"status", "exit_code"}
    assert all(isinstance(accuracy[field], int) for field in count_fields)
    assert accuracy["exit_code"] is None or isinstance(accuracy["exit_code"], int)

    performance = operator.get("performance", operator.get("benchmark"))
    assert isinstance(performance, dict)
    assert isinstance(performance.get("status"), str)
    assert isinstance(performance.get("data"), dict)
    assert performance["data"]
    for dtype_result in performance["data"].values():
        assert "speedup" in dtype_result
        assert isinstance(dtype_result["speedup"], int | float)

    assert isinstance(operator.get("customized"), bool)
    assert isinstance(operator.get("labels"), list)
    assert all(isinstance(label, str) for label in operator["labels"])
    assert {"binary", "tle"} <= set(operator["labels"])

    operator_dir = summary_path.parent / operator_name
    assert operator_dir.is_dir()
    assert _LOG_FILES <= {path.name for path in operator_dir.glob("*.log")}
    assert (operator_dir / "performance_result.json").is_file(), (
        "op-monitor requires performance_result.json as the preferred source "
        "of shape-level details when performance data is present"
    )

    # These files are intentionally outside the downstream contract. Their
    # removal or size changes must not affect op-monitor ingestion.
    assert not list(summary_path.parent.glob("summary[0-9]*.json"))
