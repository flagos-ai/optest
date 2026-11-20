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

Worker checkpoints (the current ``summary<N>.jsonl`` format and legacy
``summary<N>.json`` files), ``result.html``, and ``result.csv`` are deliberately
not part of this contract. Op-monitor does not read them.
"""

import json
import os
import shlex
import subprocess
import sys
import tarfile
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


def _package_script():
    workflow_path = _REPOSITORY_ROOT / ".github/workflows/ops-test.yaml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test-container"]["steps"]
    script = next(
        step["run"] for step in steps if step.get("name") == "Package results"
    )
    archive_assignment = (
        'ARCHIVE="${{ matrix.output_base_dir }}/${REPOSITORY_NAME}/'
        '${REPOSITORY_NAME}-${{ matrix.name }}-${DATE}${TIME_SUFFIX}.tar.gz"'
    )
    assert archive_assignment in script
    return script.replace(archive_assignment, 'ARCHIVE="${TEST_ARCHIVE}"')


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
    assert '-C "${OUTPUT_PARENT}" "${OUTPUT_NAME}"' in package_script
    assert '[[ "${OUTPUT_NAME}" != logs_results_*' in package_script
    archive_created = 'echo "[Package results] Created results archive: ${ARCHIVE}"'
    cleanup = 'rm -rf -- "${OUTPUT_PARENT}"'
    assert package_script.index('tar -czf "${ARCHIVE}"') < package_script.index(
        archive_created
    )
    assert package_script.index(archive_created) < package_script.index(cleanup)
    assert package_index < publish_index


def test_package_results_removes_unpacked_directory_after_success(tmp_path):
    output_parent = tmp_path / "2026-11-20_120000"
    output_dir = output_parent / "logs_results_contract"
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (output_parent / "runner.tmp").write_text("temporary\n", encoding="utf-8")
    archive = tmp_path / "results.tar.gz"
    github_env = tmp_path / "github-env"

    completed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _package_script()],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "DATE": "2026-11-20",
            "GITHUB_ENV": str(github_env),
            "OUTPUT_DIR": str(output_dir),
            "TEST_ARCHIVE": str(archive),
            "TIME_SUFFIX": "_120000",
        },
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not output_parent.exists()
    assert github_env.read_text(encoding="utf-8") == f"ARCHIVE={archive}\n"
    command_log = (
        f'[Package results] Command: tar -czf "{archive}" '
        f'-C "{output_parent}" "logs_results_contract"'
    )
    archive_log = f"[Package results] Created results archive: {archive}"
    cleanup_log = (
        f"[Package results] Removed unpacked result directory: {output_parent}"
    )
    assert command_log in completed.stdout
    assert archive_log in completed.stdout
    assert cleanup_log in completed.stdout
    assert completed.stdout.index(command_log) < completed.stdout.index(archive_log)
    assert completed.stdout.index(archive_log) < completed.stdout.index(cleanup_log)
    with tarfile.open(archive, "r:gz") as packaged:
        assert {
            "logs_results_contract",
            "logs_results_contract/summary.json",
        } <= set(packaged.getnames())


def test_package_results_preserves_directory_when_compression_fails(tmp_path):
    output_parent = tmp_path / "2026-11-20_120000"
    output_dir = output_parent / "logs_results_contract"
    output_dir.mkdir(parents=True)
    summary = output_dir / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tar = fake_bin / "tar"
    fake_tar.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_tar.chmod(0o755)

    completed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _package_script()],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "DATE": "2026-11-20",
            "GITHUB_ENV": str(tmp_path / "github-env"),
            "OUTPUT_DIR": str(output_dir),
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "TEST_ARCHIVE": str(tmp_path / "results.tar.gz"),
            "TIME_SUFFIX": "_120000",
        },
        text=True,
    )

    assert completed.returncode == 1
    assert summary.read_text(encoding="utf-8") == "{}\n"
    assert "[Package results] Command: tar -czf" in completed.stdout
    assert "[Package results] Created results archive:" not in completed.stdout
    assert (
        "[Package results] Removed unpacked result directory:" not in completed.stdout
    )


def test_package_results_refuses_unexpected_parent_directory(tmp_path):
    output_parent = tmp_path / "unexpected"
    output_dir = output_parent / "logs_results_contract"
    output_dir.mkdir(parents=True)
    summary = output_dir / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "results.tar.gz"

    completed = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _package_script()],
        check=False,
        capture_output=True,
        env={
            **os.environ,
            "DATE": "2026-11-20",
            "GITHUB_ENV": str(tmp_path / "github-env"),
            "OUTPUT_DIR": str(output_dir),
            "TEST_ARCHIVE": str(archive),
            "TIME_SUFFIX": "_120000",
        },
        text=True,
    )

    assert completed.returncode == 1
    assert "Refusing to remove unexpected result path" in completed.stdout
    assert summary.read_text(encoding="utf-8") == "{}\n"
    assert not archive.exists()


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
    assert accuracy.get("details_mode") == "external"
    assert "details" not in accuracy
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
        assert "details" not in dtype_result
        assert dtype_result["base"] == 2.0
        assert dtype_result["gems"] == 1.0
        assert dtype_result["shape_count"] == 1
    assert performance.get("details_mode") == "external"
    assert performance.get("speedup_source") == "origin"

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
    assert not list(summary_path.parent.glob("summary[0-9]*.jsonl"))
