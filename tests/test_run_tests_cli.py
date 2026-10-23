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
import os
import shlex
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from flagos_optest import run_tests


class ProjectDirectoryTest(unittest.TestCase):
    def run_until_operator_selection(self, argv, start_directory):
        original_directory = Path.cwd()
        original_root = run_tests.ROOT
        try:
            os.chdir(start_directory)
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                run_tests, "probe_env"
            ), mock.patch.object(run_tests, "get_ops_to_test", return_value=[]):
                with mock.patch.object(run_tests.signal, "signal"):
                    with self.assertRaisesRegex(SystemExit, "1"):
                        run_tests.main()
            return run_tests.ROOT, Path.cwd(), run_tests.CFG.base_data
        finally:
            os.chdir(original_directory)
            run_tests.ROOT = original_root

    def test_defaults_to_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root, working_directory, _ = self.run_until_operator_selection(
                ["flagos-optest"], directory
            )

        self.assertEqual(root, Path(directory).resolve())
        self.assertEqual(working_directory, Path(directory).resolve())

    def test_uses_explicit_directory_for_execution_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as parent:
            parent_path = Path(parent)
            caller = parent_path / "caller"
            target = parent_path / "target"
            caller.mkdir()
            target.mkdir()

            root, working_directory, base_data = self.run_until_operator_selection(
                ["flagos-optest", "../target", "--base-data", "baseline.json"],
                caller,
            )

        self.assertEqual(root, target.resolve())
        self.assertEqual(working_directory, target.resolve())
        self.assertEqual(base_data, str(target.resolve() / "baseline.json"))

    def test_reads_relative_operator_list_from_selected_directory(self):
        original_directory = Path.cwd()
        original_root = run_tests.ROOT
        try:
            with tempfile.TemporaryDirectory() as parent:
                parent_path = Path(parent)
                caller = parent_path / "caller"
                target = parent_path / "target"
                caller.mkdir()
                (target / "conf").mkdir(parents=True)
                (target / "conf" / "operators.yaml").write_text(
                    "ops: []\n", encoding="utf-8"
                )
                (target / "ops.txt").write_text("add\n", encoding="utf-8")
                os.chdir(caller)

                with mock.patch.object(
                    sys,
                    "argv",
                    ["flagos-optest", "../target", "--op-list-file", "ops.txt"],
                ), mock.patch.object(run_tests.signal, "signal"), mock.patch.object(
                    run_tests, "probe_env"
                ), mock.patch.object(
                    run_tests, "collect_marks", side_effect=RuntimeError("stop")
                ) as collect_marks:
                    with self.assertRaisesRegex(RuntimeError, "stop"):
                        run_tests.main()

                collect_marks.assert_called_once_with(["add"])
        finally:
            os.chdir(original_directory)
            run_tests.ROOT = original_root

    def test_creates_relative_output_in_selected_directory(self):
        original_directory = Path.cwd()
        original_root = run_tests.ROOT
        try:
            with tempfile.TemporaryDirectory() as parent:
                target = Path(parent) / "target"
                target.mkdir()
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        "flagos-optest",
                        str(target),
                        "--ops",
                        "add",
                        "--output-dir",
                        "results",
                    ],
                ), mock.patch.object(run_tests.signal, "signal"), mock.patch.object(
                    run_tests, "probe_env"
                ), mock.patch.object(
                    run_tests, "get_ops_to_test", return_value=["add"]
                ), mock.patch.object(
                    run_tests, "collect_marks", return_value=({"add"}, {"add"})
                ), mock.patch.object(
                    run_tests, "Process", return_value=mock.MagicMock()
                ), mock.patch.object(
                    run_tests, "LiveDisplay", return_value=mock.MagicMock()
                ), mock.patch.object(
                    run_tests, "display_loop"
                ), mock.patch.object(
                    run_tests, "WORKER_PROCESSES", []
                ), mock.patch.object(
                    run_tests, "ENV_INFO", {}
                ):
                    run_tests.main()

                self.assertTrue((target / "results" / "summary.json").is_file())
        finally:
            os.chdir(original_directory)
            run_tests.ROOT = original_root

    def test_rejects_missing_or_non_directory_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            regular_file = Path(directory) / "file"
            regular_file.touch()

            with self.assertRaises(argparse.ArgumentTypeError):
                run_tests.project_directory(Path(directory) / "missing")
            with self.assertRaises(argparse.ArgumentTypeError):
                run_tests.project_directory(regular_file)

    def test_test_processes_run_from_selected_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = types.SimpleNamespace(
                all_op_ids={"add"},
                base_data=None,
                dump_output=False,
                output_dir=root / "results",
                quick=False,
                skip_cpu_tests=[],
            )
            with mock.patch.object(run_tests, "ROOT", root), mock.patch.object(
                run_tests, "CFG", config
            ), mock.patch.object(run_tests, "get_env", return_value={}):
                with mock.patch.object(run_tests, "run_cmd", return_value=1) as runner:
                    run_tests.run_accuracy_q(0, "add")
                    accuracy_call = runner.call_args
                    accuracy_command = shlex.split(accuracy_call.args[1])
                    self.assertEqual(accuracy_call.kwargs["cwd"], root)
                    self.assertIn(str(root / "tests"), accuracy_command)
                    self.assertIn(
                        str(root / "tests" / "accuracy_add.json"),
                        accuracy_command,
                    )

                    runner.reset_mock()
                    run_tests.run_benchmark_q(0, "add")
                    benchmark_call = runner.call_args
                    benchmark_command = shlex.split(benchmark_call.args[1])
                    self.assertEqual(benchmark_call.kwargs["cwd"], root)
                    self.assertIn(str(root / "benchmark"), benchmark_command)
                    self.assertIn(
                        str(root / "benchmark" / "benchmark_add.json"),
                        benchmark_command,
                    )


if __name__ == "__main__":
    unittest.main()
