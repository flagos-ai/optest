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
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from flagos_optest import add_labels


PACKAGE_DIR = Path(add_labels.__file__).resolve().parent


class AddLabelsDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.original_directory = Path.cwd()
        self.original_root = add_labels.ROOT

    def tearDown(self):
        os.chdir(self.original_directory)
        add_labels.ROOT = self.original_root

    def create_project(self, parent):
        project = parent / "target"
        results = project / "results"
        (project / "conf").mkdir(parents=True)
        results.mkdir()
        (project / "conf" / "operators.yaml").write_text(
            "ops:\n  - id: add\n    labels: [binary]\n",
            encoding="utf-8",
        )
        (results / "summary.json").write_text(
            json.dumps({"result": {"add": {}}}),
            encoding="utf-8",
        )
        return project, results

    def assert_labels_added(self, summary_path):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["result"]["add"]["labels"], ["binary"])

    def test_defaults_to_current_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            project, results = self.create_project(Path(directory))
            os.chdir(project)
            with mock.patch.object(sys, "argv", ["flagos-add-labels", "results"]):
                add_labels.main()

            self.assertEqual(add_labels.ROOT, project.resolve())
            self.assertEqual(Path.cwd(), project.resolve())
            self.assert_labels_added(results / "summary.json")

    def test_uses_explicit_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            caller = parent / "caller"
            caller.mkdir()
            project, results = self.create_project(parent)
            os.chdir(caller)
            with mock.patch.object(
                sys,
                "argv",
                ["flagos-add-labels", "../target", "results"],
            ):
                add_labels.main()

            self.assertEqual(add_labels.ROOT, project.resolve())
            self.assertEqual(Path.cwd(), project.resolve())
            self.assert_labels_added(results / "summary.json")

    def test_rejects_missing_or_non_directory_project_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            regular_file = Path(directory) / "file"
            regular_file.touch()

            with self.assertRaises(argparse.ArgumentTypeError):
                add_labels.project_directory(Path(directory) / "missing")
            with self.assertRaises(argparse.ArgumentTypeError):
                add_labels.project_directory(regular_file)


class ReportDirectoryTest(unittest.TestCase):
    def create_summary(self, directory):
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "timestamp": "2026-01-01T00:00:00",
                    "env": {"os_name": "Linux", "os_release": "test"},
                    "result": {},
                }
            ),
            encoding="utf-8",
        )

    def run_report(self, script, cwd, *args):
        return subprocess.run(
            [sys.executable, str(PACKAGE_DIR / script), *map(str, args)],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_text_and_html_default_to_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            self.create_summary(results)

            text_result = self.run_report("psum_text", results, "-o", "result.csv")
            html_result = self.run_report("psum_html", results, "-o", "result.html")

            self.assertEqual(text_result.returncode, 0, text_result.stderr)
            self.assertEqual(html_result.returncode, 0, html_result.stderr)
            self.assertTrue((results / "result.csv").is_file())
            self.assertTrue((results / "result.html").is_file())

    def test_text_and_html_use_explicit_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            caller = parent / "caller"
            results = parent / "results"
            caller.mkdir()
            results.mkdir()
            self.create_summary(results)

            text_output = caller / "result.csv"
            html_output = caller / "result.html"
            text_result = self.run_report(
                "psum_text", caller, results, "-o", text_output
            )
            html_result = self.run_report(
                "psum_html", caller, results, "-o", html_output
            )

            self.assertEqual(text_result.returncode, 0, text_result.stderr)
            self.assertEqual(html_result.returncode, 0, html_result.stderr)
            self.assertTrue(text_output.is_file())
            self.assertTrue(html_output.is_file())


if __name__ == "__main__":
    unittest.main()
