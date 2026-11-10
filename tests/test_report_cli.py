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
from flagos_optest.performance_report import parse_perf_data

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
    def test_comparison_and_web_preserve_speedup_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side, ratio, source in (
                ("before", 1, "origin"),
                ("after", 2, "nv_baseline"),
            ):
                folder = root / side
                folder.mkdir()
                record = {"speedup": ratio, "speedup_source": source}
                (folder / "summary.json").write_text(
                    json.dumps(
                        {
                            "env": {"os_name": "Linux", "os_release": "test"},
                            "result": {
                                "add": {
                                    "accuracy": {
                                        "status": "Failed",
                                        "passed": 0,
                                        "failed": 1,
                                        "skipped": 0,
                                        "details": {"failed": {"case failed": []}},
                                    },
                                    "performance": {
                                        "status": "Passed",
                                        "data": {
                                            "fp32": {
                                                **record,
                                                "details": {
                                                    "[1]": {
                                                        **record,
                                                        "base": 1,
                                                        "gems": 1,
                                                    }
                                                },
                                            }
                                        },
                                    },
                                }
                            },
                        }
                    )
                )
            report = self.run_report(
                "psum_text", root, "--format", "markdown", "--single", "--compare"
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertEqual(
                report.stdout.count("1.000 (origin) / 2.000 (nv_baseline)"), 2
            )
            self.assertNotIn("1.000 -> 2.000", report.stdout)
            before = root / "before/summary.json"
            data = json.loads(before.read_text())
            dtype = data["result"]["add"]["performance"]["data"]["fp32"]
            dtype["speedup_source"] = "nv_baseline"
            dtype["details"]["[1]"]["speedup_source"] = "nv_baseline"
            before.write_text(json.dumps(data))
            report = self.run_report(
                "psum_text", root, "--format", "markdown", "--single", "--compare"
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertEqual(
                report.stdout.count("1.000 (nv_baseline) / 2.000 (nv_baseline)"), 2
            )
            self.assertNotIn("1.000 -> 2.000", report.stdout)
            (root / "after").rename(root / "H800-20261110")
            (root / "before/summary.json").unlink()
            report = self.run_report(
                "psum_web", root, "--raw-dir", root, "--output-dir", root / "web"
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            web = json.loads((root / "web/H800.json").read_text())
            self.assertEqual(
                web["operators"][0]["note"], "case failed / Speedup vs NVIDIA baseline"
            )

    def test_mixed_speedups_keep_separate_denominators(self):
        native = {"shape_detail": [2], "speedup": 10, "latency": 0.1}
        external = {"shape_detail": [1], "compared_speedup": 2, "latency": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            raw = results / "benchmark.json"
            for skip_reason in ({}, {"native_baseline_skip_reason": "unavailable"}):
                with self.subTest(skip_reason=skip_reason):
                    raw.write_text(
                        json.dumps(
                            {
                                "add": {
                                    "result": "passed",
                                    "details": [
                                        {
                                            "dtype": "torch.float32",
                                            "result": [external, native],
                                            **skip_reason,
                                        },
                                        {
                                            "dtype": "torch.bfloat16",
                                            "result": [external],
                                        },
                                        {"dtype": "torch.float16", "result": [native]},
                                    ],
                                }
                            }
                        )
                    )
                    perf = parse_perf_data("add", raw, "nv_baseline")
                    fp32 = perf["data"]["fp32"]
                    self.assertIsNone(fp32["speedup"])
                    self.assertEqual(fp32["speedup_source"], "mixed")
                    self.assertEqual(
                        fp32["speedup_by_source"], {"nv_baseline": 2, "origin": 10}
                    )
                    self.assertEqual(fp32["details"]["[2]"]["speedup_source"], "origin")

            accuracy = {"status": "Passed", "passed": 1, "failed": 0, "skipped": 0}
            for dtype_data in (
                perf["data"],
                {k: v for k, v in perf["data"].items() if k != "fp32"},
            ):
                (results / "summary.json").write_text(
                    json.dumps(
                        {
                            "env": {"os_name": "Linux", "os_release": "test"},
                            "result": {
                                "add": {
                                    "accuracy": accuracy,
                                    "performance": {
                                        "status": "Passed",
                                        "data": dtype_data,
                                    },
                                }
                            },
                        }
                    )
                )
                for script, output in (
                    ("psum_text", "report.csv"),
                    ("psum_html", "report.html"),
                ):
                    report = self.run_report(script, results, "-o", output)
                    self.assertEqual(report.returncode, 0, report.stderr)
                    content = (results / output).read_text()
                    self.assertIn("Mixed speedup sources", content)
                    self.assertNotIn("6.000", content)
                report = self.run_report(
                    "psum_text",
                    results,
                    "--format",
                    "markdown",
                    "--single",
                    "-o",
                    "report.md",
                )
                self.assertEqual(report.returncode, 0, report.stderr)
                self.assertIn(
                    "Mixed speedup sources", (results / "report.md").read_text()
                )

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
