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

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class FlagTrainWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repos = yaml.safe_load(
            (ROOT / ".github/configs/repos.yaml").read_text(encoding="utf-8")
        )
        cls.workflow = yaml.safe_load(
            (ROOT / ".github/workflows/ops-test.yaml").read_text(encoding="utf-8")
        )

    def test_flagtrain_images_and_enabled_backends(self):
        config = self.repos["repos"]["FlagTrain"]
        self.assertEqual(config["default_branch"], "main")
        self.assertNotIn("vendor_pip_args", config)
        self.assertEqual(
            config["container_images"],
            {
                "H800": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-generic-13.3:2.2.0-0.2.3",
                "BW1000": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-hygon-dtk26.04:2.2.0-0.2.3",
                "BI-V150": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-iluvatar-corex4.5.0:2.2.0-0.2.3",
                "P800": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-kunlunxin-xre5.37.1:2.2.0-0.2.3",
                "910B": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-ascend-cann9.0.0:2.2.0-0.2.3",
                "C550": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-metax-maca3.8.1.3:2.2.0-0.2.3",
                "MTT-S5000": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-mthreads-musa5.2.0:2.2.0-0.2.3",
                "S60": "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-enflame-tops1.10.6:2.2.0-0.2.3",
            },
        )
        enabled = {
            name for name, backend in config["backends"].items() if backend["enabled"]
        }
        self.assertEqual(
            enabled,
            {"H800", "BW1000", "BI-V150", "P800", "910B", "C550", "MTT-S5000", "S60"},
        )
        self.assertNotIn("deps", config)

        workflow_text = (ROOT / ".github/workflows/ops-test.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("          - FlagTrain\n", workflow_text)

    def test_gpu_probe_precedes_vendor_dependency_install(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        names = [step.get("name") for step in steps]
        self.assertLess(
            names.index("Configure package index and install optest"),
            names.index("Check GPU availability"),
        )
        self.assertLess(
            names.index("Check GPU availability"),
            names.index("Install target repository and dependencies"),
        )

        install = next(
            step
            for step in steps
            if step.get("name") == "Install target repository and dependencies"
        )["run"]
        self.assertIn(
            'uv pip install --python "${ACTIVE_PYTHON}" --editable "${TARGET_PROJECT_SPEC}"',
            install,
        )

    def test_matrix_does_not_encode_target_dependency_policy(self):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            environment = {
                **os.environ,
                "REPO_INPUT": "FlagTrain",
                "REF_INPUT": "",
                "VENDORS_INPUT": "",
                "OPS_INPUT": "",
                "UPLOAD_LOG_INPUT": "",
                "SEND_FEISHU_INPUT": "",
                "GITHUB_OUTPUT": str(output),
            }
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            matrix_line = next(
                line
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.startswith("matrix_container=")
            )
            matrix = json.loads(matrix_line.removeprefix("matrix_container="))
            self.assertTrue(matrix["include"])
            for backend in matrix["include"]:
                self.assertNotIn("vendor_pip_args", backend)

    def test_optest_install_retries_package_indexes(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Configure package index and install optest"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv = root / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then echo "uv 0.13.6"; exit 0; fi\n'
                'printf "[fake uv index=%s] %s\\n" "$UV_DEFAULT_INDEX" "$*"\n'
                'case "$UV_DEFAULT_INDEX" in *tuna*) exit 23 ;; esac\n',
                encoding="utf-8",
            )
            uv.chmod(0o755)
            github_env = root / "github-env"
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "GITHUB_ENV": str(github_env),
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "UV_DEFAULT_INDEX": "",
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "optest source installation: failed with "
                "https://pypi.tuna.tsinghua.edu.cn/simple",
                completed.stdout,
            )
            self.assertIn(
                "optest source installation: selected "
                "https://mirrors.aliyun.com/pypi/simple/",
                completed.stdout,
            )
            self.assertIn(
                "UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/",
                github_env.read_text(encoding="utf-8"),
            )

    def test_target_install_logs_dependency_sources_commands_and_results(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Install target repository and dependencies"
        )
        replacements = {
            "${{ matrix.deps }}": "",
            "${{ matrix.name }}": "H800",
        }
        for expression, value in replacements.items():
            script = script.replace(expression, value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "pyproject.toml").write_text(
                "[project]\n"
                'name = "target"\n'
                'version = "0.0.0"\n'
                "[project.optional-dependencies]\n"
                'test = ["pytest"]\n'
                'nvidia = ["deepspeed==0.19.8"]\n'
                "[tool.uv]\n"
                'exclude-dependencies = ["torch", "triton"]\n',
                encoding="utf-8",
            )
            commands = {
                "python3": f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',
                "uv": (
                    "#!/bin/sh\n"
                    'if [ "$1" = "--version" ]; then echo "uv 0.13.6"; exit 0; fi\n'
                    'case "$UV_DEFAULT_INDEX" in *tuna*) exit 23 ;; esac\n'
                    "printf '[fake uv cwd=%s] %s\\n' \"$PWD\" \"$*\"\n"
                ),
            }
            for name, content in commands.items():
                command = root / name
                command.write_text(content, encoding="utf-8")
                command.chmod(0o755)

            github_env = root / "github-env"
            common_environment = {
                **os.environ,
                "FLAGOS_GPU_VENDOR": "nvidia",
                "GITHUB_ENV": str(github_env),
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "UV_DEFAULT_INDEX": "",
            }
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=root,
                check=False,
                capture_output=True,
                env={
                    **common_environment,
                    "REPOSITORY_NAME": "FlagTrain",
                },
                text=True,
            )

            (target / "pyproject.toml").write_text(
                "[project]\n"
                'name = "target"\n'
                'version = "0.0.0"\n'
                "[project.optional-dependencies]\n"
                'test = ["pytest"]\n',
                encoding="utf-8",
            )
            test_only_completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=root,
                check=False,
                capture_output=True,
                env={
                    **common_environment,
                    "REPOSITORY_NAME": "FlagGems",
                },
                text=True,
            )
            github_env_contents = github_env.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        expected = [
            "repos.FlagTrain.deps.default + repos.FlagTrain.deps.H800",
            "Detected explicit backend dependencies: <empty>",
            "No explicit backend dependencies configured; skipping.",
            "Target project source: ./target/pyproject.toml",
            "Detected test extra: true",
            "Detected GPU vendor from flagos-gpu-check: nvidia",
            "[project.optional-dependencies].nvidia",
            "Detected vendor extra: true",
            "Exclusion policy source: [tool.uv].exclude-dependencies",
            "Detected excluded dependencies: torch, triton",
            "Selected target extras: test,nvidia",
            "Target install specification: .[test,nvidia]",
            "target repository uv installation: failed with https://pypi.tuna.tsinghua.edu.cn/simple",
            "target repository uv installation: selected https://mirrors.aliyun.com/pypi/simple/",
            "Starting: target repository uv installation via https://mirrors.aliyun.com/pypi/simple/",
            "pip install --python",
            "--editable .[test,nvidia]",
            "Finished: target repository uv installation via https://mirrors.aliyun.com/pypi/simple/ (exit_code=0)",
        ]
        for entry in expected:
            with self.subTest(entry=entry):
                self.assertIn(entry, output)

        self.assertEqual(
            test_only_completed.returncode, 0, test_only_completed.stderr
        )
        self.assertIn(
            "Detected vendor extra: false", test_only_completed.stdout
        )
        self.assertIn(
            "Selected target extras: test", test_only_completed.stdout
        )
        self.assertIn(
            "--editable .[test]", test_only_completed.stdout
        )
        self.assertIn(
            "UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/",
            github_env_contents,
        )

    def test_gpu_check_script_exports_metadata_with_unknown_values(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Check GPU availability"
        )
        replacements = {
            "${{ matrix.gpu_check_script }}": "fake-gpu-check",
            "${{ matrix.force_test_gpus }}": "false",
            "${{ matrix.test_gpus }}": "0,1",
            "${{ matrix.vendor }}": "Nvidia",
        }
        for expression, value in replacements.items():
            script = script.replace(expression, value)

        result = {
            "vendor": "nvidia",
            "device": "H800",
            "driver": None,
            "runtime": "cuda",
            "runtime_version": "13.3.0",
            "available_gpus": "0,1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gpu_check = root / "fake-gpu-check"
            gpu_check.write_text(
                "#!/bin/sh\nprintf '%s\\n' '" + json.dumps(result) + "'\n",
                encoding="utf-8",
            )
            gpu_check.chmod(0o755)
            github_env = root / "github-env"
            environment = {
                **os.environ,
                "GITHUB_ENV": str(github_env),
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
            }
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "[GPU check] Exported FLAGOS_GPU_DRIVER=<unknown>",
                completed.stdout,
            )
            exported = github_env.read_text(encoding="utf-8").splitlines()
            self.assertIn("AVAILABLE_GPUS=0,1", exported)
            self.assertIn("FLAGOS_GPU_VENDOR=nvidia", exported)
            self.assertIn("FLAGOS_GPU_DRIVER=", exported)


if __name__ == "__main__":
    unittest.main()
