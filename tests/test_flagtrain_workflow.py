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
from shutil import copytree

import yaml

from flagos_optest import run_tests

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

    def build_matrix(
        self,
        repo,
        *,
        vendors="",
        cwd=ROOT,
    ):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )
        with tempfile.TemporaryDirectory() as output_directory:
            output = Path(output_directory) / "github-output"
            environment = {
                **os.environ,
                "REPO_INPUT": repo,
                "REF_INPUT": "",
                "VENDORS_INPUT": vendors,
                "OPS_INPUT": "",
                "EXCLUDE_OPS_INPUT": "",
                "UPLOAD_LOG_INPUT": "",
                "SEND_FEISHU_INPUT": "",
                "GITHUB_OUTPUT": str(output),
            }
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=cwd,
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )
            if completed.returncode != 0:
                self.fail(completed.stderr)
            matrix_line = next(
                line
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.startswith("matrix_container=")
            )
        matrix = json.loads(matrix_line.removeprefix("matrix_container="))
        return matrix

    @staticmethod
    def upgrade_deps(backend):
        return json.loads(backend["upgrade_deps"])

    @classmethod
    def upgrade_dep(cls, backend, package_name):
        return next(
            dependency
            for dependency in cls.upgrade_deps(backend)
            if dependency["package_name"] == package_name
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
                "PPU-ZW810E": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0",
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
            {
                "H800",
                "BW1000",
                "BI-V150",
                "910B",
                "C550",
                "MTT-S5000",
                "PPU-ZW810E",
                "P800",
                "S60",
            },
        )
        self.assertNotIn("deps", config)

        workflow_text = (ROOT / ".github/workflows/ops-test.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("          - FlagTrain\n", workflow_text)

    def test_flaggems_runtime_images_and_shared_910b_flagtree(self):
        config = self.repos["repos"]["FlagGems"]
        self.assertEqual(
            self.repos["defaults"]["deps"]["flag-gems"],
            {
                "display_name": "FlagGems",
                "priority": 200,
                "version": "5.5.0.dev20261123",
                "index_url": (
                    "https://resource.flagos.net/repository/flagos-pypi-daily/simple"
                ),
                "no_deps": True,
            },
        )
        self.assertEqual(
            config["container_images"],
            {
                "PPU-ZW810E": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0",
                "W7900D": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-amd-rocm7.2.4:2.3.0",
            },
        )
        self.assertFalse(config["deps"]["default"]["flag-gems"]["enabled"])
        self.assertNotIn("910B", config["container_images"])
        self.assertNotIn("910B", config["deps"])
        self.assertEqual(
            [
                repo_name
                for repo_name, repo_config in self.repos["repos"].items()
                if repo_config["backends"]["MLU590-M9DE"]["enabled"]
            ],
            ["FlagGems"],
        )

        flaggems = self.build_matrix(
            "FlagGems",
            vendors="Nvidia,Ascend,THead,Cambricon,AMD",
        )
        flaggems_by_name = {backend["name"]: backend for backend in flaggems["include"]}
        expected_images = {
            "910B": "harbor.baai.ac.cn/flagos-app/vllm0.24.0-ascend-cann9.0.0:2.2.0-0.3.0",
            "PPU-ZW810E": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0",
            "MLU590-M9DE": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.7.2:2.3.0",
            "W7900D": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-amd-rocm7.2.4:2.3.0",
        }
        for backend_name, image in expected_images.items():
            with self.subTest(backend=backend_name):
                self.assertEqual(
                    flaggems_by_name[backend_name]["container_image"], image
                )
        self.assertEqual(
            [
                dependency["package_name"]
                for dependency in self.upgrade_deps(flaggems_by_name["910B"])
            ],
            ["flagtree"],
        )
        self.assertEqual(self.upgrade_deps(flaggems_by_name["MLU590-M9DE"]), [])
        self.assertEqual(
            [
                dependency["package_name"]
                for dependency in self.upgrade_deps(flaggems_by_name["H800"])
            ],
            ["flagtree-nvidia"],
        )
        for backend in flaggems_by_name.values():
            with self.subTest(flaggems_backend=backend["name"]):
                self.assertNotIn(
                    "flag-gems",
                    {
                        dependency["package_name"]
                        for dependency in self.upgrade_deps(backend)
                    },
                )

        flaggems_vllm = self.build_matrix(
            "FlagGems-vllm",
            vendors="Ascend,THead",
        )
        flaggems_vllm_by_name = {
            backend["name"]: backend for backend in flaggems_vllm["include"]
        }
        self.assertEqual(
            [
                dependency["package_name"]
                for dependency in self.upgrade_deps(flaggems_vllm_by_name["910B"])
            ],
            ["flagtree", "flag-gems"],
        )
        self.assertEqual(
            flaggems_vllm_by_name["PPU-ZW810E"]["container_image"],
            "harbor.baai.ac.cn/flagos-app/vllm0.32.0-thead-ppu2.1.0:2.3.0-0.5.0.dev0_g814b6d3.d20261117",
        )

    def test_configured_repositories_have_runtime_probes(self):
        configured_repos = set(self.repos["repos"])
        probed_repos = {display_name for _, display_name in run_tests._KNOWN_LIBS}

        self.assertLessEqual(configured_repos, probed_repos)

    def test_flagattention_enables_s60_backend(self):
        config = self.repos["repos"]["FlagAttention"]

        self.assertTrue(config["backends"]["S60"]["enabled"])

    def test_backend_gpu_configuration_uses_dynamic_detection_by_default(self):
        self.assertNotIn("server", self.repos)

        backend_dir = ROOT / ".github/configs/backends"
        backend_configs = {
            path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in backend_dir.glob("*.yaml")
        }
        for name, config in backend_configs.items():
            with self.subTest(backend=name):
                self.assertNotIn("use_container", config)
                self.assertNotIn("gpu_check_script", config)
                self.assertNotIn("force_test_gpus", config)
                if name == "TX8110":
                    self.assertEqual(
                        config.get("test_gpus"),
                        "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                    )
                else:
                    self.assertNotIn("test_gpus", config)

    def test_backend_flagtree_configuration(self):
        backend_dir = ROOT / ".github/configs/backends"
        backend_configs = {
            path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
            for path in backend_dir.glob("*.yaml")
        }
        daily_index = "https://resource.flagos.net/repository/flagos-pypi-daily/simple"
        expected = {
            "H800": ("flagtree-nvidia", "0.7.1.dev20261123", daily_index),
            "910B": (
                "flagtree",
                "0.7.2a1+ascend3.5",
                "https://resource.flagos.net/repository/flagos-pypi-hosted/simple",
            ),
            "C550": ("flagtree-metax", "0.7.1.dev20261123", daily_index),
            "BW1000": (
                "flagtree-hcu",
                "0.7.1.dev20261124",
                daily_index,
            ),
            "MTT-S5000": (
                "flagtree-mthreads",
                "0.7.1.dev20261123",
                daily_index,
            ),
            "BI-V150": (
                "flagtree-iluvatar",
                "0.7.1.dev20261123",
                daily_index,
            ),
            "P800": ("flagtree-kunlunxin", "0.7.1.dev20261123", daily_index),
            "TX8110": (
                "flagtree-tsingmicro",
                "0.7.1.dev20261123",
                daily_index,
            ),
            "S60": ("flagtree-enflame", "0.7.1.dev20261123", daily_index),
            "PPU-ZW810E": ("flagtree-ppu", "0.7.1.dev20261123", daily_index),
            "W7900D": ("flagtree-amd", "0.7.1.dev20261123", daily_index),
        }
        for backend_name, (package_name, version, index_url) in expected.items():
            with self.subTest(backend=backend_name):
                self.assertEqual(
                    backend_configs[backend_name]["deps"][package_name],
                    {
                        "display_name": "FlagTree",
                        "priority": 100,
                        "version": version,
                        "index_url": index_url,
                        "no_deps": True,
                        "target": "/opt/flagtree",
                    },
                )

        self.assertNotIn("deps", backend_configs["MLU590-M9DE"])
        self.assertEqual(
            backend_configs["MLU590-M9DE"]["container_image"],
            "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.7.2:2.3.0",
        )

    def test_vllm_backend_container_images(self):
        backend_dir = ROOT / ".github/configs/backends"
        expected = {
            "H800": "harbor.baai.ac.cn/flagos-dev/vllm-plugin-fl:v0.32.0-cuda-dev",
            "BW1000": "harbor.baai.ac.cn/flagos-app/vllm0.24.0-hygon-dtk26.04:2.3.0-0.3.0",
            "W7900D": "harbor.baai.ac.cn/flagos-app/vllm0.32.0-amd-rocm7.2.4:2.3.0-0.5.0.dev0_g814b6d3.d20261117",
            "PPU-ZW810E": "harbor.baai.ac.cn/flagos-app/vllm0.32.0-thead-ppu2.1.0:2.3.0-0.5.0.dev0_g814b6d3.d20261117",
            "BI-V150": "harbor.baai.ac.cn/flagos-app/vllm0.32.0-iluvatar-corex5.0.1:2.3.0-0.5.0.dev0_g814b6d3.d20261117",
        }

        for backend_name, image in expected.items():
            with self.subTest(backend=backend_name):
                config = yaml.safe_load(
                    (backend_dir / f"{backend_name}.yaml").read_text(encoding="utf-8")
                )
                self.assertEqual(config["container_image"], image)

    def test_matrix_uses_fixed_backend_flagtree_versions_without_index_queries(self):
        selected = self.build_matrix(
            "FlagAttention",
            vendors="Nvidia,Ascend,Hygon",
        )
        selected_by_name = {backend["name"]: backend for backend in selected["include"]}
        self.assertEqual(
            self.upgrade_dep(selected_by_name["H800"], "flagtree-nvidia")["version"],
            "0.7.1.dev20261123",
        )
        self.assertEqual(
            self.upgrade_dep(selected_by_name["910B"], "flagtree")["version"],
            "0.7.2a1+ascend3.5",
        )
        self.assertEqual(
            self.upgrade_dep(selected_by_name["BW1000"], "flagtree-hcu")["version"],
            "0.7.1.dev20261124",
        )

        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        build_matrix = next(
            step for step in prepare_steps if step.get("id") == "build-matrix"
        )
        self.assertNotIn("FLAGTREE_VERSION_INPUT", build_matrix["env"])
        self.assertNotIn("urlopen", build_matrix["run"])
        self.assertNotIn("flagtree_version", build_matrix["run"])

    def test_dependency_layers_enablement_precedence_and_install_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copytree(ROOT / ".github/configs", root / ".github/configs")
            repos_path = root / ".github/configs/repos.yaml"
            repos = yaml.safe_load(repos_path.read_text(encoding="utf-8"))
            repos["defaults"]["deps"] = {
                "global-only": {"version": "1.0", "priority": 300},
                "all-layers": {
                    "display_name": "Global",
                    "version": "1.0",
                    "index_url": "https://global.example/simple",
                    "no_deps": True,
                    "priority": 900,
                },
                "reenable-from-default": {
                    "enabled": False,
                    "priority": 600,
                },
                "reenable-from-backend": {
                    "enabled": True,
                    "priority": 610,
                },
                "reenable-from-repo-default": {
                    "enabled": True,
                    "priority": 620,
                },
                "disabled-at-backend": {"enabled": True},
                "disabled-at-repo-default": {"enabled": True},
                "disabled-at-repo-backend": {"enabled": True},
                "disabled-inherited": {"enabled": False, "version": "1.0"},
                "sort-z": {"priority": 200},
                "sort-a": {"priority": 200},
                "implicit-enabled": {"priority": 630},
                "default-priority": {},
            }
            backend_path = root / ".github/configs/backends/H800.yaml"
            backend = yaml.safe_load(backend_path.read_text(encoding="utf-8"))
            backend["deps"] = {
                "backend-only": {"version": "2.0", "priority": 400},
                "all-layers": {
                    "display_name": "Backend",
                    "version": "2.0",
                    "find_links": "https://backend.example/wheels",
                },
                "reenable-from-default": {"enabled": True},
                "reenable-from-backend": {"enabled": False},
                "disabled-at-backend": {"enabled": False},
                "disabled-inherited": {"version": "2.0"},
            }
            backend_path.write_text(
                yaml.safe_dump(backend, sort_keys=False),
                encoding="utf-8",
            )
            repos["repos"]["FlagAttention"]["deps"] = {
                "default": {
                    "repo-default-only": {"version": "3.0", "priority": 500},
                    "all-layers": {
                        "display_name": "Repository default",
                        "version": "3.0",
                        "index_url": "https://repo.example/simple",
                        "no_deps": False,
                    },
                    "reenable-from-backend": {"enabled": True},
                    "reenable-from-repo-default": {"enabled": False},
                    "disabled-at-repo-default": {"enabled": False},
                    "disabled-inherited": {
                        "index_url": "https://disabled.example/simple"
                    },
                },
                "H800": {
                    "repo-backend-only": {"version": "4.0", "priority": 50},
                    "all-layers": {
                        "display_name": "Repository backend",
                        "version": "4.0",
                        "priority": 100,
                    },
                    "reenable-from-repo-default": {"enabled": True},
                    "disabled-at-repo-backend": {"enabled": False},
                    "disabled-inherited": {"display_name": "Still disabled"},
                },
            }
            repos_path.write_text(
                yaml.safe_dump(repos, sort_keys=False),
                encoding="utf-8",
            )
            selected = self.build_matrix(
                "FlagAttention",
                vendors="Nvidia",
                cwd=root,
            )
            self.assertEqual(len(selected["include"]), 1)
            selected_backend = selected["include"][0]
            h800_deps = self.upgrade_deps(selected_backend)
            self.assertNotIn(
                "UPGRADE_DEPS_JSON",
                json.loads(selected_backend["test_env"]),
            )
            self.assertEqual(
                json.loads(selected_backend["upgrade_env"])["UPGRADE_DEPS_JSON"],
                selected_backend["upgrade_deps"],
            )
            self.assertEqual(
                [dependency["package_name"] for dependency in h800_deps],
                [
                    "repo-backend-only",
                    "all-layers",
                    "sort-a",
                    "sort-z",
                    "global-only",
                    "backend-only",
                    "repo-default-only",
                    "reenable-from-default",
                    "reenable-from-backend",
                    "reenable-from-repo-default",
                    "implicit-enabled",
                    "default-priority",
                ],
            )
            dependencies = {
                dependency["package_name"]: dependency for dependency in h800_deps
            }
            self.assertEqual(
                dependencies["all-layers"],
                {
                    "package_name": "all-layers",
                    "display_name": "Repository backend",
                    "priority": 100,
                    "version": "4.0",
                    "requirement": "all-layers==4.0",
                    "target": "",
                    "uv_options": [
                        "--index-url",
                        "https://repo.example/simple",
                        "--find-links",
                        "https://backend.example/wheels",
                    ],
                },
            )
            self.assertTrue(
                {
                    "global-only",
                    "backend-only",
                    "repo-default-only",
                    "repo-backend-only",
                }.issubset(dependencies)
            )
            self.assertTrue(
                {
                    "reenable-from-default",
                    "reenable-from-backend",
                    "reenable-from-repo-default",
                    "implicit-enabled",
                }.issubset(dependencies)
            )
            self.assertTrue(
                {
                    "disabled-at-backend",
                    "disabled-at-repo-default",
                    "disabled-at-repo-backend",
                    "disabled-inherited",
                }.isdisjoint(dependencies)
            )
            self.assertEqual(dependencies["default-priority"]["priority"], 1000)
            self.assertEqual(
                dependencies["implicit-enabled"]["requirement"],
                "implicit-enabled",
            )

    def test_disabled_backend_is_not_restored_by_enabled_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copytree(ROOT / ".github/configs", root / ".github/configs")
            repos_path = root / ".github/configs/repos.yaml"
            repos = yaml.safe_load(repos_path.read_text(encoding="utf-8"))
            flagattention = repos["repos"]["FlagAttention"]
            flagattention["backends"]["H800"]["enabled"] = False
            flagattention["deps"] = {"H800": {"forced-package": {"enabled": True}}}
            repos_path.write_text(
                yaml.safe_dump(repos, sort_keys=False),
                encoding="utf-8",
            )

            selected = self.build_matrix(
                "FlagAttention",
                vendors="Nvidia",
                cwd=root,
            )

        self.assertEqual(selected["include"], [])

    def test_matrix_rejects_non_normalized_dependency_target(self):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copytree(ROOT / ".github/configs", root / ".github/configs")
            backend_path = root / ".github/configs/backends/H800.yaml"
            backend = yaml.safe_load(backend_path.read_text(encoding="utf-8"))
            backend["deps"]["flagtree-nvidia"]["target"] = "/opt/../etc"
            backend_path.write_text(
                yaml.safe_dump(backend, sort_keys=False),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "REPO_INPUT": "FlagAttention",
                    "REF_INPUT": "",
                    "VENDORS_INPUT": "Nvidia",
                    "OPS_INPUT": "",
                    "EXCLUDE_OPS_INPUT": "",
                    "UPLOAD_LOG_INPUT": "",
                    "SEND_FEISHU_INPUT": "",
                    "GITHUB_OUTPUT": str(root / "github-output"),
                },
                text=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "H800: flagtree-nvidia.target must be a specific normalized "
            "absolute path such as /opt/flagtree",
            completed.stderr,
        )

    def test_gpu_probe_precedes_vendor_dependency_install(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        names = [step.get("name") for step in steps]
        self.assertLess(
            names.index("Configure package index and install optest"),
            names.index("Upgrade configured packages (FlagTree, FlagGems)"),
        )
        self.assertLess(
            names.index("Upgrade configured packages (FlagTree, FlagGems)"),
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

        upgrade = next(
            step
            for step in steps
            if step.get("name") == "Upgrade configured packages (FlagTree, FlagGems)"
        )
        self.assertEqual(
            upgrade["if"],
            "${{ matrix.upgrade_deps != '[]' }}",
        )
        self.assertEqual(upgrade["env"], "${{ fromJSON(matrix.upgrade_env) }}")
        self.assertIn("dependencies = json.loads", upgrade["run"])
        self.assertIn("dependencies", upgrade["run"])
        self.assertIn("'install',", upgrade["run"])
        self.assertIn("options[target_index] = staging", upgrade["run"])
        self.assertIn("shutil.move(staging, target)", upgrade["run"])
        self.assertIn("shutil.rmtree", upgrade["run"])
        self.assertNotIn("ignore_errors=True", upgrade["run"])
        self.assertIn("explicit_index_options.intersection(options)", upgrade["run"])
        self.assertIn("metadata.version(package)", upgrade["run"])
        self.assertIn("actual == expected", upgrade["run"])
        self.assertIn(
            'print_cache_debug "configured package install: before"',
            upgrade["run"],
        )
        self.assertIn(
            'print_cache_debug "configured package install: after"',
            upgrade["run"],
        )

    def test_flagtree_upgrade_stages_and_activates_exact_hcu_daily(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Upgrade configured packages (FlagTree, FlagGems)"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flagtree"
            staging = root / "flagtree.new"
            source.mkdir()
            (source / "marker").write_text("image source", encoding="utf-8")
            uv_log = root / "uv.log"
            uv = root / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                'args="$*"\n'
                'printf "UV_CACHE_DIR=%s UV_DEFAULT_INDEX=%s ARGS=%s\\n" '
                '"${UV_CACHE_DIR:-}" "${UV_DEFAULT_INDEX:-}" "$args" >> "$UV_LOG"\n'
                'target=""\n'
                'while [ "$#" -gt 0 ]; do\n'
                '  if [ "$1" = "--target" ]; then shift; target="$1"; fi\n'
                "  shift\n"
                "done\n"
                'version="${FAKE_FLAGTREE_VERSION:-0.7.1.dev20261124}"\n'
                'case " $args " in\n'
                '  *" pip install "*) mkdir -p "$target"; '
                'printf "hcu daily" > "$target/marker"; '
                'mkdir -p "$target/flagtree_hcu-0.7.1.dev20261124.dist-info"; '
                "printf 'Metadata-Version: 2.1\\nName: flagtree-hcu\\n"
                'Version: %s\\n\' "$version" > '
                '"$target/flagtree_hcu-0.7.1.dev20261124.dist-info/METADATA" ;;\n'
                "esac\n",
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
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "UV_CACHE_DIR": str(root / "uv-cache"),
                    "UV_DEFAULT_INDEX": "https://pypi.example/simple",
                    "UV_LOG": str(uv_log),
                    "GITHUB_ENV": str(github_env),
                    "PYTHONPATH": "",
                    "UPGRADE_DEPS_JSON": json.dumps(
                        [
                            {
                                "package_name": "flagtree-hcu",
                                "display_name": "FlagTree",
                                "priority": 100,
                                "version": "0.7.1.dev20261124",
                                "requirement": "flagtree-hcu==0.7.1.dev20261124",
                                "target": str(source),
                                "uv_options": [
                                    "--no-deps",
                                    "--target",
                                    str(source),
                                    "--index-url",
                                    "https://resource.flagos.net/repository/flagos-pypi-daily/simple",
                                ],
                            }
                        ]
                    ),
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                (source / "marker").read_text(encoding="utf-8"),
                "hcu daily",
            )
            self.assertFalse(staging.exists())
            commands = uv_log.read_text(encoding="utf-8")
            self.assertIn(
                "pip install --python ",
                commands,
            )
            self.assertIn(
                f"--no-deps --target {staging} --index-url "
                "https://resource.flagos.net/repository/flagos-pypi-daily/simple "
                "flagtree-hcu==0.7.1.dev20261124",
                commands,
            )
            install_log = next(
                line for line in commands.splitlines() if "pip install" in line
            )
            self.assertTrue(
                install_log.startswith(
                    f"UV_CACHE_DIR={root / 'uv-cache'} UV_DEFAULT_INDEX= ARGS="
                )
            )
            pre_upgrade_show = next(
                line for line in commands.splitlines() if "pip show --python" in line
            )
            self.assertTrue(pre_upgrade_show.endswith(" flagtree-hcu"))
            self.assertNotIn(f"pip show --target {staging}", commands)
            target_show_commands = [
                line
                for line in commands.splitlines()
                if f"pip show --target {source} flagtree-hcu" in line
            ]
            self.assertEqual(len(target_show_commands), 1)
            self.assertEqual(
                github_env.read_text(encoding="utf-8"),
                f"PYTHONPATH={source}\n",
            )

            self.assertIn(
                f"[FlagTree] {source} was not in PYTHONPATH; prepended it.",
                completed.stdout,
            )
            self.assertIn(
                "[flagtree-hcu] Active Python sees flagtree-hcu==0.7.1.dev20261124",
                completed.stdout,
            )

    def test_generic_upgrade_without_index_option_keeps_selected_index(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Upgrade configured packages (FlagTree, FlagGems)"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uv_log = root / "uv.log"
            uv = root / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                'printf "UV_DEFAULT_INDEX=%s ARGS=%s\\n" '
                '"${UV_DEFAULT_INDEX:-}" "$*" >> "$UV_LOG"\n',
                encoding="utf-8",
            )
            uv.chmod(0o755)
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "UV_DEFAULT_INDEX": "https://pypi.example/simple",
                    "UV_LOG": str(uv_log),
                    "UPGRADE_DEPS_JSON": json.dumps(
                        [
                            {
                                "package_name": "plain-pkg",
                                "display_name": "Plain package",
                                "priority": 1000,
                                "version": "",
                                "requirement": "plain-pkg",
                                "target": "",
                                "uv_options": [],
                            }
                        ]
                    ),
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            install_log = next(
                line
                for line in uv_log.read_text(encoding="utf-8").splitlines()
                if "ARGS=pip install " in line
            )
            self.assertTrue(
                install_log.startswith(
                    "UV_DEFAULT_INDEX=https://pypi.example/simple ARGS="
                )
            )

    def test_flagtree_upgrade_preserves_image_tree_when_staging_fails(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Upgrade configured packages (FlagTree, FlagGems)"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flagtree"
            staging = root / "flagtree.new"
            source.mkdir()
            (source / "marker").write_text("image source", encoding="utf-8")
            uv = root / "uv"
            uv.write_text(
                '#!/bin/sh\ncase " $* " in *" pip install "*) exit 42 ;; esac\n',
                encoding="utf-8",
            )
            uv.chmod(0o755)
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "UPGRADE_DEPS_JSON": json.dumps(
                        [
                            {
                                "package_name": "flagtree-nvidia",
                                "display_name": "FlagTree",
                                "priority": 100,
                                "version": "0.7.1.dev20261115",
                                "requirement": "flagtree-nvidia==0.7.1.dev20261115",
                                "target": str(source),
                                "uv_options": [
                                    "--no-deps",
                                    "--target",
                                    str(source),
                                    "--index-url",
                                    "https://resource.flagos.net/repository/flagos-pypi-daily/simple",
                                ],
                            }
                        ]
                    ),
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 42, completed.stderr)
            self.assertEqual(
                (source / "marker").read_text(encoding="utf-8"),
                "image source",
            )
            self.assertFalse(staging.exists())
            self.assertNotIn(
                "show installed flagtree-nvidia in target directory",
                completed.stdout,
            )

    def test_vendor_install_script_precedes_target_install(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        names = [step.get("name") for step in steps]
        self.assertLess(
            names.index("Run vendor install script"),
            names.index("Configure package index and install optest"),
        )
        self.assertLess(
            names.index("Run vendor install script"),
            names.index("Install target repository and dependencies"),
        )

        run = next(
            step for step in steps if step.get("name") == "Run vendor install script"
        )["run"]
        self.assertIn("VENDOR=$(printf", run)
        self.assertIn(".github/scripts/install/${VENDOR}.py", run)
        self.assertIn(".github/scripts/install/${VENDOR}.sh", run)
        self.assertIn("*.py)", run)

    def test_vendor_specific_work_does_not_use_hardcoded_vendor_branches(self):
        profile = next(
            step
            for step in self.workflow["jobs"]["test-container"]["steps"]
            if step.get("name") == "Copy root profile to GitHub Actions home"
        )["run"]
        self.assertNotIn("matrix.vendor", profile)
        self.assertNotIn("ASCEND_WORK_PATH", profile)

    def test_install_scripts_exist_for_prepared_vendors(self):
        directory = ROOT / ".github/scripts/install"
        self.assertEqual(
            sorted(path.name for path in directory.iterdir()),
            ["ascend.sh", "hygon.sh", "mthreads.sh"],
        )
        for script in directory.iterdir():
            text = script.read_text(encoding="utf-8")
            with self.subTest(script=script.name):
                self.assertNotIn("${{", text)
                self.assertEqual(script.suffix, ".sh")
                self.assertEqual(
                    subprocess.run(
                        ["bash", "-n", str(script)],
                        check=False,
                        capture_output=True,
                    ).returncode,
                    0,
                )

    def test_matrix_resolves_command_timeout_without_dependency_policy(self):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )

        for repo, expected_timeout in (
            ("FlagTrain", 1800),
            ("FlagGems", 1800),
            ("FlagAttention", 900),
        ):
            with self.subTest(repo=repo), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "github-output"
                environment = {
                    **os.environ,
                    "REPO_INPUT": repo,
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
                    self.assertEqual(backend["op_cmd_timeout"], expected_timeout)

                if repo == "FlagTrain":
                    backends = {
                        backend["name"]: backend for backend in matrix["include"]
                    }
                    self.assertIn("P800", backends)
                    self.assertIn("S60", backends)
                    ppu = next(
                        backend
                        for backend in matrix["include"]
                        if backend["name"] == "PPU-ZW810E"
                    )
                    self.assertEqual(
                        ppu["container_image"],
                        "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0",
                    )
                    self.assertEqual(
                        backends["P800"]["container_image"],
                        "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-kunlunxin-xre5.37.1:2.2.0-0.2.3",
                    )
                    self.assertEqual(
                        backends["S60"]["container_image"],
                        "harbor.baai.ac.cn/flagos-app/megatron_rl0.17.1-enflame-tops1.10.6:2.2.0-0.2.3",
                    )

    def test_matrix_sets_persistent_per_vendor_cache(self):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )

        # FlagGems-vllm enables backends across every vendor, including THead
        # (PPU-ZW810E), which has no /data mount and therefore bases its durable
        # output (and cache) under /home instead.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            environment = {
                **os.environ,
                "REPO_INPUT": "FlagGems-vllm",
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
        saw_data_backend = False
        saw_home_backend = False
        for backend in matrix["include"]:
            vendor_key = backend["vendor"].lower()
            cache_base = backend["output_base_dir"].rstrip("/")
            # test_env is serialized to a JSON string in the matrix payload.
            test_env = json.loads(backend["test_env"])
            self.assertEqual(
                test_env["UV_CACHE_DIR"],
                f"{cache_base}/uv-cache/{vendor_key}",
            )
            self.assertEqual(
                test_env["PIP_CACHE_DIR"],
                f"{cache_base}/pip-cache/{vendor_key}",
            )
            self.assertEqual(test_env["UV_LINK_MODE"], "copy")
            if backend["name"] == "PPU-ZW810E":
                saw_home_backend = True
                self.assertTrue(test_env["UV_CACHE_DIR"].startswith("/home/"))
            if cache_base.startswith("/data/"):
                saw_data_backend = True

        # Cover both the common /data path and the THead /home fallback.
        self.assertTrue(saw_data_backend)
        self.assertTrue(saw_home_backend)

    def test_matrix_cache_env_preserves_existing_vendor_variables(self):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            environment = {
                **os.environ,
                "REPO_INPUT": "FlagGems",
                "REF_INPUT": "",
                "VENDORS_INPUT": "Metax",
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

        # The Metax backend (C550) carries pre-existing test_env variables that
        # must survive alongside the injected cache variables.
        c550 = next(b for b in matrix["include"] if b["name"] == "C550")
        test_env = json.loads(c550["test_env"])
        self.assertEqual(test_env["FLAGTREE_BACKEND"], "metax")
        self.assertEqual(test_env["MACA_DIRECT_DISPATCH"], 1)
        self.assertIn("UV_CACHE_DIR", test_env)
        self.assertIn("PIP_CACHE_DIR", test_env)
        self.assertEqual(test_env["UV_LINK_MODE"], "copy")

    def test_per_repo_command_timeout_configuration(self):
        self.assertEqual(self.repos["defaults"]["op_cmd_timeout"], 600)
        self.assertEqual(self.repos["repos"]["FlagGems"]["op_cmd_timeout"], 1800)
        self.assertEqual(self.repos["repos"]["FlagTrain"]["op_cmd_timeout"], 1800)
        self.assertEqual(self.repos["repos"]["FlagAttention"]["op_cmd_timeout"], 900)
        for repo in (
            "FlagGems-vllm",
            "FlagGems-sglang",
        ):
            self.assertNotIn("op_cmd_timeout", self.repos["repos"][repo])

        steps = self.workflow["jobs"]["test-container"]["steps"]
        run_ops_tests = next(
            step["run"] for step in steps if step.get("name") == "Run ops tests"
        )
        self.assertIn('--timeout "${{ matrix.op_cmd_timeout }}"', run_ops_tests)

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
                'if [ "$*" = "pip show --help" ]; then echo "  --target"; exit 0; fi\n'
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

    def test_old_uv_self_updates_for_target_show_support(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Configure package index and install optest"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            updated = root / "uv-updated"
            uv_log = root / "uv.log"
            uv = root / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                'printf "UV_CACHE_DIR=%s ARGS=%s\\n" '
                '"${UV_CACHE_DIR:-}" "$*" >> "$UV_LOG"\n'
                'if [ "$1" = "--version" ]; then\n'
                '  if [ -f "$UV_UPDATED" ]; then echo "uv 0.11.30"; '
                'else echo "uv 0.7.12"; fi\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$*" = "pip show --help" ]; then\n'
                '  if [ -f "$UV_UPDATED" ]; then echo "  --target"; fi\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$*" = "self update 0.11.30" ]; then '
                'touch "$UV_UPDATED"; exit 0; fi\n'
                "exit 0\n",
                encoding="utf-8",
            )
            uv.chmod(0o755)
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "GITHUB_ENV": str(root / "github-env"),
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "UV_CACHE_DIR": str(root / "uv-cache"),
                    "UV_LOG": str(uv_log),
                    "UV_UPDATED": str(updated),
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            uv_output = uv_log.read_text(encoding="utf-8")
            self.assertIn(f"UV_CACHE_DIR={root / 'uv-cache'}", uv_output)
            self.assertIn("ARGS=self update 0.11.30", uv_output)
            self.assertIn("uv version: uv 0.11.30", completed.stdout)
            self.assertNotIn("uv installation or upgrade", completed.stdout)

    def test_old_uv_falls_back_to_cached_pip_upgrade(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Configure package index and install optest"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pip_log = root / "pip.log"
            uv = root / "uv"
            uv.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then echo "uv 0.7.12"; exit 0; fi\n'
                'if [ "$*" = "pip show --help" ]; then exit 0; fi\n'
                'if [ "$*" = "self update 0.11.30" ]; then exit 42; fi\n'
                "exit 0\n",
                encoding="utf-8",
            )
            uv.chmod(0o755)
            scripts_dir = root / "new-bin"
            scripts_dir.mkdir()
            new_uv = scripts_dir / "uv"
            new_uv.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "--version" ]; then echo "uv 0.11.30"; exit 0; fi\n'
                'if [ "$*" = "pip show --help" ]; then echo "  --target"; exit 0; fi\n'
                "exit 0\n",
                encoding="utf-8",
            )
            new_uv.chmod(0o755)
            python = root / "python3"
            python.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
                '  printf "PIP_CACHE_DIR=%s ARGS=%s\\n" '
                '"${PIP_CACHE_DIR:-}" "$*" >> "$PIP_LOG"\n'
                "  exit 0\n"
                "fi\n"
                'if [ "$1" = "-c" ]; then echo "$UV_SCRIPTS_DIR"; exit 0; fi\n'
                'exec "$REAL_PYTHON" "$@"\n',
                encoding="utf-8",
            )
            python.chmod(0o755)
            pip_cache = root / "pip-cache"
            github_path = root / "github-path"
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "GITHUB_ENV": str(root / "github-env"),
                    "GITHUB_PATH": str(github_path),
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                    "PIP_CACHE_DIR": str(pip_cache),
                    "PIP_LOG": str(pip_log),
                    "REAL_PYTHON": sys.executable,
                    "UV_SCRIPTS_DIR": str(scripts_dir),
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            pip_output = pip_log.read_text(encoding="utf-8")
            self.assertIn(f"PIP_CACHE_DIR={pip_cache}", pip_output)
            self.assertIn("-m pip install --upgrade uv>=0.11.30", pip_output)
            self.assertIn("uv version: uv 0.11.30", completed.stdout)
            self.assertEqual(
                github_path.read_text(encoding="utf-8"),
                f"{scripts_dir}\n",
            )

    def test_target_install_logs_dependency_sources_commands_and_results(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Install target repository and dependencies"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The target repo is now checked out into a per-repo directory
            # named after the library (TARGET_DIR), not a fixed "target".
            flagtrain_dir = root / "FlagTrain"
            flagtrain_dir.mkdir()
            (flagtrain_dir / "pyproject.toml").write_text(
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
                    'printf \'[fake uv cwd=%s] %s\\n\' "$PWD" "$*"\n'
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
                    "TARGET_DIR": "FlagTrain",
                },
                text=True,
            )

            # Second run uses REPOSITORY_NAME/TARGET_DIR=FlagGems, so the target
            # library lives in its own ./FlagGems directory.
            flaggems_dir = root / "FlagGems"
            flaggems_dir.mkdir()
            (flaggems_dir / "pyproject.toml").write_text(
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
                    "TARGET_DIR": "FlagGems",
                },
                text=True,
            )
            github_env_contents = github_env.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = completed.stdout
        expected = [
            "Target project source: ./FlagTrain/pyproject.toml",
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

        self.assertEqual(test_only_completed.returncode, 0, test_only_completed.stderr)
        self.assertIn("Detected vendor extra: false", test_only_completed.stdout)
        self.assertIn("Selected target extras: test", test_only_completed.stdout)
        self.assertIn("--editable .[test]", test_only_completed.stdout)
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
            "flagos-gpu-check": "fake-gpu-check",
            "${{ matrix.test_gpus }}": "",
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

    def test_configured_test_gpus_override_detected_devices(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"]
            for step in steps
            if step.get("name") == "Check GPU availability"
        )
        replacements = {
            "flagos-gpu-check": "fake-gpu-check",
            "${{ matrix.test_gpus }}": "8,9",
            "${{ matrix.vendor }}": "TsingMicro",
        }
        for expression, value in replacements.items():
            script = script.replace(expression, value)

        result = {
            "vendor": "tsingmicro",
            "device": "TX8110",
            "driver": "1.0",
            "runtime": "tsingmicro",
            "runtime_version": "1.0",
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
            completed = subprocess.run(
                ["bash", "-e", "-o", "pipefail", "-c", script],
                check=False,
                capture_output=True,
                env={
                    **os.environ,
                    "GITHUB_ENV": str(github_env),
                    "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                "[GPU check] Using configured test_gpus override: 8,9",
                completed.stdout,
            )
            self.assertIn(
                "AVAILABLE_GPUS=8,9",
                github_env.read_text(encoding="utf-8").splitlines(),
            )


if __name__ == "__main__":
    unittest.main()
