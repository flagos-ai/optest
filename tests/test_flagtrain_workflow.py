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
import re
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
        flagtree_version="",
        vendors="",
        cwd=ROOT,
        return_output=False,
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
                "FLAGTREE_VERSION_INPUT": flagtree_version,
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
        if return_output:
            return matrix, completed.stdout
        return matrix

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

    def test_flaggems_runtime_images_and_910b_flagtree_exception(self):
        config = self.repos["repos"]["FlagGems"]
        self.assertEqual(
            config["container_images"],
            {
                "910B": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1:2.3.0",
                "PPU-ZW810E": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0",
                "W7900D": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-amd-rocm7.2.4:2.3.0",
            },
        )
        self.assertEqual(config["skip_flagtree_upgrade"], ["910B"])
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
            "910B": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.2.1:2.3.0",
            "PPU-ZW810E": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0:2.3.0",
            "MLU590-M9DE": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.7.2:2.3.0",
            "W7900D": "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-amd-rocm7.2.4:2.3.0",
        }
        for backend_name, image in expected_images.items():
            with self.subTest(backend=backend_name):
                self.assertEqual(
                    flaggems_by_name[backend_name]["container_image"], image
                )
        self.assertTrue(flaggems_by_name["910B"]["skip_flagtree_upgrade"])
        self.assertFalse(flaggems_by_name["H800"]["skip_flagtree_upgrade"])
        self.assertFalse(flaggems_by_name["PPU-ZW810E"]["skip_flagtree_upgrade"])
        self.assertFalse(flaggems_by_name["W7900D"]["skip_flagtree_upgrade"])
        self.assertFalse(flaggems_by_name["MLU590-M9DE"]["skip_flagtree_upgrade"])
        self.assertEqual(flaggems_by_name["MLU590-M9DE"]["flagtree_package_name"], "")

        flaggems_vllm = self.build_matrix(
            "FlagGems-vllm",
            vendors="Ascend,THead",
        )
        flaggems_vllm_by_name = {
            backend["name"]: backend for backend in flaggems_vllm["include"]
        }
        self.assertFalse(flaggems_vllm_by_name["910B"]["skip_flagtree_upgrade"])
        self.assertEqual(
            flaggems_vllm_by_name["PPU-ZW810E"]["container_image"],
            "harbor.baai.ac.cn/flaggems/flaggems-test-thead-ppu-zw810e:20261015",
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
            "H800": ("flagtree-nvidia", "0.7.1.dev20261115", daily_index),
            "910B": (
                "flagtree-ascend-cann9.0.0",
                "0.7.0.dev20261115",
                daily_index,
            ),
            "C550": ("flagtree-metax", "0.7.1.dev20261115", daily_index),
            "BW1000": (
                "flagtree-hcu",
                "0.7.1.dev20261104",
                daily_index,
            ),
            "MTT-S5000": (
                "flagtree-mthreads",
                "0.7.1.dev20261115",
                daily_index,
            ),
            "BI-V150": (
                "flagtree-iluvatar",
                "0.7.1.dev20261115",
                daily_index,
            ),
            "P800": ("flagtree-kunlunxin", "0.7.1.dev20261112", daily_index),
            "TX8110": (
                "flagtree-tsingmicro",
                "0.7.1.dev20261115",
                daily_index,
            ),
            "S60": ("flagtree-enflame", "0.7.1.dev20261115", daily_index),
            "PPU-ZW810E": ("flagtree-ppu", "0.7.1.dev20261115", daily_index),
            "W7900D": ("flagtree-amd", "0.7.1.dev20261115", daily_index),
        }
        for backend_name, (package_name, version, index_url) in expected.items():
            with self.subTest(backend=backend_name):
                self.assertEqual(
                    backend_configs[backend_name]["flagtree"],
                    {
                        "package_name": package_name,
                        "version": version,
                        "index_url": index_url,
                    },
                )

        self.assertNotIn("flagtree", backend_configs["MLU590-M9DE"])
        self.assertEqual(
            backend_configs["MLU590-M9DE"]["container_image"],
            "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.7.2:2.3.0",
        )

    def test_matrix_checks_overrides_and_falls_back_per_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copytree(ROOT / ".github/configs", root / ".github/configs")
            index_root = root / "simple"
            available_versions = {
                "flagtree-nvidia": {
                    "0.7.1.dev20261015",
                    "0.7.1.dev20261115",
                    "0.8.0.dev20261201",
                },
                "flagtree-ascend-cann9.0.0": {"0.8.0.dev20261201"},
                "flagtree-hcu": {
                    "0.7.1.dev20261015",
                    "0.7.1.dev20261104",
                    "0.7.1a2+hcu3.6",
                },
            }
            for package_name, versions in available_versions.items():
                normalized_name = re.sub(r"[-_.]+", "-", package_name).lower()
                wheel_name = re.sub(r"[-_.]+", "_", package_name).lower()
                package_dir = index_root / normalized_name
                package_dir.mkdir(parents=True)
                wheel_urls = [
                    f"../../packages/{normalized_name}/"
                    f"{wheel_name}-{version}-py3-none-any.whl"
                    for version in sorted(versions)
                ]
                if package_name == "flagtree-hcu":
                    index_contents = json.dumps(
                        {"files": [{"url": url} for url in wheel_urls]}
                    )
                else:
                    index_contents = "\n".join(
                        f'<a href="{url}">{url}</a>' for url in wheel_urls
                    )
                (package_dir / "index.html").write_text(
                    index_contents,
                    encoding="utf-8",
                )

            index_url = index_root.as_uri()
            for backend_name in ("H800", "910B", "BW1000"):
                backend_path = root / f".github/configs/backends/{backend_name}.yaml"
                backend = yaml.safe_load(backend_path.read_text(encoding="utf-8"))
                backend["flagtree"]["index_url"] = index_url
                backend_path.write_text(
                    yaml.safe_dump(backend, sort_keys=False),
                    encoding="utf-8",
                )

            defaults = self.build_matrix(
                "FlagGems",
                vendors="Nvidia,Ascend,Hygon",
                cwd=root,
            )
            default_by_name = {
                backend["name"]: backend for backend in defaults["include"]
            }
            self.assertEqual(
                default_by_name["H800"]["flagtree_version"],
                "0.7.1.dev20261115",
            )
            self.assertEqual(
                default_by_name["910B"]["flagtree_version"],
                "0.7.0.dev20261115",
            )
            self.assertEqual(
                default_by_name["BW1000"]["flagtree_version"],
                "0.7.1.dev20261104",
            )

            dated, dated_output = self.build_matrix(
                "FlagGems",
                flagtree_version="20261015",
                vendors="Nvidia,Ascend,Hygon",
                cwd=root,
                return_output=True,
            )
            dated_by_name = {backend["name"]: backend for backend in dated["include"]}
            self.assertEqual(
                dated_by_name["H800"]["flagtree_version"],
                "0.7.1.dev20261015",
            )
            self.assertEqual(
                dated_by_name["910B"]["flagtree_version"],
                "0.7.0.dev20261115",
            )
            self.assertEqual(
                dated_by_name["BW1000"]["flagtree_version"],
                "0.7.1.dev20261015",
            )
            self.assertIn("::warning title=FlagTree version fallback::", dated_output)
            self.assertIn(
                "910B: flagtree-ascend-cann9.0.0==0.7.0.dev20261015 is unavailable",
                dated_output,
            )
            full_daily = "0.8.0.dev20261201"
            overridden, overridden_output = self.build_matrix(
                "FlagGems",
                flagtree_version=full_daily,
                vendors="Nvidia,Ascend,Hygon",
                cwd=root,
                return_output=True,
            )
            overridden_by_name = {
                backend["name"]: backend for backend in overridden["include"]
            }
            self.assertEqual(
                overridden_by_name["H800"]["flagtree_version"],
                full_daily,
            )
            self.assertEqual(
                overridden_by_name["910B"]["flagtree_version"],
                full_daily,
            )
            self.assertEqual(
                overridden_by_name["BW1000"]["flagtree_version"],
                "0.7.1.dev20261104",
            )
            self.assertIn(
                "BW1000: flagtree-hcu==0.8.0.dev20261201 is unavailable",
                overridden_output,
            )

            release_version = "0.7.1a2+hcu3.6"
            released, released_output = self.build_matrix(
                "FlagGems",
                flagtree_version=release_version,
                vendors="Nvidia,Ascend,Hygon",
                cwd=root,
                return_output=True,
            )
            released_by_name = {
                backend["name"]: backend for backend in released["include"]
            }
            self.assertEqual(
                released_by_name["BW1000"]["flagtree_version"],
                release_version,
            )
            self.assertEqual(
                released_by_name["H800"]["flagtree_version"],
                "0.7.1.dev20261115",
            )
            self.assertEqual(
                released_by_name["910B"]["flagtree_version"],
                "0.7.0.dev20261115",
            )
            self.assertIn(
                "H800: flagtree-nvidia==0.7.1a2+hcu3.6 is unavailable",
                released_output,
            )

            prefix_collision, collision_output = self.build_matrix(
                "FlagGems",
                flagtree_version="0.7.1",
                vendors="Nvidia",
                cwd=root,
                return_output=True,
            )
            self.assertEqual(
                prefix_collision["include"][0]["flagtree_version"],
                "0.7.1.dev20261115",
            )
            self.assertIn(
                "H800: flagtree-nvidia==0.7.1 is unavailable",
                collision_output,
            )

            h800_path = root / ".github/configs/backends/H800.yaml"
            h800 = yaml.safe_load(h800_path.read_text(encoding="utf-8"))
            h800["flagtree"]["index_url"] = (root / "missing-index").as_uri()
            h800_path.write_text(
                yaml.safe_dump(h800, sort_keys=False),
                encoding="utf-8",
            )
            failed_query, failed_query_output = self.build_matrix(
                "FlagGems",
                flagtree_version=full_daily,
                vendors="Nvidia",
                cwd=root,
                return_output=True,
            )
            self.assertEqual(
                failed_query["include"][0]["flagtree_version"],
                "0.7.1.dev20261115",
            )
            self.assertIn(
                "H800: failed to check flagtree-nvidia==0.8.0.dev20261201",
                failed_query_output,
            )
            self.assertIn(
                "using configured version 0.7.1.dev20261115", failed_query_output
            )

    def test_repo_can_skip_flagtree_upgrade_globally_or_by_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copytree(ROOT / ".github/configs", root / ".github/configs")
            repos_path = root / ".github/configs/repos.yaml"
            repos = yaml.safe_load(repos_path.read_text(encoding="utf-8"))

            repos["repos"]["FlagAttention"]["skip_flagtree_upgrade"] = ["H800"]
            repos_path.write_text(
                yaml.safe_dump(repos, sort_keys=False),
                encoding="utf-8",
            )
            selected = self.build_matrix(
                "FlagAttention",
                vendors="Nvidia,Ascend",
                cwd=root,
            )
            selected_by_name = {
                backend["name"]: backend for backend in selected["include"]
            }
            self.assertTrue(selected_by_name["H800"]["skip_flagtree_upgrade"])
            self.assertFalse(selected_by_name["910B"]["skip_flagtree_upgrade"])

            repos["repos"]["FlagAttention"]["skip_flagtree_upgrade"] = True
            repos_path.write_text(
                yaml.safe_dump(repos, sort_keys=False),
                encoding="utf-8",
            )
            skipped = self.build_matrix(
                "FlagAttention",
                vendors="Nvidia,Ascend",
                cwd=root,
            )
            self.assertTrue(
                all(backend["skip_flagtree_upgrade"] for backend in skipped["include"])
            )

    def test_gpu_probe_precedes_vendor_dependency_install(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        names = [step.get("name") for step in steps]
        self.assertLess(
            names.index("Configure package index and install optest"),
            names.index("Upgrade Flagtree"),
        )
        self.assertLess(
            names.index("Upgrade Flagtree"),
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

        upgrade = next(step for step in steps if step.get("name") == "Upgrade Flagtree")
        self.assertEqual(
            upgrade["if"],
            "${{ matrix.flagtree_package_name != '' && matrix.skip_flagtree_upgrade != true }}",
        )
        self.assertEqual(upgrade["env"], "${{ fromJSON(matrix.test_env) }}")
        self.assertNotIn("flagtree_backup", upgrade["run"])
        self.assertNotIn("uv pip uninstall", upgrade["run"])
        self.assertIn("uv pip install", upgrade["run"])
        self.assertNotIn("uv pip install -v", upgrade["run"])
        self.assertIn("--no-deps", upgrade["run"])
        self.assertIn('--target "${FLAGTREE_STAGING_DIR}"', upgrade["run"])
        self.assertIn('--index-url "${FLAGTREE_INDEX_URL}"', upgrade["run"])
        self.assertNotIn("--no-cache", upgrade["run"])
        self.assertIn('rm -rf -- "${FLAGTREE_STAGING_DIR}"', upgrade["run"])
        self.assertIn('rm -rf -- "${FLAGTREE_INSTALL_DIR}"', upgrade["run"])
        self.assertIn(
            'mv "${FLAGTREE_STAGING_DIR}" "${FLAGTREE_INSTALL_DIR}"',
            upgrade["run"],
        )
        self.assertIn("uv pip show", upgrade["run"])
        self.assertIn('print_cache_debug "FlagTree install: before"', upgrade["run"])
        self.assertIn('print_cache_debug "FlagTree install: after"', upgrade["run"])
        self.assertIn(
            'uv pip show --python "${ACTIVE_PYTHON}" flagtree', upgrade["run"]
        )
        self.assertNotIn('"show staged ${FLAGTREE_PACKAGE_NAME}"', upgrade["run"])
        self.assertIn(
            'uv pip show --target "${FLAGTREE_INSTALL_DIR}" "${FLAGTREE_PACKAGE_NAME}"',
            upgrade["run"],
        )
        self.assertNotIn(
            '"show installed ${FLAGTREE_PACKAGE_NAME} through uv environment"',
            upgrade["run"],
        )
        self.assertNotIn('pip show "${FLAGTREE_PACKAGE_NAME}"', upgrade["run"])
        self.assertIn(
            '":${PYTHONPATH:-}:"',
            upgrade["run"],
        )
        self.assertIn(
            'export PYTHONPATH="${FLAGTREE_INSTALL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"',
            upgrade["run"],
        )
        self.assertIn(
            'printf \'PYTHONPATH=%s\\n\' "${PYTHONPATH}" >> "${GITHUB_ENV}"',
            upgrade["run"],
        )
        self.assertIn('"${ACTIVE_PYTHON}" -c', upgrade["run"])
        self.assertIn("actual = metadata.version(package)", upgrade["run"])
        self.assertIn("actual == expected", upgrade["run"])
        self.assertLess(
            upgrade["run"].index('"show installed flagtree before upgrade"'),
            upgrade["run"].index('rm -rf -- "${FLAGTREE_INSTALL_DIR}"'),
        )

    def test_flagtree_upgrade_stages_and_activates_exact_hcu_daily(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"] for step in steps if step.get("name") == "Upgrade Flagtree"
        )
        replacements = {
            "${{ matrix.flagtree_package_name }}": "flagtree-hcu",
            "${{ matrix.flagtree_version }}": "0.7.1.dev20261104",
            "${{ matrix.flagtree_index_url }}": (
                "https://resource.flagos.net/repository/flagos-pypi-daily/simple"
            ),
            "${{ matrix.name }}": "BW1000",
        }
        for expression, value in replacements.items():
            script = script.replace(expression, value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flagtree"
            staging = root / "flagtree.new"
            source.mkdir()
            (source / "marker").write_text("image source", encoding="utf-8")
            script = script.replace("/opt/flagtree.new", str(staging)).replace(
                "/opt/flagtree", str(source)
            )
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
                'version="${FAKE_FLAGTREE_VERSION:-0.7.1.dev20261104}"\n'
                'case " $args " in\n'
                '  *" pip install "*) mkdir -p "$target"; '
                'printf "hcu daily" > "$target/marker"; '
                'mkdir -p "$target/flagtree_hcu-0.7.1.dev20261104.dist-info"; '
                "printf 'Metadata-Version: 2.1\\nName: flagtree-hcu\\n"
                'Version: %s\\n\' "$version" > '
                '"$target/flagtree_hcu-0.7.1.dev20261104.dist-info/METADATA" ;;\n'
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
                "flagtree-hcu==0.7.1.dev20261104",
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
            self.assertTrue(pre_upgrade_show.endswith(" flagtree"))
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
                f"[FlagTree] {source} is not present in PYTHONPATH; prepending it.",
                completed.stdout,
            )
            self.assertIn(
                f"[FlagTree] Effective PYTHONPATH: {source}", completed.stdout
            )
            self.assertIn(
                "[FlagTree] Active Python sees flagtree-hcu==0.7.1.dev20261104",
                completed.stdout,
            )

            github_env.write_text("", encoding="utf-8")
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
                    "PYTHONPATH": str(source),
                },
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn(
                f"[FlagTree] {source} already exists in PYTHONPATH; skipping prepend.",
                completed.stdout,
            )
            self.assertIn(
                f"[FlagTree] Effective PYTHONPATH: {source}", completed.stdout
            )
            self.assertEqual(
                github_env.read_text(encoding="utf-8"),
                f"PYTHONPATH={source}\n",
            )

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
                    "PYTHONPATH": str(source),
                    "FAKE_FLAGTREE_VERSION": "0.7.1.dev20261103",
                },
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "flagtree-hcu: active Python sees version 0.7.1.dev20261103, "
                "expected 0.7.1.dev20261104",
                completed.stderr,
            )

    def test_flagtree_upgrade_preserves_image_tree_when_staging_fails(self):
        steps = self.workflow["jobs"]["test-container"]["steps"]
        script = next(
            step["run"] for step in steps if step.get("name") == "Upgrade Flagtree"
        )
        replacements = {
            "${{ matrix.flagtree_package_name }}": "flagtree-nvidia",
            "${{ matrix.flagtree_version }}": "0.7.1.dev20261115",
            "${{ matrix.flagtree_index_url }}": (
                "https://resource.flagos.net/repository/flagos-pypi-daily/simple"
            ),
            "${{ matrix.name }}": "H800",
        }
        for expression, value in replacements.items():
            script = script.replace(expression, value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flagtree"
            staging = root / "flagtree.new"
            source.mkdir()
            (source / "marker").write_text("image source", encoding="utf-8")
            script = script.replace("/opt/flagtree.new", str(staging)).replace(
                "/opt/flagtree", str(source)
            )
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
                "Starting: remove image-provided FlagTree directory",
                completed.stdout,
            )

    def test_matrix_resolves_command_timeout_without_dependency_policy(self):
        prepare_steps = self.workflow["jobs"]["prepare"]["steps"]
        script = next(
            step["run"] for step in prepare_steps if step.get("id") == "build-matrix"
        )

        for repo, expected_timeout in (("FlagTrain", 1800), ("FlagGems", 1800)):
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
        for repo in (
            "FlagGems-vllm",
            "FlagGems-sglang",
            "FlagAttention",
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
            "repos.FlagTrain.deps.default + repos.FlagTrain.deps.H800",
            "Detected explicit backend dependencies: <empty>",
            "No explicit backend dependencies configured; skipping.",
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
