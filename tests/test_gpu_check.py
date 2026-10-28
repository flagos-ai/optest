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

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from flagos_optest import gpu_check, gpu_metadata

ILUVATAR_TABLE = """Timestamp    Wed Oct 28 11:22:04 2026
+--------------------------------------------------------------------------------+
| IX-ML: 5.0.0       Driver Version: 5.0.0       CUDA Version: 10.2              |
| 0    Iluvatar BI-V150 OAM        | 00000000:0C:00.0 | 500MHz 1600MHz            |
| N/A 25C P0 N/A / N/A             | 68MiB / 32768MiB | 0% Default                |
| 1    Iluvatar BI-V150 OAM        | 00000000:0F:00.0 | 500MHz 1600MHz            |
| N/A 25C P0 63W / 600W            | 68MiB / 32768MiB | 0% Default                |
| Processes: No running processes found                                          |
+--------------------------------------------------------------------------------+
"""


def command_runner(outputs):
    def run(cmd, env=None, timeout=None):
        value = outputs.get(tuple(cmd))
        return (0, value, "") if value is not None else (127, "", "not found")
    return run


class GPUCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Isolate metadata lookup from any SDKs installed on the test host.
        runtimes = {
            key: (name, variables, (str(self.root / name),))
            for key, (name, variables, _) in gpu_metadata.RUNTIMES.items()
        }
        patcher = mock.patch.dict(gpu_metadata.RUNTIMES, runtimes)
        patcher.start()
        self.addCleanup(patcher.stop)

    def install_version(self, runtime, filename, text):
        path = self.root / runtime / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def iluvatar_runner(self):
        return command_runner({
            ("ixsmi",): ILUVATAR_TABLE,
            ("ixsmi", "--query-gpu=name", "--format=csv,noheader"):
                "Iluvatar BI-V150 OAM\nIluvatar BI-V150 OAM\n",
            ("ixsmi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"):
                "68\n68\n",
            ("ixsmi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"):
                "32768\n32768\n",
        })

    def call_main(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["flagos-gpu-check", "--vendor", "iluvatar", *args]), \
                mock.patch.object(gpu_check, "_run", side_effect=self.iluvatar_runner()), \
                mock.patch.object(gpu_check, "_vendor_env", return_value={}), \
                redirect_stdout(stdout), redirect_stderr(stderr):
            status = gpu_check.main()
        return status, stdout.getvalue(), stderr.getvalue()

    def test_iluvatar_text_and_independent_versions(self):
        self.install_version("corex", "version.txt", "CoreX Version: 4.5.0\n")
        status, out, err = self.call_main([])
        self.assertEqual(status, 0)
        self.assertEqual(err, "")
        self.assertEqual(out, """Vendor: iluvatar
Device: BI-V150
Driver: 5.0.0
Runtime: corex
Runtime Version: 4.5.0
Detected 2 Iluvatar BI-V150 GPU(s).
Available GPUs: 0,1
GPUs Usage:
 GPU  Total (MiB)  Used (MiB)  Free (MiB)
   0       32,768          68      32,700
   1       32,768          68      32,700
""")

    def test_structured_stdout_contains_only_document(self):
        self.install_version("corex", "version.txt", "4.5.0\n")
        for fmt, load in (("json", json.loads), ("yaml", yaml.safe_load)):
            with self.subTest(fmt=fmt):
                status, out, err = self.call_main(["--format", fmt])
                self.assertEqual(status, 0)
                self.assertEqual(err, "")
                result = load(out)
                self.assertEqual(result["device"], "BI-V150")
                self.assertEqual(result["driver"], "5.0.0")
                self.assertNotIn("device_name", result)
                self.assertNotIn("driver_version", result)
                self.assertEqual(result["runtime_version"], "4.5.0")
                self.assertEqual(result["device_count"], 2)
                self.assertEqual(result["memory_unit"], "MiB")
                self.assertEqual(result["available_gpus"], "0,1")
                self.assertEqual(result["gpus"][0], {
                    "id": 0, "total": 32768, "used": 68, "free": 32700,
                })

    def test_output_file_matches_redirected_stdout_in_every_format(self):
        for fmt in ("text", "json", "yaml"):
            with self.subTest(fmt=fmt):
                path = self.root / f"status.{fmt}"
                _, expected, _ = self.call_main(["--format", fmt])
                status, out, err = self.call_main(["--format", fmt, "--output", str(path)])
                self.assertEqual((status, out, err), (0, "", ""))
                self.assertEqual(path.read_text(encoding="utf-8"), expected)

    def test_output_error_is_nonzero_and_only_on_stderr(self):
        status, out, err = self.call_main(["--format", "json", "--output", str(self.root)])
        self.assertEqual(status, 1)
        self.assertEqual(out, "")
        self.assertIn("cannot write", err)

    def test_unknown_corex_version_does_not_use_ixml_or_cuda(self):
        _, out, _ = self.call_main(["--format", "json"])
        self.assertIsNone(json.loads(out)["runtime_version"])

    def test_nvidia_uses_installed_cuda_not_smi_capability(self):
        run = command_runner({
            ("nvidia-smi",): "Driver Version: 580.95.05    CUDA Version: 13.3",
            ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"):
                "NVIDIA H800 80GB HBM3\nNVIDIA H800 80GB HBM3\n",
            ("nvcc", "--version"): "Cuda compilation tools, release 12.8, V12.8.93",
        })
        result = gpu_metadata.collect_metadata("nvidia", {}, run)
        self.assertEqual(result, {
            "device": "H800", "driver": "580.95.05",
            "runtime": "cuda", "runtime_version": "12.8",
        })
        self.install_version("cuda", "version.json", '{"cuda": {"version": "13.3.0"}}')
        self.assertEqual(gpu_metadata.collect_metadata("nvidia", {}, run)["runtime_version"], "13.3.0")

    def test_all_vendor_model_and_runtime_mappings(self):
        examples = [
            ("ascend", "| 0  910B  OK |", "910B", "cann", "9.0.0"),
            ("cambricon", "| 0 MLU590-M9DE |", "MLU590-M9DE", "neuware", "4.7.2"),
            ("amd", "Card series: AMD Radeon PRO W7900D", "Radeon PRO W7900D", "rocm", "6.4.0"),
            ("enflame", "| 0 Enflame S60 |", "S60", "tops", "1.10.6"),
            ("hygon", "Card Series: BW1000", "BW1000", "dtk", "26.04"),
            ("kunlunxin", "| 0 KunlunXin P800 |", "P800", "xre", "5.37.1"),
            ("metax", "| 0 MetaX C500 |", "C500", "maca", "3.8.1.3"),
            ("mthreads", "GPU 0: MTT S5000", "MTT-S5000", "musa", "5.2.0"),
            ("sunrise", "Device Name: Sunrise P1000", "P1000", "tangrt", "1.0.0"),
            ("thead", "GPU 0: PPU-ZW810E", "PPU-ZW810E", "ppu", "2.0.0"),
            ("tsingmicro", "| TX8110 |", "TX8110", "tsm", "1.0.0"),
        ]
        for key, table, name, runtime, version in examples:
            with self.subTest(vendor=key):
                self.install_version(runtime, "version.txt", version)
                cmd = [gpu_metadata.COMMANDS[key]]
                if key == "ascend":
                    cmd += ["info"]
                run = command_runner({tuple(cmd): table + "\nDriver Version: 5.0.0"})
                result = gpu_metadata.collect_metadata(key, {}, run)
                self.assertEqual(result["device"], name)
                self.assertEqual(result["driver"], "5.0.0")
                self.assertEqual(result["runtime"], runtime)
                self.assertEqual(result["runtime_version"], version)

    def test_amd_device_model_falls_back_to_exact_sysfs_pci_identity(self):
        devices = self.root / "sys" / "bus" / "pci" / "devices"
        gpu = devices / "0000:03:00.0"
        gpu.mkdir(parents=True)
        for field, value in {
            "vendor": "0x1002", "device": "0x744b", "class": "0x030000",
            "subsystem_vendor": "0x1002", "subsystem_device": "0x0e0c",
        }.items():
            (gpu / field).write_text(value + "\n", encoding="ascii")
        runner = command_runner({
            ("rocm-smi",): "Card Model: N/A\n",
            ("rocm-smi", "--showproductname", "--showdriverversion"):
                "GPU[0]: get_name, Error when calling libdrm\nDriver version: 6.16.13\n",
        })
        with mock.patch.object(gpu_metadata, "AMD_SYSFS_ROOT", devices):
            result = gpu_metadata.collect_metadata("amd", {}, runner)
        self.assertEqual(result["device"], "W7900D")
        self.assertEqual(result["driver"], "6.16.13")

    def test_amd_sysfs_fallback_does_not_guess_unknown_or_mixed_models(self):
        devices = self.root / "sys" / "bus" / "pci" / "devices"
        for number, device_id in enumerate(("0x744b", "0x744c")):
            gpu = devices / f"0000:0{number}:00.0"
            gpu.mkdir(parents=True)
            for field, value in {
                "vendor": "0x1002", "device": device_id, "class": "0x030000",
                "subsystem_vendor": "0x1002", "subsystem_device": "0x0e0c",
            }.items():
                (gpu / field).write_text(value + "\n", encoding="ascii")
        with mock.patch.object(gpu_metadata, "AMD_SYSFS_ROOT", devices):
            self.assertIsNone(gpu_metadata._amd_model_from_sysfs())

    def test_explicit_installation_root_precedes_default(self):
        self.install_version("corex", "version.txt", "5.0.0")
        self.install_version("custom", "version.txt", "4.5.0")
        result = gpu_metadata.collect_metadata(
            "iluvatar", {"COREX_HOME": str(self.root / "custom")}, self.iluvatar_runner(),
        )
        self.assertEqual(result["runtime_version"], "4.5.0")
        self.install_version("cuda", "version.json", '{"cuda": {"version": "13.3.0"}}')
        cuda_root = str(self.root / "selected-cuda")
        run = command_runner({
            (f"{cuda_root}/bin/nvcc", "--version"): "Cuda compilation tools, release 12.8",
        })
        result = gpu_metadata.collect_metadata("nvidia", {"CUDA_HOME": cuda_root}, run)
        self.assertEqual(result["runtime_version"], "12.8")

    def test_ascend_install_info_and_cuda_component_isolation(self):
        self.install_version("cann", "aarch64-linux/ascend_toolkit_install.info",
                             "arch=aarch64\nversion=9.0.0\n")
        run = command_runner({("npu-smi", "info"): "| 0 910B OK |"})
        result = gpu_metadata.collect_metadata("ascend", {}, run)
        self.assertEqual(result["runtime_version"], "9.0.0")
        self.install_version("cuda", "version.json", '{"cuda_nvcc": {"version": "13.3.0"}}')
        result = gpu_metadata.collect_metadata("nvidia", {}, command_runner({}))
        self.assertIsNone(result["runtime_version"])

    def test_cann_runtime_manifest_ignores_components_and_requirements(self):
        run = command_runner({("npu-smi", "info"): "| 0 910B4 OK |"})
        self.install_version("cann", "share/info/msprof/version.info",
                             "[PACKAGE]\nVersion=26.0.0\n")
        self.install_version("cann", "share/info/runtime/version.info",
                             'required_package_runtime_version="9.0"\n')
        self.assertIsNone(gpu_metadata.collect_metadata("ascend", {}, run)["runtime_version"])
        self.install_version("cann", "share/info/runtime/version.info",
                             "Version=9.0.0\nversion_dir=cann\n")
        self.assertEqual(gpu_metadata.collect_metadata("ascend", {}, run)["runtime_version"],
                         "9.0.0")

    def test_ppu_runtime_uses_sdk_manifest_not_smi_tool_version(self):
        run = command_runner({
            ("ppu-smi",): "Driver Version: 1.3.2-d7f5a2\nGPU 0: PPU-ZW810E",
            ("ppu-smi", "--version"): "ppu-smi version: 1.28",
        })
        self.install_version("ppu", "VERSION.txt", "ppu_sdk_detection_magic\n")
        self.assertIsNone(gpu_metadata.collect_metadata("thead", {}, run)["runtime_version"])
        self.install_version("ppu", "release.yaml", "version: 2.1.0-a5f865\n")
        self.assertEqual(gpu_metadata.collect_metadata("thead", {}, run)["runtime_version"],
                         "2.1.0-a5f865")

    def test_mixed_models_are_not_silently_reduced_to_first_model(self):
        run = command_runner({("mx-smi",): "| 0 C500 |\n| 1 C550 |\n| 2 C500 |"})
        self.assertEqual(gpu_metadata.collect_metadata("metax", {}, run)["device"], "C500, C550")

    def test_xre_uses_runtime_header_not_driver_compatibility(self):
        self.install_version("xre", "version.txt", "Driver compatible version: 4.0.0\n")
        run = command_runner({("xpu-smi",): "| 0 P800 |"})
        self.assertIsNone(gpu_metadata.collect_metadata("kunlunxin", {}, run)["runtime_version"])
        self.install_version("xre", "include/xpu/version.h",
                             '#define XPURT_VERSION_STR "5.37.1"\n')
        self.assertEqual(gpu_metadata.collect_metadata("kunlunxin", {}, run)["runtime_version"],
                         "5.37.1")

    def test_release_suffix_and_timestamp_versions(self):
        for key, runtime, version in (("ascend", "cann", "9.0.0_RC1"),
                                      ("tsingmicro", "tsm", "260610164501")):
            with self.subTest(vendor=key):
                self.install_version(runtime, "version.txt", version)
                result = gpu_metadata.collect_metadata(key, {}, command_runner({}))
                self.assertEqual(result["runtime_version"], version)

    def test_missing_metadata_does_not_fail_availability(self):
        run = self.iluvatar_runner()
        def without_metadata(cmd, env=None, timeout=None):
            if cmd == ["ixsmi"]:
                return 1, "", "unsupported"
            return run(cmd, env)
        with mock.patch.object(gpu_check, "_run", side_effect=without_metadata), \
                mock.patch.object(gpu_check, "_vendor_env", return_value={}):
            result = gpu_check.run_vendor_check("iluvatar")
        self.assertIsNone(result["driver"])
        self.assertEqual(result["available_gpus"], "0,1")

    def test_retry_diagnostics_do_not_contaminate_structured_stdout(self):
        for fmt in ("json", "yaml"):
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.object(gpu_check, "MEM_THRESHOLD", 40000), \
                    mock.patch.object(gpu_check, "MAX_WAIT", 1), \
                    mock.patch.object(gpu_check.time, "sleep"), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                status, out, err = self.call_main(["--format", fmt])
            self.assertEqual(status, 1)
            self.assertEqual(out, "")
            self.assertIn("Timed out", err)
            self.assertNotIn("Timestamp", err)

    def test_sunrise_memory_is_normalized_to_mib(self):
        run = command_runner({
            ("pt_smi",): "0B / 64GiB\n1048577B / 67108864KiB\n1GiB / 65536MiB",
        })
        with mock.patch.object(gpu_check, "_run", side_effect=run):
            self.assertEqual(gpu_check.detect_sunrise({}), 3)
            gpus, available = gpu_check.sample_sunrise({}, 3)
        self.assertEqual([g["used"] for g in gpus], [0, 2, 1024])
        self.assertEqual(available, [0, 1, 2])

    def test_inaccessible_metadata_paths_are_optional(self):
        with mock.patch.object(Path, "glob", side_effect=PermissionError("denied")):
            result = gpu_metadata.collect_metadata("iluvatar", {}, self.iluvatar_runner())
        self.assertIsNone(result["runtime_version"])

    def test_optional_metadata_commands_have_short_timeouts(self):
        run = self.iluvatar_runner()
        timeouts = []
        def track(cmd, env=None, timeout=None):
            if cmd == ["ixsmi"] or "--query-gpu=driver_version" in cmd:
                timeouts.append(timeout)
                return 124, "", "timed out"
            return run(cmd, env)
        with mock.patch.object(gpu_check, "_run", side_effect=track), \
                mock.patch.object(gpu_check, "_vendor_env", return_value={}):
            result = gpu_check.run_vendor_check("iluvatar")
        self.assertEqual(timeouts, [5, 5])
        self.assertEqual(result["available_gpus"], "0,1")


if __name__ == "__main__":
    unittest.main()
