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

import subprocess
import sys
import types
from fnmatch import fnmatch
from unittest import mock

import pytest

from flagos_optest import gpu_check


def _runner(returncode=0, stdout="", stderr=""):
    return lambda *_args, **_kwargs: (returncode, stdout, stderr)


def test_run_and_vendor_environments(monkeypatch):
    completed = types.SimpleNamespace(returncode=3, stdout="out", stderr="err")
    monkeypatch.setattr(gpu_check.subprocess, "run", lambda *_a, **_k: completed)
    assert gpu_check._run(["tool"]) == (3, "out", "err")
    monkeypatch.setattr(
        gpu_check.subprocess,
        "run",
        mock.Mock(side_effect=subprocess.TimeoutExpired("tool", 1)),
    )
    assert gpu_check._run(["tool"])[0] == 124
    monkeypatch.setattr(
        gpu_check.subprocess, "run", mock.Mock(side_effect=OSError("missing"))
    )
    assert gpu_check._run(["tool"])[0] == 127

    monkeypatch.setenv("PATH", "/existing")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/libraries")
    for vendor, variable in (
        ("iluvatar", "LD_LIBRARY_PATH"),
        ("kunlunxin", "XPU_HOME"),
        ("mthreads", "MUSA_INSTALL_PATH"),
        ("tsingmicro", "KUIPER_HOME"),
        ("nvidia", "PATH"),
    ):
        assert variable in gpu_check._vendor_env(vendor)


@pytest.mark.parametrize(
    ("function", "output", "returncode", "which"),
    [
        (gpu_check.detect_nvidia, "", 0, None),
        (gpu_check.detect_nvidia, "", 0, "/bin/tool"),
        (gpu_check.detect_ascend, "", 1, "/bin/tool"),
        (gpu_check.detect_ascend, "nothing", 0, "/bin/tool"),
        (gpu_check.detect_cambricon, "", 127, "/bin/tool"),
        (gpu_check.detect_cambricon, "nothing", 0, "/bin/tool"),
        (gpu_check.detect_enflame, "nothing", 0, "/bin/tool"),
        (gpu_check.detect_hygon, "", 0, None),
        (gpu_check.detect_hygon, "nothing", 0, "/bin/tool"),
        (gpu_check.detect_iluvatar, "", 0, "/bin/tool"),
        (gpu_check.detect_kunlunxin, "", 0, "/bin/tool"),
        (gpu_check.detect_mthreads, "", 0, "/bin/tool"),
        (gpu_check.detect_sunrise, "", 127, "/bin/tool"),
        (gpu_check.detect_sunrise, "", 0, "/bin/tool"),
        (gpu_check.detect_thead, "", 0, None),
        (gpu_check.detect_thead, "", 0, "/bin/tool"),
        (gpu_check.detect_tsingmicro, "Chip count: a b c invalid", 0, "/bin/tool"),
        (gpu_check.detect_tsingmicro, "", 0, "/bin/tool"),
    ],
)
def test_detection_errors(monkeypatch, function, output, returncode, which):
    monkeypatch.setattr(gpu_check, "_run", _runner(returncode, output))
    monkeypatch.setattr(gpu_check.shutil, "which", lambda *_a, **_k: which)
    with pytest.raises(gpu_check.VendorCheckError):
        function({})


def test_amd_detection_error(monkeypatch):
    monkeypatch.setattr(gpu_check.shutil, "which", lambda *_a, **_k: None)
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.detect_amd({})


@pytest.mark.parametrize(
    "output",
    [
        "nothing",
        "Attached x x invalid",
        "Attached x x 0",
    ],
)
def test_metax_detection_errors(monkeypatch, output):
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout=output))
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.detect_metax({})
    monkeypatch.setattr(gpu_check, "_run", _runner(127, output))
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.detect_metax({})


def test_sample_error_and_defensive_paths(monkeypatch):
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="invalid"))
    gpus, available = gpu_check._sample_mib_pair([], {}, 1, "GPU", "bad {i}", None)
    assert (gpus, available) == ([], [])

    monkeypatch.setattr(gpu_check, "_run", _runner(1))
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.sample_ascend({}, 1)

    monkeypatch.setattr(
        gpu_check,
        "_run",
        _runner(stdout="0000:00:00.0 1 / 2"),
    )
    original_findall = gpu_check.re.findall
    monkeypatch.setattr(
        gpu_check.re,
        "findall",
        lambda pattern, text: (
            ["/"] if "\\d+\\s*/" in pattern else original_findall(pattern, text)
        ),
    )
    assert gpu_check.sample_ascend({}, 1) == ([], [])
    monkeypatch.setattr(gpu_check.re, "findall", original_findall)

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="1 MiB/ 2 MiB"))
    monkeypatch.setattr(gpu_check.re, "match", lambda *_a: None)
    assert gpu_check.sample_cambricon({}, 1) == ([], [])

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="not-json"))
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.sample_amd({}, None)
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout='{"card": {"other": "1"}}'))
    assert gpu_check.sample_amd({}, None) == ([], [])

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="Device Mem Info\nBAR1"))
    assert gpu_check.sample_enflame({}, 1) == ([], [])
    assert (
        gpu_check._enflame_mem_field("Device Mem Info\nBAR1\nTotal Size 4", "Total")
        == ""
    )

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="Used Memory:a:1"))
    assert gpu_check.sample_hygon({}, 2) == ([], [])
    assert gpu_check._hygon_mem_values("Used Memory:short", "Used") == []

    monkeypatch.setattr(gpu_check, "_csv_values", mock.Mock(side_effect=[["1"], []]))
    assert gpu_check.sample_iluvatar({}, 2) == ([], [])

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="MiB / 2MiB"))
    assert gpu_check.sample_kunlunxin({}, 1) == ([], [])

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="x MiB a b c d e f /"))
    assert gpu_check.sample_metax({}, 1) == ([], [])

    monkeypatch.setattr(
        gpu_check,
        "_run",
        _runner(stdout="FB Memory Usage\nTotal invalid\nUsed invalid"),
    )
    assert gpu_check.sample_mthreads({}, 1) == ([], [])
    assert gpu_check._mthreads_field(["Total none"], "Total") == ""

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout=""))
    assert gpu_check.sample_thead({}, 1) == ([], [])
    assert gpu_check.sample_tsingmicro({}, 1) == ([], [])

    monkeypatch.setattr(
        gpu_check,
        "_run",
        _runner(stdout="0000:00:00.0 1 / 2\nnon-device line"),
    )
    assert gpu_check.sample_ascend({}, 1)[0]
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="0000: no memory pair"))
    assert gpu_check.sample_ascend({}, 1) == ([], [])
    assert gpu_check._enflame_mem_field("ordinary line", "Total") == ""

    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="Attached x\n"))
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.detect_metax({})
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="matched MiB extra"))
    assert gpu_check.sample_metax({}, 1) == ([], [])
    monkeypatch.setattr(
        gpu_check,
        "_run",
        _runner(stdout="a b c d e f g h 90/100 MiB extra\nordinary line"),
    )
    assert gpu_check.sample_metax({}, 1)[1] == []
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="plain lines"))
    assert gpu_check.sample_mthreads({}, 1) == ([], [])
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="Chip count: short"))
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.detect_tsingmicro({})
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="x | short"))
    assert gpu_check.sample_tsingmicro({}, 1) == ([], [])


def test_below_threshold_sampling_branches(monkeypatch):
    monkeypatch.setattr(gpu_check, "MEM_THRESHOLD", 100)
    cases = [
        (
            gpu_check._sample_mib_pair,
            ([], {}, 1, "GPU", "bad {i}", None),
            "90MiB / 100MiB",
        ),
        (gpu_check.sample_ascend, ({}, 1), "0000:00:00.0 90 / 100"),
        (gpu_check.sample_cambricon, ({}, 1), "90 MiB/ 100 MiB"),
        (
            gpu_check.sample_enflame,
            ({}, 1),
            "Device Mem Info\nTotal Size 100\nFree Size 10\nBAR1",
        ),
        (gpu_check.sample_hygon, ({}, 1), "Used Memory:a:90\nTotal Memory:a:100"),
        (gpu_check.sample_iluvatar, ({}, 1), None),
        (gpu_check.sample_kunlunxin, ({}, 1), "a b c d e f g h 90MiB / 100MiB"),
        (gpu_check.sample_metax, ({}, 1), "a b c d e f g h 90/100 MiB extra"),
        (gpu_check.sample_mthreads, ({}, 1), "FB Memory Usage\nTotal 100\nUsed 90"),
        (gpu_check.sample_sunrise, ({}, 1), "90MiB / 100MiB"),
        (gpu_check.sample_thead, ({}, 1), "90MiB / 100MiB"),
        (gpu_check.sample_tsingmicro, ({}, 1), "x | a b c d e f 90M x 100M"),
    ]
    for function, args, output in cases:
        if function is gpu_check.sample_iluvatar:
            monkeypatch.setattr(
                gpu_check, "_csv_values", mock.Mock(side_effect=[["90"], ["100"]])
            )
        else:
            monkeypatch.setattr(gpu_check, "_run", _runner(stdout=output))
        _gpus, available = function(*args)
        assert available == []


def test_devices_commands_and_vendor_probe(monkeypatch):
    patterns = {
        "hygon": "/dev/mkfd",
        "amd": "/dev/kfd",
        "thead": "/dev/alixpu_ppu0",
        "nvidia": "/dev/nvidiactl",
        "iluvatar": "/dev/iluvatar0",
        "kunlunxin": "/dev/xpuctrl",
        "enflame": "/dev/gcuctl",
        "tsingmicro": "/dev/accel_drv_mgr",
        "metax": "/dev/mxcd",
        "cambricon": "/dev/cambricon_dev0",
        "mthreads": "/dev/mtgpu.0",
        "ascend": "/dev/davinci_manager",
        "sunrise": "/dev/ptpuctrl",
    }
    for vendor, present in patterns.items():
        monkeypatch.setattr(
            gpu_check.glob,
            "glob",
            lambda pattern, p=present: [p] if fnmatch(p, pattern) else [],
        )
        assert gpu_check._device_present(vendor)
    monkeypatch.setattr(gpu_check.glob, "glob", lambda _pattern: [])
    assert not gpu_check._device_present("unknown")
    assert not gpu_check._device_present("amd")

    monkeypatch.setattr(gpu_check.shutil, "which", lambda *_a, **_k: "/tool")
    assert gpu_check._command_present("nvidia")

    monkeypatch.setattr(gpu_check, "VENDOR_ORDER", ["a", "b"])
    monkeypatch.setattr(gpu_check, "_device_present", lambda key: key == "b")
    monkeypatch.setattr(gpu_check, "_command_present", lambda _key: True)
    assert gpu_check.probe_vendor() == "b"
    monkeypatch.setattr(gpu_check, "_command_present", lambda _key: False)
    assert gpu_check.probe_vendor() == "b"
    monkeypatch.setattr(gpu_check, "_device_present", lambda _key: False)
    assert gpu_check.probe_vendor() is None


def test_metadata_retry_render_and_auto_main(monkeypatch, capsys):
    monkeypatch.setattr(gpu_check, "probe_vendor", lambda: None)
    assert gpu_check.probe_accelerator_metadata()["vendor"] is None
    monkeypatch.setattr(gpu_check, "probe_vendor", lambda: "nvidia")
    monkeypatch.setattr(gpu_check, "_vendor_env", lambda _key: {})
    monkeypatch.setattr(
        gpu_check,
        "collect_metadata",
        lambda key, env, run: {
            "device": "GPU",
            "driver": None,
            "runtime": None,
            "runtime_version": None,
        },
    )
    monkeypatch.setattr(gpu_check, "_run", _runner(stdout="metadata"))
    assert gpu_check.probe_accelerator_metadata()["device"] == "GPU"

    vendor = {
        "unit": "GPU",
        "detect": lambda _env: None,
        "sample": lambda _env, _count: (
            [{"index": 0, "total": 2, "used": 0, "free": 2}],
            [0],
        ),
    }
    monkeypatch.setitem(gpu_check.VENDORS, "fake", vendor)
    monkeypatch.setitem(gpu_check.VENDOR_NAMES, "fake", "Fake")
    monkeypatch.setattr(
        gpu_check,
        "probe_accelerator_metadata",
        lambda _key: {
            "vendor": "fake",
            "device": None,
            "driver": None,
            "runtime": None,
            "runtime_version": None,
        },
    )
    result = gpu_check.run_vendor_check("fake")
    assert result["device_count"] == 1
    assert "unknown" in gpu_check.render_text(result)

    monkeypatch.setattr(sys, "argv", ["flagos-gpu-check"])
    monkeypatch.setattr(gpu_check, "probe_vendor", lambda: None)
    assert gpu_check.main() == 1

    samples = iter([([], []), ([], [])])
    monkeypatch.setitem(
        gpu_check.VENDORS,
        "wait",
        {
            "unit": "GPU",
            "detect": lambda _env: 1,
            "sample": lambda _env, _count: next(samples),
        },
    )
    monkeypatch.setattr(gpu_check, "SLEEP_TIME", 1)
    monkeypatch.setattr(gpu_check, "MAX_WAIT", 2)
    monkeypatch.setattr(gpu_check.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        gpu_check,
        "probe_accelerator_metadata",
        lambda _key: {"vendor": "wait"},
    )
    with pytest.raises(gpu_check.VendorCheckError):
        gpu_check.run_vendor_check("wait")

    monkeypatch.setattr(sys, "argv", ["flagos-gpu-check"])
    monkeypatch.setattr(gpu_check, "probe_vendor", lambda: "fake")
    monkeypatch.setattr(gpu_check, "run_vendor_check", lambda _key: result)
    assert gpu_check.main() == 0
    assert "no supported vendor" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["flagos-gpu-check", "--vendor", "fake"])
    monkeypatch.setattr(
        gpu_check, "run_vendor_check", mock.Mock(side_effect=gpu_check.VendorCheckError)
    )
    assert gpu_check.main() == 1
