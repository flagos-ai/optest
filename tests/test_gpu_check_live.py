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

"""Replay sanitized tool output captured on vendor GPU machines."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from flagos_optest import gpu_check, gpu_metadata

FIXTURES = Path(__file__).parent / "fixtures" / "gpu_check"
CAPTURES = sorted(FIXTURES.glob("*.json"))


@pytest.fixture(params=CAPTURES, ids=lambda path: path.stem)
def capture(request, tmp_path, monkeypatch):
    data = json.loads(request.param.read_text(encoding="utf-8"))
    for name, text in data["files"].items():
        path = tmp_path / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def isolated_path(name):
        return tmp_path / str(name).lstrip("/")

    outputs = {tuple(c["argv"]): (c["returncode"], c["stdout"], c["stderr"])
               for c in data["commands"]}

    def run(argv, env=None, timeout=None):
        # Installation-root compilers are redirected through the temporary tree.
        argv = [arg.removeprefix(str(tmp_path)) for arg in argv]
        return outputs.get(tuple(argv), (127, "", "not present in capture"))

    monkeypatch.setattr(gpu_metadata, "Path", isolated_path)
    monkeypatch.setattr(gpu_metadata, "AMD_SYSFS_ROOT", isolated_path("/sys/bus/pci/devices"))
    monkeypatch.setattr(gpu_check, "_run", run)
    monkeypatch.setattr(gpu_check, "_vendor_env", lambda vendor: data["environment"])
    monkeypatch.setattr(gpu_check.shutil, "which", lambda *args, **kwargs: "/captured/tool")
    monkeypatch.setattr(gpu_check, "MAX_WAIT", 0)
    return data


def test_live_metadata_and_memory(capture):
    result = gpu_check.run_vendor_check(capture["vendor"])
    assert result == capture["expected"]


@pytest.mark.parametrize("fmt", ["text", "json", "yaml"])
def test_live_cli_documents(capture, fmt, capsys, tmp_path):
    argv = ["flagos-gpu-check", "--vendor", capture["vendor"], "--format", fmt]
    with mock.patch.object(sys, "argv", argv):
        assert gpu_check.main() == 0
    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    if fmt == "text":
        assert stdout == capture["text"]
    else:
        load = json.loads if fmt == "json" else yaml.safe_load
        assert load(stdout) == capture["expected"]
    target = tmp_path / ("result." + fmt)
    with mock.patch.object(sys, "argv", [*argv, "--output", str(target)]):
        assert gpu_check.main() == 0
    assert capsys.readouterr() == ("", "")
    assert target.read_text(encoding="utf-8") == stdout
