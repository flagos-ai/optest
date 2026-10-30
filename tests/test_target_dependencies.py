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

import sys
from unittest import mock

import pytest

from flagos_optest import target_dependencies


@pytest.fixture
def target(tmp_path):
    def write(extra=""):
        (tmp_path / "pyproject.toml").write_text(
            "[project]\n"
            'name = "target"\n'
            'version = "0.0.0"\n'
            "[project.optional-dependencies]\n"
            f"{extra}\n",
            encoding="utf-8",
        )
        return tmp_path

    return write


def test_missing_vendor_extra_is_skipped(target, capsys):
    with mock.patch.object(target_dependencies.subprocess, "run") as run:
        installed = target_dependencies.install_vendor_dependencies(target(), "nvidia")

    assert installed is False
    run.assert_not_called()
    assert "skipping vendor dependencies" in capsys.readouterr().out


def test_vendor_extra_uses_normal_target_install(target):
    root = target('nvidia = ["deepspeed==0.19.8", "hjson==3.1.0"]')
    with mock.patch.object(target_dependencies.subprocess, "run") as run:
        installed = target_dependencies.install_vendor_dependencies(root, "nvidia")

    assert installed is True
    run.assert_called_once_with(
        [sys.executable, "-m", "pip", "install", f"{root}[nvidia]"], check=True
    )


def test_no_deps_installs_extra_requirements_directly(target, capsys):
    root = target(
        'nvidia = ["deepspeed==0.19.8", "deepspeed-kernels", '
        '"hjson==3.1.0", "py-cpuinfo==9.0.0"]'
    )
    with mock.patch.object(target_dependencies.subprocess, "run") as run:
        installed = target_dependencies.install_vendor_dependencies(
            root, "nvidia", no_deps=True
        )

    assert installed is True
    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "deepspeed==0.19.8",
            "deepspeed-kernels",
            "hjson==3.1.0",
            "py-cpuinfo==9.0.0",
        ],
        check=True,
    )
    output = capsys.readouterr().out
    assert f"Vendor extra source: {root / 'pyproject.toml'}" in output
    assert "[project.optional-dependencies].nvidia" in output
    assert "Detected vendor extra requirements (4):" in output
    assert "Effective vendor pip arguments: --no-deps" in output
    assert "Command:" in output
    assert "deepspeed==0.19.8" in output
    assert "Finished: resolved vendor requirement installation (exit_code=0)" in output


def test_vendor_install_failure_logs_exit_code(target, capsys):
    root = target('nvidia = ["deepspeed==0.19.8"]')
    error = target_dependencies.subprocess.CalledProcessError(7, "pip")
    with mock.patch.object(
        target_dependencies.subprocess, "run", side_effect=error
    ), pytest.raises(target_dependencies.subprocess.CalledProcessError):
        target_dependencies.install_vendor_dependencies(
            root, "nvidia", no_deps=True
        )

    output = capsys.readouterr().out
    assert "Finished: resolved vendor requirement installation (exit_code=7)" in output
