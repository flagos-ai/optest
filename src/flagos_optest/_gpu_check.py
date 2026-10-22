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

"""Console entry points for the packaged vendor GPU checks."""

import subprocess
import sys
from pathlib import Path


def _run(script_name):
    """Run a packaged shell check while preserving its arguments and status."""
    script = Path(__file__).resolve().with_name(script_name)
    if not script.is_file():
        print(f"GPU check script not found: {script}", file=sys.stderr)
        return 1
    return subprocess.run(["bash", str(script), *sys.argv[1:]], check=False).returncode


def gpu_check_amd():
    return _run("gpu_check_amd.sh")


def gpu_check_ascend():
    return _run("gpu_check_ascend.sh")


def gpu_check_cambricon():
    return _run("gpu_check_cambricon.sh")


def gpu_check_enflame():
    return _run("gpu_check_enflame.sh")


def gpu_check_hygon():
    return _run("gpu_check_hygon.sh")


def gpu_check_iluvatar():
    return _run("gpu_check_iluvatar.sh")


def gpu_check_kunlunxin():
    return _run("gpu_check_kunlunxin.sh")


def gpu_check_metax():
    return _run("gpu_check_metax.sh")


def gpu_check_mthreads():
    return _run("gpu_check_mthreads.sh")


def gpu_check_nvidia():
    return _run("gpu_check_nvidia.sh")


def gpu_check_sunrise():
    return _run("gpu_check_sunrise.sh")


def gpu_check_thead():
    return _run("gpu_check_thead.sh")


def gpu_check_tsingmicro():
    return _run("gpu_check_tsingmicro.sh")
