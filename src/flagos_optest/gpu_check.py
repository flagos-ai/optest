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

# -*- coding: utf-8 -*-
"""
Unified vendor GPU availability check: ``flagos-gpu-check``.

This single command replaces the thirteen per-vendor ``gpu_check_<vendor>.sh``
scripts. It probes the host vendor, collects device and runtime metadata, and
reports memory availability once at least one device has enough free memory.

Probing uses two signals, in this priority order:

  1. a vendor-unique ``/dev`` device node (primary, most reliable), and
  2. the vendor's SMI command resolving on ``PATH`` (secondary confirmation).

Device-node probing is preferred because several vendors ship an
``nvidia-smi``-compatible shim, so command name alone is ambiguous. See
``_device_present`` and ``VENDOR_ORDER`` for the collision rules; do not reorder
them without reading the inline comments.

Output format is selectable with ``--format {text,json,yaml}`` (default
``text`` for a concise human-readable summary), and the
rendered result can be saved to a file with ``--output PATH``.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import yaml

if __package__:
    from .gpu_metadata import VENDOR_NAMES, collect_metadata
else:
    from gpu_metadata import VENDOR_NAMES, collect_metadata

# Configuration parameters (identical to the original shell scripts).
MEM_THRESHOLD = 30000  # Minimum free memory required (MiB)
SLEEP_TIME = 120       # Wait time between retries (seconds)
MAX_WAIT = 600         # Maximum total wait time (seconds)


class VendorCheckError(Exception):
    """Raised after the relevant diagnostic has been printed to stderr."""


def eprint(msg=""):
    """Print to stderr (used where the original script wrote to stderr)."""
    print(msg, file=sys.stderr)


def _run(cmd, env=None, timeout=None):
    """Run a command, returning (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True,
                              env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out: {cmd[0]}"
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _table_header(unit):
    """Return the per-GPU table header, e.g. ' GPU  Total (MiB)  ...'."""
    return f" {unit}  Total (MiB)  Used (MiB)  Free (MiB)"


def _table_row(index, total, used, free):
    """Reproduce the shell ``printf "%4d%'13d%'12d%'12d"`` (thousands grouped)."""
    return f"{index:4d}{total:13,d}{used:12,d}{free:12,d}"


# --------------------------------------------------------------------------- #
# Per-vendor environment
# --------------------------------------------------------------------------- #
def _vendor_env(key):
    """Return an environment dict with the vendor's toolchain paths exported.

    A few vendors install their SMI tool outside the default PATH; these mirror
    the ``export`` lines from the original per-vendor scripts so both the probe
    (``command -v``) and the queries resolve the right binary.
    """
    env = os.environ.copy()

    def prepend(var, value):
        existing = env.get(var, "")
        env[var] = f"{value}:{existing}" if existing else value

    if key == "iluvatar":
        prepend("LD_LIBRARY_PATH", "/usr/local/corex/lib")
    elif key == "kunlunxin":
        root = env.setdefault("XPU_HOME", "/usr/local/xpu")
        prepend("PATH", f"{root}/bin")
        prepend("LD_LIBRARY_PATH", f"{root}/lib")
    elif key == "mthreads":
        root = env.setdefault("MUSA_INSTALL_PATH", "/usr/local/musa")
        prepend("PATH", f"{root}/bin")
        prepend("LD_LIBRARY_PATH", f"{root}/lib")
    elif key == "tsingmicro":
        root = env.setdefault("KUIPER_HOME", "/home/secure/runtime/kuiper")
        prepend("PATH", f"{root}/bin")
        prepend("LD_LIBRARY_PATH", f"{root}/lib")
    return env


# --------------------------------------------------------------------------- #
# Vendor detect() / sample() implementations.
#
# detect(env)        -> device count (int) or None; raises VendorCheckError
#                       after printing a diagnostic to stderr.
# sample(env, count) -> (gpus, available) where gpus is a list of
#                       {"index","total","used","free"} dicts and available is a
#                       list of indices with free >= MEM_THRESHOLD.
# --------------------------------------------------------------------------- #

# ---- nvidia --------------------------------------------------------------- #
def detect_nvidia(env):
    if shutil.which("nvidia-smi", path=env.get("PATH")) is None:
        eprint("Error: nvidia-smi command not found.")
        raise VendorCheckError()
    _, out, _ = _run(["nvidia-smi", "-L"], env=env)
    count = sum(1 for line in out.splitlines() if "NVIDIA " in line)
    if count == 0:
        eprint("No NVIDIA GPU cards detected.")
        raise VendorCheckError()
    return count


def _sample_mib_pair(cmd, env, count, unit, warn_query, warn_parse):
    """Shared sampler for nvidia/thead style ``<cmd> -i N`` '<used>MiB / <total>MiB'."""
    gpus = []
    available = []
    for i in range(count):
        _, out, _ = _run(cmd + [str(i)], env=env)
        m = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", out)
        if not m:
            eprint(warn_query.format(i=i))
            continue
        used = int(m.group(1))
        total = int(m.group(2))
        free = total - used
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


def sample_nvidia(env, count):
    return _sample_mib_pair(
        ["nvidia-smi", "-i"], env, count, "GPU",
        "Warning: Failed to query memory on GPU {i}.",
        None,
    )


# ---- ascend (huawei) ------------------------------------------------------ #
def detect_ascend(env):
    rc, out, err = _run(["npu-smi", "info"], env=env)
    combined = out + err  # original runs `npu-smi info 2>&1`
    if rc != 0:
        eprint("Failed to run npu-smi. Please check if npu-smi is installed and working correctly.")
        raise VendorCheckError()
    count = sum(1 for line in combined.splitlines() if "OK" in line)
    if count == 0:
        eprint("No Ascend NPUs detected. Please ensure you have Ascend NPUs installed and properly configured.")
        raise VendorCheckError()
    return count


def sample_ascend(env, count):
    rc, out, _ = _run(["npu-smi", "info"], env=env)
    if rc != 0:
        eprint("Failed to query NPU information.")
        raise VendorCheckError()
    # Chip lines contain a PCIe address like "0000:"; the HBM usage is the last
    # "<used> / <total>" pair on the line.
    hbm = []
    for line in out.splitlines():
        if "0000:" in line:
            pairs = re.findall(r"\d+\s*/\s*\d+", line)
            if pairs:
                hbm.append(pairs[-1])
    gpus = []
    available = []
    for i, entry in enumerate(hbm):
        parts = entry.split("/")
        used_s = parts[0].strip()
        total_s = parts[1].strip() if len(parts) > 1 else ""
        if not used_s or not total_s:
            eprint(f"Warning: Failed to parse memory info for chip {i}.")
            continue
        used = int(used_s)
        total = int(total_s)
        free = total - used
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- cambricon ------------------------------------------------------------ #
def detect_cambricon(env):
    rc, out, _ = _run(["cnmon"], env=env)
    if rc == 127:
        eprint("Failed to run cnmon. Please check if cnmon is installed and working correctly.")
        raise VendorCheckError()
    count = len(re.findall(r"\d+ MiB/ \d+ MiB", out))
    # NOTE: the original shell used `$?` of `cnmon | grep -c`, so a zero match
    # (grep exit 1) actually printed the "Failed to run cnmon" message. Here we
    # distinguish the two cases: a missing/erroring cnmon vs. a healthy cnmon
    # that simply reports no cards, which is clearer and keeps both messages.
    if count == 0:
        eprint("No Cambricon MLUs detected. Please ensure you have MLU cards installed and properly configured.")
        raise VendorCheckError()
    return count


def sample_cambricon(env, count):
    _, out, _ = _run(["cnmon"], env=env)
    lines = re.findall(r"\d+ MiB/ \d+ MiB", out)
    gpus = []
    available = []
    i = 0
    for line in lines:
        used_m = re.match(r"^\d+", line)
        total_m = re.search(r"\d+(?= MiB$)", line)
        if not used_m or not total_m:
            eprint(f"Warning: Failed to parse memory info for card {i}.")
            i += 1
            continue
        used = int(used_m.group(0))
        total = int(total_m.group(0))
        free = total - used
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
        i += 1
    return gpus, available


# ---- amd ------------------------------------------------------------------ #
def detect_amd(env):
    if shutil.which("rocm-smi", path=env.get("PATH")) is None:
        eprint("Error: rocm-smi command not found. Please check if ROCm is installed.")
        raise VendorCheckError()
    # AMD iterates over the JSON cards; no separate count/preamble.


def sample_amd(env, count):
    # rocm-smi reports VRAM in bytes via JSON; keys differ across ROCm versions,
    # so match on lowercased substrings ("total memory" / "used memory") and
    # convert to MiB. Rendering is handled after sampling succeeds.
    _, out, _ = _run(["rocm-smi", "--showmeminfo", "vram", "--json"], env=env)
    try:
        data = json.loads(out)
    except ValueError:
        eprint("Failed to parse rocm-smi JSON output.")
        eprint("Failed to query GPU memory information via rocm-smi.")
        raise VendorCheckError()
    gpus = []
    available = []
    for i, (_card, info) in enumerate(sorted(data.items())):
        total_b = used_b = None
        for k, v in info.items():
            kl = k.lower()
            if "total memory" in kl:
                total_b = int(v)
            elif "used memory" in kl:
                used_b = int(v)
        if total_b is None or used_b is None:
            continue
        total = total_b // (1024 * 1024)
        used = used_b // (1024 * 1024)
        free = total - used
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- enflame -------------------------------------------------------------- #
def detect_enflame(env):
    _, out, _ = _run(["efsmi", "-L"], env=env)
    count = sum(1 for line in out.splitlines()
                if re.match(r"^\s*\d+\s+.*\b[\da-fA-F]{4}:[\da-fA-F]{2}:[\da-fA-F]{2}\.\d\b", line))
    if count == 0:
        eprint("No Enflame GPUs detected. Please ensure you have Enflame GPUs installed and properly configured.")
        raise VendorCheckError()
    return count


def _enflame_mem_field(text, label):
    """Within the 'Device Mem Info'..'BAR1' block, return the digits of a line."""
    in_section = False
    for line in text.splitlines():
        if "Device Mem Info" in line:
            in_section = True
        if in_section and "BAR1" in line:
            break
        if in_section and label in line:
            return re.sub(r"\D", "", line)
    return ""


def sample_enflame(env, count):
    gpus = []
    available = []
    for i in range(count):
        _, out, _ = _run(["efsmi", "-i", str(i), "-q", "-d", "MEMORY"], env=env)
        total_s = _enflame_mem_field(out, "Total Size")
        free_s = _enflame_mem_field(out, "Free Size")
        if not free_s or not total_s:
            eprint(f"Warning: Failed to parse memory info for chip {i}.")
            continue
        total = int(total_s)
        free = int(free_s)
        used = total - free
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- hygon ---------------------------------------------------------------- #
def detect_hygon(env):
    if shutil.which("hy-smi", path=env.get("PATH")) is None:
        eprint("Error: hy-smi command not found. Please check if DTK is installed.")
        raise VendorCheckError()
    _, out, _ = _run(["hy-smi"], env=env)
    count = sum(1 for line in out.splitlines() if "Normal" in line)
    if count == 0:
        eprint("No Hygon DCU cards detected.")
        raise VendorCheckError()
    return count


def _hygon_mem_values(text, label):
    """Collect the 3rd ':'-delimited field of each line matching ``label``."""
    values = []
    for line in text.splitlines():
        if label in line:
            fields = line.split(":")
            if len(fields) >= 3:
                values.append(fields[2].strip())
    return values


def sample_hygon(env, count):
    _, out, _ = _run(["hy-smi", "--showmeminfo", "vram"], env=env)
    usage = _hygon_mem_values(out, "Used Memory")
    total = _hygon_mem_values(out, "Total Memory")
    gpus = []
    available = []
    for i in range(count):
        if i >= len(total) or i >= len(usage):
            continue
        total_i = int(total[i])
        used_i = int(usage[i])
        free_i = total_i - used_i
        gpus.append({"index": i, "total": total_i, "used": used_i, "free": free_i})
        if free_i >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- iluvatar (tianshu) --------------------------------------------------- #
def detect_iluvatar(env):
    _, out, _ = _run(["ixsmi", "--query-gpu=name", "--format=csv,noheader"], env=env)
    count = sum(1 for line in out.splitlines() if line.strip())
    if count == 0:
        eprint("No Iluvatar GPUs detected. Please ensure you have Iluvatar GPUs installed and properly configured.")
        raise VendorCheckError()
    return count


def _csv_values(cmd, env):
    _, out, _ = _run(cmd, env=env)
    return [line.strip() for line in out.splitlines() if line.strip()]


def sample_iluvatar(env, count):
    usage = _csv_values(["ixsmi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], env)
    total = _csv_values(["ixsmi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], env)
    gpus = []
    available = []
    for i in range(count):
        if i >= len(usage) or i >= len(total):
            continue
        used_i = int(usage[i])
        total_i = int(total[i])
        free_i = total_i - used_i
        gpus.append({"index": i, "total": total_i, "used": used_i, "free": free_i})
        if free_i >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- kunlunxin ------------------------------------------------------------ #
def detect_kunlunxin(env):
    _, out, _ = _run(["xpu-smi"], env=env)
    count = sum(1 for line in out.splitlines() if "P800" in line)
    if count == 0:
        eprint("No KLX GPUs detected. Please ensure you have KLX cards installed and properly configured.")
        raise VendorCheckError()
    return count


def sample_kunlunxin(env, count):
    _, out, _ = _run(["xpu-smi"], env=env)
    usage = []
    total = []
    # awk '/MiB \/ [0-9]*MiB/ { print $9 (used), $11 (total) }'
    for line in out.splitlines():
        if re.search(r"MiB / [0-9]*MiB", line):
            fields = line.split()
            if len(fields) > 10:
                usage.append(fields[8])
                total.append(fields[10])
    gpus = []
    available = []
    for i in range(count):
        if i >= len(total) or i >= len(usage):
            continue
        # Strip the trailing unit/suffix (shell: ${var%???} drops 3 chars).
        total_i = int(total[i][:-3])
        used_i = int(usage[i][:-3])
        free_i = total_i - used_i
        gpus.append({"index": i, "total": total_i, "used": used_i, "free": free_i})
        if free_i >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- metax ---------------------------------------------------------------- #
def detect_metax(env):
    rc, out, _ = _run(["mx-smi"], env=env)
    if rc == 127:
        eprint("Failed to run mx-smi. Please check if mx-smi is installed and working correctly.")
        raise VendorCheckError()
    count = None
    for line in out.splitlines():
        if "Attached" in line:
            fields = line.split()
            if len(fields) > 3:
                count = fields[3]
            break
    if count is None:
        eprint("Failed to run mx-smi. Please check if mx-smi is installed and working correctly.")
        raise VendorCheckError()
    try:
        count = int(count)
    except ValueError:
        count = 0
    if count == 0:
        eprint("No MetaX GPUs detected. Please ensure you have MetaX GPUs installed and properly configured.")
        raise VendorCheckError()
    return count


def sample_metax(env, count):
    _, out, _ = _run(["mx-smi"], env=env)
    entries = []
    for line in out.splitlines():
        if re.search(r"MiB\s| A", line):
            fields = line.split()
            if len(fields) > 8:
                entries.append(fields[8])
    gpus = []
    available = []
    i = 0
    for entry in entries:
        parts = entry.split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        used_i = int(parts[0])
        total_i = int(parts[1])
        free_i = total_i - used_i
        gpus.append({"index": i, "total": total_i, "used": used_i, "free": free_i})
        if free_i >= MEM_THRESHOLD:
            available.append(i)
        i += 1
    return gpus, available


# ---- mthreads ------------------------------------------------------------- #
def detect_mthreads(env):
    _, out, _ = _run(["mthreads-gmi", "-L"], env=env)
    count = sum(1 for line in out.splitlines() if "GPU " in line)
    if count == 0:
        eprint("No Moore Threads GPUs detected. Please ensure you have GPUs installed and properly configured.")
        raise VendorCheckError()
    return count


def _mthreads_field(section_lines, keyword):
    for line in section_lines:
        if keyword in line:
            m = re.search(r"\d+", line)
            if m:
                return m.group(0)
    return ""


def sample_mthreads(env, count):
    gpus = []
    available = []
    for i in range(count):
        _, out, _ = _run(["mthreads-gmi", "-q", "-d", "MEMORY", "-i", str(i)], env=env)
        lines = out.splitlines()
        section = []
        for idx, line in enumerate(lines):
            if "FB Memory Usage" in line:
                section = lines[idx:idx + 4]  # matching line + next 3 (grep -A 3)
                break
        total_s = _mthreads_field(section, "Total")
        used_s = _mthreads_field(section, "Used")
        if not used_s or not total_s:
            eprint(f"Warning: Failed to query GPU {i} memory information.")
            continue
        total_i = int(total_s)
        used_i = int(used_s)
        free_i = total_i - used_i
        gpus.append({"index": i, "total": total_i, "used": used_i, "free": free_i})
        if free_i >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- sunrise -------------------------------------------------------------- #
def detect_sunrise(env):
    rc, out, _ = _run(["pt_smi"], env=env)
    if rc == 127:
        eprint("Failed to run pt_smi. Please check if pt_smi is installed and working correctly.")
        raise VendorCheckError()
    count = len(_sunrise_memory(out))
    if count == 0:
        eprint("No Sunrise GPUs detected. Please ensure you have Sunrise GPUs installed and properly configured.")
        raise VendorCheckError()
    return count


def _sunrise_memory(text):
    return re.findall(
        r"(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB)\s*/\s*"
        r"(\d+(?:\.\d+)?)\s*(B|KiB|MiB|GiB)", text,
    )


def _to_mib(value, unit, rounding):
    scale = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}[unit]
    return int((Decimal(value) * scale / 1024**2).to_integral_value(rounding=rounding))


def sample_sunrise(env, count):
    _, out, _ = _run(["pt_smi"], env=env)
    lines = _sunrise_memory(out)
    gpus = []
    available = []
    for i, (used_s, used_unit, total_s, total_unit) in enumerate(lines):
        # Round usage up to whole MiB so availability is conservative.
        used = _to_mib(used_s, used_unit, ROUND_CEILING)
        total = _to_mib(total_s, total_unit, ROUND_FLOOR)
        free = total - used
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- thead ---------------------------------------------------------------- #
def detect_thead(env):
    if shutil.which("ppu-smi", path=env.get("PATH")) is None:
        eprint("Error: ppu-smi command not found.")
        raise VendorCheckError()
    _, out, _ = _run(["ppu-smi", "-L"], env=env)
    count = sum(1 for line in out.splitlines() if "PPU-" in line)
    if count == 0:
        eprint("No T-Head PPU cards detected.")
        raise VendorCheckError()
    return count


def sample_thead(env, count):
    gpus = []
    available = []
    for i in range(count):
        _, out, _ = _run(["ppu-smi", "-i", str(i)], env=env)
        m = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", out)
        if not m:
            eprint(f"Warning: Failed to query PPU {i} memory.")
            continue
        used = int(m.group(1))
        total = int(m.group(2))
        free = total - used
        gpus.append({"index": i, "total": total, "used": used, "free": free})
        if free >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# ---- tsingmicro ----------------------------------------------------------- #
def detect_tsingmicro(env):
    _, out, _ = _run(["tsm_smi"], env=env)
    count = None
    for line in out.splitlines():
        if "Chip count:" in line:
            fields = line.split()
            if len(fields) > 5:
                count = fields[5]
            break
    try:
        count = int(count) if count is not None else 0
    except ValueError:
        count = 0
    if count == 0:
        eprint("No TsingMicro cards detected. Please ensure you have TsingMicro cards installed and properly configured.")
        raise VendorCheckError()
    return count


def sample_tsingmicro(env, count):
    _, out, _ = _run(["tsm_smi"], env=env)
    usage = []
    total = []
    # Mirror the two original awk programs (single pass; qualifying lines are
    # identical for the used/total passes):
    #   used : $2=="|" -> $9 ; $4=="|" && $11!="|" -> $11
    #   total: $2=="|" -> $11; $4=="|" && $11!="|" -> $13
    for line in out.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] == "|":
            if len(fields) > 10:
                usage.append(fields[8])
                total.append(fields[10])
        elif len(fields) > 12 and fields[3] == "|" and fields[10] != "|":
            usage.append(fields[10])
            total.append(fields[12])
    gpus = []
    available = []
    for i in range(count):
        if i >= len(total) or i >= len(usage):
            continue
        # Strip the trailing unit char (shell: ${var%?} drops 1 char).
        total_i = int(total[i][:-1])
        used_i = int(usage[i][:-1])
        free_i = total_i - used_i
        gpus.append({"index": i, "total": total_i, "used": used_i, "free": free_i})
        if free_i >= MEM_THRESHOLD:
            available.append(i)
    return gpus, available


# --------------------------------------------------------------------------- #
# Vendor registry
#
# ``key`` is the "*" part of the retired gpu_check_*.sh scripts. ``command`` is
# the SMI binary used for the secondary (command -v) probe signal; ``unit`` is
# the noun shown in the table header and the periodic/timeout messages.
# --------------------------------------------------------------------------- #
VENDORS = {
    "nvidia": {"unit": "GPU", "command": "nvidia-smi", "detect": detect_nvidia, "sample": sample_nvidia},
    "ascend": {"unit": "NPU", "command": "npu-smi", "detect": detect_ascend, "sample": sample_ascend},
    "cambricon": {"unit": "MLU", "command": "cnmon", "detect": detect_cambricon, "sample": sample_cambricon},
    "amd": {"unit": "GPU", "command": "rocm-smi", "detect": detect_amd, "sample": sample_amd},
    "enflame": {"unit": "GPU", "command": "efsmi", "detect": detect_enflame, "sample": sample_enflame},
    "hygon": {"unit": "DCU", "command": "hy-smi", "detect": detect_hygon, "sample": sample_hygon},
    "iluvatar": {"unit": "GPU", "command": "ixsmi", "detect": detect_iluvatar, "sample": sample_iluvatar},
    "kunlunxin": {"unit": "GPU", "command": "xpu-smi", "detect": detect_kunlunxin, "sample": sample_kunlunxin},
    "metax": {"unit": "GPU", "command": "mx-smi", "detect": detect_metax, "sample": sample_metax},
    "mthreads": {"unit": "GPU", "command": "mthreads-gmi", "detect": detect_mthreads, "sample": sample_mthreads},
    "sunrise": {"unit": "GPU", "command": "pt_smi", "detect": detect_sunrise, "sample": sample_sunrise},
    "thead": {"unit": "PPU", "command": "ppu-smi", "detect": detect_thead, "sample": sample_thead},
    "tsingmicro": {"unit": "GPU", "command": "tsm_smi", "detect": detect_tsingmicro, "sample": sample_tsingmicro},
}

# Probe priority order. Every vendor is uniquely identifiable by its /dev node,
# but the order still matters for the two collisions below, so keep it fixed and
# keep nvidia last (several vendors ship an nvidia-smi-compatible shim):
#   * hygon BEFORE amd  -- both expose /dev/kfd; only hygon also has /dev/mkfd.
#   * thead  BEFORE nvidia -- thead ships an nvidia-smi shim; its /dev node is
#                             unique, so device-based probing avoids the clash.
#   * nvidia LAST -- genuine NVIDIA hosts fall through to here.
VENDOR_ORDER = [
    "hygon",      # must precede amd (see _device_present)
    "amd",
    "thead",      # must precede nvidia
    "iluvatar",
    "kunlunxin",
    "enflame",
    "tsingmicro",
    "metax",
    "cambricon",
    "mthreads",
    "ascend",
    "sunrise",
    "nvidia",     # keep last
]


def _device_present(key):
    """Return True if the vendor's unique /dev device node(s) are present.

    This is the *primary* probe signal. Patterns come from live-verified device
    listings. The hygon/amd pair both expose /dev/kfd, so they are disambiguated
    by /dev/mkfd (present only on Hygon DCU hosts).
    """
    g = glob.glob
    if key == "hygon":
        # Hygon DCU exposes a modified-KFD node in addition to /dev/kfd.
        return bool(g("/dev/mkfd"))
    if key == "amd":
        # Plain ROCm: /dev/kfd present, but no Hygon /dev/mkfd.
        return bool(g("/dev/kfd")) and not bool(g("/dev/mkfd"))
    if key == "thead":
        return bool(g("/dev/alixpu_ppu*"))
    if key == "nvidia":
        return bool(g("/dev/nvidia[0-9]*")) or bool(g("/dev/nvidiactl"))
    if key == "iluvatar":
        return bool(g("/dev/iluvatar*"))
    if key == "kunlunxin":
        return bool(g("/dev/xpu[0-9]*")) or bool(g("/dev/xpuctrl"))
    if key == "enflame":
        return bool(g("/dev/gcu[0-9]*")) or bool(g("/dev/gcuctl"))
    if key == "tsingmicro":
        return bool(g("/dev/accel_drv_mgr"))
    if key == "metax":
        return bool(g("/dev/mxcd"))
    if key == "cambricon":
        return bool(g("/dev/cambricon_dev*"))
    if key == "mthreads":
        return bool(g("/dev/mtgpu.*"))
    if key == "ascend":
        return bool(g("/dev/davinci[0-9]*")) or bool(g("/dev/davinci_manager"))
    if key == "sunrise":
        return bool(g("/dev/ptpu[0-9]*")) or bool(g("/dev/ptpuctrl"))
    return False


def _command_present(key):
    """Secondary probe signal: the vendor's SMI command resolves on PATH."""
    env = _vendor_env(key)
    return shutil.which(VENDORS[key]["command"], path=env.get("PATH")) is not None


def probe_vendor():
    """Return the detected vendor key, or None if no supported GPU is present.

    First pass requires BOTH signals (device node AND command) for the highest
    confidence. If nothing matches both -- e.g. the device is present but its
    toolkit is not installed -- a second pass accepts a device-only match so the
    selected vendor's own check can emit its specific "command not found" error.
    """
    for key in VENDOR_ORDER:
        if _device_present(key) and _command_present(key):
            return key
    for key in VENDOR_ORDER:
        if _device_present(key):
            return key
    return None


def probe_accelerator_metadata(key=None):
    """Return normalized vendor, device, driver and runtime metadata."""
    key = key or probe_vendor()
    if key is None:
        return {
            "vendor": None,
            "device": None,
            "driver": None,
            "runtime": None,
            "runtime_version": None,
        }
    env = _vendor_env(key)

    def metadata_run(cmd, env):
        return _run(cmd, env=env, timeout=5)

    return {"vendor": key, **collect_metadata(key, env, metadata_run)}


# --------------------------------------------------------------------------- #
# Retry loop and output rendering
# --------------------------------------------------------------------------- #
def run_vendor_check(key):
    """Run detect + the retry/sample loop for ``key``; return the result dict."""
    vendor = VENDORS[key]
    env = _vendor_env(key)
    unit = vendor["unit"]
    count = vendor["detect"](env)
    metadata = probe_accelerator_metadata(key)
    waited = 0
    while True:
        gpus, available = vendor["sample"](env, count)
        if available:
            avail_str = ",".join(str(i) for i in available)
            return {
                "vendor": key,
                **metadata,
                "unit": unit,
                "device_count": count if count is not None else len(gpus),
                "memory_unit": "MiB",
                "available_gpus": avail_str,
                "gpus": [
                    {"id": g["index"], "total": g["total"],
                     "used": g["used"], "free": g["free"]}
                    for g in gpus
                ],
            }
        eprint(f"No {unit} has sufficient memory, waiting for {SLEEP_TIME} seconds...")
        time.sleep(SLEEP_TIME)
        waited += SLEEP_TIME
        if waited >= MAX_WAIT:
            eprint(f"Error: Timed out waiting for available {unit}.")
            raise VendorCheckError()


def render_text(result):
    """Render the same concise summary for stdout and saved text files."""
    lines = [f"Vendor: {result['vendor']}"]
    for label, field in (
        ("Device", "device"),
        ("Driver", "driver"),
        ("Runtime", "runtime"),
        ("Runtime Version", "runtime_version"),
    ):
        lines.append(f"{label}: {result[field] or 'unknown'}")
    device = f" {result['device']}" if result["device"] else ""
    lines.extend([
        (f"Detected {result['device_count']} {VENDOR_NAMES[result['vendor']]}"
         f"{device} {result['unit']}(s)."),
        f"Available GPUs: {result['available_gpus']}",
        "GPUs Usage:",
        _table_header(result["unit"]),
    ])
    for g in result["gpus"]:
        lines.append(_table_row(g["id"], g["total"], g["used"], g["free"]))
    return "\n".join(lines) + "\n"


def render_output(result, fmt):
    if fmt == "json":
        return json.dumps(result, indent=2) + "\n"
    if fmt == "yaml":
        return yaml.safe_dump(result, sort_keys=False, allow_unicode=True)
    return render_text(result)


def main():
    parser = argparse.ArgumentParser(
        prog="flagos-gpu-check",
        description="Probe the host GPU vendor and report available GPUs.",
    )
    parser.add_argument(
        "-f", "--format", choices=["text", "json", "yaml"], default="text",
        help="output format (default: text); diagnostics always go to stderr",
    )
    parser.add_argument(
        "-o", "--output", metavar="PATH",
        help="save the rendered output to a file (in the chosen format)",
    )
    parser.add_argument(
        "--vendor", choices=sorted(VENDORS), metavar="VENDOR",
        help=f"skip auto-probe and force a specific vendor (one of: {', '.join(sorted(VENDORS))})",
    )
    args = parser.parse_args()

    if args.vendor:
        key = args.vendor
    else:
        key = probe_vendor()
        if key is None:
            eprint("Error: no supported vendor GPU detected on this host.")
            return 1

    try:
        result = run_vendor_check(key)
    except VendorCheckError:
        return 1

    rendered = render_output(result, args.format)
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(rendered)
        except OSError as exc:
            eprint(f"Error: cannot write {args.output}: {exc}")
            return 1
    else:
        sys.stdout.write(rendered)

    return 0


if __name__ == "__main__":
    sys.exit(main())
