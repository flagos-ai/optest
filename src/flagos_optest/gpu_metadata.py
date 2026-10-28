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

"""Best-effort accelerator metadata, without importing vendor Python stacks."""

import json
import re
from pathlib import Path

VENDOR_NAMES = {
    "nvidia": "NVIDIA", "ascend": "Ascend", "cambricon": "Cambricon",
    "amd": "AMD", "enflame": "Enflame", "hygon": "Hygon",
    "iluvatar": "Iluvatar", "kunlunxin": "KunlunXin", "metax": "MetaX",
    "mthreads": "Moore Threads", "sunrise": "Sunrise", "thead": "T-Head",
    "tsingmicro": "TsingMicro",
}

# Runtime means the vendor software distribution installed in the container,
# rather than the maximum compatibility level advertised by the host driver.
# Paths are checked in order, with explicit installation roots taking priority.
RUNTIMES = {
    "nvidia": ("cuda", ("CUDA_HOME", "CUDA_PATH"), ("/usr/local/cuda",)),
    "ascend": ("cann", ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"),
               ("/usr/local/Ascend/ascend-toolkit/latest", "/usr/local/Ascend/latest")),
    "cambricon": ("neuware", ("NEUWARE_HOME",), ("/usr/local/neuware",)),
    "amd": ("rocm", ("ROCM_PATH", "ROCM_HOME"), ("/opt/rocm",)),
    "enflame": ("tops", ("TOPS_HOME", "TOPS_PATH"), ("/opt/tops", "/usr/local/tops")),
    "hygon": ("dtk", ("DTK_HOME", "DTK_PATH", "DTKROOT"), ("/opt/dtk",)),
    "iluvatar": ("corex", ("COREX_HOME", "COREX_PATH", "COREX_ROOT"), ("/usr/local/corex",)),
    "kunlunxin": ("xre", ("XRE_HOME", "XPU_HOME"), ("/usr/local/xpu",)),
    "metax": ("maca", ("MACA_PATH", "MACA_HOME"), ("/opt/maca", "/usr/local/maca")),
    "mthreads": ("musa", ("MUSA_HOME", "MUSA_INSTALL_PATH"), ("/usr/local/musa",)),
    "sunrise": ("tangrt", ("TANGRT_HOME",), ("/usr/local/tangrt", "/opt/tangrt")),
    "thead": ("ppu", ("PPU_HOME", "PPU_SDK_HOME"),
              ("/usr/local/PPU_SDK", "/usr/local/ppu")),
    "tsingmicro": ("tsm", ("TSM_HOME", "KUIPER_PATH", "KUIPER_HOME"),
                   ("/usr/local/kuiper", "/home/secure/runtime/kuiper", "/usr/local/tsm")),
}

COMMANDS = {
    "nvidia": "nvidia-smi", "ascend": "npu-smi", "cambricon": "cnmon",
    "amd": "rocm-smi", "enflame": "efsmi", "hygon": "hy-smi",
    "iluvatar": "ixsmi", "kunlunxin": "xpu-smi", "metax": "mx-smi",
    "mthreads": "mthreads-gmi", "sunrise": "pt_smi", "thead": "ppu-smi",
    "tsingmicro": "tsm_smi",
}

# Prefer model tokens over marketing names, memory sizes and form factors.
MODELS = {
    "nvidia": r"\b(?:[ABHVL]\d{2,3}[A-Z]?)(?:\s+NVL)?\b",
    "ascend": r"\b(?:Ascend\s*)?(9\d{2}[A-Z]?\d?)\b",
    "cambricon": r"\bMLU[\w-]+\b",
    "amd": r"\b(?:Instinct\s+)?(?:MI\d+[A-Z]*|Radeon\s+(?:PRO\s+)?[\w-]+)\b",
    "enflame": r"\b(?:S\d+|ZIXIAO[\w-]+|GCU[\w-]+)\b",
    "hygon": r"\b(?:BW|K|Z)\d+[A-Z0-9-]*\b",
    "iluvatar": r"\bBI-[A-Z0-9-]+\b",
    "kunlunxin": r"\b(?:P\d{3,4}[A-Z]*|R\d{3,4}[A-Z]*)\b",
    "metax": r"\bC\d{3,4}[A-Z]*\b",
    "mthreads": r"\bMTT[-\s]+S\d+[A-Z]*\b",
    "sunrise": r"\b(?:Sunrise[-\s]+)?(P(?:T)?\d{3,4}[A-Z]*)\b",
    "thead": r"\bPPU-[A-Z0-9-]+\b",
    "tsingmicro": r"\bTX\d+[A-Z]*\b",
}
VERSION = r"\d+(?:\.\d+)+(?:[._-]?(?:RC|SP|beta)\d+)?(?:[-+][\w.-]+)?"
# TSM releases use a timestamp-style integer rather than a dotted version.
TSM_VERSION = rf"(?:{VERSION}|\d{{12}})"


def _unique(values):
    return ", ".join(dict.fromkeys(values)) or None


def _model_names(key, text):
    if key == "hygon":
        # A board may be described as "BW200, UBB BW1000". Prefer its UBB
        # product designation over the chip family on the same line.
        text = re.sub(r"\bBW\w+,\s*UBB\s+(BW\w+)", r"\1", text)
    matches = re.findall(MODELS[key], text, re.IGNORECASE)
    if key == "mthreads":
        matches = [re.sub(r"[-\s]+", "-", name) for name in matches]
    return _unique(matches)


def _label_version(text, label, version_pattern=VERSION):
    match = re.search(
        rf"\b(?:{label})\s*(?:version|ver|release)?\s*[:=|]?\s*[vV]?({version_pattern})",
        text, re.IGNORECASE,
    )
    return match.group(1) if match else None


def _file_version(path, runtime):
    """Read only distribution-level metadata, never component/driver versions."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    if runtime == "xre" and path.name == "version.h":
        match = re.search(rf'^\s*#\s*define\s+XPURT_VERSION_STR\s+"({VERSION})"',
                          text, re.MULTILINE)
        return match.group(1) if match else None
    if runtime == "dtk" and path.name == ".dtk_version":
        match = re.search(rf"^DTK-({VERSION})\s*$", text, re.MULTILINE)
        return match.group(1) if match else None
    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        distribution = "musa_toolkits" if runtime == "musa" else runtime
        value = data.get(distribution, data)
        if isinstance(value, dict):
            value = value.get("version")
        return str(value) if isinstance(value, (str, int, float)) else None
    # XRE version.txt can also list a driver-compatible version. Require an
    # explicit runtime label there; a bare release-only file is still accepted.
    label = r"xre|xpurt" if runtime == "xre" else rf"{re.escape(runtime)}|version"
    pattern = TSM_VERSION if runtime == "tsm" else VERSION
    value = _label_version(text, label, pattern)
    if value:
        return value
    match = re.fullmatch(rf"\s*[vV]?({pattern})\s*", text)
    return match.group(1) if match else None


def _runtime_version(key, env, run, smi_text):
    runtime, env_vars, defaults = RUNTIMES[key]
    roots = list(dict.fromkeys([env[v] for v in env_vars if env.get(v)] + list(defaults)))
    filenames = ["version.json", "version.txt", "version", "VERSION", ".info/version"]
    if key == "ascend":
        filenames += ["ascend_toolkit_install.info", "*/ascend_toolkit_install.info",
                      "version.cfg", "*/version.cfg"]
    elif key == "kunlunxin":
        filenames = ["include/xpu/version.h", "xre/include/xpu/version.h"] + filenames + [
            "xre/version.txt", "xre/version",
        ]
    elif key == "hygon":
        filenames = [".dtk_version"] + filenames
    for root in roots:
        for filename in filenames:
            try:
                paths = sorted(Path(root).glob(filename))
            except (OSError, ValueError):
                continue
            for path in paths:
                value = _file_version(path, runtime)
                if value:
                    return value
        # Try the selected root's compiler before inspecting other installations.
        if key == "nvidia":
            value = _cuda_version(str(Path(root) / "bin/nvcc"), env, run)
            if value:
                return value
        elif key == "iluvatar":
            rc, out, err = run([str(Path(root) / "bin/clang"), "--version"], env=env)
            if rc == 0:
                # CoreX embeds its release and build date in Clang's vendor
                # build string; Clang's own version is a different product.
                match = re.search(r"\((\d+\.\d+\.\d+)\.\d{8}\b", out + err)
                if match:
                    return match.group(1)
    # CUDA's SMI header reports driver capability, NOT the installed toolkit.
    # Likewise IX-ML and the CUDA compatibility version are not CoreX releases.
    if key == "nvidia":
        value = _cuda_version("nvcc", env, run)
        if value:
            return value
        # Runtime-only CUDA images omit nvcc and version.json. The libraries
        # metapackage records the distribution release, unlike cudart's build.
        rc, out, _ = run(["dpkg-query", "-W", "-f=${Package}\\t${Version}\\n",
                          "cuda-libraries-*"], env=env)
        if rc == 0:
            releases = re.findall(r"(?m)^cuda-libraries-\d+-\d+\s+(\d+\.\d+\.\d+)(?:-\S+)?$", out)
            if len(set(releases)) == 1:
                return releases[0]
        return None
    if key == "enflame":
        rc, out, _ = run(["dpkg-query", "-W", "-f=${Version}", "topsruntime"], env=env)
        if rc == 0 and re.fullmatch(VERSION, out.strip()):
            return out.strip()
    label = r"tsm|runtime" if key == "tsingmicro" else re.escape(runtime)
    return _label_version(smi_text, label,
                          TSM_VERSION if runtime == "tsm" else VERSION)


def _cuda_version(command, env, run):
    rc, out, err = run([command, "--version"], env=env)
    return _label_version(out + err, "release") if rc == 0 else None


def collect_metadata(key, env, run):
    """Return nullable metadata; unavailable optional queries do not block use."""
    command = COMMANDS[key]
    base = [command, "info"] if key == "ascend" else [command]
    rc, table, _ = run(base, env=env)
    if rc != 0:
        table = ""
    if key == "tsingmicro":
        rc, out, _ = run([command, "--version"], env=env)
        if rc == 0:
            table += "\n" + out
    name = _model_names(key, table)
    driver = _label_version(table, r"driver(?:\s+version)?",
                            TSM_VERSION if key == "tsingmicro" else VERSION)
    if key in ("nvidia", "iluvatar"):
        # Separate fields: older SMI releases may support name but not driver.
        for field in ("name", "driver_version"):
            rc, out, _ = run(
                [command, f"--query-gpu={field}", "--format=csv,noheader"], env=env,
            )
            if rc != 0:
                continue
            values = [line.strip() for line in out.splitlines() if line.strip()]
            if field == "name":
                name = _model_names(key, out) or _unique(
                    re.sub(r"^(?:NVIDIA|Iluvatar)\s+", "", v, flags=re.IGNORECASE)
                    for v in values if v.upper() not in ("N/A", "[N/A]", "NOT SUPPORTED")
                ) or name
            else:
                driver = _unique(v for v in values if re.fullmatch(VERSION, v)) or driver
    elif key in ("amd", "hygon"):
        rc, out, _ = run([command, "--showproductname", "--showdriverversion"], env=env)
        if rc == 0:
            name = _model_names(key, out) or name
            driver = _label_version(out, r"driver(?:\s+version)?") or driver
    elif key in ("mthreads", "thead", "enflame"):
        rc, out, _ = run([command, "-L"], env=env)
        if rc == 0:
            name = _model_names(key, out) or name
    if not name:
        match = re.search(
            r"(?:Product|Device|Board|Card)\s+Name\s*:\s*([^|\n]+)",
            table, re.IGNORECASE,
        )
        name = match.group(1).strip() if match else None
    # Ascend records the installed host driver separately from CANN.
    if key == "ascend" and not driver:
        driver = _file_version(Path("/usr/local/Ascend/driver/version.info"), "driver")
    return {
        "device_name": name,
        "driver_version": driver,
        "runtime": RUNTIMES[key][0],
        "runtime_version": _runtime_version(key, env, run, table),
    }
