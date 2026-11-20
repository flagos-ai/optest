# FlagOS Operator Test Tools

English | [简体中文](README.zh-CN.md)

[![Python checks](https://github.com/flagos-ai/optest/actions/workflows/python-checks.yaml/badge.svg?branch=main)](https://github.com/flagos-ai/optest/actions/workflows/python-checks.yaml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/flagos-ai/optest/actions/workflows/python-checks.yaml)

`flagos-optest` is a shared tool for accuracy and performance testing across
FlagOS operator libraries. The package also provides report-generation,
result-labeling, web-data conversion, and vendor-specific GPU checking toolsuite.

## Menu

- [Install](#install)
- [Requirements](#requirements)
- [`flagos-optest`](#usage-flagos-optest)
  - [Purpose](#purpose)
  - [Results and `summary.json`](#results-and-summaryjson)
  - [Console output](#console-output-example)
  - [Result helper commands](#result-helper-commands)
- [`flagos-gpu-check`](#usage-flagos-gpu-check)
  - [Purpose and behavior](#purpose-and-behavior)
  - [Options](#options)
  - [Device and runtime metadata](#device-and-runtime-metadata)
  - [Examples](#examples)
- [Target repository dependencies with uv](#target-repository-dependencies-with-uv)
  - [Configure test and vendor extras](#configure-test-and-vendor-extras)
  - [Centralized workflow behavior](#centralized-workflow-behavior)
  - [FlagTree upgrade](#flagtree-upgrade)
  - [FlagGems upgrade](#flaggems-upgrade)
  - [Persistent package cache](#persistent-package-cache)
- [Repository layout](#repository-layout)
  - [Repository configuration](#repository-configuration)
  - [Backend configuration](#backend-configuration)

## Install

Install `optest` into the same Python environment as the operator library under
test:

```bash
uv pip install /path/to/optest
```

You can install tool in editable mode as well:

```bash
git clone https://github.com/flagos-ai/optest
cd optest
uv pip install -e .
```

The package installs the following console commands:

```text
flagos-optest       Run operator accuracy and performance tests
flagos-add-labels   Add catalog labels to an existing summary.json file
flagos-psum-text    Generate CSV or Markdown reports
flagos-psum-html    Generate a self-contained HTML report
flagos-psum-web     Convert summaries to web-site data JSON
flagos-gpu-check    Detect the accelerator vendor and available devices
```

The target operator repository is expected to provide its own runtime and test
dependencies, including torch, the operator library, pytest, and the plugins that
provide options such as `--record` and `--collect-marks`.

## Requirements

The operator library under test must provide the following files and directorys.
following the [FlagGems](https://github.com/flagos-ai/FlagGems) repository layout
as a reference:

```text
conf/operators.yaml   Operator inventory used for selection, stages, and labels
tests/                Pytest accuracy tests
benchmark/            Pytest performance benchmarks
```

The exact names are significant: the scheduler reads `conf/operators.yaml` and
runs `pytest` against `tests/` for accuracy testing and `benchmark/` for performance
benchmarking.

Each `operators.yaml` entry should follow the complete FlagGems schema and
define fields such as `id`, `description`, `for`, `labels`, `kind`, and
`stages`. In particular, `id` identifies the operator, `stages` controls
selection by `--stages`, and `labels` is consumed by `flagos-add-labels`.
Missing or malformed operator metadata fields can prevent operators from being
selected or reported correctly.

## Usage: `flagos-optest`

### Purpose

`flagos-optest` discovers the target environment, selects operators from
`conf/operators.yaml`, and initiates the accuracy and benchmark tests for
operators specified, across one or more devices. Like `pytest`, its optional
positional `DIRECTORY` defaults to the current directory.

Run from the operator repository:

```bash
cd /path/to/operator-library
flagos-optest --gpus 0,1 --stages stable --dump-output
```

You can also pass the repository explicitly:

```bash
flagos-optest /path/to/operator-library \
  --gpus 0,1 --stages stable --dump-output
```

Common options:

| Option | Purpose |
| --- | --- |
| `DIRECTORY` | Target operator repository. Defaults to the current directory. |
| `--ops add,sum` | Test an explicit comma-separated operator list. This takes priority over `--op-list-file` and `--stages`. |
| `--op-list-file FILE` | Read operator IDs from a file, one per line; lines beginning with `#` are comments. |
| `--exclude-ops add,sum` | Omit a comma-separated list from the selected operators. This takes priority over `--exclude-op-list-file`. |
| `--exclude-op-list-file FILE` | Read operator IDs to omit from a file, one per line; lines beginning with `#` are comments. |
| `--start OP_ID` | Resume from an operator ID by testing IDs lexicographically greater than or equal to it. |
| `--gpus 0,1` | Select device IDs. Use `--gpus all` to use every device detected by PyTorch. Default: `0`. |
| `--stages stable,beta` | Select inventory stages. Supported values are `alpha`, `beta`, `stable`, `all`, and `removed`. Default: `stable`. |
| `--output-dir DIR` | Result directory. Relative paths are resolved from the target repository. Default: `logs_results_YYYYMMDD_HHMM`. |
| `--timeout SECONDS` | Timeout for each accuracy or performance subprocess. Default: `1800` seconds. |
| `--dump-output` | Preserve each accuracy and benchmark subprocess's stdout and stderr. |
| `--quick` | Ask the target tests to use reduced parameter combinations. |
| `--color auto\|always\|never` | Control ANSI color output. Default: `auto`. |
| `--base-data FILE` | Supply an optional performance-baseline JSON file. |

Examples:

```bash
# Run all stable operators on every detected device.
flagos-optest --gpus all --dump-output

# Run three operators and write results to a specific directory.
flagos-optest --ops abs,add,sum --gpus 0,1,2 \
  --output-dir /tmp/optest-results

# Run stable and beta operators in another checkout.
flagos-optest /work/FlagGems --stages stable,beta --gpus 0,1

# Allow each accuracy and performance subprocess up to one hour.
flagos-optest --timeout 3600 --gpus all

# Run all stable operators except those listed as unsupported.
flagos-optest --exclude-ops index_add,scatter --gpus all

# Apply a file-based exclusion list to an explicit include list.
flagos-optest --op-list-file ops.txt --exclude-op-list-file unsupported.txt

# Resume a list from the operator named "mul".
flagos-optest --start mul --stages all --gpus 0
```

Explicit include and exclude lists are deduplicated while preserving their
original order. Exclusions use exact, case-sensitive operator ID matching. When
any list option is supplied, the startup log reports the parsed include and
exclude counts, the number of exclusions that matched, and the final number of
operators scheduled. It also records every parsed exclusion, including names
that did not match the selected operators:

```text
[INFO] Operator selection ...
[INFO]   Included ... 3 (--ops)
[INFO]   Exclusions ... 2 (--exclude-ops)
[INFO]   Matched exclusions ... 1
[INFO]   Selected ... 2
[INFO] Operator exclusions ... --exclude-ops (2 entries):
[INFO]   mul
[INFO]   unsupported_op
[INFO] Testing 2 operators ...
```

For file-based lists, the source filename is included in the summary and list
heading. Each parsed exclusion is printed on its own line for readability:

```text
[INFO] Operator selection ...
[INFO]   Included ... 5 (--op-list-file: ops.txt)
[INFO]   Exclusions ... 3 (--exclude-op-list-file: unsupported_ops.txt)
[INFO]   Matched exclusions ... 2
[INFO]   Selected ... 3
[INFO] Operator exclusions ... unsupported_ops.txt (3 entries):
[INFO]   add
[INFO]   sum
[INFO]   unsupported_op
[INFO] Testing 3 operators ...
```

### Results and `summary.json`

After a successful run, the result directory contains the aggregated summary
and one directory per operator:

```text
logs_results_YYYYMMDD_HHMM/
├── summary.json
└── <operator>/
    ├── accuracy_result.json
    ├── performance_result.json
    ├── accuracy_stdout.log       # with --dump-output
    ├── accuracy_stderr.log       # with --dump-output
    ├── performance_stdout.log    # with --dump-output
    └── performance_stderr.log    # with --dump-output
```

During an active run, each worker appends compact `start`, `operator`, and
`complete` records to a `summary<N>.jsonl` checkpoint. Every operator record is
flushed and synced once, avoiding the quadratic write amplification caused by
rewriting all earlier results after each operator. The checkpoints are removed
only after every expected worker checkpoint is complete, duplicate-free, and
exactly covers the selected operators, and the final `summary.json` has been
atomically published and verified. Incomplete or failed runs retain all
existing checkpoints for recovery. Worker startup atomically replaces its own
checkpoint, so reusing an output directory cannot mix records from an older
run.

`summary.json` contains the run timestamp and duration, `total_ops` (the number
of operators successfully aggregated into `result`), operator-level
`accuracy_counts` and `performance_counts`, an `env` object, and
the per-operator accuracy and performance results. Each operator entry also
carries a `customized` flag and a `labels` list. `labels` merges the manifest
labels from `conf/operators.yaml` (applied by `flagos-add-labels`) with any
labels detected during the run. In particular it includes `"tle"` when the
implementation this chip actually dispatches to uses Triton experimental
language extensions (its source file, or a kernel submodule it imports, imports
`triton.experimental.tle.language`). The `"tle"` entry is resolved from the live
operator, so a vendor kernel that uses TLE is tagged on that vendor while a
generic TLE path is tagged everywhere; it is therefore per-chip. Status counts
include zero values for `passed`, `failed`, `skipped`, `not_found`, `error`, and
`timeout`.
The `env` object centralizes the test-critical environment and dependency
versions detected by the scheduler:

- CPU architecture, operating system, and Python version;
- PyTorch version, accelerator availability, device model, and device count;
- FlagTree and Triton versions and Triton capability information;
- every recognized FlagOS library that can be imported, including its version,
  vendor, and device backend;
- detected optional frameworks such as vLLM or SGLang; and
- the normalized accelerator vendor, device, driver, runtime and runtime
  version, using the same metadata fields as `flagos-gpu-check`.

This includes all test-environment dependency and version fields probed by
optest, but it is not a complete `pip freeze` of every installed package. A
shortened `env` example looks like this (OS and Python values vary with the
selected image):

```json
{
  "timestamp": "2026-10-23 18:50:37",
  "total_duration": "0:01:47",
  "total_ops": 3,
  "accuracy_counts": {
    "passed": 3,
    "failed": 0,
    "skipped": 0,
    "not_found": 0,
    "error": 0,
    "timeout": 0
  },
  "performance_counts": {
    "passed": 3,
    "failed": 0,
    "skipped": 0,
    "not_found": 0,
    "error": 0,
    "timeout": 0
  },
  "env": {
    "architecture": "x86_64",
    "os_name": "ubuntu",
    "os_release": "<detected OS release>",
    "python": "<detected Python version>",
    "torch": {
      "version": "2.11.0+cu130",
      "cuda_available": true,
      "device_name": "NVIDIA H800",
      "device_count": 8
    },
    "flagtree": "0.7.0",
    "triton": {
      "version": "3.6.0",
      "has_config": true
    },
    "flag_gems": {
      "library_name": "FlagGems",
      "version": "0.0+ge510c6bbe",
      "vendor": "nvidia",
      "device": "cuda"
    },
    "vllm": {
      "version": "0.24.0"
    },
    "vendor": "nvidia",
    "device": "H800",
    "driver": "610.57.04",
    "runtime": "cuda",
    "runtime_version": "13.3.0"
  }
}
```

### Console output example

The following is a real H800 run for `abs`, `add`, and `sum`. Operator output
can be much longer for a full suite; the important lifecycle is test start,
environment probing, operator accuracy/benchmark start and end, total elapsed
time, and test completion.

```text
[INFO] Project root ... /__w/optest/optest/FlagGems
[INFO] Test started at ... 2026-10-23 18:48:49
[INFO] Command timeout ... 1800 seconds
[INFO] PyTorch detected ... 2.11.0+cu130
[INFO] PyTorch CUDA support ... True
[INFO] PyTorch device name ... NVIDIA H800
[INFO] PyTorch device count ... 8
[INFO] FlagTree (flagtree) detected ... 0.7.0
[INFO] Triton (triton) detected ... 3.6.0
[INFO] Triton (triton) has Config ... [True]
[INFO] FlagGems (flag_gems) detected ... 0.0+ge510c6bbe (vendor=nvidia, device=cuda)
[INFO] Canonical device context ... vendor=nvidia, device=cuda (from flag_gems)
[INFO] vllm detected ... 0.24.0
[INFO] Accelerator ... nvidia / H800
[INFO] Driver ... 610.57.04
[INFO] Runtime ... cuda 13.3.0
[INFO] Operator selection ...
[INFO]   Included ... 3 (--ops)
[INFO]   Exclusions ... 0 (none)
[INFO]   Matched exclusions ... 0
[INFO]   Selected ... 3
[INFO] Testing 3 operators ...
[INFO] Only 3 operators requested, skipping mark collection
[INFO] [18:48:59][GPU  0] accuracy  abs ...
[INFO] [18:48:59][GPU  1] accuracy  add ...
[INFO] [18:48:59][GPU  2] accuracy  sum ...
[INFO] [18:49:35][GPU  0] accuracy  abs [OK         35.2s]  (  0% 0/3 ops)
[INFO] [18:49:35][GPU  0] benchmark abs ...
[INFO] [18:49:36][GPU  2] accuracy  sum [OK         36.4s]  ( 33% 1/3 ops)
[INFO] [18:49:36][GPU  2] benchmark sum ...
[INFO] [18:49:45][GPU  1] accuracy  add [OK         45.3s]  ( 33% 1/3 ops)
[INFO] [18:49:45][GPU  1] benchmark add ...
[INFO] [18:50:09][GPU  2] benchmark sum [OK         33.6s]  ( 66% 2/3 ops)
[INFO] [18:50:16][GPU  0] benchmark abs [OK         41.0s]  ( 66% 2/3 ops)
[INFO] [18:50:37][GPU  1] benchmark add [OK         52.3s]  (100% 3/3 ops)
[INFO] Total operators ... 3
[INFO] Accuracy operators ... passed=3
[INFO] Performance operators ... passed=3
[INFO] Total elapsed time ... 0:01:47 (107.62s)
[INFO] Test completed.
```

The final console summary uses the same operator counts written to
`summary.json`. Statuses with a zero count are omitted from the accuracy and
performance lines.

### Result helper commands

Run these commands after `flagos-optest` has created `summary.json`.

#### `flagos-add-labels`

This command copies each operator's `labels` from the target repository's
`conf/operators.yaml` into the corresponding `summary.json` result. Use it
before publishing reports when downstream filtering or grouping needs catalog
labels.

The first positional directory is optional and defaults to the current target
repository. `RESULT_DIR` is required:

```bash
# Current directory is the operator repository.
cd /path/to/operator-library
flagos-add-labels /tmp/optest-results

# Run from anywhere and pass both directories explicitly.
flagos-add-labels /path/to/operator-library /tmp/optest-results
```

#### `flagos-psum-text`

This command converts `summary.json` into CSV or Markdown. Use CSV for data
processing and spreadsheets; use Markdown for issues, pull requests, and other
text-based reports. The optional result directory defaults to the current
directory.

```bash
# Current directory contains summary.json.
cd /tmp/optest-results
flagos-psum-text -o result.csv
flagos-psum-text --format markdown -o result.md

# Pass the result directory explicitly.
flagos-psum-text /tmp/optest-results \
  -o /tmp/optest-results/result.csv
```

Additional modes include `--metric {speedup,base}`, `--single` for a detailed
single-operator Markdown report, and `--compare` for a `before/` versus
`after/` single-operator comparison.

#### `flagos-psum-html`

This command creates a self-contained HTML report with environment details,
operator results, and filtering. The report contains summary data only by
default. Pass `--include-logs` to also embed each operator's stdout and stderr
logs. Result JSON files are never embedded. The optional result directory
defaults to the current directory.

```bash
# Current directory contains summary.json.
cd /tmp/optest-results
flagos-psum-html -o result.html

# Pass the result directory explicitly.
flagos-psum-html /tmp/optest-results \
  -o /tmp/optest-results/result.html

# Opt in to embedding accuracy/performance stdout and stderr logs.
flagos-psum-html /tmp/optest-results --include-logs \
  -o /tmp/optest-results/result-with-logs.html
```

#### `flagos-psum-web`

This command converts a directory of per-platform summaries into the JSON data
format consumed by the FlagOS documentation site. It is intended for website
publishing pipelines rather than a single local report:

```bash
flagos-psum-web --raw-dir /path/to/platform-results \
  --output-dir /path/to/web-data
```

## Usage: `flagos-gpu-check`

### Purpose and behavior

`flagos-gpu-check` is a unified installed command for all supported
accelerator vendors. It detects the host vendor, queries device memory, waits
when no device has enough free memory, and returns the available device IDs for
test scheduling.

A device is available when it has at least 30,000 MiB of free memory. If no
device meets that threshold, the command retries every 120 seconds for up to
600 seconds and exits nonzero on timeout. These values are currently fixed by
the implementation rather than exposed as command-line parameters.

Auto-detection first looks for vendor-specific `/dev` nodes and confirms the
vendor's management command on `PATH`. Device nodes are the primary signal
because several vendors provide NVIDIA-compatible command shims. Use
`--vendor` when auto-detection is unavailable inside a constrained container or
when a specific implementation must be checked.

### Options

| Option | Purpose and typical use |
| --- | --- |
| `-f, --format text` | Concise device, driver, runtime and memory summary (default). |
| `-f, --format json` | A single JSON document on stdout, suitable for direct parsing or redirection. |
| `-f, --format yaml` | A single YAML document on stdout. |
| `-o, --output PATH` | Write the same final document to a UTF-8 file instead of stdout, in any format. |
| `--vendor VENDOR` | Skip auto-detection and force a supported vendor key below. |
| `-h, --help` | Show command help. |

Successful checks print no raw management-tool tables, timestamps or process
lists. Retry messages, warnings and errors go only to stderr in every format.
With `--output`, stdout is empty. Failures exit nonzero without emitting a
partial result document.

### Device and runtime metadata

The name `runtime` identifies the vendor software environment installed in the
container for FlagGems, FlagTree and similar consumers. It is an umbrella term
for the distribution, which can also include SDK headers, compilers and
libraries. `runtime_version` is its release version; `driver` describes
the host driver separately.

| Vendor key | Accelerator vendor | Management command | Unit | Runtime |
| --- | --- | --- | --- | --- |
| `amd` | AMD | `rocm-smi` | GPU | `rocm` |
| `ascend` | Huawei Ascend | `npu-smi` | NPU | `cann` |
| `cambricon` | Cambricon | `cnmon` | MLU | `neuware` |
| `enflame` | Enflame | `efsmi` | GPU | `tops` |
| `hygon` | Hygon | `hy-smi` | DCU | `dtk` |
| `iluvatar` | Iluvatar | `ixsmi` | GPU | `corex` |
| `kunlunxin` | KunlunXin | `xpu-smi` | GPU | `xre` |
| `metax` | MetaX | `mx-smi` | GPU | `maca` |
| `mthreads` | Moore Threads | `mthreads-gmi` | GPU | `musa` |
| `nvidia` | NVIDIA | `nvidia-smi` | GPU | `cuda` |
| `sunrise` | Sunrise | `pt_smi` | GPU | `tangrt` |
| `thead` | T-Head | `ppu-smi` | PPU | `ppu` |
| `tsingmicro` | TsingMicro | `tsm_smi` | GPU | `tsm` |

Metadata is best effort and does not require importing PyTorch or vendor
Python modules. Model and driver information comes from management-tool
queries; Ascend also checks its driver installation metadata. Device names
omit known vendor and form-factor decorations, for example
`Iluvatar BI-V150 OAM` becomes `BI-V150`. Distinct models or driver versions on
a mixed host are joined with `, ` rather than silently selecting the first.

Runtime versions are read from distribution metadata such as `version.json`,
`version.txt`, `version`, `VERSION` and `.info/version` under the installation
root. CANN also checks `ascend_toolkit_install.info`, `version.cfg` and the
CANN 9 `share/info/runtime/version.info` distribution manifest, not tool
component versions or dependency requirements. T-Head reads the PPU SDK's
`release.yaml`, independently of the `ppu-smi` tool version. XRE
prefers `XPURT_VERSION_STR` in `include/xpu/version.h`, avoiding driver
compatibility versions in `version.txt`, and also checks its `xre/` subdirectory.
Explicit installation environment variables
(such as `COREX_ROOT`, `DTKROOT`, `CUDA_HOME`, `ASCEND_HOME_PATH`, `MACA_PATH` and
`MUSA_INSTALL_PATH`) take priority over conventional installation directories.
CUDA additionally supports `nvcc --version` and the `cuda-libraries-*`
distribution package in runtime-only images. CoreX releases are read from the
vendor build string in `bin/clang --version`, DTK from `.dtk_version`, MUSA
from `musa_toolkits.version` in `version.json`, and TOPS from the `topsruntime`
package. TsingMicro exposes separate driver and runtime versions through
`tsm_smi --version`; other vendors can supply an explicit runtime-labelled
version in their management-tool output.

The CUDA compatibility version in an SMI header is **not** the installed CUDA
release. Likewise, IX-ML and the Iluvatar driver's CUDA compatibility version
are **not** the CoreX release. Missing metadata is `null` in JSON/YAML and
`unknown` in text; it does not prevent checking memory availability. Runtime
images that omit these metadata sources may therefore report an unknown
version.

Real-output regression captures and the verified hardware/runtime matrix are
listed in [tests/fixtures/gpu_check](tests/fixtures/gpu_check/README.md).

JSON and YAML share these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `vendor` | string | Stable vendor key from the table above. |
| `device` | string or null | Normalized device model, e.g. `BI-V150`, `H800`, `MTT-S5000`, `C500`. |
| `driver` | string or null | Installed driver release reported by the vendor. |
| `runtime` | string | Vendor software environment name, e.g. `corex`, `cuda`, `cann`. |
| `runtime_version` | string or null | Installed environment release, independent of the host driver. |
| `unit` | string | Accelerator unit, e.g. GPU, NPU or DCU. |
| `device_count` | integer | Number of detected devices. |
| `memory_unit` | string | Always `MiB`; applies to all per-device memory values. |
| `available_gpus` | string | Comma-separated available indices; retained for existing CI consumers. |
| `gpus` | array | Entries with integer `id`, `total`, `used` and `free` fields. `id` is the device index used in `available_gpus`; existing memory keys are preserved. |

The GPU-check schema uses `device`, `driver` and per-device `id` (not
`device_name`, `driver_version` or `index`). The existing `available_gpus` name
is retained for CI compatibility; `unit` identifies the vendor's accelerator
type even when it is called NPU, DCU, MLU or PPU.

### Examples

```bash
# Auto-detect the vendor and print a human-readable summary.
flagos-gpu-check

# Either form produces a standard JSON file.
flagos-gpu-check --format json > gpu-status.json
flagos-gpu-check --format json --output gpu-status.json

# YAML supports both redirection and --output too.
flagos-gpu-check --format yaml --output gpu-status.yaml

# Force a vendor when device-node auto-detection is unavailable.
flagos-gpu-check --vendor ascend
```

The following representative two-device Iluvatar example illustrates separate
driver and container runtime versions. Counts, versions and memory usage vary
by host.

```text
Vendor: iluvatar
Device: BI-V150
Driver: 5.0.0
Runtime: corex
Runtime Version: 4.5.0
Detected 2 Iluvatar BI-V150 GPU(s).
Available GPUs: 0,1
 GPU  Total (MiB)  Used (MiB)  Free (MiB)
   0       32,768          68      32,700
   1       32,768          68      32,700
```

With `--format json`:

```json
{
  "vendor": "iluvatar",
  "device": "BI-V150",
  "driver": "5.0.0",
  "runtime": "corex",
  "runtime_version": "4.5.0",
  "unit": "GPU",
  "device_count": 2,
  "memory_unit": "MiB",
  "available_gpus": "0,1",
  "gpus": [
    {
      "id": 0,
      "total": 32768,
      "used": 68,
      "free": 32700
    },
    {
      "id": 1,
      "total": 32768,
      "used": 68,
      "free": 32700
    }
  ]
}
```

With `--format yaml` (versions remain strings, including values such as
`26.04`):

```yaml
vendor: iluvatar
device: BI-V150
driver: 5.0.0
runtime: corex
runtime_version: 4.5.0
unit: GPU
device_count: 2
memory_unit: MiB
available_gpus: 0,1
gpus:
- id: 0
  total: 32768
  used: 68
  free: 32700
- id: 1
  total: 32768
  used: 68
  free: 32700
```

## Target Repository Dependencies with uv

The target operator library or repository defines its own runtime and
vendor-specific dependencies in `pyproject.toml`. The `optest` workflow reads
that configuration, then automatically selects and installs the dependency
groups that match the detected vendor. This lets developers and CI install a
reproducible test environment directly from the target repository.

> **Important:** Use `uv pip install`, not `pip install`, for target repository
> extras that rely on `[tool.uv].exclude-dependencies`. Hardware images often
> provide vendor-specific builds of packages such as torch and Triton, while a
> framework may declare generic versions as transitive dependencies. uv can
> exclude those selected distributions from resolution and still install the
> framework's other transitive dependencies. pip does not read `[tool.uv]` and
> currently has no equivalent project setting for this selective exclusion.

### Configure test and vendor extras

Use the normalized vendor name reported by `flagos-gpu-check` as the extra
name. For example:

```toml
[project.optional-dependencies]
test = [
    "pytest",
    "pytest-xdist",
    # Add the plugins that provide --record and --collect-marks.
]
nvidia = [
    "deepspeed==0.19.8",
    "deepspeed-kernels",
]
ascend = [
    # Ascend-specific runtime packages.
]
kunlunxin = [
    # KunlunXin-specific runtime packages.
]

[tool.uv]
# These hardware-specific packages are already supplied by the runtime image.
exclude-dependencies = [
    "torch",
    "triton",
]
```

Only exclude a dependency when every test image already provides a compatible
replacement. Excluding a package that is not present leaves the target runtime
incomplete.

Install the shared test dependencies, a vendor environment, or both with uv:

```bash
uv pip install -e '.[test]'
uv pip install -e '.[nvidia]'
uv pip install -e '.[test,nvidia]'
uv pip install -e '.[test,kunlunxin]'
```

The extra names are owned by the target repository. A project may define
groups for every supported vendor or only the vendors it implements. Keep the
generic test tools in `test` and vendor-specific frameworks, SDK bindings, and
runtime packages in the corresponding vendor group. This lets developers use
the same installation commands locally and in CI without adding heavyweight
target dependencies to `flagos-optest` itself.

### Centralized workflow behavior

The centralized workflow runs `flagos-gpu-check` before installing the target
repository. It exports the detected metadata for later steps as
`FLAGOS_GPU_VENDOR`, `FLAGOS_GPU_DEVICE`, `FLAGOS_GPU_DRIVER`,
`FLAGOS_GPU_RUNTIME`, and `FLAGOS_GPU_RUNTIME_VERSION`.

The dependency step then reads the target `pyproject.toml`. It selects `test`
when present and independently selects the extra whose name exactly matches
`FLAGOS_GPU_VENDOR` when present. It combines the selected extras into one
incremental command, for example:

```bash
uv pip install --python /path/to/active/python -e '.[test,nvidia]'
```

If only `test` exists, the command uses `.[test]`; if only the vendor extra
exists, it uses that group. When neither group exists, it installs the base
project with `-e .`. Using `uv pip install` instead of `uv sync` is intentional:
install preserves already satisfied packages in the prebuilt image, whereas
sync reconciles the entire environment to uv's resolved versions and may
replace packages such as numpy. Target-specific package names and exclusion
policy therefore stay in the target repository's `pyproject.toml`, not in
`repos.yaml` or the workflow.

The target repository owns its application and test dependency contract:

- Put dependencies required on every backend in `[project].dependencies`.
- Put shared test-only dependencies in `[project.optional-dependencies].test`.
- Put backend/vendor-specific dependencies in the optional group named after
  the normalized vendor, such as `nvidia`, `ascend`, or `kunlunxin`.
- Use `[tool.uv].exclude-dependencies` only when the runtime image or the
  configured shared environment intentionally supplies a compatible build that
  uv must leave untouched.

Do not add a dependency to optest's central `deps` map merely because one
target repository needs it. Central `deps` exists for packages that prepare the
shared test environment, are used by multiple target repositories, or require
hardware/repository exceptions that cannot live in a target project. Keeping
project-owned requirements in the target `pyproject.toml` makes local installs
and CI resolve the same environment.

### FlagTree upgrade

After installing `optest` and before probing GPUs, the generic package-upgrade
step replaces the image-provided FlagTree source tree with the backend-specific
configured wheel. A backend declares the distribution under `deps`, using the
package name as the mapping key:

```yaml
# .github/configs/backends/H800.yaml
deps:
  flagtree-nvidia:
    display_name: FlagTree
    priority: 100
    version: 0.7.1.dev20261115
    index_url: https://resource.flagos.net/repository/flagos-pypi-daily/simple
    no_deps: true
    target: /opt/flagtree
```

Most backends use a complete daily version. This matters for runtimes with a different version base; for example,
the CANN 9.0.0 distribution uses `0.7.0.devYYYYMMDD`, while most other daily
distributions use `0.7.1.devYYYYMMDD`.

BW1000 uses `flagtree-hcu==0.7.1.dev20261104` from the FlagOS daily index. Its
backend YAML retains the previous compatible `flagtree==0.7.1a1+hcu3.6`
release mapping as a commented fallback.

The backend YAML is the sole source of the FlagTree version. Updating a wheel
therefore requires changing that backend's exact `version` and confirming that
the artifact exists at its configured `index_url`. The workflow-dispatch form
does not expose a runtime FlagTree version override, and the prepare job does
not query package indexes.

The `Upgrade configured packages (FlagTree, FlagGems)` step installs the exact
vendor wheel with `uv pip install --no-deps --target
/opt/flagtree.new` and only after installation succeeds removes the old
`/opt/flagtree` and moves the staged directory into its place. A failed
resolution or installation therefore leaves the original tree intact. The
command inherits the persistent per-vendor `UV_CACHE_DIR`;
unlike the previous manual pip flow, it does not disable the package cache.
The step prints release metadata before replacement and, afterward, shows the
backend distribution through its target directory. It then checks whether
`/opt/flagtree` is already on `PYTHONPATH`, logs and keeps an existing entry, or
prepends it when missing. The effective value is written to `GITHUB_ENV` so all
later Python and pytest commands use the replacement. Finally, `ACTIVE_PYTHON`
must resolve the configured distribution to the exact requested version; this
avoids treating a bare `pip show`, which cannot see an unactivated target
directory, as an installation check.
Backends with no FlagTree entry under `deps` retain the image-provided
tree; this currently applies only to Cambricon.

A target repository disables an inherited package in its most-specific
dependency layer:

```yaml
repos:
  FlagGems:
    deps:
      910B:
        flagtree-ascend-cann9.0.0:
          enabled: false
```

The current `FlagGems` 910B exception uses the CANN 9.2.1 runtime image and
retains its image-provided FlagTree because the shared 910B daily wheel targets
CANN 9.0.0. Other FlagGems backends and the 910B backend in other repositories
continue to use their normal FlagTree upgrade configuration. `FlagGems` also
uses its current T-Head PPU 2.1.0 and AMD ROCm 7.2.4 runtime images through
repository-specific overrides; both still receive their configured FlagTree
daily wheels. Its Cambricon backend similarly uses the current Neuware 4.7.2
shared backend image, but retains the image-provided FlagTree because no
Cambricon daily distribution is configured. Only FlagGems currently enables
this backend, so it does not need a repository-specific image override.

### FlagGems upgrade

After the backend-specific FlagTree step and before GPU detection, the workflow
installs a shared FlagGems daily wheel into the active Python environment. The
wheel is vendor-independent, so its configuration lives once under
`defaults.deps` in `.github/configs/repos.yaml` instead of being duplicated
across backend YAML files:

```yaml
defaults:
  deps:
    flag-gems:
      display_name: FlagGems
      priority: 200
      version: 5.5.0.dev20261118
      index_url: https://resource.flagos.net/repository/flagos-pypi-daily/simple
      no_deps: true
```

The configured artifact is
`packages/flag-gems/5.5.0.dev20261118/flag_gems-5.5.0.dev20261118-py3-none-any.whl`.
The generic upgrade step processes FlagTree first because its priority is 100,
then FlagGems at priority 200. It uses `uv pip install --no-deps` with the
configured index and verifies through the
active Python interpreter that the installed distribution exactly matches the
configured version.

A repository that must not receive the daily wheel uses a package override:

```yaml
repos:
  FlagGems:
    deps:
      default:
        flag-gems:
          enabled: false
```

This exception applies to FlagGems itself because its checked-out source is
installed editable later in the job.

The generic dependency map is resolved independently for every package name.
Use the exact same package key in multiple layers to override it:

| Precedence | Configuration layer | Scope |
| ---: | --- | --- |
| 1 (lowest) | `defaults.deps` | Every enabled backend of every repository. |
| 2 | `.github/configs/backends/<backend>.yaml` `deps` | That hardware backend in every repository where it is enabled. |
| 3 | `repos.<repo>.deps.default` | Every enabled backend of one repository. |
| 4 (highest) | `repos.<repo>.deps.<backend>` | One repository/backend combination. |

Merging is field-by-field, not whole-package replacement. A field omitted by a
later layer keeps its inherited value. The final merged `enabled` value decides
whether the package is installed: it defaults to `true` when no layer sets it;
an inherited `false` remains false through later layers that omit `enabled`;
and only a later explicit `enabled: true` re-enables it. Conversely, a later
explicit `enabled: false` always disables values inherited from lower layers.
This is independent from `repos.<repo>.backends.<name>.enabled`: a disabled
backend is removed from the matrix before package resolution, so package-level
`enabled: true` cannot restore it.

Merge precedence does not determine installation order. After disabled
packages are removed, enabled packages are installed by ascending `priority`
(default `1000`), then lexicographically by package name for equal priorities.

`version` is metadata used to build `package==version`; when omitted, the bare
package name is installed. Control keys (`enabled`, `display_name`, `priority`,
and `version`) are not passed to uv. Every other key becomes a
`uv pip install` option with underscores changed to hyphens: `index_url` becomes
`--index-url VALUE`, `no_deps: true` becomes `--no-deps`, lists repeat the
option, and omitted keys produce no argument. `target` additionally enables
staged replacement at `<target>.new` and persists that path on `PYTHONPATH`.

#### Adding and excluding configured packages

Use these recipes only for shared test-environment packages. The package name
is the YAML mapping key and must be identical in every layer that overrides it.

Add a package to every enabled backend of every repository:

```yaml
# .github/configs/repos.yaml
defaults:
  deps:
    shared-profiler:
      display_name: Shared profiler
      priority: 300
      version: 2.4.1
      index_url: https://example.invalid/simple
      no_deps: true
```

Add a hardware-specific shared package to every repository that enables that
backend:

```yaml
# .github/configs/backends/H800.yaml
deps:
  nvidia-debug-runtime:
    display_name: NVIDIA debug runtime
    priority: 400
    version: 1.8.0
```

Disable an inherited package for an entire repository, or only for one of its
backends:

```yaml
# .github/configs/repos.yaml
repos:
  SomeRepo:
    deps:
      default:
        shared-profiler:
          enabled: false       # all enabled SomeRepo backends
      H800:
        nvidia-debug-runtime:
          enabled: false       # SomeRepo/H800 only
```

To skip the entire backend job for a repository, use the separate backend
switch instead:

```yaml
repos:
  SomeRepo:
    backends:
      H800: {enabled: false}    # removes SomeRepo/H800 from the matrix
```

`backends.H800.enabled: false` skips every package and the test job for that
combination. `deps.H800.<package>.enabled: false` skips only that package and
still runs SomeRepo on H800.

Add an exceptional central package for one repository, or one repository and
backend combination:

```yaml
repos:
  SomeRepo:
    deps:
      default:
        shared-ci-helper:
          version: 3.1.0       # every enabled SomeRepo backend
          priority: 500
      910B:
        ascend-ci-helper:
          version: 4.2.0       # SomeRepo/910B only
          priority: 500
```

These repository-scoped central entries are exceptions. Prefer
`SomeRepo/pyproject.toml` for packages owned by SomeRepo. To re-enable a package
disabled by a lower layer, set `enabled: true` at the more specific layer; an
omitted `enabled` inherits the earlier value.

The package fields have these meanings:

| Field | Behavior |
| --- | --- |
| `<package-name>` | Exact distribution key used to merge the four configuration layers and build the install requirement. Hyphenated and underscored spellings are different configuration keys. |
| `display_name` | Human-readable log label only. It does not change the distribution name, import name, merge identity, or install order. Defaults to the package key. |
| `priority` | Integer controlling configured-package install order: lower values run first; default `1000`; equal values sort by package key. It does not change configuration merge precedence. |
| `enabled` | Final merged boolean controlling whether the package is installed. Defaults to `true`; the last explicit value wins. |
| `version` | Produces the exact requirement `<package>==<version>`. Omitting it installs the bare package requirement. |
| `target` | Normalized, specific absolute install directory such as `/opt/flagtree`. The workflow installs into `<target>.new`, replaces the old directory only after success, and prepends `target` to `PYTHONPATH`. Omit it for normal active-environment packages. |
| Other fields | Converted to uv options: for example `index_url` becomes `--index-url`, `no_deps: true` becomes `--no-deps`, and list values repeat the option. |

`target` is not a generic virtual-environment prefix. A target-directory package
can coexist with a distribution in the active Python environment, and the
prepended `PYTHONPATH` may take import precedence. Avoid declaring the same
distribution in a target repository unless that coexistence is deliberate.

#### Installation order and same-name packages

The workflow builds the environment in this order:

| Phase | Source | Result |
| ---: | --- | --- |
| 1 | Backend container image | Supplies the initial Python environment and vendor runtime. |
| 2 | optest checkout | Installs the scheduler and its own package metadata. |
| 3 | Merged central `deps` | Installs enabled packages by ascending `priority`, then package key. Exact `version` pins are verified immediately. |
| 4 | GPU detection | Selects the target repository's vendor extra. |
| 5 | Target repository `pyproject.toml` | Installs the base project editable plus `test` and detected-vendor extras. This is the final dependency-resolution phase before tests. |

For normal active-environment packages, the later target install has final
authority over requirements declared by the target project, but uv is
incremental rather than an unconditional overwrite:

- If the target declares an exact or constrained version that the installed
  package does not satisfy, uv installs a satisfying version, replacing the
  earlier image or central-`deps` version when necessary.
- If the installed version already satisfies the target requirement, uv can
  keep it; “installed later” does not by itself force a reinstall.
- If the target project itself has the same distribution name as an installed
  wheel, the editable target install becomes the active project distribution.
  This is why the `FlagGems` repository disables the global `flag-gems` entry.
- A name in `[tool.uv].exclude-dependencies` is omitted from target dependency
  resolution. The compatible version supplied by the image or central `deps`
  remains active; the target pin does not replace it.

Therefore, keep one owner for each requirement. Use `pyproject.toml` as the
authority for repository-owned dependencies, use central `deps` for shared
environment preparation, and use exclusions only for intentional image/shared
runtime ownership. `priority` orders central install commands; it is not a
general dependency-conflict resolver, and the target install always occurs
after all central priorities.

### Persistent package cache

Target dependencies include large wheels (for example `vllm`,
`flashinfer-cubin`, and `cupy`, hundreds of MB each). To avoid re-downloading
them on every run, the workflow points `uv` and `pip` at a persistent cache on
each self-hosted runner's durable storage. The `prepare` job derives the cache
location per vendor and injects it into the matrix test environment:

```text
UV_CACHE_DIR  = <output_base_dir>/uv-cache/<vendor>
PIP_CACHE_DIR = <output_base_dir>/pip-cache/<vendor>
UV_LINK_MODE  = copy
```

- `<output_base_dir>` is the backend's own durable output path from
  `backends/<NAME>.yaml`: `/data/github-outputs` on every backend, and
  `/home/github-outputs` on THead (`PPU-ZW810E`), which has no `/data` mount.
- `<vendor>` is the lowercased `FLAGOS_GPU_VENDOR` (for example `nvidia`,
  `metax`, `ascend`). The cache is keyed by vendor only, not by repository:
  different target repositories on the same vendor install nearly identical
  vendor packages, so a single per-vendor cache lets them all share one set of
  wheels and maximizes reuse.
- `UV_LINK_MODE=copy` is required because the cache and the environment live on
  different filesystems; it also silences the hardlink-fallback warning.

The optest-install step also checks that the runner's `uv` supports
`uv pip show --target` (known supported baseline: `0.11.30`). An older
image-provided binary first runs `uv self update 0.11.30`, which inherits
`UV_CACHE_DIR`; if self-update is unavailable, the step installs
`uv>=0.11.30` through the existing pip index fallback and `PIP_CACHE_DIR`, then
activates that Python environment's scripts directory for later steps.

How this behaves across runs:

- **First run (cold cache):** `uv` downloads each wheel once from the configured
  package index and stores it under `UV_CACHE_DIR`.
- **Later runs (warm cache):** `uv` reuses the stored wheels and performs no
  network download. `uv` keys its cache by package version and wheel URL or
  hash, so an unchanged version is always a cache hit, while a version bump or a
  changed direct-URL requirement re-downloads only the changed wheel
  automatically. No manual cache invalidation is needed.

The optest, generic package-upgrade, and target-dependency install steps use
normal `uv pip install` output. All three print a `[Cache debug]` report
(the resolved `UV_CACHE_DIR`/`PIP_CACHE_DIR`/`UV_LINK_MODE`, and whether the
cache is `COLD` or `WARM` with its size and file count) before and after
installing, so the step log clearly shows which wheels were reused from the
local cache and which were downloaded.

### FlagGems per-vendor operator lists

When the target repository is `FlagGems`, the operator-test step probes each
runner for vendor-specific operator lists before scheduling. The probe is
vendor-agnostic, so every vendor automatically discovers its own files on its
own runner. For other repositories the step keeps the default stage selection
and performs no probe.

The probed paths use the lowercased vendor name:

```text
/data/flaggems/<vendor>/ops_include.txt   # include list -> --op-list-file
/data/flaggems/<vendor>/ops_exclude.txt   # exclude list -> --exclude-op-list-file
```

Here `<vendor>` is `FLAGOS_GPU_VENDOR`, the vendor reported by
`flagos-gpu-check` and exported earlier in the job. It is already lowercase and
matches the runner directory layout, for example `ascend`, `cambricon`, or
`enflame`. Probe behavior:

- If `ops_include.txt` exists, the step appends `--op-list-file <path>` and the
  run tests exactly that list instead of `--stages all`.
- If `ops_exclude.txt` exists, the step appends `--exclude-op-list-file <path>`.
- If both exist, both arguments are appended.
- A missing file is simply skipped; no file is required and the step never
  fails because a list is absent.

An explicit `exclude_ops` workflow input still wins over a probed exclude file,
matching the scheduler priority `--exclude-ops > --exclude-op-list-file`.
Likewise an explicit `ops` input (anything other than `all`) takes priority
over the include probe.

The step logs the probed candidate paths, whether each file was found, and the
final operator and exclusion arguments, so a run log shows exactly which lists
were loaded and took effect. Each vendor maintains its own lists on its runner;
the workflow carries no hardcoded per-vendor path.

## Repository Layout

```text
.github/configs/repos.yaml
                          Repository-level workflow configuration
.github/configs/backends/ Shared hardware and vendor defaults
.github/workflows/        Centralized GitHub Actions workflow
pyproject.toml            Package metadata and console-script definitions
src/flagos_optest/        Python package, scheduler, reports, and GPU checks
```

### Repository configuration

`.github/configs/repos.yaml` defines which operator repositories the shared
workflow can test. Its top-level `org` is combined with each `repos` key to
form the GitHub checkout target. Each repository entry can define:

| Field | Meaning |
| --- | --- |
| `default_branch` | Target branch used when the workflow does not receive an explicit ref. |
| `backends.<name>.enabled` | Enables or disables a shared hardware backend for that repository. |
| `container_images.<backend>` | Repository-specific image override for a backend. |
| `deps.default.<package>` | Repository-wide package override or addition for every enabled backend; fields override global and backend values. |
| `deps.<backend>.<package>` | Highest-precedence package override for one backend; explicit `enabled: true` can re-enable and `false` can disable the inherited package. |
| `base_data` | Optional performance-baseline JSON path in the target checkout. |
| `op_cmd_timeout` | Timeout in seconds for each accuracy or performance subprocess. Inherits `defaults.op_cmd_timeout`. |
| `ops_test_timeout_minutes` | Per-repository timeout for the operator-test step. |

The image keys intentionally differ by level:

```yaml
# .github/configs/backends/H800.yaml: shared default
container_image: harbor.example/default-h800:tag

# .github/configs/repos.yaml: repository-specific override
repos:
  ExampleOperatorLibrary:
    container_images:
      H800: harbor.example/example-h800:tag
```

`repos.<repo>.container_images.<backend>` takes precedence over the backend's
`container_image`. If no repository override exists, the shared backend image
is used. Package maps are merged by exact package name and then by field, from
least to most specific:

```text
defaults.deps -> backend deps -> repo deps.default -> repo deps.<backend>
```

The last explicit value for each field wins. In particular, the last explicit
`enabled` value wins; omitting it inherits the prior state. Package
`priority` controls only the resulting install order and never changes this
configuration precedence.

Normal target-specific vendor dependencies should not use these maps. The
workflow instead selects the target's optional-dependency group from the vendor
key reported by `flagos-gpu-check`.

### Backend configuration

Each `.github/configs/backends/<hardware>.yaml` file supplies the default
hardware configuration shared by all repositories:

| Field | Meaning |
| --- | --- |
| `name` | Hardware/backend identifier referenced by `repos.yaml`. |
| `vendor` | Vendor filter name accepted by the workflow. |
| `runner_labels` | Self-hosted GitHub Actions runner label. |
| `container_image` | Default image, unless the selected repository overrides it. |
| `container_volumes` | Host paths mounted into the job container. |
| `container_options` | Device mappings, privileges, IPC, shared memory, and other container runtime options. |
| `deps.<package>` | Hardware-specific package defaults merged into the generic upgrade map. |
| `test_gpus` | Optional device-ID override; when omitted, `flagos-gpu-check` supplies the available devices. |
| `test_env` | Environment variables injected into the operator-test step. |
| `output_base_dir` | Base directory for result folders and archives. |

Current backend defaults cover H800/NVIDIA, 910B/Ascend, C550/MetaX,
BW1000/Hygon, MTT-S5000/Moore Threads, BI-V150/Iluvatar, PPU-ZW810E/T-Head,
MLU590-M9DE/Cambricon, P800/KunlunXin, W7900D/AMD, TX8110/TsingMicro, and
S60/Enflame. Their current scheduling defaults are summarized below; the image,
volume, container-option, and environment details remain authoritative in each
backend YAML file.

| Backend | Vendor | Runner label | FlagTree package | Default test devices | Output base |
| --- | --- | --- | --- | --- | --- |
| `H800` | Nvidia | `h800-flaggems-weekly` | `flagtree-nvidia` | detected | `/data/github-outputs` |
| `910B` | Ascend | `910b-flaggems-weekly` | `flagtree-ascend-cann9.0.0` | detected | `/data/github-outputs` |
| `C550` | Metax | `c550-flaggems-weekly` | `flagtree-metax` | detected | `/data/github-outputs` |
| `BW1000` | Hygon | `bw1000-flaggems-weekly` | `flagtree-hcu` | detected | `/data/github-outputs` |
| `MTT-S5000` | MThreads | `s5000-flaggems-weekly` | `flagtree-mthreads` | detected | `/data/github-outputs` |
| `BI-V150` | Iluvatar | `v150-flaggems-weekly` | `flagtree-iluvatar` | detected | `/data/github-outputs` |
| `PPU-ZW810E` | THead | `810e-flaggems-weekly` | `flagtree-ppu` | detected | `/home/github-outputs` |
| `MLU590-M9DE` | Cambricon | `mlu590-m9de-flaggems-weekly` | image-provided | detected | `/data/github-outputs` |
| `P800` | KunLunXin | `p800-flaggems-weekly` | `flagtree-kunlunxin` | detected | `/data/github-outputs` |
| `W7900D` | AMD | `w7900d-flaggems-weekly` | `flagtree-amd` | detected | `/data/github-outputs` |
| `TX8110` | TsingMicro | `tx8110-flaggems-weekly` | `flagtree-tsingmicro` | `0-15` (override) | `/data/github-outputs` |
| `S60` | Enflame | `s60-flaggems-weekly` | `flagtree-enflame` | detected | `/data/github-outputs` |

All current backends use containers, and the workflow always invokes
`flagos-gpu-check`. Keep repository-specific policy in `repos.yaml`; change a
backend file only when the shared hardware default itself changes.
