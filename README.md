# FlagOS Operator Test Tools

`flagos-optest` is the shared operator accuracy and performance test scheduler
used by FlagOS operator libraries. The package also provides report-generation,
result-labeling, web-data conversion, and cross-vendor GPU availability tools.

## Install

Install optest into the same Python environment as the operator library under
test:

```bash
python -m pip install /path/to/optest
```

When developing optest itself, install the local checkout:

```bash
cd /path/to/optest
python -m pip install .
```

The package installs these console commands:

```text
flagos-optest       Run operator accuracy and performance tests
flagos-add-labels   Add catalog labels to an existing summary.json
flagos-psum-text    Generate CSV or Markdown reports
flagos-psum-html    Generate a self-contained HTML report
flagos-psum-web     Convert summaries to web-site data JSON
flagos-gpu-check    Detect the accelerator vendor and available devices
```

The target operator repository supplies its own runtime and test dependencies,
including PyTorch, the FlagOS operator library, pytest, and the plugins that
provide options such as `--record` and `--collect-marks`.

## Requirements

The operator repository under test must provide these three resources. Follow
the [FlagGems](https://github.com/flagos-ai/FlagGems) repository as the reference
layout and schema:

```text
conf/operators.yaml   Operator inventory used for selection, stages, and labels
tests/                Pytest accuracy tests
benchmark/            Pytest performance benchmarks
```

The exact names are significant: the scheduler reads `conf/operators.yaml` and
runs pytest against `tests/` and `benchmark/`.

Each `operators.yaml` entry should follow the complete FlagGems schema and
define fields such as `id`, `description`, `for`, `labels`, `kind`, and
`stages`. In particular, `id` identifies the operator, `stages` controls
selection by `--stages`, and `labels` is consumed by `flagos-add-labels`.
Missing or malformed scheduling fields can prevent operators from being
selected or reported correctly.

## Usage: `flagos-optest`

### Purpose

`flagos-optest` discovers the target environment, selects operators from
`conf/operators.yaml`, and schedules each operator's accuracy and benchmark
tests across one or more devices. Like `pytest`, its optional positional
`DIRECTORY` defaults to the current directory.

Run from the operator repository:

```bash
cd /path/to/operator-library
flagos-optest --gpus 0,1 --stages stable --dump-output
```

Or pass the repository explicitly:

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
| `--start OP_ID` | Resume from an operator ID by testing IDs lexicographically greater than or equal to it. |
| `--gpus 0,1` | Select device IDs. Use `--gpus all` to use every device detected by PyTorch. Default: `0`. |
| `--stages stable,beta` | Select inventory stages. Supported values are `alpha`, `beta`, `stable`, `all`, and `removed`. Default: `stable`. |
| `--output-dir DIR` | Result directory. Relative paths are resolved from the target repository. Default: `logs_results_YYYYMMDD_HHMM`. |
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

# Resume a list from the operator named "mul".
flagos-optest --start mul --stages all --gpus 0
```

### Results and `summary.json`

The result directory contains the aggregated summary, per-device intermediate
summaries, and one directory per operator:

```text
logs_results_YYYYMMDD_HHMM/
├── summary.json
├── summary0.json
└── <operator>/
    ├── accuracy_result.json
    ├── performance_result.json
    ├── accuracy_stdout.log       # with --dump-output
    ├── accuracy_stderr.log       # with --dump-output
    ├── performance_stdout.log    # with --dump-output
    └── performance_stderr.log    # with --dump-output
```

`summary.json` contains the run timestamp and duration, an `env` object, and
the per-operator accuracy and performance results. The `env` object centralizes
the test-critical environment and dependency versions detected by the
scheduler:

- CPU architecture, operating system, and Python version;
- PyTorch version, accelerator availability, device model, and device count;
- FlagTree and Triton versions and Triton capability information;
- every recognized FlagOS library that can be imported, including its version,
  vendor, and device backend;
- detected optional frameworks such as vLLM or SGLang; and
- the accelerator SDK name and version, such as CUDA, CANN, XRE, TOPS, PPU,
  COREX, MACA, MUSA, or DTK.

This includes all test-environment dependency and version fields probed by
optest, but it is not a complete `pip freeze` of every installed package. A
shortened `env` example looks like this (OS and Python values vary with the
selected image):

```json
{
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
    "sdk": {
      "name": "CUDA",
      "version": "13.0"
    }
  }
}
```

### Console output example

The following is a real H800 run for `abs`, `add`, and `sum`. Operator output
can be much longer for a full suite; the important lifecycle is test start,
environment probing, operator accuracy/benchmark start and end, total elapsed
time, and test completion.

```text
[INFO] Project root ... /__w/optest/optest/target
[INFO] Test started at ... 2026-10-23 18:48:49
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
[INFO] SDK detection ... CUDA 13.0
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
[INFO] Total elapsed time ... 0:01:47 (107.62s)
[INFO] Test completed.
```

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
operator results, filtering, and preserved per-operator logs when available.
The optional result directory defaults to the current directory.

```bash
# Current directory contains summary.json.
cd /tmp/optest-results
flagos-psum-html -o result.html

# Pass the result directory explicitly.
flagos-psum-html /tmp/optest-results \
  -o /tmp/optest-results/result.html
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

`flagos-gpu-check` is one unified installed command for every supported
accelerator vendor. It detects the host vendor, queries device memory, waits
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
| `-f, --format text` | Human-readable progress and memory table. This is the default and is best for interactive use and CI logs. |
| `-f, --format json` | Machine-readable output for workflow parsing. Progress remains visible on stderr while stdout contains only JSON. |
| `-f, --format yaml` | Structured output for configuration-oriented tools and inspection. Progress remains on stderr. |
| `-o, --output PATH` | Save the final rendered result in the selected format. In text mode, progress still streams to stdout. |
| `--vendor VENDOR` | Skip auto-detection and force one of the supported vendor keys below. |
| `-h, --help` | Show command help. |

Supported vendors:

| Vendor key | Accelerator vendor | Management command | Reported unit |
| --- | --- | --- | --- |
| `amd` | AMD | `rocm-smi` | GPU |
| `ascend` | Huawei Ascend | `npu-smi` | NPU |
| `cambricon` | Cambricon | `cnmon` | MLU |
| `enflame` | Enflame | `efsmi` | GPU |
| `hygon` | Hygon | `hy-smi` | DCU |
| `iluvatar` | Iluvatar CoreX | `ixsmi` | GPU |
| `kunlunxin` | KunlunXin | `xpu-smi` | GPU |
| `metax` | MetaX | `mx-smi` | GPU |
| `mthreads` | Moore Threads | `mthreads-gmi` | GPU |
| `nvidia` | NVIDIA | `nvidia-smi` | GPU |
| `sunrise` | Sunrise | `pt_smi` | GPU |
| `thead` | T-Head | `ppu-smi` | PPU |
| `tsingmicro` | TsingMicro | `tsm_smi` | GPU |

Examples:

```bash
# Auto-detect the vendor and print a human-readable table.
flagos-gpu-check

# Produce clean JSON on stdout for a CI parser.
flagos-gpu-check --format json

# Save YAML while retaining progress messages on stderr.
flagos-gpu-check --format yaml --output /tmp/gpu-status.yaml

# Force a vendor when device-node auto-detection is not available.
flagos-gpu-check --vendor ascend
```

### NVIDIA output example

This is the complete text output captured from an eight-device NVIDIA H800
runner. Memory usage changes over time, so the values on another host will
differ.

```text
Available Vendor: nvidia
Detected 8 NVIDIA GPU card(s).
 GPU  Total (MiB)  Used (MiB)  Free (MiB)
   0       81,559           0      81,559
   1       81,559           0      81,559
   2       81,559           0      81,559
   3       81,559           0      81,559
   4       81,559           0      81,559
   5       81,559           0      81,559
   6       81,559           0      81,559
   7       81,559           0      81,559
Available GPUs: 0,1,2,3,4,5,6,7
```

The same run with `--format json` produces this structured result on stdout:

```json
{
  "vendor": "nvidia",
  "unit": "GPU",
  "available_gpus": "0,1,2,3,4,5,6,7",
  "gpus": [
    {
      "index": 0,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 1,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 2,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 3,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 4,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 5,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 6,
      "total": 81559,
      "used": 0,
      "free": 81559
    },
    {
      "index": 7,
      "total": 81559,
      "used": 0,
      "free": 81559
    }
  ]
}
```

## Target Repository Test Dependencies

The operator library, rather than the optest wheel, owns heavyweight runtime
and vendor dependencies. Define optional dependency groups in the target
repository's `pyproject.toml` so developers and CI can install a reproducible
test environment directly from that repository.

For example:

```toml
[project.optional-dependencies]
test = [
  "pytest",
  "pytest-xdist",
  # Add the plugins that provide --record and --collect-marks.
]
nvidia = [
  # NVIDIA-specific runtime packages.
]
ascend = [
  # Ascend-specific runtime packages.
]
kunlunxin = [
  # KunlunXin-specific runtime packages.
]
```

Install the shared test dependencies, a vendor environment, or both:

```bash
python -m pip install '.[test]'
python -m pip install '.[nvidia]'
python -m pip install '.[test,nvidia]'
python -m pip install '.[test,kunlunxin]'
```

The extra names are owned by the target repository. A project may define
groups for every supported vendor or only the vendors it implements. Keep the
generic test tools in `test` and vendor-specific frameworks, SDK bindings, and
runtime packages in the corresponding vendor group. This lets developers use
the same installation commands locally and in CI without adding heavyweight
target dependencies to `flagos-optest` itself.

## Repository Layout

```text
src/flagos_optest/        Python package, scheduler, reports, and GPU checks
.github/configs/repos.yaml
                          Repository-level workflow configuration
.github/configs/backends/ Shared hardware and vendor defaults
.github/workflows/        Centralized GitHub Actions workflow
pyproject.toml            Package metadata and console-script definitions
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
| `deps.default` | Extra packages installed for every enabled backend. |
| `deps.<backend>` | Extra packages installed only for one backend. |
| `base_data` | Optional performance-baseline JSON path in the target checkout. |
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
is used. Dependency lists are additive rather than replacing one another:

```text
backend deps + repo deps.default + repo deps.<backend>
```

### Backend configuration

Each `.github/configs/backends/<hardware>.yaml` file supplies the default
hardware configuration shared by all repositories:

| Field | Meaning |
| --- | --- |
| `name` | Hardware/backend identifier referenced by `repos.yaml`. |
| `vendor` | Vendor filter name accepted by the workflow. |
| `runner_labels` | Self-hosted GitHub Actions runner label. |
| `use_container` | Whether the test job runs in a container. |
| `container_image` | Default image, unless the selected repository overrides it. |
| `container_volumes` | Host paths mounted into the job container. |
| `container_options` | Device mappings, privileges, IPC, shared memory, and other container runtime options. |
| `test_gpus` | Configured device IDs for this hardware. |
| `force_test_gpus` | When true, use `test_gpus` even if GPU probing reports a different set. |
| `test_env` | Environment variables injected into the operator-test step. |
| `deps` | Hardware-specific packages prepended to repository dependency groups. |
| `gpu_check_script` | Availability command; current backends use `flagos-gpu-check`. |
| `output_base_dir` | Base directory for result folders and archives. |

Current backend defaults cover H800/NVIDIA, 910B/Ascend, C550/MetaX,
BW1000/Hygon, MTT-S5000/Moore Threads, BI-V150/Iluvatar, PPU-ZW810E/T-Head,
MLU590-M9DE/Cambricon, P800/KunlunXin, W7900D/AMD, TX8110/TsingMicro, and
S60/Enflame. Their current scheduling defaults are summarized below; the image,
volume, container-option, and environment details remain authoritative in each
backend YAML file.

| Backend | Vendor | Runner label | Default test devices | Output base |
| --- | --- | --- | --- | --- |
| `H800` | Nvidia | `h800-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `910B` | Ascend | `910b-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `C550` | Metax | `c550-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `BW1000` | Hygon | `bw1000-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `MTT-S5000` | MThreads | `s5000-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `BI-V150` | Iluvatar | `v150-flaggems-weekly` | `0-15` | `/data/github-outputs` |
| `PPU-ZW810E` | THead | `810e-flaggems-weekly` | `0-15` | `/home/github-outputs` |
| `MLU590-M9DE` | Cambricon | `mlu590-m9de-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `P800` | KunLunXin | `p800-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `W7900D` | AMD | `w7900d-flaggems-weekly` | `0-7` | `/data/github-outputs` |
| `TX8110` | TsingMicro | `tx8110-flaggems-weekly` | `0-15` (forced) | `/data/github-outputs` |
| `S60` | Enflame | `s60-flaggems-weekly` | `0-7` | `/data/github-outputs` |

All current backends use containers and `flagos-gpu-check`. Keep
repository-specific policy in `repos.yaml`; change a backend file only when the
shared hardware default itself changes.
