# FlagOS 算子测试工具

[English](README.md) | 简体中文

[![Python checks](https://github.com/flagos-ai/optest/actions/workflows/python-checks.yaml/badge.svg?branch=main)](https://github.com/flagos-ai/optest/actions/workflows/python-checks.yaml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/flagos-ai/optest/actions/workflows/python-checks.yaml)

`flagos-optest` 是供多个 FlagOS 算子库共用的算子精度与性能测试工具。
本软件包还提供报告生成、结果标签添加、Web 数据转换以及厂商专用的 GPU 检查工具集。

## 目录

- [安装](#安装)
- [要求](#要求)
- [`flagos-optest`](#flagos-optest-用法)
  - [用途](#用途)
  - [结果与 `summary.json`](#结果与-summaryjson)
  - [控制台输出示例](#控制台输出示例)
  - [结果辅助命令](#结果辅助命令)
- [`flagos-gpu-check`](#flagos-gpu-check-用法)
  - [用途与行为](#用途与行为)
  - [选项](#选项)
  - [设备与运行时元数据](#设备与运行时元数据)
  - [示例](#示例)
- [使用 uv 管理目标仓库依赖](#使用-uv-管理目标仓库依赖)
  - [配置测试与厂商 extra](#配置测试与厂商-extra)
  - [集中式工作流行为](#集中式工作流行为)
  - [FlagTree 升级](#flagtree-升级)
  - [FlagGems 升级](#flaggems-升级)
  - [持久化软件包缓存](#持久化软件包缓存)
- [仓库布局](#仓库布局)
  - [仓库配置](#仓库配置)
  - [后端配置](#后端配置)

## 安装

将 `optest` 安装到与待测试算子库相同的 Python 环境中：

```bash
uv pip install /path/to/optest
```

也可以以可编辑模式安装该工具：

```bash
git clone https://github.com/flagos-ai/optest
cd optest
uv pip install -e .
```

本软件包会安装以下控制台命令：

```text
flagos-optest       运行算子精度和性能测试
flagos-add-labels   为现有 summary.json 文件添加清单标签
flagos-psum-text    生成 CSV 或 Markdown 报告
flagos-psum-html    生成独立的 HTML 报告
flagos-psum-web     将汇总结果转换为网站数据 JSON
flagos-gpu-check    检测加速器厂商和可用设备
```

目标算子仓库应自行提供运行时和测试依赖，包括 torch、算子库、pytest，
以及提供 `--record`、`--collect-marks` 等选项的插件。

## 要求

待测试的算子库必须提供以下文件和目录。可参考
[FlagGems](https://github.com/flagos-ai/FlagGems) 的仓库布局：

```text
conf/operators.yaml   用于算子选择、阶段和标签的算子清单
tests/                Pytest 精度测试
benchmark/            Pytest 性能基准测试
```

这些名称有明确含义：调度器读取 `conf/operators.yaml`，对 `tests/` 运行
pytest 以执行精度测试，并对 `benchmark/` 运行 pytest 以执行性能基准测试。

`operators.yaml` 中的每个条目都应遵循完整的 FlagGems schema，并定义
`id`、`description`、`for`、`labels`、`kind` 和 `stages` 等字段。
其中，`id` 标识算子，`stages` 控制 `--stages` 的选择范围，`labels` 由
`flagos-add-labels` 使用。算子元数据字段缺失或格式错误，可能导致算子无法被正确选择或报告。

## `flagos-optest` 用法

### 用途

`flagos-optest` 会探测目标环境，从 `conf/operators.yaml` 中选择算子，
并在一个或多个设备上对指定算子启动精度测试和基准测试。与 `pytest` 类似，
可选的位置参数 `DIRECTORY` 默认为当前目录。

在算子仓库中运行：

```bash
cd /path/to/operator-library
flagos-optest --gpus 0,1 --stages stable --dump-output
```

也可以显式传入仓库路径：

```bash
flagos-optest /path/to/operator-library \
  --gpus 0,1 --stages stable --dump-output
```

常用选项：

| 选项 | 用途 |
| --- | --- |
| `DIRECTORY` | 目标算子仓库。默认为当前目录。 |
| `--ops add,sum` | 测试显式指定的逗号分隔算子列表。优先级高于 `--op-list-file` 和 `--stages`。 |
| `--op-list-file FILE` | 从文件读取算子 ID，每行一个；以 `#` 开头的行为注释。 |
| `--exclude-ops add,sum` | 从已选算子中排除逗号分隔的列表。优先级高于 `--exclude-op-list-file`。 |
| `--exclude-op-list-file FILE` | 从文件读取要排除的算子 ID，每行一个；以 `#` 开头的行为注释。 |
| `--start OP_ID` | 从某个算子 ID 恢复，测试按字典序大于或等于该 ID 的算子。 |
| `--gpus 0,1` | 选择设备 ID。使用 `--gpus all` 可使用 PyTorch 检测到的所有设备。默认值：`0`。 |
| `--stages stable,beta` | 选择清单阶段。支持 `alpha`、`beta`、`stable`、`all` 和 `removed`。默认值：`stable`。 |
| `--output-dir DIR` | 结果目录。相对路径以目标仓库为基准解析。默认值：`logs_results_YYYYMMDD_HHMM`。 |
| `--timeout SECONDS` | 每个精度或性能子进程的超时时间。默认值：`1800` 秒。 |
| `--dump-output` | 保留每个精度和基准测试子进程的 stdout 与 stderr。 |
| `--quick` | 要求目标测试使用精简的参数组合。 |
| `--color auto\|always\|never` | 控制 ANSI 彩色输出。默认值：`auto`。 |
| `--base-data FILE` | 提供可选的性能基线 JSON 文件。 |

示例：

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

显式包含和排除列表会在保持原始顺序的同时去重。排除项使用区分大小写的精确算子 ID 匹配。
只要提供任一列表选项，启动日志就会报告解析出的包含项和排除项数量、匹配的排除项数量，
以及最终调度的算子数量。日志还会记录每个解析出的排除项，包括未匹配所选算子的名称：

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

使用基于文件的列表时，汇总和列表标题中会包含源文件名。为提高可读性，
每个解析出的排除项会单独占一行：

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

### 结果与 `summary.json`

成功运行后，结果目录包含聚合汇总以及每个算子对应的目录：

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

运行期间，每个 worker 会把紧凑的 `start`、`operator` 和 `complete` 记录追加到
`summary<N>.jsonl` 检查点。每个算子记录只写一次，并立即 flush 和 fsync，避免每完成
一个算子就重写此前全部结果所导致的二次方写放大。只有当所有预期 worker 检查点都
完整、无重复且精确覆盖本次选择的算子，并且最终 `summary.json` 已经原子发布并通过
校验后，才会删除这些检查点。未完成或失败的运行会保留所有现有检查点，以便恢复。
worker 启动时会原子替换自己的检查点，因此复用输出目录也不会混入旧运行的记录。

`summary.json` 包含运行时间戳和持续时间、`total_ops`（成功聚合到 `result`
中的算子数量）、算子级别的 `accuracy_counts` 和 `performance_counts`、
一个 `env` 对象，以及每个算子的精度与性能结果。每个算子条目还带有
`customized` 标记和 `labels` 列表。`labels` 合并了来自 `conf/operators.yaml`
的清单标签（由 `flagos-add-labels` 写入）与运行时检测到的标签。其中,当该芯片
实际 dispatch 到的实现使用了 Triton 实验语言扩展（其源码文件、或它导入的 kernel
子模块 import 了 `triton.experimental.tle.language`）时,会包含 `"tle"`。
`"tle"` 从实际运行的算子解析得到：厂商自己的 kernel 用了 tle 就只在该厂商芯片上标注,
公共实现用了 tle 则所有芯片都标注,因此按芯片区分。
状态计数包含
`passed`、`failed`、`skipped`、`not_found`、`error` 和 `timeout`，即使值为零也会保留。

生成的汇总不再把完整精度 case 和性能 shape 明细写入 `summary.json`；这些明细只保存
在每个算子的 `accuracy_result.json` 和 `performance_result.json` 中。精度记录
保留单行 `note`，性能 dtype 记录保留聚合后的 `speedup`、`base`、`gems` 和
`shape_count`。`data_file` 与 `details_mode: "external"` 指向规范的原始明细文件。
`flagos-psum-text --single` 和 `--compare` 只会按需读取选中算子的文件，同时继续兼容
包含内联 `details` 的旧汇总。如果复制结果后仍需生成详细报告，必须将算子目录与
`summary.json` 一起保留。

`env` 对象集中记录调度器检测到的关键测试环境和依赖版本：

- CPU 架构、操作系统和 Python 版本；
- PyTorch 版本、加速器可用性、设备型号和设备数量；
- FlagTree 和 Triton 版本，以及 Triton 能力信息；
- 所有可导入的已识别 FlagOS 库，包括其版本、厂商和设备后端；
- 检测到的可选框架，例如 vLLM 或 SGLang；
- 标准化的加速器厂商、设备、驱动、运行时和运行时版本，字段与
  `flagos-gpu-check` 使用的元数据相同。

其中包含 optest 探测的所有测试环境依赖和版本字段，但并非对所有已安装软件包执行完整的
`pip freeze`。下面是精简的 `env` 示例（操作系统和 Python 值会随所选镜像而变化）：

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
    "arch": "x86_64",
    "os": "Ubuntu 24.04.3 LTS",
    "python": "3.12.12",
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

### 控制台输出示例

以下是针对 `abs`、`add` 和 `sum` 的真实 H800 运行记录。完整测试套件的算子输出可能长得多；
重要的生命周期包括测试开始、环境探测、算子精度/基准测试开始与结束、总耗时和测试完成。

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

最终控制台汇总使用与 `summary.json` 相同的算子计数。精度和性能行会省略计数为零的状态。

### 结果辅助命令

在 `flagos-optest` 创建 `summary.json` 后运行以下命令。

#### `flagos-add-labels`

此命令会将目标仓库 `conf/operators.yaml` 中每个算子的 `labels` 复制到
`summary.json` 中对应的结果里。当下游筛选或分组需要清单标签时，请在发布报告前运行它。

第一个位置目录参数可选，默认为当前目标仓库；`RESULT_DIR` 为必填参数：

```bash
# Current directory is the operator repository.
cd /path/to/operator-library
flagos-add-labels /tmp/optest-results

# Run from anywhere and pass both directories explicitly.
flagos-add-labels /path/to/operator-library /tmp/optest-results
```

#### `flagos-psum-text`

此命令将 `summary.json` 转换为 CSV 或 Markdown。CSV 适用于数据处理和电子表格；
Markdown 适用于 issue、pull request 和其他文本报告。可选的结果目录默认为当前目录。

```bash
# Current directory contains summary.json.
cd /tmp/optest-results
flagos-psum-text -o result.csv
flagos-psum-text --format markdown -o result.md

# Pass the result directory explicitly.
flagos-psum-text /tmp/optest-results \
  -o /tmp/optest-results/result.csv
```

其他模式包括 `--metric {speedup,base}`、用于生成详细单算子 Markdown 报告的
`--single`，以及用于比较 `before/` 和 `after/` 单算子结果的 `--compare`。多算子报告
直接使用 `summary.json` 中的聚合值；详细的单算子/对比报告才会按需读取引用的原始
结果文件。

#### `flagos-psum-html`

此命令会创建独立的 HTML 报告，其中包含环境详情、算子结果和筛选功能。默认报告
只包含汇总数据。传入 `--include-logs` 后，还会内嵌每个算子的 stdout 和 stderr
日志；结果 JSON 文件始终不会被内嵌。可选的结果目录默认为当前目录。

```bash
# Current directory contains summary.json.
cd /tmp/optest-results
flagos-psum-html -o result.html

# Pass the result directory explicitly.
flagos-psum-html /tmp/optest-results \
  -o /tmp/optest-results/result.html

# 显式选择内嵌精度和性能测试的 stdout/stderr 日志。
flagos-psum-html /tmp/optest-results --include-logs \
  -o /tmp/optest-results/result-with-logs.html
```

#### `flagos-psum-web`

此命令将包含各平台汇总结果的目录转换为 FlagOS 文档网站使用的 JSON 数据格式。
它适用于网站发布流水线，而不是单个本地报告：

```bash
flagos-psum-web --raw-dir /path/to/platform-results \
  --output-dir /path/to/web-data
```

## `flagos-gpu-check` 用法

### 用途与行为

`flagos-gpu-check` 是面向所有受支持加速器厂商的统一安装命令。它会检测主机厂商、
查询设备显存，在没有设备具备足够空闲显存时等待，并返回可用于测试调度的设备 ID。

当设备至少有 30,000 MiB 空闲显存时，即视为可用。如果没有设备达到该阈值，
命令会每 120 秒重试一次，最长等待 600 秒，并在超时后以非零状态退出。
这些值目前固定在实现中，尚未作为命令行参数开放。

自动检测会先查找厂商专用的 `/dev` 节点，再确认该厂商的管理命令位于 `PATH` 中。
设备节点是主要判断依据，因为有些厂商会提供与 NVIDIA 兼容的命令 shim。
当受限容器中无法自动检测，或必须检查特定实现时，请使用 `--vendor`。

### 选项

| 选项 | 用途和典型使用方式 |
| --- | --- |
| `-f, --format text` | 简洁的设备、驱动、运行时和显存汇总（默认）。 |
| `-f, --format json` | 在标准输出中生成单个 JSON 文档，适合直接解析或重定向。 |
| `-f, --format yaml` | 在标准输出中生成单个 YAML 文档。 |
| `-o, --output PATH` | 不输出到标准输出，而是将相同的最终文档以任意格式写入 UTF-8 文件。 |
| `--vendor VENDOR` | 跳过自动检测，强制使用下表中受支持的厂商键。 |
| `-h, --help` | 显示帮助信息并退出。 |

检查成功时，不会打印原始的管理工具表格、时间戳或进程列表。在所有格式下，
重试消息、警告和错误都只写入标准错误。使用 `--output` 时，标准输出为空。
失败时会以非零状态退出，且不会生成不完整的结果文档。

### 设备与运行时元数据

`runtime` 这一名称指容器中为 FlagGems、FlagTree 及类似使用方安装的厂商软件环境。
它是整个发行版的统称，其中还可能包含 SDK 头文件、编译器和库。
`runtime_version` 是该环境的发行版本；`driver` 则单独描述主机驱动。

| 厂商键 | 加速器厂商 | 管理命令 | 单元 | 运行时 |
| --- | --- | --- | --- | --- |
| `amd` | AMD | `rocm-smi` | GPU | `rocm` |
| `ascend` | 华为昇腾 | `npu-smi` | NPU | `cann` |
| `cambricon` | 寒武纪 | `cnmon` | MLU | `neuware` |
| `enflame` | 燧原 | `efsmi` | GPU | `tops` |
| `hygon` | 海光 | `hy-smi` | DCU | `dtk` |
| `iluvatar` | 天数智芯 | `ixsmi` | GPU | `corex` |
| `kunlunxin` | 昆仑芯 | `xpu-smi` | GPU | `xre` |
| `metax` | 沐曦 | `mx-smi` | GPU | `maca` |
| `mthreads` | 摩尔线程 | `mthreads-gmi` | GPU | `musa` |
| `nvidia` | NVIDIA | `nvidia-smi` | GPU | `cuda` |
| `sunrise` | 日初 | `pt_smi` | GPU | `tangrt` |
| `thead` | 平头哥 | `ppu-smi` | PPU | `ppu` |
| `tsingmicro` | 清微智能 | `tsm_smi` | GPU | `tsm` |

元数据采用尽力探测方式，不要求导入 PyTorch 或厂商 Python 模块。型号和驱动信息来自
管理工具查询；昇腾还会检查其驱动安装元数据。设备名称会移除已知的厂商和形态修饰，
例如 `Iluvatar BI-V150 OAM` 会变为 `BI-V150`。如果混合主机上存在不同型号或驱动版本，
会使用 `, ` 连接，而不是静默选择第一个。

运行时版本取自安装根目录下的发行版元数据，例如 `version.json`、`version.txt`、
`version`、`VERSION` 和 `.info/version`。CANN 还会检查
`ascend_toolkit_install.info`、`version.cfg` 以及 CANN 9 的
`share/info/runtime/version.info` 发行版清单，而不是工具组件版本或依赖要求。
T-Head 会读取 PPU SDK 的 `release.yaml`，与 `ppu-smi` 工具版本无关。XRE 优先读取
`include/xpu/version.h` 中的 `XPURT_VERSION_STR`，避免使用 `version.txt` 中的驱动兼容版本，
并检查其 `xre/` 子目录。显式安装环境变量（例如 `COREX_ROOT`、`DTKROOT`、
`CUDA_HOME`、`ASCEND_HOME_PATH`、`MACA_PATH` 和 `MUSA_INSTALL_PATH`）优先于
常规安装目录。CUDA 还支持通过 `nvcc --version`，以及运行时专用镜像中的
`cuda-libraries-*` 发行包探测版本。CoreX 从 `bin/clang --version` 的厂商构建字符串
读取发行版本，DTK 从 `.dtk_version` 读取，MUSA 从 `version.json` 中的
`musa_toolkits.version` 读取，TOPS 从 `topsruntime` 软件包读取。TsingMicro 通过
`tsm_smi --version` 分别提供驱动和运行时版本；其他厂商可以在其管理工具输出中
提供带有明确运行时标签的版本。

SMI 头部中的 CUDA 兼容版本**并非**已安装的 CUDA 发行版本。同样，IX-ML 和天数智芯
驱动的 CUDA 兼容版本也**并非** CoreX 发行版本。缺失的元数据在 JSON/YAML 中为
`null`，在文本中为 `unknown`；这不会阻止显存可用性检查。因此，省略这些元数据源的
运行时镜像可能会报告未知版本。

真实输出的回归样本和已验证的硬件/运行时矩阵列在
[tests/fixtures/gpu_check](tests/fixtures/gpu_check/README.md) 中。

JSON 和 YAML 共用以下字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `vendor` | string | 上表中的稳定厂商键。 |
| `device` | string 或 null | 标准化的设备型号，例如 `BI-V150`。 |
| `driver` | string 或 null | 主机驱动版本。 |
| `runtime` | string 或 null | 标准化的运行时名称，例如 `cuda` 或 `corex`。 |
| `runtime_version` | string 或 null | 容器或用户空间运行时版本。 |
| `unit` | string | 厂商使用的加速器单元名称，例如 `GPU`、`NPU`、`DCU`、`MLU` 或 `PPU`。 |
| `device_count` | integer | 检测到的设备总数。 |
| `memory_unit` | string | 始终为 `MiB`；适用于所有设备的显存值。 |
| `available_gpus` | string | 逗号分隔的可用索引；为现有 CI 使用方保留。 |
| `gpus` | array | 包含整数 `id`、`total`、`used` 和 `free` 字段的条目。`id` 是 `available_gpus` 中使用的设备索引；保留现有显存键。 |

GPU 检查 schema 使用 `device`、`driver` 和每个设备的 `id`（而不是
`device_name`、`driver_version` 或 `index`）。为兼容 CI，保留现有的
`available_gpus` 名称；即使厂商将加速器称为 NPU、DCU、MLU 或 PPU，
`unit` 也会标明其加速器类型。

### 示例

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

下面这个具有代表性的双设备天数智芯示例展示了相互独立的驱动版本和容器运行时版本。
设备数量、版本和显存使用情况会因主机而异。

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

使用 `--format json` 时：

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

使用 `--format yaml` 时（包括 `26.04` 等值在内的版本号仍为字符串）：

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

## 使用 uv 管理目标仓库依赖

目标算子库或仓库在自己的 `pyproject.toml` 中定义运行时和厂商专用依赖。
`optest` 工作流读取该配置，并根据检测到的厂商自动选择和安装匹配的依赖组。
这样，开发者和 CI 就能直接从目标仓库安装可复现的测试环境。

> **重要：** 对依赖 `[tool.uv].exclude-dependencies` 的目标仓库 extra，请使用
> `uv pip install`，不要使用 `pip install`。硬件镜像通常会提供 torch 和 Triton
> 等软件包的厂商专用构建版本，而框架可能将通用版本声明为传递依赖。uv 可以从解析中
> 排除这些选定的发行包，同时仍安装该框架的其他传递依赖。pip 不会读取 `[tool.uv]`，
> 目前也没有与这种选择性排除等效的项目设置。

### 配置测试与厂商 extra

使用 `flagos-gpu-check` 报告的标准化厂商名称作为 extra 名称。例如：

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

只有当每个测试镜像都已经提供兼容替代项时，才应排除某项依赖。
如果被排除的软件包并不存在，目标运行时将不完整。

使用 uv 安装共享测试依赖、厂商环境或两者：

```bash
uv pip install -e '.[test]'
uv pip install -e '.[nvidia]'
uv pip install -e '.[test,nvidia]'
uv pip install -e '.[test,kunlunxin]'
```

extra 名称由目标仓库管理。项目可以为每个受支持的厂商定义依赖组，也可以只定义其实现的厂商。
通用测试工具应放在 `test` 中，厂商专用框架、SDK 绑定和运行时软件包则放在对应的厂商组中。
这样，开发者在本地和 CI 中可以使用相同的安装命令，而不必把重量级目标依赖加入
`flagos-optest` 本身。

### 集中式工作流行为

集中式工作流会在安装目标仓库之前运行 `flagos-gpu-check`，并将检测到的元数据导出为
`FLAGOS_GPU_VENDOR`、`FLAGOS_GPU_DEVICE`、`FLAGOS_GPU_DRIVER`、
`FLAGOS_GPU_RUNTIME` 和 `FLAGOS_GPU_RUNTIME_VERSION`，供后续步骤使用。

依赖安装步骤随后读取目标仓库的 `pyproject.toml`。如果存在 `test`，则选择它；
如果存在名称与 `FLAGOS_GPU_VENDOR` 完全匹配的 extra，则独立选择该 extra。
被选中的 extra 会合并到一条增量安装命令中，例如：

```bash
uv pip install --python /path/to/active/python -e '.[test,nvidia]'
```

如果只有 `test`，命令使用 `.[test]`；如果只有厂商 extra，则使用该依赖组；
如果两者都不存在，则使用 `-e .` 安装基础项目。选择 `uv pip install` 而非
`uv sync` 是有意为之：install 会保留预构建镜像中已满足的依赖，而 sync 会使整个环境
与 uv 的解析版本保持一致，可能替换 numpy 等软件包。因此，目标专用的软件包名称和排除策略
应保留在目标仓库的 `pyproject.toml` 中，而不是放入 `repos.yaml` 或工作流。

目标仓库负责维护自身应用和测试依赖契约：

- 所有 backend 都需要的依赖放入 `[project].dependencies`。
- 共享的测试专用依赖放入 `[project.optional-dependencies].test`。
- backend/厂商专用依赖放入以标准化厂商名命名的 optional group，例如 `nvidia`、
  `ascend` 或 `kunlunxin`。
- 只有当运行时镜像或通用配置环境有意提供兼容构建，并且需要 uv 保持其不变时，才使用
  `[tool.uv].exclude-dependencies`。

不要因为只有一个目标仓库需要某项依赖，就把它加入 optest 的中央 `deps` map。中央
`deps` 用于准备共享测试环境、供多个目标仓库共同使用的软件包，或无法放入目标项目的
硬件/仓库例外。将项目自身依赖保留在目标仓库 `pyproject.toml` 中，可以让本地安装和 CI
解析出相同环境。

### FlagTree 升级

工作流会在安装 `optest` 之后、探测 GPU 之前，由通用 package 升级步骤用后端配置的
wheel 替换镜像自带的 FlagTree 源码目录。后端在 `deps` 中以 package name 为 key：

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

大多数后端使用完整的默认 daily 版本。不同运行时可能具有不同的
版本基线：例如 CANN 9.0.0 使用 `0.7.0.devYYYYMMDD`，而大多数 daily distribution
使用 `0.7.1.devYYYYMMDD`。

BW1000 使用 FlagOS daily index 中的 `flagtree-hcu==0.7.1.dev20261104`。
其 backend YAML 将之前兼容的 `flagtree==0.7.1a1+hcu3.6` release 配置保留为
注释状态，以便必要时回退。

backend YAML 是 FlagTree 版本的唯一来源。升级 wheel 时需要修改对应 backend 的精确
`version`，并确认制品存在于配置的 `index_url`。workflow-dispatch 页面不提供运行时
FlagTree 版本覆盖，prepare job 也不会查询 package index。

`Upgrade configured packages (FlagTree, FlagGems)` 通用步骤通过
`uv pip install --no-deps --target /opt/flagtree.new` 将精确配置的厂商 wheel
安装到 staging 目录，安装成功后才删除原来的 `/opt/flagtree` 并将 staging 目录移动到
该路径。因此解析或安装失败时，镜像原有目录保持不变。安装命令继承按厂商持久化的
`UV_CACHE_DIR`；与之前的手工 pip 流程不同，不会禁用软件包缓存。步骤会在替换前输出
release 元数据，并在替换后通过 target 目录输出后端 distribution 信息。随后检查
`/opt/flagtree` 是否已在 `PYTHONPATH` 中：存在时打印并保留，缺失时将其前置；最终值写入
`GITHUB_ENV`，确保后续 Python 和 pytest 命令默认使用替换后的目录。最后再通过
`ACTIVE_PYTHON` 校验配置的 distribution 是否为精确请求版本，避免将无法看到未激活
target 目录的裸 `pip show` 误当成安装检查。没有在 `deps` 中配置 FlagTree 的后端会保留镜像自带目录；目前只有
Cambricon 使用此行为。

`FlagGems` 继承共享的 910B CANN 9.0.0 镜像，因此会安装配置的
`flagtree-ascend-cann9.0.0` daily wheel。它仍通过仓库专用覆盖使用当前的 T-Head
PPU 2.1.0 和 AMD ROCm 7.2.4 runtime 镜像，这两个后端也会安装各自配置的 FlagTree
daily wheel。其 Cambricon 后端同样使用当前的 Neuware 4.7.2 共享后端镜像，但由于未
配置 Cambricon daily distribution，会保留镜像自带的 FlagTree。目前只有 FlagGems
启用该后端，因此无需仓库专用镜像覆盖。

### FlagGems 升级

工作流会在后端专用的 FlagTree 步骤之后、GPU 探测之前，将共享的 FlagGems daily
wheel 安装到活动 Python 环境。该 wheel 不区分厂商，因此只需在
`.github/configs/repos.yaml` 的 `defaults.deps` 中配置一次，无需在每个 backend
YAML 中重复：

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

当前配置对应的制品为
`packages/flag-gems/5.5.0.dev20261118/flag_gems-5.5.0.dev20261118-py3-none-any.whl`。
通用升级步骤按 priority 100 先处理 FlagTree，再按 priority 200 处理 FlagGems。它使用
配置的 index 执行 `uv pip install --no-deps`，并通过活动 Python 解释器校验已安装
distribution 与配置版本完全一致。

必须跳过 daily wheel 的仓库使用 package override：

```yaml
repos:
  FlagGems:
    deps:
      default:
        flag-gems:
          enabled: false
```

FlagGems 自身需要该例外，因为任务稍后会以 editable 模式安装其 checkout 源码。

通用依赖映射按 package name 独立解析；需要覆盖同一 package 时，各层必须使用完全相同的
package key：

| 优先级 | 配置层 | 作用范围 |
| ---: | --- | --- |
| 1（最低） | `defaults.deps` | 所有仓库的所有已启用 backend。 |
| 2 | `.github/configs/backends/<backend>.yaml` 中的 `deps` | 在启用了该硬件 backend 的所有仓库中生效。 |
| 3 | `repos.<repo>.deps.default` | 指定仓库的所有已启用 backend。 |
| 4（最高） | `repos.<repo>.deps.<backend>` | 指定仓库和 backend 的组合。 |

合并按字段进行，而不是用整个 package mapping 替换前层。后层未提供的字段继续继承。
最终合并后的 `enabled` 决定是否安装：所有层均未设置时默认为 `true`；前层的 `false`
会穿过未设置 `enabled` 的后层继续生效；只有后层显式设置 `enabled: true` 才能重新启用；
反过来，后层显式设置 `enabled: false` 总会禁用从前层继承的 package。
这与 `repos.<repo>.backends.<name>.enabled` 相互独立：被禁用的 backend 会在 package
解析前从矩阵中移除，因此 package 的 `enabled: true` 不能恢复整个 backend。

配置合并优先级不决定安装顺序。移除禁用项后，package 按 `priority` 升序（默认
`1000`）安装；相同 priority 再按 package name 的字典序安装。

`version` 只用于生成 `package==version`；省略时安装裸 package name。控制字段
（`enabled`、`display_name`、`priority`、`version`）不会传给 uv。
其他字段会将下划线转换为连字符并成为 `uv pip install` 参数：`index_url` 转为
`--index-url VALUE`，`no_deps: true` 转为 `--no-deps`，列表值会重复该选项，省略字段则
不产生参数。`target` 还会启用 `<target>.new` staging 替换，并将该路径持久写入
`PYTHONPATH`。

#### 新增和排除通用 package

以下配置只用于共享测试环境 package。package name 是 YAML mapping key；任何覆盖层都必须
使用完全相同的 key。

为所有仓库的所有已启用 backend 添加 package：

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

为所有启用了某个硬件 backend 的仓库添加共享 package：

```yaml
# .github/configs/backends/H800.yaml
deps:
  nvidia-debug-runtime:
    display_name: NVIDIA debug runtime
    priority: 400
    version: 1.8.0
```

为整个仓库排除继承的 package，或只排除该仓库的某个 backend：

```yaml
# .github/configs/repos.yaml
repos:
  SomeRepo:
    deps:
      default:
        shared-profiler:
          enabled: false       # SomeRepo 的所有已启用 backend
      H800:
        nvidia-debug-runtime:
          enabled: false       # 仅 SomeRepo/H800
```

若要为某个仓库跳过整个 backend job，应使用独立的 backend 开关：

```yaml
repos:
  SomeRepo:
    backends:
      H800: {enabled: false}    # 从 matrix 删除 SomeRepo/H800
```

`backends.H800.enabled: false` 会跳过该组合的所有 package 和测试 job；
`deps.H800.<package>.enabled: false` 只跳过该 package，SomeRepo 的 H800 测试仍会运行。

为一个仓库，或一个仓库/backend 组合新增例外的中央 package：

```yaml
repos:
  SomeRepo:
    deps:
      default:
        shared-ci-helper:
          version: 3.1.0       # SomeRepo 的所有已启用 backend
          priority: 500
      910B:
        ascend-ci-helper:
          version: 4.2.0       # 仅 SomeRepo/910B
          priority: 500
```

这些仓库范围的中央配置属于例外。SomeRepo 自身拥有的 package 应优先写入
`SomeRepo/pyproject.toml`。若要重新启用在较低层被禁用的 package，应在更具体层显式设置
`enabled: true`；省略 `enabled` 会继承前层状态。

package 字段定义如下：

| 字段 | 行为 |
| --- | --- |
| `<package-name>` | 用于合并四层配置并生成安装 requirement 的精确 distribution key。连字符和下划线写法是不同的配置 key。 |
| `display_name` | 仅用于人类可读的日志标签；不会改变 distribution name、import name、合并标识或安装顺序。默认值为 package key。 |
| `priority` | 控制通用 package 安装顺序的整数：数值较小者先执行，默认 `1000`；相同值按 package key 排序。它不会改变配置合并优先级。 |
| `enabled` | 最终合并后的布尔值，决定是否安装。默认 `true`，最后一个显式值生效。 |
| `version` | 生成精确 requirement `<package>==<version>`；省略时安装裸 package requirement。 |
| `target` | 规范化的具体绝对安装目录，例如 `/opt/flagtree`。工作流先安装至 `<target>.new`，成功后才替换旧目录，并将 `target` 前置到 `PYTHONPATH`。普通活动环境 package 应省略该字段。 |
| 其他字段 | 转为 uv 参数，例如 `index_url` 变成 `--index-url`、`no_deps: true` 变成 `--no-deps`，列表值会重复该参数。 |

`target` 不是普通虚拟环境前缀。target-directory package 可以与活动 Python 环境中的同名
distribution 共存，并且前置的 `PYTHONPATH` 可能拥有 import 优先级。除非明确需要这种共存，
否则不要在目标仓库中再次声明同一 distribution。

#### 安装顺序和同名 package

工作流按以下顺序构建环境：

| 阶段 | 来源 | 结果 |
| ---: | --- | --- |
| 1 | backend 容器镜像 | 提供初始 Python 环境和厂商运行时。 |
| 2 | optest checkout | 安装调度器及其自身 package metadata。 |
| 3 | 合并后的中央 `deps` | 按 `priority` 升序、再按 package key 安装启用项，并立即校验精确 `version` pin。 |
| 4 | GPU 探测 | 选择目标仓库的厂商 extra。 |
| 5 | 目标仓库 `pyproject.toml` | 以 editable 模式安装基础项目以及 `test` 和检测到的厂商 extra；这是测试前最后一次依赖解析。 |

对于普通活动环境 package，较晚执行的目标安装对目标项目声明的 requirement 具有最终控制权，
但 uv 是增量安装，而不是无条件覆盖：

- 如果目标仓库声明的精确版本或版本范围与当前安装版本不兼容，uv 会安装满足条件的版本，
  必要时替换镜像或中央 `deps` 先前安装的版本。
- 如果当前版本已经满足目标 requirement，uv 可以保留它；“安装阶段更晚”本身不会强制重装。
- 如果目标项目自身的 distribution name 与已安装 wheel 相同，目标项目的 editable 安装会成为
  活动的项目 distribution。`FlagGems` 仓库因此禁用了全局 `flag-gems` 配置。
- `[tool.uv].exclude-dependencies` 中的名称不会参与目标依赖解析。镜像或中央 `deps` 提供的
  兼容版本会保持活动状态，目标 pin 不会替换它。

因此每个 requirement 应只有一个明确 owner：仓库自身依赖以 `pyproject.toml` 为准，中央
`deps` 负责共享环境准备，exclusion 只用于有意由镜像/共享运行时负责的软件包。`priority`
只控制中央安装命令的顺序，不是通用依赖冲突解析器；目标仓库安装始终发生在所有中央
priority 之后。

### 持久化软件包缓存

目标依赖包含体积很大的 wheel（例如 `vllm`、`flashinfer-cubin`、`cupy`，
每个都有数百 MB）。为避免每次运行都重新下载，工作流会将 `uv` 和 `pip` 指向每个
自托管 runner 持久存储上的缓存目录。`prepare` 作业按厂商推导出缓存位置，
并将其注入矩阵的测试环境：

```text
UV_CACHE_DIR  = <output_base_dir>/uv-cache/<vendor>
PIP_CACHE_DIR = <output_base_dir>/pip-cache/<vendor>
UV_LINK_MODE  = copy
```

- `<output_base_dir>` 取自后端自身在 `backends/<NAME>.yaml` 中的持久输出路径：
  每个后端默认为 `/data/github-outputs`；THead（`PPU-ZW810E`）没有 `/data`
  挂载，因此为 `/home/github-outputs`。
- `<vendor>` 是小写的 `FLAGOS_GPU_VENDOR`（例如 `nvidia`、`metax`、`ascend`）。
  缓存仅按厂商区分，不按仓库区分：同一厂商上的不同目标仓库安装的厂商软件包几乎相同，
  因此单一的按厂商缓存可让它们共享同一套 wheel，最大化复用。
- 需要设置 `UV_LINK_MODE=copy`，因为缓存目录与环境位于不同的文件系统；它同时也会
  抑制 hardlink 回退的告警。

安装 optest 的步骤还会检查 runner 中的 `uv` 是否支持 `uv pip show --target`
（已验证的版本基线为 `0.11.30`）。镜像自带版本过旧时，先执行继承
`UV_CACHE_DIR` 的 `uv self update 0.11.30`；若 self-update 不可用，则通过现有的
pip 镜像 fallback 和 `PIP_CACHE_DIR` 安装 `uv>=0.11.30`，并为后续步骤启用该
Python 环境的 scripts 目录。

跨运行的行为如下：

- **首次运行（冷缓存）：** `uv` 从配置的软件包索引下载每个 wheel 一次，并存入
  `UV_CACHE_DIR`。
- **后续运行（热缓存）：** `uv` 直接复用已缓存的 wheel，不进行任何网络下载。
  `uv` 以软件包版本以及 wheel 的 URL 或哈希作为缓存键，因此版本不变时始终命中缓存；
  版本升级或直接 URL 依赖发生变化时，仅自动重新下载发生变化的 wheel，无需手动清理缓存。

optest、通用 package 升级和目标依赖安装步骤都使用正常的 `uv pip install` 输出。三个步骤都会在
安装前后打印 `[Cache debug]` 报告
（解析出的 `UV_CACHE_DIR`/`PIP_CACHE_DIR`/`UV_LINK_MODE`，以及缓存是 `COLD`
还是 `WARM`，并附上其大小与文件数量），因此步骤日志能够清晰显示哪些 wheel 来自本地
缓存复用、哪些是从网络下载的。

### FlagGems 各厂商算子列表

当目标仓库为 `FlagGems` 时，算子测试步骤会在调度之前于各 runner 上探测厂商专用的
算子列表。探测逻辑与厂商无关，因此每个厂商都会在自己的 runner 上自动发现自己的文件。
对于其他仓库，该步骤保持默认的 stage 选择，不做任何探测。

探测的路径使用小写的厂商名称：

```text
/data/flaggems/<vendor>/ops_include.txt   # include 列表 -> --op-list-file
/data/flaggems/<vendor>/ops_exclude.txt   # exclude 列表 -> --exclude-op-list-file
```

其中 `<vendor>` 即 `FLAGOS_GPU_VENDOR`，为 `flagos-gpu-check` 上报并在任务前序步骤
导出的厂商名。它本身即为小写，且与 runner 的目录结构一致，例如 `ascend`、`cambricon`
或 `enflame`。探测行为如下：

- 若存在 `ops_include.txt`，步骤追加 `--op-list-file <path>`，运行时精确测试该列表，
  而不再使用 `--stages all`。
- 若存在 `ops_exclude.txt`，步骤追加 `--exclude-op-list-file <path>`。
- 若两者都存在，则两个参数都追加。
- 文件缺失时直接跳过；列表并非必需，步骤不会因缺少列表而失败。

显式的 `exclude_ops` 工作流输入仍优先于探测到的 exclude 文件，与调度器优先级
`--exclude-ops > --exclude-op-list-file` 一致。同理，显式的 `ops` 输入（非 `all`）
优先于 include 探测。

步骤会打印探测到的候选路径、各文件是否命中，以及最终的算子参数和排除参数，因此运行日志
会清晰显示到底加载并生效了哪些列表。各厂商在自己的 runner 上维护各自的列表；工作流中不再
含有任何硬编码的厂商路径。

## 仓库布局

```text
.github/configs/repos.yaml
                          仓库级工作流配置
.github/configs/backends/ 共享硬件和厂商默认配置
.github/workflows/        集中式 GitHub Actions 工作流
pyproject.toml            软件包元数据和控制台脚本定义
src/flagos_optest/        Python 软件包、调度器、报告和 GPU 检查
```

### 仓库配置

`.github/configs/repos.yaml` 定义共享工作流可以测试哪些算子仓库。顶层 `org`
与每个 `repos` 键组合成 GitHub checkout 目标。每个仓库条目可以定义：

| 字段 | 含义 |
| --- | --- |
| `default_branch` | 工作流没有收到显式 ref 时使用的目标分支。 |
| `backends.<name>.enabled` | 为该仓库启用或禁用共享硬件后端。 |
| `container_images.<backend>` | 某个后端的仓库专用镜像覆盖项。 |
| `deps.default.<package>` | 为所有已启用 backend 提供仓库级覆盖或新增项；字段覆盖全局和 backend 值。 |
| `deps.<backend>.<package>` | 单个 backend 的最高优先级覆盖；显式 `enabled: true` 可重新启用，`false` 可禁用继承项。 |
| `base_data` | 目标 checkout 中可选的性能基线 JSON 路径。 |
| `op_cmd_timeout` | 每个精度或性能子进程的超时时间（秒）。继承 `defaults.op_cmd_timeout`。 |
| `ops_test_timeout_minutes` | 每个仓库的算子测试步骤超时时间。 |

不同层级的镜像键有意采用不同形式：

```yaml
# .github/configs/backends/H800.yaml: shared default
container_image: harbor.example/default-h800:tag

# .github/configs/repos.yaml: repository-specific override
repos:
  ExampleOperatorLibrary:
    container_images:
      H800: harbor.example/example-h800:tag
```

`repos.<repo>.container_images.<backend>` 的优先级高于后端的 `container_image`。
如果不存在仓库覆盖项，则使用共享后端镜像。package map 按完全相同的 package name 和
字段由低到高合并：

```text
defaults.deps -> backend deps -> repo deps.default -> repo deps.<backend>
```

每个字段最后一次显式设置的值生效。尤其是 `enabled`：最后一次显式值生效，省略则继承
前一层状态。package 的 `priority` 只控制最终安装顺序，不会改变上述配置覆盖优先级。

普通的目标专用厂商依赖不应使用这些 map。工作流会根据 `flagos-gpu-check` 报告的厂商键，
选择目标仓库的可选依赖组。

有些镜像在安装原生算子库之前需要先做一些修补：C++ 工具链、厂商 torch 链接的系统库、
动态加载器不认识的运行时目录，或对已安装厂商栈的补丁。这类工作在前面单独一步执行，
时机在目标仓库安装之前，而且不需要任何配置——它是对后端 vendor 名称的约定：

```text
.github/scripts/install/<vendor>.py    （或 <vendor>.sh）
<vendor> = 后端 vendor 名的小写：mthreads、hygon、ascend ...
```

文件存在就执行，不存在就跳过，因此原生算子库不需要对镜像做修补的厂商不需要任何额外声明。
脚本用 `bash` 执行（`*.py` 时用 `python3`），并继承 job 环境：`REPOSITORY_NAME`、
`flagos-gpu-check` 导出的 `FLAGOS_GPU_*`，以及后端的 `test_env`。
脚本必须幂等。

厂商特例的环境 patch 也放在这里，而不是在工作流里写成硬编码的 `if vendor == ...` 分支。

### 后端配置

每个 `.github/configs/backends/<hardware>.yaml` 文件都提供供所有仓库共享的默认硬件配置：

| 字段 | 含义 |
| --- | --- |
| `name` | `repos.yaml` 引用的硬件/后端标识符。 |
| `vendor` | 工作流接受的厂商筛选名称。 |
| `runner_labels` | 自托管 GitHub Actions runner 标签。 |
| `container_image` | 默认镜像，除非所选仓库覆盖它。 |
| `container_volumes` | 挂载到作业容器中的主机路径。 |
| `container_options` | 设备映射、权限、IPC、共享内存和其他容器运行时选项。 |
| `deps.<package>` | 合并到通用升级映射中的硬件专用 package 默认值。 |
| `test_gpus` | 可选的设备 ID 覆盖项；省略时由 `flagos-gpu-check` 提供可用设备。 |
| `test_env` | 注入算子测试步骤的环境变量。 |
| `output_base_dir` | 结果目录和归档文件的基础目录。 |

当前后端默认配置覆盖 H800/NVIDIA、910B/昇腾、C550/沐曦、BW1000/海光、
MTT-S5000/摩尔线程、BI-V150/天数智芯、PPU-ZW810E/平头哥、
MLU590-M9DE/寒武纪、P800/昆仑芯、W7900D/AMD、TX8110/清微智能和 S60/燧原。
下表汇总其当前调度默认值；每个后端 YAML 文件中的镜像、卷、容器选项和环境细节仍为权威配置。

| 后端 | 厂商 | Runner 标签 | FlagTree 包 | 默认测试设备 | 输出基础目录 |
| --- | --- | --- | --- | --- | --- |
| `H800` | Nvidia | `h800-flaggems-weekly` | `flagtree-nvidia` | 自动检测 | `/data/github-outputs` |
| `910B` | Ascend | `910b-flaggems-weekly` | `flagtree-ascend-cann9.0.0` | 自动检测 | `/data/github-outputs` |
| `C550` | Metax | `c550-flaggems-weekly` | `flagtree-metax` | 自动检测 | `/data/github-outputs` |
| `BW1000` | Hygon | `bw1000-flaggems-weekly` | `flagtree-hcu` | 自动检测 | `/data/github-outputs` |
| `MTT-S5000` | MThreads | `s5000-flaggems-weekly` | `flagtree-mthreads` | 自动检测 | `/data/github-outputs` |
| `BI-V150` | Iluvatar | `v150-flaggems-weekly` | `flagtree-iluvatar` | 自动检测 | `/data/github-outputs` |
| `PPU-ZW810E` | THead | `810e-flaggems-weekly` | `flagtree-ppu` | 自动检测 | `/home/github-outputs` |
| `MLU590-M9DE` | Cambricon | `mlu590-m9de-flaggems-weekly` | 镜像自带 | 自动检测 | `/data/github-outputs` |
| `P800` | KunLunXin | `p800-flaggems-weekly` | `flagtree-kunlunxin` | 自动检测 | `/data/github-outputs` |
| `W7900D` | AMD | `w7900d-flaggems-weekly` | `flagtree-amd` | 自动检测 | `/data/github-outputs` |
| `TX8110` | TsingMicro | `tx8110-flaggems-weekly` | `flagtree-tsingmicro` | `0-15`（覆盖） | `/data/github-outputs` |
| `S60` | Enflame | `s60-flaggems-weekly` | `flagtree-enflame` | 自动检测 | `/data/github-outputs` |

当前所有后端都使用容器，工作流始终会调用 `flagos-gpu-check`。仓库专用策略应保留在
`repos.yaml` 中；只有在共享硬件默认配置本身发生变化时，才应修改后端文件。
