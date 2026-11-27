# Container Images

This directory contains optest-managed application container images for
vLLM, Megatron, and sglang across all supported vendor stacks.

## Directory layout

```
container/
├── configs.yaml             Central config: apps → stacks → versions
├── vllm/
│   ├── Dockerfile           Single Dockerfile for all vllm stacks
│   └── vllm-serve           Launcher script (sources vendor env, execs vllm)
├── megatron/
│   └── Dockerfile           Single Dockerfile for all megatron stacks
├── sglang/
│   ├── Dockerfile           Single Dockerfile for all sglang stacks
│   └── sglang-serve         Launcher script (execs sglang.launch.server)
├── scripts/
│   ├── gen_matrix.py        Generate GitHub Actions matrix from configs.yaml
│   └── verify_image.py      Op gate: per-operator check against verify/<repo>/<stack>.json
└── verify/
    ├── verify-vllm-backend.sh     Serve test + dependency matrix check for vllm
    ├── verify-sglang-backend.sh   Serve test + dependency matrix check for sglang
    └── <repo>/
        └── <stack>.json           Per-operator passing ops list
```

`<repo>` is the target repository name (e.g. `flaggems`).
`<stack>` is the stack key (e.g. `ascend-cann9.0.0`, `nvidia-cuda13.0`).
Example: `container/verify/flaggems/ascend-cann9.0.0.json`.

## Workflow

`.github/workflows/app-image.yaml` builds, verifies, and optionally pushes
images. Trigger via workflow_dispatch with:

| Input | Description |
| --- | --- |
| `app` | `vllm`, `megatron`, `sglang`, or `all` |
| `stack` | Comma-separated stack keys (empty = all stacks for the app) |
| `push` | `true` to push to Harbor after the verify gate passes |

`image_version` is auto-computed as `YYYYMMDD-sha8` from the optest checkout;
it is not a manual input so every build is distinct and traceable.

## Stack key convention

Pattern: `{vendor}-{runtime}{version}`, matching build-infra naming.
Stacks that differ only in hardware generation (910b vs 910c, S60 vs L600)
are **separate entries** — their runtimes are incompatible.

Examples: `ascend-cann9.0.0`, `metax-maca3.8.1.3`, `nvidia-cuda13.0`

## Two-layer version model

- **Baked**: FlagTree / FlagGems / plugin-fl versions in `configs.yaml`
  are installed at docker build time. Update by rebuilding the image.
- **Hot-patch**: `ops-test.yaml`'s "Upgrade configured packages" step
  reinstalls a newer daily version at test time without a full rebuild.
  Use for minor bumps; bake a new image for breaking changes.

## Torch / runtime protection

The Dockerfiles generate `/tmp/protect.txt` from
`/etc/flagos/flaggems-constraints.txt` (written by the runtime base image)
plus a live `uv pip freeze` snapshot. Every `uv pip install` passes
`-c /tmp/protect.txt` so torch, triton, and other pinned runtime packages
are never replaced.

## BuildKit cache

Each `RUN --mount=type=cache` block is keyed by vendor package name and
points to `/root/.cache/uv` inside the container. Docker manages this
cache under the BuildKit storage on the runner host.

The `--cache-from/--cache-to type=local` layer cache is stored at
`<output_base_dir>/docker-build-cache/<vendor>` on the runner.

## Verify process

The verify step runs in two stages:

**Stage 1 — Serve test + dependency matrix check** (vllm and sglang only):
`container/verify/verify-<app>-backend.sh` starts the app container, waits
for the server readiness signal, sends a sample inference request, and
checks that the dependency matrix (torch, triton, flagtree, flag_gems,
numpy) is byte-identical between the runtime base and the app image. Megatron
has no serve launcher so this stage is skipped.

**Stage 2 — Op gate**:
`container/scripts/verify_image.py` runs a temporary container from the
just-built image, installs flagos-optest, and runs FlagGems tests. Results
are parsed **per individual operator** from `summary.json` — the overall
flagos-optest exit code is not used. All operators listed in
`container/verify/<repo>/<stack>.json` must have `result == "passed"` for
the image to be pushed.

If no `.json` file exists for a stack, the gate is skipped and the image is
pushed unconditionally (useful when onboarding a new stack before a baseline
has been confirmed).

## Updating passing_ops lists

After a successful run on known-good hardware:

```bash
python - <<'EOF'
import json, sys
with open("output/FlagGems/summary.json") as f:
    s = json.load(f)
passed = [op for op, d in s.get("ops", {}).items() if d.get("result") == "passed"]
print(json.dumps({"ops": sorted(passed)}, indent=2))
EOF
```

Write the output to `container/verify/<repo>/<stack>.json` and commit it.
Example: `container/verify/flaggems/ascend-cann9.0.0.json`.

## Published image location

`harbor.baai.ac.cn/flagos-optest/<app><app_version>-<stack>:<image_version>`

Example: `harbor.baai.ac.cn/flagos-optest/vllm0.32.0-ascend-cann9.0.0:20261127-c0f9bdc8`

## Updating backend container_image references

After publishing a new image, update the `container_image` field in the
relevant backend YAML:

```yaml
# .github/configs/backends/910B.yaml
container_image: harbor.baai.ac.cn/flagos-optest/vllm0.32.0-ascend-cann9.0.0:20261127-c0f9bdc8
```

For repos that override per-backend images in `repos.yaml`
(FlagGems-sglang, FlagTrain), update their `container_images:` block.
