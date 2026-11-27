#!/usr/bin/env python3
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

"""Verify that a built app image passes the gate for a given stack.

The verify gate:
  1. Runs a temporary container from the just-built image.
  2. Inside the container: installs flagos-optest from the optest checkout,
     checks out FlagAttention and FlagGems, runs their op tests, and collects
     summary.json from each.
  3. Reads the per-stack passing-ops list from
     container/verify/<repo>/<stack>.json (e.g. verify/flaggems/ascend-cann9.0.0.json).
  4. Parses summary.json per individual operator. All operators that appear in
     the passing-ops list must have result == "passed" in the new run.
     The overall flagos-optest exit code is NOT used as the gate — individual
     operator results are checked explicitly.

Proxy:
  http_proxy / https_proxy / no_proxy are inherited from the runner environment
  and forwarded to the container via --env without echoing their values in logs.

Exit code:
  0  all required ops passed
  1  one or more required ops failed or were missing from summary.json
  2  usage / setup error

Usage:
    python container/scripts/verify_image.py \\
        --image harbor.baai.ac.cn/flagos-optest/vllm0.32.0-ascend-cann9.0.0:1.0.0 \\
        --app vllm \\
        --stack ascend-cann9.0.0 \\
        --vendor ascend \\
        --optest-dir /path/to/optest \\
        --work-dir /tmp/verify-$$
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Docker run flags per vendor (mirrors build-infra build-config.yml run section).
# Only the minimal flags needed to bring the accelerator up inside the temp
# container. These are not the full ops-test flags — just enough for imports.
# ---------------------------------------------------------------------------
_VENDOR_RUN_FLAGS: dict[str, str] = {
    "nvidia": "--gpus all",
    "ascend": "--privileged --shm-size=1g -e ASCEND_VISIBLE_DEVICES=0,1",
    "hygon": "-e DCU_VISIBLE_DEVICES=all",
    "metax": "",   # metax-docker wrapper used on runner; fallback to plain docker here
    "mthreads": "--runtime mthreads --env MTHREADS_VISIBLE_DEVICES=all",
    "iluvatar": "--runtime iluvatar --env IX_VISIBLE_DEVICES=all",
    "kunlunxin": "--runtime xpu -e CXPU_VISIBLE_DEVICES=0",
    "enflame": "--network host -e ENFLAME_VISIBLE_DEVICES=all -e TENCENT_VISIBLE_DEVICES=all",
    "cambricon": "",
    "amd": "--device /dev/kfd --device /dev/dri",
    "thead": "--privileged --shm-size=512g --ipc host --network host",
}


def run(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, **kwargs)


def load_passing_ops(verify_dir: Path, repo: str, stack: str) -> list[str]:
    """Load the pre-confirmed passing ops list for this repo/stack.

    Path: container/verify/<repo>/<stack>.json
    e.g.  container/verify/flaggems/ascend-cann9.0.0.json
    """
    ops_file = verify_dir / repo / f"{stack}.json"
    if not ops_file.exists():
        print(
            f"[verify] WARNING: no passing ops file at {ops_file}; "
            "skipping operator gate (image will be pushed).",
            file=sys.stderr,
        )
        return []
    with open(ops_file, encoding="utf-8") as f:
        data = json.load(f)
    ops = data if isinstance(data, list) else data.get("ops", [])
    print(f"[verify] Loaded {len(ops)} required ops from {ops_file}", file=sys.stderr)
    return ops


def parse_summary(summary_path: Path) -> dict[str, str]:
    """Return op_name -> result mapping from a flagos-optest summary.json."""
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    results: dict[str, str] = {}
    # summary.json structure: {"ops": {"op_name": {"result": "passed"|"failed"|..}}}
    ops = summary.get("ops", {})
    for op_name, op_data in ops.items():
        if isinstance(op_data, dict):
            results[op_name] = op_data.get("result", "unknown")
    return results


def check_ops(required_ops: list[str], results: dict[str, str], repo: str) -> list[str]:
    """Return list of required ops that did not pass."""
    failures = []
    for op in required_ops:
        status = results.get(op, "MISSING")
        if status != "passed":
            print(
                f"[verify] FAIL {repo} op={op!r} result={status!r}",
                file=sys.stderr,
            )
            failures.append(op)
        else:
            print(f"[verify] OK   {repo} op={op!r}", file=sys.stderr)
    return failures


def proxy_env_flags() -> list[str]:
    """Return --env flags for proxy vars without exposing their values in logs."""
    flags = []
    for var in ("http_proxy", "https_proxy", "no_proxy",
                 "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        if os.environ.get(var):
            # Pass only the name; docker inherits the value from the runner env.
            flags += ["--env", var]
    return flags


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify built app image gate.")
    parser.add_argument("--image", required=True, help="Fully-qualified image to test.")
    parser.add_argument("--app", required=True, help="App name (vllm|megatron|sglang).")
    parser.add_argument("--stack", required=True, help="Stack key, e.g. ascend-cann9.0.0.")
    parser.add_argument("--vendor", required=True, help="Vendor key, e.g. ascend.")
    parser.add_argument(
        "--optest-dir",
        required=True,
        help="Path to the optest repository checkout on the runner.",
    )
    parser.add_argument(
        "--work-dir",
        default="",
        help="Host directory mounted into the container for output. "
             "Created if absent. Defaults to a tmpdir.",
    )
    args = parser.parse_args()

    optest_dir = Path(args.optest_dir).resolve()
    verify_dir = optest_dir / "container" / "verify"

    work_dir = Path(args.work_dir) if args.work_dir else Path(
        tempfile.mkdtemp(prefix="flagos-verify-")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = work_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Load required ops lists per repo (path: verify/<repo>/<stack>.json).
    fa_ops = load_passing_ops(verify_dir, "flagattention", args.stack)
    fg_ops = load_passing_ops(verify_dir, "flaggems", args.stack)

    if not fa_ops and not fg_ops:
        print("[verify] No passing_ops.json found for either repo; push allowed.", file=sys.stderr)
        sys.exit(0)

    vendor_flags = _VENDOR_RUN_FLAGS.get(args.vendor, "").split()
    proxy_flags = proxy_env_flags()

    # Build the in-container script: install optest, run tests, dump summary.json.
    verify_script = r"""#!/bin/bash
set -e
export PATH="/flagos/bin:${PATH}"

# Source full vendor env (LD_LIBRARY_PATH, etc.) — same as vllm-serve.
if [ -d /etc/profile.d ]; then
    for f in /etc/profile.d/*.sh; do [ -r "$f" ] && . "$f"; done
fi

OPTEST_DIR=/optest
OUTPUT=/output

echo "[verify] Installing flagos-optest from checkout..."
uv pip install --python /flagos/bin/python --no-deps -e "${OPTEST_DIR}"

echo "[verify] Running FlagAttention tests..."
flagos-optest \
    --repo FlagAttention \
    --output-dir "${OUTPUT}/FlagAttention" \
    --stages accuracy \
    --ops all \
    || true   # individual results checked in host Python, not exit code

echo "[verify] Running FlagGems tests (~100 ops)..."
flagos-optest \
    --repo FlagGems \
    --output-dir "${OUTPUT}/FlagGems" \
    --stages accuracy \
    --ops all \
    --max-ops 100 \
    || true
"""

    script_path = work_dir / "verify.sh"
    script_path.write_text(verify_script, encoding="utf-8")

    docker_cmd = [
        "docker", "run", "--rm",
        # Mount optest checkout read-only.
        "-v", f"{optest_dir}:/optest:ro",
        # Mount output dir for summary.json collection.
        "-v", f"{output_dir}:/output",
        # Mount the verify script.
        "-v", f"{script_path}:/verify.sh:ro",
    ]
    docker_cmd += vendor_flags
    docker_cmd += proxy_flags
    docker_cmd += [args.image, "bash", "/verify.sh"]

    print(f"[verify] Starting temp container for {args.image}", file=sys.stderr)
    result = run(docker_cmd, check=False)
    if result.returncode != 0:
        print(
            f"[verify] Container exited with code {result.returncode}; "
            "treating as gate failure.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse results per repo.
    all_failures: list[str] = []

    fa_summary = output_dir / "FlagAttention" / "summary.json"
    if fa_ops and fa_summary.exists():
        fa_results = parse_summary(fa_summary)
        all_failures += check_ops(fa_ops, fa_results, "FlagAttention")
    elif fa_ops:
        print("[verify] FlagAttention summary.json missing; treating all ops as FAIL.", file=sys.stderr)
        all_failures += [f"FlagAttention/{op}" for op in fa_ops]

    fg_summary = output_dir / "FlagGems" / "summary.json"
    if fg_ops and fg_summary.exists():
        fg_results = parse_summary(fg_summary)
        all_failures += check_ops(fg_ops, fg_results, "FlagGems")
    elif fg_ops:
        print("[verify] FlagGems summary.json missing; treating all ops as FAIL.", file=sys.stderr)
        all_failures += [f"FlagGems/{op}" for op in fg_ops]

    if all_failures:
        print(
            f"[verify] GATE FAILED: {len(all_failures)} required op(s) did not pass:",
            file=sys.stderr,
        )
        for f in all_failures:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    print("[verify] GATE PASSED: all required ops passed.", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
