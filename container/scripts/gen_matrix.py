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

"""Generate a GitHub Actions matrix for the app-image workflow.

Reads container/configs.yaml and .github/configs/backends/<BACKEND>.yaml
and writes a JSON matrix to stdout (one include entry per app/stack).

Usage:
    python container/scripts/gen_matrix.py \\
        --app vllm \\
        --stack ascend-cann9.0.0 \\
        --backends-dir .github/configs/backends

Pass --app all to include every app. Omit --stack to include every stack
for the selected app(s). Omit --image-version to auto-compute YYYYMMDD-sha8
from the current git checkout. The workflow uses the JSON to populate
matrix.include.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: uv pip install PyYAML")


def default_image_version() -> str:
    """Return YYYYMMDD-<sha8> from the current git checkout, or YYYYMMDD-unknown."""
    today = date.today().strftime("%Y%m%d")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        sha = "unknown"
    return f"{today}-{sha}"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def backend_runner_labels(backend_name: str, backends_dir: Path) -> list[str]:
    """Return runner_labels list from the backend YAML, or [] if not found."""
    backend_file = backends_dir / f"{backend_name}.yaml"
    if not backend_file.exists():
        return []
    data = load_yaml(backend_file)
    labels = data.get("runner_labels", "")
    if isinstance(labels, list):
        return labels
    return [str(labels)] if labels else []


def backend_output_base_dir(backend_name: str, backends_dir: Path) -> str:
    backend_file = backends_dir / f"{backend_name}.yaml"
    if not backend_file.exists():
        return "/data/github-outputs"
    data = load_yaml(backend_file)
    return data.get("output_base_dir", "/data/github-outputs")


def _app_build_args(app_name: str, app_version: str, stack_cfg: dict) -> dict:
    """Return per-app Dockerfile ARG name → value mapping for docker build."""
    if app_name == "vllm":
        return {
            "VLLM_VERSION": app_version,
            "PLUGIN_FL_VERSION": stack_cfg.get("plugin_fl_version", ""),
            "FLAGCX_VERSION": stack_cfg.get("flagcx_version", ""),
        }
    if app_name == "megatron":
        return {
            "MEGATRON_VERSION": app_version,
        }
    if app_name == "sglang":
        return {
            "SGLANG_VERSION": app_version,
            # plugin_fl_version in configs.yaml is the sglang-plugin-FL ref.
            "SGLANG_FL_VERSION": stack_cfg.get("plugin_fl_version", ""),
            "SGL_KERNEL_SHIM_VERSION": stack_cfg.get("sgl_kernel_shim_version", ""),
        }
    return {}


def build_matrix(
    images_yaml: Path,
    backends_dir: Path,
    apps_filter: list[str],
    stacks_filter: list[str],
    image_version: str,
) -> list[dict]:
    config = load_yaml(images_yaml)
    registry_host = config["registry"]["host"]
    registry_project = config["registry"]["project"]
    global_components = config.get("components", {})

    matrix = []
    apps = config.get("apps", {})

    for app_name, app_cfg in apps.items():
        if apps_filter and app_name not in apps_filter:
            continue

        app_version = str(app_cfg.get("version", ""))
        stacks = app_cfg.get("stacks", {})

        for stack_key, stack_cfg in stacks.items():
            if stacks_filter and stack_key not in stacks_filter:
                continue

            backend = stack_cfg.get("backend", "")
            vendor = stack_cfg.get("vendor", "")
            skip_build = stack_cfg.get("skip_build", False)

            # Published image tag: <app><app_version>-<stack>:<image_version>
            image_tag = (
                f"{registry_host}/{registry_project}/"
                f"{app_name}{app_version}-{stack_key}:{image_version}"
            )

            # Runner labels come from the backend YAML.
            runner_labels = backend_runner_labels(backend, backends_dir)
            output_base_dir = backend_output_base_dir(backend, backends_dir)

            # docker build cache dir lives under output_base_dir, separate
            # from ops-test's uv-cache/<vendor> so the two don't conflict.
            docker_cache_dir = f"{output_base_dir}/docker-build-cache/{vendor}"

            # Vendor PyPI index.
            vendor_pypi = (
                f"https://resource.flagos.net/repository/flagos-pypi-{vendor}/simple"
            )

            # FlagTree: stack override or global default.
            ft_global = global_components.get("flagtree", {})
            flagtree_version = stack_cfg.get(
                "flagtree_version", ft_global.get("version", "")
            )
            flagtree_package = stack_cfg.get("flagtree_package", "flagtree")

            # FlagGems: stack override or global default.
            fg_global = global_components.get("flaggems", {})
            flaggems_version = stack_cfg.get(
                "flaggems_version", fg_global.get("version", "")
            )

            entry: dict = {
                "app": app_name,
                "app_version": app_version,
                "stack": stack_key,
                "backend": backend,
                "vendor": vendor,
                "image_tag": image_tag,
                "runner_labels": runner_labels,
                "docker_cache_dir": docker_cache_dir,
                "skip_build": skip_build,
                "flagos_pypi": vendor_pypi,
                "flagtree_version": flagtree_version,
                "flagtree_package": flagtree_package,
                "flaggems_version": flaggems_version,
                # Flat fields used by per-app verify scripts.
                "plugin_fl_version": stack_cfg.get("plugin_fl_version", ""),
                "flagcx_version": stack_cfg.get("flagcx_version", ""),
                "sgl_kernel_shim_version": stack_cfg.get("sgl_kernel_shim_version", ""),
                "app_env": stack_cfg.get("app_env", ""),
                # Per-app build args: passed as --build-arg by the workflow.
                # vllm   → VLLM_VERSION / PLUGIN_FL_VERSION / FLAGCX_VERSION
                # megatron → MEGATRON_VERSION
                # sglang → SGLANG_VERSION / SGLANG_FL_VERSION / SGL_KERNEL_SHIM_VERSION
                "build_args": _app_build_args(
                    app_name, app_version, stack_cfg
                ),
            }

            if skip_build:
                entry["existing_image"] = stack_cfg.get("existing_image", "")
            else:
                entry["base_image"] = stack_cfg.get("base_image", "")

            matrix.append(entry)

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate app-image workflow matrix.")
    parser.add_argument(
        "--app",
        default="all",
        help="App to build (vllm|megatron|sglang|all). Default: all.",
    )
    parser.add_argument(
        "--stack",
        default="",
        help="Comma-separated stack keys to build. Empty = all stacks for the app.",
    )
    parser.add_argument(
        "--image-version",
        default=None,
        help=(
            "Image tag version suffix (YYYYMMDD-sha8). "
            "Defaults to auto-computed value from the current git checkout."
        ),
    )
    parser.add_argument(
        "--images-yaml",
        default="container/configs.yaml",
        help="Path to container/configs.yaml.",
    )
    parser.add_argument(
        "--backends-dir",
        default=".github/configs/backends",
        help="Path to backends directory.",
    )
    args = parser.parse_args()

    apps_filter = [] if args.app == "all" else [args.app]
    stacks_filter = [s.strip() for s in args.stack.split(",") if s.strip()]

    image_version = args.image_version or default_image_version()

    repo_root = Path(
        os.environ.get("GITHUB_WORKSPACE", Path(__file__).resolve().parents[2])
    )
    images_yaml = repo_root / args.images_yaml
    backends_dir = repo_root / args.backends_dir

    if not images_yaml.exists():
        sys.exit(f"configs.yaml not found: {images_yaml}")

    matrix = build_matrix(
        images_yaml=images_yaml,
        backends_dir=backends_dir,
        apps_filter=apps_filter,
        stacks_filter=stacks_filter,
        image_version=image_version,
    )

    if not matrix:
        sys.exit("No matching stacks found for the given app/stack filters.")

    print(json.dumps({"include": matrix}, ensure_ascii=False))


if __name__ == "__main__":
    main()
