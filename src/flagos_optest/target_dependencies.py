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

"""Install a target repository's vendor dependencies from ``pyproject.toml``."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


def _load_vendor_requirements(target: Path, vendor: str):
    pyproject_path = target / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    requirements = (
        pyproject.get("project", {}).get("optional-dependencies", {}).get(vendor)
    )
    if requirements is None:
        return None
    if not isinstance(requirements, list) or not all(
        isinstance(requirement, str) for requirement in requirements
    ):
        raise ValueError(
            f"{pyproject_path}: project.optional-dependencies.{vendor} "
            "must be a list of requirement strings"
        )
    return requirements


def _run_pip(arguments):
    command = [sys.executable, "-m", "pip", "install", *arguments]
    print("[Install dependencies] Starting: resolved vendor requirement installation", flush=True)
    print(f"[Install dependencies] Command: {shlex.join(command)}", flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        print(
            "[Install dependencies] Finished: resolved vendor requirement "
            f"installation (exit_code={error.returncode})",
            flush=True,
        )
        raise
    print(
        "[Install dependencies] Finished: resolved vendor requirement "
        "installation (exit_code=0)",
        flush=True,
    )


def install_vendor_dependencies(target: Path, vendor: str, *, no_deps=False):
    pyproject_path = target / "pyproject.toml"
    print(
        f"[Install dependencies] Vendor extra source: {pyproject_path} "
        f"[project.optional-dependencies].{vendor}",
        flush=True,
    )
    requirements = _load_vendor_requirements(target, vendor)
    if requirements is None:
        print(
            f"[Install dependencies] Target does not define {vendor!r} "
            "optional-dependency group; skipping vendor dependencies.",
            flush=True,
        )
        return False
    if not requirements:
        print(
            f"[Install dependencies] Target defines an empty {vendor!r} "
            "optional-dependency group; skipping vendor dependencies.",
            flush=True,
        )
        return False

    print(
        f"[Install dependencies] Detected vendor extra requirements "
        f"({len(requirements)}): {shlex.join(requirements)}",
        flush=True,
    )
    pip_arguments = ["--no-deps"] if no_deps else []
    print(
        "[Install dependencies] Effective vendor pip arguments: "
        + (shlex.join(pip_arguments) if pip_arguments else "<empty>"),
        flush=True,
    )

    if no_deps:
        # Installing target[vendor] with --no-deps would skip the extra itself.
        # Install the extra's requirements directly so only their transitive
        # dependency resolution is disabled.
        _run_pip([*pip_arguments, *requirements])
    else:
        _run_pip([f"{target}[{vendor}]"])
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Install target vendor dependencies declared in pyproject.toml."
    )
    parser.add_argument("target", type=Path, help="target repository directory")
    parser.add_argument("vendor", help="normalized flagos-gpu-check vendor key")
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="install the vendor requirements without transitive dependencies",
    )
    args = parser.parse_args()
    try:
        install_vendor_dependencies(
            args.target.resolve(), args.vendor, no_deps=args.no_deps
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        parser.exit(1, f"flagos-install-target-deps: error: {error}\n")


if __name__ == "__main__":
    main()
