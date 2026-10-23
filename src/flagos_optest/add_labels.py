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

# -*- coding: utf-8 -*-
"""
Standalone script to add operator labels from operators.yaml into summary.json.

Usage:
    flagos-add-labels [project_directory] <result_dir>

Where <result_dir> is the output directory produced by run_tests.py containing
summary.json. The script reads conf/operators.yaml relative to the project root,
builds a mapping of op_id -> labels, and injects the "labels" field into each
operator entry in summary.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

# Project root holding conf/operators.yaml. main() always sets this from the
# optional project-directory argument before labels are loaded.
ROOT = Path.cwd()


def pinfo(msg):
    print(f"\033[32m[INFO]\033[0m {msg}")


def perror(msg):
    print(f"\033[31m[ERROR]\033[0m {msg}")


def project_directory(value):
    """Return an absolute project directory or raise an argparse error."""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"project directory does not exist: {value}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"project directory is not a directory: {value}")
    return path


def load_op_labels():
    """Load operator labels from conf/operators.yaml."""
    op_labels = {}
    op_inventory = ROOT / "conf" / "operators.yaml"
    try:
        with open(str(op_inventory), "r") as f:
            data = yaml.safe_load(f)
            catalog = data.get("ops", [])
    except Exception as e:
        perror(f"Failed to load operator inventory: {e}")
        return op_labels

    for op_entry in catalog:
        op_id = op_entry.get("id", "")
        labels = op_entry.get("labels", [])
        op_labels[op_id] = labels

    return op_labels


def main():
    parser = argparse.ArgumentParser(
        description="Add operator labels from operators.yaml into summary.json"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=project_directory("."),
        type=project_directory,
        metavar="DIRECTORY",
        help="Target project directory. Default: current directory.",
    )
    parser.add_argument(
        "result_dir",
        metavar="RESULT_DIR",
        help="Path to the result directory containing summary.json",
    )
    args = parser.parse_args()

    global ROOT
    ROOT = args.directory
    os.chdir(ROOT)

    result_dir = Path(args.result_dir)
    summary_path = result_dir / "summary.json"

    if not summary_path.exists():
        perror(f"summary.json not found in {result_dir}")
        sys.exit(1)

    # Load summary
    with summary_path.open("r") as f:
        summary = json.load(f)

    op_data = summary.get("result", {})
    if not op_data:
        perror("No 'result' field found in summary.json")
        sys.exit(1)

    # Load labels from operators.yaml
    op_labels = load_op_labels()
    if not op_labels:
        perror("No labels loaded from operators.yaml")
        sys.exit(1)

    # Inject labels into each operator entry
    updated = 0
    for op_id, op_entry in op_data.items():
        labels = op_labels.get(op_id, [])
        op_entry["labels"] = labels
        updated += 1

    # Write back
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    pinfo(f"Updated {updated} operators with labels in {summary_path}")


if __name__ == "__main__":
    main()
