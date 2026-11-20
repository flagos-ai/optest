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

"""External operator-detail parsing and compact summary helpers."""

import json
from pathlib import Path
from typing import Any

if __package__:
    from .performance_report import parse_perf_data
else:
    from performance_report import parse_perf_data  # type: ignore[no-redef]

DETAILS_MODE_EXTERNAL = "external"


def parse_accuracy_data(result_file):
    """Parse a raw accuracy result into status counts and grouped details."""
    raw_data = {}
    try:
        with result_file.open("r", encoding="utf-8") as input_file:
            raw_data = json.load(input_file)
    except (json.JSONDecodeError, ValueError):
        return {
            "total": 0,
            "skipped": 0,
            "failed": 0,
            "passed": 0,
            "errors": 1,
            "status": "Error",
            "details": {"error": f"Invalid JSON in {result_file}"},
        }

    passed = []
    skipped: dict[str, Any] = {}
    failed: dict[str, Any] = {}
    num_skipped = 0
    num_failed = 0
    num_passed = 0
    skipped_with_issue = False
    for test_case, item in raw_data.items():
        case_str = test_case[: test_case.find("[")]
        result = item.get("result", "")
        params = [case_str]
        for value in item.get("params", {}).values():
            params.append(str(value).replace(" ", ""))
        param_str = ":".join(params)

        if result == "passed":
            passed.append(param_str)
            num_passed += 1
        elif result == "skipped":
            reason = item.get("reason", "Unknown")
            if "Issue" in reason:
                skipped_with_issue = True
            skipped.setdefault(reason, set())
            skipped[reason].add(param_str)
            num_skipped += 1
        else:
            reason = item.get("reason", "Unknown")
            failed.setdefault(reason, set())
            failed[reason].add(param_str)
            num_failed += 1

    num_total = num_passed + num_skipped + num_failed
    parsed: dict[str, Any] = {
        "total": num_total,
        "skipped": num_skipped,
        "failed": num_failed,
        "passed": num_passed,
        "errors": 0,
        "details": {},
    }
    if not skipped and not failed:
        parsed["status"] = "NotFound" if not passed else "Passed"
        return parsed

    if num_failed > 0:
        parsed["status"] = "Failed"
        parsed["details"]["failed"] = {
            reason: list(cases) for reason, cases in failed.items()
        }
        return parsed

    if num_passed > 0:
        parsed["status"] = "Passed"
    elif skipped_with_issue:
        parsed["status"] = "Failed"
    else:
        parsed["status"] = "Skipped"

    parsed["details"]["skipped"] = {
        reason: list(cases) for reason, cases in skipped.items()
    }
    return parsed


def _first_reason(details, kind):
    reasons = details.get(kind, {}) if isinstance(details, dict) else {}
    if not isinstance(reasons, dict):
        return ""
    for reason in reasons:
        return str(reason).split("\n", maxsplit=1)[0]
    return ""


def compact_accuracy_result(result):
    """Remove inline cases while retaining the note needed by summary views."""
    details = result.pop("details", {})
    note = ""
    if result.get("failed", 0) > 0:
        note = _first_reason(details, "failed")
    elif result.get("skipped", 0) > 0 and result.get("passed", 0) == 0:
        note = _first_reason(details, "skipped")
    elif isinstance(details, dict) and details.get("error"):
        note = str(details["error"]).split("\n", maxsplit=1)[0]
    if note:
        result["note"] = note
    result["details_mode"] = DETAILS_MODE_EXTERNAL
    return result


def _mean_metric(details, metric):
    values: list[float] = []
    for shape_result in details.values():
        if not isinstance(shape_result, dict):
            continue
        value = shape_result.get(metric)
        if isinstance(value, (int, float)):
            values.append(value)
    return sum(values) / len(values) if values else None


def compact_performance_result(result, *, speedup_source):
    """Remove shape details while retaining dtype-level report aggregates."""
    data = result.get("data", {})
    if isinstance(data, dict):
        for dtype_result in data.values():
            if not isinstance(dtype_result, dict):
                continue
            details = dtype_result.pop("details", {})
            if not isinstance(details, dict):
                details = {}
            dtype_result["base"] = _mean_metric(details, "base")
            dtype_result["gems"] = _mean_metric(details, "gems")
            dtype_result["shape_count"] = len(details)
    result["speedup_source"] = speedup_source
    result["details_mode"] = DETAILS_MODE_EXTERNAL
    return result


def _external_data_path(summary_dir, record):
    data_file = record.get("data_file")
    if not isinstance(data_file, str) or not data_file:
        raise ValueError("data_file is missing")
    relative_path = Path(data_file)
    if relative_path.is_absolute():
        raise ValueError("data_file must be relative")
    root = Path(summary_dir).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("data_file escapes the result directory") from error
    if not candidate.is_file():
        raise ValueError(f"data_file does not exist: {data_file}")
    return candidate


def hydrate_operator_details(summary_dir, op, operator_result):
    """Load external details into one operator record for detailed reports.

    The record is updated in place. A list of warnings is returned so report
    commands can remain usable when an archive is incomplete.
    """
    warnings = []
    accuracy = operator_result.get("accuracy", {})
    if (
        isinstance(accuracy, dict)
        and accuracy.get("details_mode") == DETAILS_MODE_EXTERNAL
    ):
        try:
            accuracy_path = _external_data_path(summary_dir, accuracy)
            parsed_accuracy = parse_accuracy_data(accuracy_path)
            accuracy["details"] = parsed_accuracy.get("details", {})
            if parsed_accuracy.get("status") == "Error":
                warnings.append(f"{op} accuracy details are invalid")
        except (OSError, ValueError) as error:
            warnings.append(f"{op} accuracy details unavailable: {error}")
            accuracy["details"] = {}

    performance = operator_result.get("performance", {})
    if (
        isinstance(performance, dict)
        and performance.get("details_mode") == DETAILS_MODE_EXTERNAL
    ):
        try:
            performance_path = _external_data_path(summary_dir, performance)
            parsed_performance = parse_perf_data(
                op,
                performance_path,
                speedup_source=performance.get("speedup_source", "origin"),
            )
            parsed_data = parsed_performance.get("data", {})
            performance["data"] = parsed_data if isinstance(parsed_data, dict) else {}
            if parsed_performance.get("status") == "Error":
                warnings.append(f"{op} performance details are invalid")
        except (OSError, ValueError) as error:
            warnings.append(f"{op} performance details unavailable: {error}")
            performance["data"] = {}

    return warnings
