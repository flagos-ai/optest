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

import json
from unittest import mock

import pytest

from flagos_optest import result_details


@pytest.mark.parametrize(
    ("result", "expected_note"),
    [
        (
            {
                "failed": 1,
                "details": {"failed": {"failure\ntrace": ["case"]}},
            },
            "failure",
        ),
        (
            {
                "failed": 0,
                "skipped": 1,
                "passed": 0,
                "details": {"skipped": {"skip\ntrace": ["case"]}},
            },
            "skip",
        ),
        (
            {"details": {"error": "invalid\ntrace"}},
            "invalid",
        ),
        (
            {"failed": 1, "details": {"failed": []}},
            None,
        ),
        (
            {"failed": 1, "details": "invalid"},
            None,
        ),
    ],
)
def test_compact_accuracy_result_removes_details_and_keeps_note(result, expected_note):
    compact = result_details.compact_accuracy_result(result)

    assert "details" not in compact
    assert compact["details_mode"] == "external"
    assert compact.get("note") == expected_note


def test_compact_accuracy_result_handles_empty_reason_map():
    assert result_details._first_reason({"failed": {}}, "failed") == ""


def test_compact_performance_result_keeps_aggregates_without_details():
    result = {
        "data": {
            "fp32": {
                "result": "OK",
                "speedup": 2.0,
                "details": {
                    "[1]": {"base": 4.0, "gems": 2.0},
                    "[2]": {"base": 2.0, "gems": "unknown"},
                    "invalid": "not-an-object",
                },
            },
            "empty": {"details": []},
            "invalid": "not-an-object",
        }
    }

    compact = result_details.compact_performance_result(
        result, speedup_source="nv_baseline"
    )

    assert compact["details_mode"] == "external"
    assert compact["speedup_source"] == "nv_baseline"
    assert compact["data"]["fp32"] == {
        "result": "OK",
        "speedup": 2.0,
        "base": 3.0,
        "gems": 2.0,
        "shape_count": 3,
    }
    assert compact["data"]["empty"] == {
        "base": None,
        "gems": None,
        "shape_count": 0,
    }

    no_data = result_details.compact_performance_result(
        {"data": []}, speedup_source="origin"
    )
    assert no_data["details_mode"] == "external"


def test_compact_results_do_not_scale_with_case_or_shape_details():
    accuracy = result_details.compact_accuracy_result(
        {
            "failed": 1000,
            "details": {
                "failed": {"mismatch": [f"case-{index}" for index in range(1000)]}
            },
        }
    )
    performance = result_details.compact_performance_result(
        {
            "data": {
                "fp32": {
                    "speedup": 2.0,
                    "details": {
                        f"[{index}]": {
                            "base": 2.0,
                            "gems": 1.0,
                            "speedup": 2.0,
                        }
                        for index in range(1000)
                    },
                }
            }
        },
        speedup_source="origin",
    )

    encoded = json.dumps({"accuracy": accuracy, "performance": performance})
    assert '"details"' not in encoded
    assert len(encoded) < 1000
    assert performance["data"]["fp32"]["shape_count"] == 1000


def test_external_data_path_rejects_unsafe_or_missing_paths(tmp_path):
    existing = tmp_path / "add" / "accuracy_result.json"
    existing.parent.mkdir()
    existing.write_text("{}", encoding="utf-8")
    assert (
        result_details._external_data_path(
            tmp_path, {"data_file": "add/accuracy_result.json"}
        )
        == existing
    )

    for data_file, message in (
        (None, "missing"),
        (str(existing), "relative"),
        ("../outside.json", "escapes"),
        ("add/missing.json", "does not exist"),
    ):
        with pytest.raises(ValueError, match=message):
            result_details._external_data_path(tmp_path, {"data_file": data_file})


def test_hydrate_operator_details_loads_external_accuracy_and_performance(tmp_path):
    operator_dir = tmp_path / "add"
    operator_dir.mkdir()
    (operator_dir / "accuracy_result.json").write_text(
        json.dumps(
            {
                "test_add[x]": {
                    "result": "failed",
                    "reason": "mismatch",
                    "params": {"shape": [1]},
                }
            }
        ),
        encoding="utf-8",
    )
    (operator_dir / "performance_result.json").write_text(
        json.dumps(
            {
                "add": {
                    "result": "passed",
                    "details": [
                        {
                            "dtype": "torch.float32",
                            "result": [
                                {
                                    "shape_detail": [1],
                                    "latency_base": 2.0,
                                    "latency": 1.0,
                                    "speedup": 2.0,
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    operator_result = {
        "accuracy": {
            "details_mode": "external",
            "data_file": "add/accuracy_result.json",
        },
        "performance": {
            "details_mode": "external",
            "data_file": "add/performance_result.json",
            "speedup_source": "origin",
            "data": {"fp32": {"speedup": 2.0}},
        },
    }

    warnings = result_details.hydrate_operator_details(tmp_path, "add", operator_result)

    assert warnings == []
    assert operator_result["accuracy"]["details"] == {
        "failed": {"mismatch": ["test_add:[1]"]}
    }
    assert operator_result["performance"]["data"]["fp32"]["details"] == {
        "[1]": {
            "base": 2.0,
            "gems": 1.0,
            "speedup": 2.0,
            "speedup_source": "origin",
        }
    }


def test_hydrate_operator_details_keeps_inline_legacy_data(tmp_path):
    operator_result = {
        "accuracy": {"details": {"failed": {"old": ["case"]}}},
        "performance": {"data": {"fp32": {"details": {"[1]": {}}}}},
    }

    assert (
        result_details.hydrate_operator_details(tmp_path, "add", operator_result) == []
    )
    assert operator_result["accuracy"]["details"]["failed"] == {"old": ["case"]}
    assert (
        result_details.hydrate_operator_details(
            tmp_path, "add", {"accuracy": [], "performance": []}
        )
        == []
    )


def test_hydrate_operator_details_reports_invalid_and_unavailable_data(
    tmp_path, monkeypatch
):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    operator_result = {
        "accuracy": {
            "details_mode": "external",
            "data_file": "invalid.json",
        },
        "performance": {
            "details_mode": "external",
            "data_file": "invalid.json",
        },
    }

    warnings = result_details.hydrate_operator_details(tmp_path, "add", operator_result)
    assert warnings == [
        "add accuracy details are invalid",
        "add performance details are invalid",
    ]

    unavailable = {
        "accuracy": {"details_mode": "external"},
        "performance": {"details_mode": "external"},
    }
    warnings = result_details.hydrate_operator_details(tmp_path, "add", unavailable)
    assert len(warnings) == 2
    assert all("unavailable" in warning for warning in warnings)
    assert unavailable["accuracy"]["details"] == {}
    assert unavailable["performance"]["data"] == {}

    with mock.patch.object(
        result_details, "_external_data_path", side_effect=OSError("denied")
    ):
        warnings = result_details.hydrate_operator_details(
            tmp_path, "add", operator_result
        )
    assert warnings == [
        "add accuracy details unavailable: denied",
        "add performance details unavailable: denied",
    ]


def test_hydrate_operator_details_rejects_non_mapping_performance_data(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw.json"
    raw.write_text("{}", encoding="utf-8")
    operator_result = {
        "performance": {
            "details_mode": "external",
            "data_file": "raw.json",
        }
    }
    monkeypatch.setattr(
        result_details,
        "parse_perf_data",
        lambda *_args, **_kwargs: {"status": "Passed", "data": []},
    )

    assert (
        result_details.hydrate_operator_details(tmp_path, "add", operator_result) == []
    )
    assert operator_result["performance"]["data"] == {}
