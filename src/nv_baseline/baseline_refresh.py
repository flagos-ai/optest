#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Decide which operators a baseline run must re-collect.

The published baseline (data/op_<short>_perf_baseline.json) pins, per operator,
the framework version it was collected against in its "label" (for example
{"vllm": "0.32.0"}). A refresh does not need to re-measure an operator whose
label already equals the framework version installed in the collecting
container, so this module turns (baseline, candidates, current label) into a
plan: which operators to collect now, and which baseline entries to carry over
into the new output unchanged.

Two modes:

- incremental (default): for each candidate operator,
    * absent from the baseline            -> collect
    * present, stored label == current    -> retain the stored entry
    * present, stored label != current    -> collect (the entry, label included,
                                             is replaced by the fresh result)
    * present, either label unknown/empty -> collect (nothing to compare)
  Baseline operators that are not candidates in this run (removed from the
  target repo, or filtered out by --ops) are retained as well, so a partial
  collection never drops data from the published file.

- full: ignore the baseline entirely and collect every candidate, which is the
  behaviour of a collector without these flags. Nothing is retained, so a full
  run narrowed by --ops publishes only the operators it collected.

The caller seeds its results dict with `plan.retained` before collecting, so
the output file is the merged baseline at every flush, not just this run's
operators.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .collection_environment import _version
except ImportError:  # pragma: no cover - flat-script fallback, see collector
    from collection_environment import _version

MODE_INCREMENTAL = "incremental"
MODE_FULL = "full"
MODES = (MODE_INCREMENTAL, MODE_FULL)

# Why an operator is collected.
REASON_FULL_MODE = "full-mode"
REASON_ABSENT = "absent-from-baseline"
REASON_LABEL_CHANGED = "label-changed"
REASON_LABEL_UNKNOWN = "label-unknown"
# Why a baseline entry is retained.
REASON_LABEL_MATCH = "label-unchanged"
REASON_PRESENT = "present-no-label-to-compare"
REASON_NOT_CANDIDATE = "not-collected-this-run"

# Framework packages whose version is pinned per operator as a label. The
# baseline is always collected inside the target framework's own container, so
# whichever of these is installed identifies what the baseline tracks: vllm for
# FlagGems-vllm, sglang for FlagGems-sglang, deepspeed for FlagTrain. torch and
# triton are deliberately excluded as shared infrastructure, not the framework.
# Add a package here to start labeling a new framework; detection is automatic.
LABEL_FRAMEWORKS = ("vllm", "sglang", "deepspeed")


def detect_label(frameworks=LABEL_FRAMEWORKS):
    """Probe the environment for installed framework packages, no args needed.

    Returns a {framework: version} map for every candidate in `frameworks`
    actually installed in the collecting environment. Absent candidates are
    omitted (not null) so the label carries only frameworks really present. An
    empty result means no known framework is installed: the operator entry gets
    no label, and an incremental refresh falls back to presence-only matching.
    """
    detected = {}
    for name in frameworks:
        version = _version(name)
        if version is not None:
            detected[name] = version
    return detected


def format_label(label):
    """Render a label map for logs: "vllm=0.32.0", or "(none)" when empty."""
    if not label:
        return "(none)"
    return ", ".join(f"{k}={v}" for k, v in sorted(label.items()))


def load_baseline(path):
    """Read a published baseline JSON; a missing file is an empty baseline.

    A file that exists but does not parse, or does not hold a JSON object, is an
    error: silently treating it as empty would turn an operator mix-up or a
    truncated download into a full re-collection that then overwrites the
    (recoverable) published file with a differently-scoped one.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"baseline {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"baseline {path} must hold a JSON object")
    return data


def operator_entries(baseline):
    """Operator entries of a baseline, without "_"-prefixed bookkeeping keys
    (_environment, _scaling_factors, _incremental). A non-dict entry is dropped:
    it cannot carry a label or be published as an operator."""
    return {
        name: entry
        for name, entry in baseline.items()
        if not name.startswith("_") and isinstance(entry, dict)
    }


def stored_label(entry):
    """The label of a baseline entry, or None when it has none/an unusable one."""
    label = entry.get("label")
    return label if isinstance(label, dict) and label else None


@dataclass
class RefreshPlan:
    """What a refresh run collects and what it carries over unchanged."""

    mode: str
    label: dict
    baseline_path: str | None
    collect: list = field(default_factory=list)
    retained: dict = field(default_factory=dict)
    reasons: dict = field(default_factory=dict)
    # Baseline entries of operators this run re-collects, kept so a failed
    # re-collection can fall back to the published entry (see restore_missing).
    previous: dict = field(default_factory=dict)
    restored: list = field(default_factory=list)
    # True when the run measured nothing, so `_environment` and
    # `_scaling_factors` were carried over from the baseline rather than
    # produced by a collecting child.
    environment_from_baseline: bool = False

    @property
    def collect_set(self):
        return set(self.collect)

    def summary_lines(self):
        """Human-readable plan, one line per group plus a per-operator breakdown
        of what is collected (the retained set is only counted: it is routinely
        the whole published baseline)."""
        lines = [
            f"[refresh] mode={self.mode}  label={format_label(self.label)}"
            f"  baseline={self.baseline_path or '(none)'}",
            f"[refresh] collect {len(self.collect)}, "
            f"retain {len(self.retained)} baseline operator(s)",
        ]
        if self.mode == MODE_INCREMENTAL and not self.label:
            lines.append(
                "[refresh] no framework version detected in this container; "
                "matching on presence only (an operator already in the "
                "baseline is not re-collected)"
            )
        by_reason = {}
        for name in self.collect:
            by_reason.setdefault(self.reasons.get(name), []).append(name)
        for reason, names in sorted(by_reason.items(), key=lambda kv: str(kv[0])):
            lines.append(f"[refresh]   collect ({reason}): {', '.join(sorted(names))}")
        return lines

    def metadata(self):
        """The `_incremental` block recorded in the output baseline, so a
        published file states which of its operators this run measured and
        which came from the previous one."""
        return {
            "mode": self.mode,
            "baseline": self.baseline_path,
            "label": dict(self.label),
            "collected": sorted(self.collect),
            "retained": sorted(self.retained),
            "restored_after_failure": sorted(self.restored),
            "environment_from_baseline": self.environment_from_baseline,
            "decisions": dict(self.reasons),
        }


def plan_refresh(
    candidates, baseline, label, mode=MODE_INCREMENTAL, baseline_path=None
):
    """Build the plan for one refresh run.

    candidates: operator names this run could collect (after --ops filtering).
    baseline:   the parsed published baseline (see load_baseline).
    label:      {framework: version} of the collecting container (detect_label).
    mode:       MODE_INCREMENTAL or MODE_FULL.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")

    candidates = sorted(set(candidates))
    plan = RefreshPlan(mode=mode, label=dict(label or {}), baseline_path=baseline_path)

    if mode == MODE_FULL:
        plan.collect = list(candidates)
        plan.reasons = {name: REASON_FULL_MODE for name in candidates}
        return plan

    entries = operator_entries(baseline)
    for name in candidates:
        entry = entries.get(name)
        if entry is None:
            plan.collect.append(name)
            plan.reasons[name] = REASON_ABSENT
            continue
        previous = stored_label(entry)
        if not plan.label:
            # Nothing to compare against: keep what the baseline already has
            # rather than re-measuring every operator on every run.
            plan.retained[name] = entry
            plan.reasons[name] = REASON_PRESENT
        elif previous is None:
            plan.collect.append(name)
            plan.previous[name] = entry
            plan.reasons[name] = REASON_LABEL_UNKNOWN
        elif previous == plan.label:
            plan.retained[name] = entry
            plan.reasons[name] = REASON_LABEL_MATCH
        else:
            plan.collect.append(name)
            plan.previous[name] = entry
            plan.reasons[name] = (
                f"{REASON_LABEL_CHANGED} ({format_label(previous)} -> "
                f"{format_label(plan.label)})"
            )

    # Baseline operators outside this run's candidate set: removed from the
    # target repo, or filtered out by --ops. Retained so a narrowed run does
    # not drop them from the published file.
    for name, entry in entries.items():
        if name not in plan.reasons:
            plan.retained[name] = entry
            plan.reasons[name] = REASON_NOT_CANDIDATE
    return plan


def restore_missing(plan, results):
    """Put back the published entry of every operator this run meant to
    re-collect but produced nothing for (subprocess failed, skipped, or timed
    out), and return those names.

    An incremental refresh updates a published baseline in place, so a failed
    re-collection must not delete a known-good (if stale) entry: the operator
    would silently disappear from data/ and from every downstream comparison.
    The restored entry keeps its original label, so the next run sees the same
    label mismatch and tries again. Full mode keeps its existing behaviour
    (`plan.previous` is empty), where a failed operator is simply absent.
    """
    restored = []
    for name, entry in plan.previous.items():
        if name not in results:
            results[name] = entry
            restored.append(name)
    plan.restored = sorted(restored)
    return plan.restored
