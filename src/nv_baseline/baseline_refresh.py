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
    * present in the baseline             -> retain the stored entry unchanged
  Baseline operators that are not candidates in this run (removed from the
  target repo, or filtered out by --ops) are retained as well, so a partial
  collection never drops data from the published file. Version labels are
  ignored; presence alone determines whether to retain or collect.

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
# Why a baseline entry is retained.
REASON_PRESENT = "present-in-baseline"
REASON_NOT_CANDIDATE = "not-collected-this-run"

# Framework packages whose version is recorded per operator as a label for
# provenance tracking. The baseline is collected inside the target framework's
# own container, so whichever of these is installed identifies what the baseline
# tracks: vllm for FlagGems-vllm, sglang for FlagGems-sglang, deepspeed for
# FlagTrain, torch for FlagGems.
LABEL_FRAMEWORKS = ("vllm", "sglang", "deepspeed", "torch")


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
    # Bookkeeping keys (`_environment`, `_scaling_factors`) taken from the
    # previous baseline because no collecting child reported one. Non-empty
    # means the published provenance describes the earlier run, not this one.
    carried_over: list = field(default_factory=list)

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
            # Which bookkeeping keys describe the previous run rather than this
            # one, because no collecting child reported them.
            "carried_over_metadata": list(self.carried_over),
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
    # Incremental mode: retain any operator already in the baseline,
    # collect only those absent from it. Version labels are ignored.
    for name in candidates:
        entry = entries.get(name)
        if entry is None:
            plan.collect.append(name)
            plan.reasons[name] = REASON_ABSENT
        else:
            plan.retained[name] = entry
            plan.reasons[name] = REASON_PRESENT

    # Baseline operators outside this run's candidate set: removed from the
    # target repo, or filtered out by --ops. Retained so a narrowed run does
    # not drop them from the published file.
    for name, entry in entries.items():
        if name not in plan.reasons:
            plan.retained[name] = entry
            plan.reasons[name] = REASON_NOT_CANDIDATE
    return plan


BOOKKEEPING_FROM_CHILD = ("_environment", "_scaling_factors")
# Key order of a published baseline: the provenance of the run comes first, then
# the operators. Same in both modes, so a reader (and a diff) always finds the
# environment at the top of the file instead of somewhere among the operators.
PUBLISHED_FIRST = ("_environment", "_scaling_factors")
# Bookkeeping that stays out of the published baseline: the refresh plan is
# run-time state and lives in the sidecar status.json.
DROPPED_FROM_PUBLISHED = ("_incremental",)


def publishable(results):
    """The results dict as it should be written to the baseline file.

    Puts PUBLISHED_FIRST at the top in that order, then everything else in its
    existing order, and drops run-time bookkeeping: the refresh plan belongs in
    the sidecar status.json, not in the published data. Python dicts keep
    insertion order and json.dumps follows it, so rebuilding the dict is all
    the ordering takes.
    """
    ordered = {key: results[key] for key in PUBLISHED_FIRST if key in results}
    for key, value in results.items():
        if key in ordered or key in DROPPED_FROM_PUBLISHED:
            continue
        ordered[key] = value
    return ordered


def carry_over_metadata(plan, results, baseline, keys=BOOKKEEPING_FROM_CHILD):
    """Fill the bookkeeping keys no collecting child reported from `baseline`,
    and return the keys that needed it.

    `_environment` and `_scaling_factors` describe the machine a run measured
    on, so only a collecting child produces them. Two cases leave them absent:
    an incremental run with nothing stale (no child at all), and a run whose
    every queued operator died before flushing its own metadata. Both would
    otherwise republish a file that still holds operators -- restored or
    retained -- but has lost all provenance. A value a child did report always
    wins; this only fills gaps.
    """
    carried = []
    for key in keys:
        if key not in results and key in baseline:
            results[key] = baseline[key]
            carried.append(key)
    plan.carried_over = carried
    return carried


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
