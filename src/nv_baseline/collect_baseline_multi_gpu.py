#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dynamically schedule isolated operator collectors, one consumer per GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Directory of this collector package (holds collect_baseline_nvidia.py and
# hardware_specs.json).
PKG_DIR = Path(__file__).resolve().parent
# Default per-operator module dir; overridable via --ops-dir (e.g. a checked-out
# target repo's baseline/). Mirrors collect_baseline_nvidia.OPS_DIR.
OPS_DIR = ROOT / "baseline"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_devices(value):
    devices = [v.strip() for v in value.split(",")]
    if not all(devices) or len(devices) != len(set(devices)):
        raise ValueError("devices must be nonempty and unique")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not set(devices).issubset(visible.split(",")):
        raise ValueError(
            "--devices must select tokens from inherited CUDA_VISIBLE_DEVICES"
        )
    return devices


def name_list(value):
    if value is None:
        return None
    path = Path(value)
    # A comma-separated selection can exceed the filesystem filename limit.
    try:
        is_file = path.is_file()
    except OSError:
        is_file = False
    content = path.read_text() if is_file else value.replace(",", "\n")
    return {s for line in content.splitlines() if (s := line.split("#")[0].strip())}


def _iter_op_stems(ops_dir):
    """Operator module stems directly under ops_dir, plus one level down in
    per-library subdirs (e.g. FlagTrain's baseline/deepspeed, baseline/megatron)."""
    stems = [p.stem for p in ops_dir.glob("*.py") if not p.stem.startswith("__")]
    for sub in ops_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("__"):
            stems += [p.stem for p in sub.glob("*.py") if not p.stem.startswith("__")]
    return stems


def select_operators(only=None, ops_dir=OPS_DIR):
    ops_dir = Path(ops_dir)
    if not ops_dir.is_dir():
        raise ValueError(f"operator directory not found: {ops_dir}")
    names = sorted(set(_iter_op_stems(ops_dir)))
    unknown = (only or set()) - set(names)
    if unknown:
        raise ValueError(f"unknown operator modules: {sorted(unknown)}")
    return [n for n in names if only is None or n in only]


def run_process(command, environment, log_path, timeout, stop):
    """Terminate the entire group, including NCU and its Python child."""
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if stop.wait(0.2) or time.monotonic() >= deadline:
                    return "interrupted" if stop.is_set() else "timeout"
            return process.returncode
        finally:
            # NCU descendants can outlive their direct parent after a crash.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def collect(args, *, execute=run_process):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.with_suffix(output.suffix + ".lock")).open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another collector owns this output") from exc
        return _collect(args, execute=execute)


def record_quality(entry):
    """Identify measured workloads and whether each has profiler evidence."""
    return {
        (shape, dtype): bool(record.get("num_kernels"))
        for shape, values in entry.get("shapes", {}).items()
        for dtype, record in values.items()
    }


def covers_previous(previous, current):
    old, new = record_quality(previous), record_quality(current)
    return old.keys() <= new.keys() and all(
        not profiled or new[key] for key, profiled in old.items()
    )


def report_directory(report_root, output, operator, attempt_directory):
    if not report_root:
        return attempt_directory / "ncu_reports"
    # Basenames alone collide for /run/a/data.json and /run/b/data.json.
    digest = hashlib.sha256(str(output).encode()).hexdigest()[:16]
    namespace = f"{output.name[:80]}-{digest}"
    return Path(report_root).resolve() / namespace / operator / attempt_directory.name


def merge_event(results, origins, tasks, name, record, data):
    """Checkpoint latest task state without degrading a retained result snapshot.

    Running events have data=None; terminal events carry the child JSON (possibly
    empty after failure). Attempts are kept separately rather than mixing records
    collected on different GPUs/environments into one operator entry. Task state
    is kept in a sidecar map (status.json), separate from the published baseline
    `results`, which carries no run-time bookkeeping.
    """
    previous = tasks.get(name, {})
    history = list(previous.get("previous_attempts", []))
    if data is None and previous:
        history.append(
            {
                key: value
                for key, value in previous.items()
                if key != "previous_attempts"
            }
        )
    if history:
        record["previous_attempts"] = history
    if data is None and name in results:
        record["retained_result"] = origins[name]
    if data is not None:
        candidate = data.get(name)
        if candidate is not None and (
            name not in results or covers_previous(results[name], candidate)
        ):
            results[name] = candidate
            origins[name] = {
                "device": record["device"],
                "result": record["result"],
                "environment": data.get("_environment"),
            }
        elif name in results:
            record["retained_result"] = origins[name]
        if "_environment" in data:
            # Each child's _environment is already reduced to a single
            # machine-level descriptor (no per-card identity), so keep one
            # representative rather than a per-card map.
            results.setdefault("_environment", data["_environment"])
        if "_scaling_factors" in data:
            results["_scaling_factors"] = data["_scaling_factors"]
        print(f"[{record['device']}] {name}: {record['status']}", flush=True)
    tasks[name] = record


_TERMINAL_STATUSES = ("completed", "failed", "skipped", "profiling_incomplete")


def _print_progress(tasks, total):
    """Emit one overall-progress line: how many operators have reached a
    terminal state out of the total, broken down by status, plus which
    operators are still running on which card."""
    counts = {s: 0 for s in _TERMINAL_STATUSES}
    running = []
    for name, rec in tasks.items():
        status = rec.get("status")
        if status in counts:
            counts[status] += 1
        elif status == "running":
            running.append(f"{name}@cuda{rec.get('device')}")
    done = sum(counts.values())
    breakdown = " ".join(f"{s}={counts[s]}" for s in _TERMINAL_STATUSES)
    line = f"[progress] {done}/{total} done ({breakdown})"
    if running:
        line += f" | running: {', '.join(sorted(running))}"
    print(line, flush=True)


def _collect(args, *, execute=run_process):
    if not math.isfinite(args.op_timeout) or args.op_timeout <= 0:
        raise ValueError("--op-timeout must be finite and positive")
    ops_dir = Path(getattr(args, "ops_dir", None) or OPS_DIR)
    devices = parse_devices(args.devices)
    names = select_operators(name_list(args.ops), ops_dir=ops_dir)
    if not names:
        raise ValueError("no operators selected")
    output = Path(args.output).resolve()
    work = output.with_suffix(output.suffix + ".work")
    if output.exists():
        raise ValueError("output exists; choose another path")
    # The published baseline carries no run-time bookkeeping; per-operator task
    # state is tracked separately and written to a sidecar status.json under the
    # .work dir, which the workflow reads to grade the run.
    results = {}
    tasks = {}
    status_path = work / "status.json"
    origins = {}
    pending = queue.Queue()
    for name in names:
        pending.put(name)
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(output, results)
    atomic_json(status_path, {"tasks": tasks})
    stop = threading.Event()
    events = queue.Queue()

    def consume(device):
        while not stop.is_set():
            try:
                name = pending.get_nowait()
            except queue.Empty:
                return
            folder = work / name / uuid.uuid4().hex
            folder.mkdir(parents=True, exist_ok=True)
            part = folder / "result.json"
            command = [
                sys.executable,
                str(PKG_DIR / "collect_baseline_nvidia.py"),
                "--ops",
                name,
                "--ops-dir",
                str(ops_dir),
                "--output",
                str(part),
                "--report-dir",
                str(
                    report_directory(
                        getattr(args, "report_dir", None), output, name, folder
                    )
                ),
                "--op-timeout",
                str(args.op_timeout),
                "--reference-chip",
                args.reference_chip,
            ]
            # BASELINE_TASK_EXIT=1 makes the child collector return a 0/1 exit
            # code (internal signal passed via environment variable).
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=device,
                PYTHONUNBUFFERED="1",
                BASELINE_TASK_EXIT="1",
            )
            events.put(
                (
                    name,
                    {
                        "status": "running",
                        "device": device,
                        "log": str(folder / "collector.log"),
                        "result": str(part),
                    },
                    None,
                )
            )
            try:
                code = execute(
                    command,
                    environment,
                    folder / "collector.log",
                    args.op_timeout + 120,
                    stop,
                )
                data = json.loads(part.read_text()) if part.exists() else {}
                entry = data.get(name)
                status = (
                    "failed"
                    if code != 0
                    else ("skipped" if entry is None else "completed")
                )
                if status == "completed":
                    records = [
                        r
                        for shape in entry.get("shapes", {}).values()
                        for r in shape.values()
                    ]
                    if not records:
                        status = "skipped"
                    elif any(not r.get("num_kernels") for r in records):
                        status = "profiling_incomplete"
                record = {
                    "status": status,
                    "device": device,
                    "returncode": code,
                    "log": str(folder / "collector.log"),
                    "result": str(part),
                }
                events.put((name, record, data))
            except Exception as exc:
                events.put(
                    (
                        name,
                        {
                            "status": "failed",
                            "device": device,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        {},
                    )
                )

    previous = {}

    def interrupted(signum, frame):
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(consume, device) for device in devices]
            try:
                while not all(f.done() for f in futures) or not events.empty():
                    for future in futures:
                        if future.done():
                            future.result()
                    try:
                        name, record, data = events.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    merge_event(results, origins, tasks, name, record, data)
                    atomic_json(output, results)
                    atomic_json(status_path, {"tasks": tasks})
                    # Print an overall progress line on each terminal event
                    # (running events carry data=None and are skipped here).
                    if data is not None:
                        _print_progress(tasks, len(names))
                for future in futures:
                    future.result()
            except BaseException:
                # Set this before executor.__exit__ joins the consumers. Otherwise
                # a broken checkpoint leaves them draining the entire GPU queue.
                stop.set()
                raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    # The exit status and the final task state live only in the sidecar
    # status.json; the published baseline carries no run-time bookkeeping.
    exit_status = int(
        stop.is_set()
        or any(tasks.get(n, {}).get("status") != "completed" for n in names)
    )
    atomic_json(status_path, {"tasks": tasks, "interrupted": stop.is_set()})
    atomic_json(output, results)
    return exit_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--devices", required=True, help="GPU visibility tokens, e.g. 0,1,2,3"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--report-dir", help="Optional report root (partitioned by output and operator)"
    )
    parser.add_argument("--ops")
    parser.add_argument(
        "--ops-dir",
        default=None,
        help="directory of per-operator baseline modules (e.g. a checked-out "
        "target repo's baseline/). Defaults to the baseline/ dir next to this "
        "collector package",
    )
    parser.add_argument("--op-timeout", type=float, default=1800)
    parser.add_argument("--reference-chip", default="H800")
    args = parser.parse_args()
    if args.op_timeout <= 0:
        parser.error("--op-timeout must be positive")
    try:
        return collect(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
