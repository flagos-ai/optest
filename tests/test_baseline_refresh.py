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

"""Focused tests for the incremental baseline refresh plan.

baseline_refresh is pure logic (no torch, no GPU), so it is exercised directly.
The module lives in src/nv_baseline, which is not part of the flagos_optest
wheel, so it is loaded by path.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "nv_baseline" / "baseline_refresh.py"


def _load_module():
    package_dir = str(MODULE_PATH.parent)
    if package_dir not in sys.path:
        sys.path.insert(0, package_dir)
    spec = importlib.util.spec_from_file_location("baseline_refresh", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so register first.
    sys.modules["baseline_refresh"] = module
    spec.loader.exec_module(module)
    return module


refresh = _load_module()


def entry(label=None, shapes=None):
    out = {"native_api": "ops.x", "shapes": shapes or {"[1]": {}}}
    if label is not None:
        out["label"] = label
    return out


VLLM_32 = {"vllm": "0.32.0"}
VLLM_24 = {"vllm": "0.24.0"}


def test_absent_operator_is_collected():
    plan = refresh.plan_refresh(["new_op"], {}, VLLM_32)
    assert plan.collect == ["new_op"]
    assert plan.retained == {}
    assert plan.reasons["new_op"] == refresh.REASON_ABSENT


def test_same_label_is_retained_not_recollected():
    baseline = {"a": entry(VLLM_32)}
    plan = refresh.plan_refresh(["a"], baseline, VLLM_32)
    assert plan.collect == []
    assert plan.retained == {"a": baseline["a"]}
    assert plan.reasons["a"] == refresh.REASON_LABEL_MATCH


def test_changed_label_is_recollected_and_reason_shows_both_versions():
    baseline = {"a": entry(VLLM_24)}
    plan = refresh.plan_refresh(["a"], baseline, VLLM_32)
    assert plan.collect == ["a"]
    assert plan.retained == {}
    assert plan.reasons["a"].startswith(refresh.REASON_LABEL_CHANGED)
    assert "vllm=0.24.0 -> vllm=0.32.0" in plan.reasons["a"]
    # The stale entry is remembered as the fallback for a failed re-collection.
    assert plan.previous == {"a": baseline["a"]}


def test_unlabeled_entry_is_recollected():
    baseline = {"a": entry(), "b": entry({}), "c": entry("0.24.0")}
    plan = refresh.plan_refresh(["a", "b", "c"], baseline, VLLM_32)
    assert plan.collect == ["a", "b", "c"]
    for name in ("a", "b", "c"):
        assert plan.reasons[name] == refresh.REASON_LABEL_UNKNOWN


def test_no_detected_framework_matches_on_presence_only():
    baseline = {"a": entry(VLLM_24)}
    plan = refresh.plan_refresh(["a", "b"], baseline, {})
    assert plan.collect == ["b"]
    assert plan.retained == {"a": baseline["a"]}
    assert plan.reasons["a"] == refresh.REASON_PRESENT
    assert any("presence only" in line for line in plan.summary_lines())


def test_baseline_operators_outside_the_run_are_retained():
    baseline = {"kept": entry(VLLM_24), "a": entry(VLLM_32)}
    plan = refresh.plan_refresh(["a"], baseline, VLLM_32)
    assert plan.collect == []
    assert set(plan.retained) == {"a", "kept"}
    assert plan.reasons["kept"] == refresh.REASON_NOT_CANDIDATE


def test_bookkeeping_and_non_dict_keys_are_not_operators():
    baseline = {
        "_environment": {"hardware": {}},
        "_scaling_factors": {"chips": {}},
        "broken": ["not", "an", "entry"],
        "a": entry(VLLM_32),
    }
    plan = refresh.plan_refresh(["a"], baseline, VLLM_32)
    assert set(plan.retained) == {"a"}
    assert "_environment" not in plan.reasons
    assert "broken" not in plan.reasons


def test_full_mode_ignores_the_baseline():
    baseline = {"a": entry(VLLM_32), "gone": entry(VLLM_32)}
    plan = refresh.plan_refresh(["a"], baseline, VLLM_32, mode=refresh.MODE_FULL)
    assert plan.collect == ["a"]
    assert plan.retained == {}
    assert plan.previous == {}
    assert plan.reasons == {"a": refresh.REASON_FULL_MODE}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        refresh.plan_refresh(["a"], {}, VLLM_32, mode="delta")


def test_candidates_are_deduplicated_and_sorted():
    plan = refresh.plan_refresh(["b", "a", "b"], {}, VLLM_32)
    assert plan.collect == ["a", "b"]
    assert plan.collect_set == {"a", "b"}


def test_restore_missing_keeps_the_published_entry_on_failure():
    baseline = {"a": entry(VLLM_24), "b": entry(VLLM_24)}
    plan = refresh.plan_refresh(["a", "b"], baseline, VLLM_32)
    results = {"a": entry(VLLM_32, shapes={"[2]": {}})}
    restored = refresh.restore_missing(plan, results)
    assert restored == ["b"]
    # The freshly collected operator is untouched; the failed one falls back to
    # its published entry, keeping its old label so the next run retries it.
    assert results["a"]["shapes"] == {"[2]": {}}
    assert results["b"] == baseline["b"]
    assert plan.metadata()["restored_after_failure"] == ["b"]


def test_restore_missing_is_a_noop_in_full_mode():
    baseline = {"a": entry(VLLM_24)}
    plan = refresh.plan_refresh(["a"], baseline, VLLM_32, mode=refresh.MODE_FULL)
    results = {}
    assert refresh.restore_missing(plan, results) == []
    assert results == {}


def test_load_baseline_missing_file_is_empty(tmp_path):
    assert refresh.load_baseline(tmp_path / "nope.json") == {}
    assert refresh.load_baseline(None) == {}


def test_load_baseline_rejects_unusable_files(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        refresh.load_baseline(broken)
    listed = tmp_path / "list.json"
    listed.write_text("[]")
    with pytest.raises(ValueError, match="must hold a JSON object"):
        refresh.load_baseline(listed)


def test_load_baseline_reads_a_published_file(tmp_path):
    path = tmp_path / "op_vllm_perf_baseline.json"
    path.write_text(json.dumps({"_environment": {}, "a": entry(VLLM_32)}))
    assert set(refresh.load_baseline(path)) == {"_environment", "a"}


def test_detect_label_reports_installed_frameworks(monkeypatch):
    monkeypatch.setattr(
        refresh, "_version", lambda name: "0.32.0" if name == "vllm" else None
    )
    assert refresh.detect_label() == VLLM_32
    monkeypatch.setattr(refresh, "_version", lambda name: None)
    assert refresh.detect_label() == {}


def test_format_label():
    assert refresh.format_label({}) == "(none)"
    assert refresh.format_label(None) == "(none)"
    assert (
        refresh.format_label({"vllm": "0.32.0", "deepspeed": "0.20.1"})
        == "deepspeed=0.20.1, vllm=0.32.0"
    )


def test_metadata_and_summary_describe_the_plan(tmp_path):
    baseline = {"kept": entry(VLLM_32), "stale": entry(VLLM_24)}
    plan = refresh.plan_refresh(
        ["kept", "stale", "fresh"],
        baseline,
        VLLM_32,
        baseline_path=str(tmp_path / "data.json"),
    )
    meta = plan.metadata()
    assert meta["mode"] == refresh.MODE_INCREMENTAL
    assert meta["collected"] == ["fresh", "stale"]
    assert meta["retained"] == ["kept"]
    assert meta["label"] == VLLM_32
    assert meta["baseline"].endswith("data.json")
    text = "\n".join(plan.summary_lines())
    assert "collect 2, retain 1" in text
    assert "fresh" in text and "stale" in text


def test_real_published_baselines_are_fully_labeled():
    """Every committed baseline must carry a comparable label on every
    operator; otherwise an incremental refresh degenerates into a full one."""
    for path in sorted((ROOT / "data").glob("op_*_perf_baseline.json")):
        baseline = refresh.load_baseline(path)
        entries = refresh.operator_entries(baseline)
        assert entries, f"{path.name} holds no operator entries"
        unlabeled = [n for n, e in entries.items() if refresh.stored_label(e) is None]
        assert not unlabeled, f"{path.name} has unlabeled operators: {unlabeled}"
        # A refresh at the published label re-collects nothing.
        label = refresh.stored_label(next(iter(entries.values())))
        plan = refresh.plan_refresh(list(entries), baseline, label, baseline_path=path)
        assert plan.collect == []
        assert len(plan.retained) == len(entries)


class TestSchedulerIntegration:
    """End-to-end over the multi-GPU scheduler, the path the workflow runs,
    with the per-operator collector subprocess stubbed out."""

    @staticmethod
    def _scheduler():
        path = ROOT / "src" / "nv_baseline" / "collect_baseline_multi_gpu.py"
        spec = importlib.util.spec_from_file_location(
            "collect_baseline_multi_gpu", path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["collect_baseline_multi_gpu"] = module
        spec.loader.exec_module(module)
        # The planner probes the collecting container for an installed
        # framework; pin it so the test does not depend on this host.
        module.baseline_refresh.detect_label = lambda: dict(VLLM_32)
        return module

    @staticmethod
    def _args(tmp_path, ops_dir, baseline=None, mode=refresh.MODE_INCREMENTAL):
        return types.SimpleNamespace(
            devices="0",
            output=str(tmp_path / "out.json"),
            ops=None,
            ops_dir=str(ops_dir),
            op_timeout=60.0,
            reference_chip="H800",
            report_dir=None,
            baseline=str(baseline) if baseline else None,
            mode=mode,
        )

    @staticmethod
    def _ops_dir(tmp_path, names):
        ops = tmp_path / "baseline"
        ops.mkdir()
        for name in names:
            (ops / f"{name}.py").write_text("")
        return ops

    def _executor(self, collected, *, fail=()):
        """Stand in for the collector subprocess: write the result.json a real
        child would, or fail without writing anything."""

        def execute(command, environment, log_path, timeout, stop):
            name = command[command.index("--ops") + 1]
            output = Path(command[command.index("--output") + 1])
            log_path.write_text("stub\n")
            if name in fail:
                return 1
            output.write_text(
                json.dumps(
                    {
                        name: {
                            "label": VLLM_32,
                            "native_api": f"ops.{name}",
                            "shapes": {"[8]": {"torch.float32": {"num_kernels": 1}}},
                        },
                        "_environment": {"reference_chip": "H800"},
                        "_scaling_factors": {"chips": {}},
                    }
                )
            )
            collected.append(name)
            return 0

        return execute

    def test_incremental_run_collects_only_stale_operators(self, tmp_path):
        scheduler = self._scheduler()
        ops_dir = self._ops_dir(tmp_path, ["fresh", "stale", "unchanged"])
        baseline = tmp_path / "data.json"
        baseline.write_text(
            json.dumps(
                {
                    "_environment": {"reference_chip": "H800"},
                    "unchanged": entry(VLLM_32),
                    "stale": entry(VLLM_24),
                    "retired": entry(VLLM_32),
                }
            )
        )
        collected = []
        args = self._args(tmp_path, ops_dir, baseline=baseline)
        code = scheduler.collect(args, execute=self._executor(collected))
        assert code == 0
        # "unchanged" matched the container label and "retired" is no longer an
        # operator module, so neither was measured; the new and the stale one
        # were.
        assert sorted(collected) == ["fresh", "stale"]
        published = json.loads(Path(args.output).read_text())
        assert set(published) - {
            "_environment",
            "_scaling_factors",
            "_incremental",
        } == {
            "fresh",
            "stale",
            "unchanged",
            "retired",
        }
        assert published["stale"]["label"] == VLLM_32
        assert published["unchanged"] == entry(VLLM_32)
        meta = published["_incremental"]
        assert meta["mode"] == refresh.MODE_INCREMENTAL
        assert meta["collected"] == ["fresh", "stale"]
        assert set(meta["retained"]) == {"unchanged", "retired"}

    def test_failed_recollection_keeps_the_published_entry(self, tmp_path):
        scheduler = self._scheduler()
        ops_dir = self._ops_dir(tmp_path, ["stale"])
        baseline = tmp_path / "data.json"
        baseline.write_text(json.dumps({"stale": entry(VLLM_24)}))
        args = self._args(tmp_path, ops_dir, baseline=baseline)
        code = scheduler.collect(args, execute=self._executor([], fail=("stale",)))
        assert code == 1
        published = json.loads(Path(args.output).read_text())
        # The operator keeps its old entry and old label, so it stays in the
        # baseline and the next run retries it.
        assert published["stale"] == entry(VLLM_24)
        assert published["_incremental"]["restored_after_failure"] == ["stale"]

    def test_nothing_stale_publishes_the_baseline_with_its_provenance(self, tmp_path):
        scheduler = self._scheduler()
        ops_dir = self._ops_dir(tmp_path, ["a"])
        baseline = tmp_path / "data.json"
        baseline.write_text(
            json.dumps(
                {
                    "_environment": {"reference_chip": "H800"},
                    "_scaling_factors": {"chips": {"H800": 1.0}},
                    "a": entry(VLLM_32),
                }
            )
        )
        collected = []
        args = self._args(tmp_path, ops_dir, baseline=baseline)
        code = scheduler.collect(args, execute=self._executor(collected))
        assert code == 0
        assert collected == []
        published = json.loads(Path(args.output).read_text())
        assert published["a"] == entry(VLLM_32)
        # No child ran, so environment and scaling factors come from the
        # baseline and are flagged as such.
        assert published["_environment"] == {"reference_chip": "H800"}
        assert published["_scaling_factors"] == {"chips": {"H800": 1.0}}
        assert published["_incremental"]["environment_from_baseline"] is True
        status = json.loads((Path(f"{args.output}.work") / "status.json").read_text())
        assert status["tasks"] == {}
        assert status["incremental"]["collected"] == []

    def test_full_mode_recollects_everything(self, tmp_path):
        scheduler = self._scheduler()
        ops_dir = self._ops_dir(tmp_path, ["a", "b"])
        baseline = tmp_path / "data.json"
        baseline.write_text(json.dumps({"a": entry(VLLM_32), "b": entry(VLLM_32)}))
        collected = []
        args = self._args(tmp_path, ops_dir, baseline=baseline, mode=refresh.MODE_FULL)
        code = scheduler.collect(args, execute=self._executor(collected))
        assert code == 0
        assert sorted(collected) == ["a", "b"]
        published = json.loads(Path(args.output).read_text())
        assert published["_incremental"]["retained"] == []

    def test_corrupt_baseline_is_a_usage_error(self, tmp_path):
        scheduler = self._scheduler()
        ops_dir = self._ops_dir(tmp_path, ["a"])
        baseline = tmp_path / "data.json"
        baseline.write_text("{broken")
        args = self._args(tmp_path, ops_dir, baseline=baseline)
        with pytest.raises(ValueError, match="not valid JSON"):
            scheduler.collect(args, execute=self._executor([]))


class TestWorkflowWiring:
    @classmethod
    def setup_class(cls):
        cls.workflow = yaml.safe_load(
            (ROOT / ".github/workflows/nv_baseline.yaml").read_text(encoding="utf-8")
        )
        cls.steps = cls.workflow["jobs"]["collect"]["steps"]

    def step(self, name):
        return next(s for s in self.steps if s.get("name") == name)

    def test_mode_input_defaults_to_incremental(self):
        inputs = self.workflow[True]["workflow_dispatch"]["inputs"]
        assert inputs["mode"]["default"] == refresh.MODE_INCREMENTAL
        assert set(inputs["mode"]["options"]) == set(refresh.MODES)

    def test_collect_step_passes_mode_and_baseline(self):
        run = self.step("Run NV baseline collection")["run"]
        assert '--mode "${MODE}"' in run
        assert '--baseline "${BASE_JSON}"' in run
        # The comparison baseline is the committed data/ copy, a different path
        # from the output the collector writes (which must not already exist).
        assert 'BASE_JSON="${GITHUB_WORKSPACE}/data/${OUTPUT}"' in run

    def test_mode_env_falls_back_to_incremental(self):
        env = self.workflow["jobs"]["collect"]["env"]
        assert env["MODE"] == "${{ github.event.inputs.mode || 'incremental' }}"

    def test_classify_passes_a_no_op_incremental_run(self):
        script = self.step("Classify collection result")["run"]
        namespace = {"__name__": "classify"}
        status = {"tasks": {}, "incremental": {"collected": [], "retained": ["a"]}}
        self.run_classify(script, namespace, status, expect_exit=0)

    def test_classify_still_fails_an_empty_full_run(self):
        script = self.step("Classify collection result")["run"]
        status = {"tasks": {}}
        self.run_classify(script, {"__name__": "classify"}, status, expect_exit=1)

    def run_classify(self, script, namespace, status, expect_exit):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "op_test_perf_baseline.json"
            output.write_text("{}")
            work = Path(f"{output}.work")
            work.mkdir()
            (work / "status.json").write_text(json.dumps(status))
            github_output = Path(directory) / "github-output"
            github_output.write_text("")
            import os
            from unittest import mock

            with mock.patch.dict(
                os.environ,
                {"OUTPUT": str(output), "GITHUB_OUTPUT": str(github_output)},
            ):
                with pytest.raises(SystemExit) as excinfo:
                    exec(compile(script, "classify", "exec"), namespace)
            assert excinfo.value.code == expect_exit
            return github_output.read_text()

    def test_summary_step_reports_the_refresh_decision(self):
        run = self.step("Summarize collected operators")["run"]
        assert 'incremental = status.get("incremental")' in run
        assert "re-collected now" in run
        assert "carried over from the committed baseline" in run

    # The data/ update step guards against publishing a baseline that lost
    # operators. Exercised as a real bash script (it is a heredoc inside an
    # `if`, which a string check would not validate).
    def update_guard(self, workspace, mode):
        import os
        import subprocess

        run = self.step("Update baseline in data/")["run"]
        # Only the guard prefix: the rest commits and pushes.
        prefix = run.split("mkdir -p")[0]
        return subprocess.run(
            ["bash", "-e", "-o", "pipefail"],
            input=prefix,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "GITHUB_WORKSPACE": str(workspace),
                "OUTPUT": "op_t.json",
                "MODE": mode,
            },
        )

    @staticmethod
    def _workspace(tmp_path, produced, committed):
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "op_t.json").write_text(json.dumps(produced))
        (tmp_path / "data" / "op_t.json").write_text(json.dumps(committed))
        return tmp_path

    def test_update_step_is_valid_bash(self):
        import subprocess

        run = self.step("Update baseline in data/")["run"]
        assert subprocess.run(["bash", "-n"], input=run, text=True).returncode == 0

    def test_update_guard_allows_a_complete_baseline(self, tmp_path):
        both = {"a": entry(VLLM_32), "b": entry(VLLM_32)}
        workspace = self._workspace(
            tmp_path, {**both, "_environment": {}}, {"a": entry(VLLM_24), "b": entry()}
        )
        done = self.update_guard(workspace, refresh.MODE_INCREMENTAL)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "no operator lost" in done.stdout

    def test_update_guard_refuses_a_shrunken_baseline(self, tmp_path):
        workspace = self._workspace(
            tmp_path, {"a": entry(VLLM_32)}, {"a": entry(VLLM_24), "b": entry(VLLM_24)}
        )
        done = self.update_guard(workspace, refresh.MODE_INCREMENTAL)
        assert done.returncode == 1
        assert "drops 1 operator(s)" in done.stdout
        assert ": b;" in done.stdout

    def test_update_guard_exempts_full_mode(self, tmp_path):
        # A full run intentionally omits operators that failed; that commit is
        # flagged as forced instead of being refused.
        workspace = self._workspace(
            tmp_path, {"a": entry(VLLM_32)}, {"a": entry(VLLM_24), "b": entry(VLLM_24)}
        )
        assert self.update_guard(workspace, refresh.MODE_FULL).returncode == 0

    def test_update_guard_proceeds_without_a_committed_baseline(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "op_t.json").write_text(json.dumps({"a": entry(VLLM_32)}))
        done = self.update_guard(tmp_path, refresh.MODE_INCREMENTAL)
        assert done.returncode == 0
        assert "no readable committed baseline" in done.stdout
