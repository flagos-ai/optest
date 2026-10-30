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

import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / ".github" / "scripts" / "notify_feishu.py"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ops-test.yaml"


def load_notify_feishu():
    spec = importlib.util.spec_from_file_location("notify_feishu", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class NotifyFeishuRefTest(unittest.TestCase):
    def setUp(self):
        self.module = load_notify_feishu()
        self.requests = []

    def urlopen(self, request):
        self.requests.append(request)
        if request.full_url.endswith("tenant_access_token/internal"):
            return Response({"tenant_access_token": "token"})
        return Response({"code": 0})

    def run_notification(self, **overrides):
        environment = {
            "FEISHU_APP_ID": "app-id",
            "FEISHU_APP_SECRET": "app-secret",
            "FEISHU_CHAT_ID": "chat-id",
            "CI_STATUS": "success",
            "GITHUB_WORKFLOW": "ops-test",
            "GITHUB_REPOSITORY": "flagos-ai/optest",
            "GITHUB_REF_NAME": "optest-branch",
            "GITHUB_SHA": "0123456789abcdef",
            "GITHUB_ACTOR": "tester",
            "GITHUB_RUN_ID": "123",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
        }
        environment.update(overrides)
        with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
            self.module.urllib.request, "urlopen", side_effect=self.urlopen
        ):
            self.module.main()

        message = json.loads(self.requests[1].data)
        return json.loads(message["content"])

    def test_target_ref_overrides_optest_and_legacy_refs(self):
        card = self.run_notification(
            TARGET_REF="target-tag", OVERRIDE_REF="legacy-target-branch"
        )
        fields = card["elements"][0]["fields"]
        contents = [field["text"]["content"] for field in fields]
        self.assertIn("**Ref:** target-tag", contents)
        self.assertNotIn("**Ref:** optest-branch", contents)
        self.assertNotIn("**Ref:** legacy-target-branch", contents)

    def test_legacy_override_ref_remains_supported(self):
        card = self.run_notification(OVERRIDE_REF="target-commit")
        fields = card["elements"][0]["fields"]
        contents = [field["text"]["content"] for field in fields]
        self.assertIn("**Ref:** target-commit", contents)

    def test_workflow_passes_resolved_target_checkout_ref(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        prepare_steps = workflow["jobs"]["prepare"]["steps"]
        build_matrix = next(
            step for step in prepare_steps if step.get("id") == "build-matrix"
        )
        self.assertIn(
            "checkout_ref = os.environ.get('REF_INPUT', '').strip() "
            "or rc['default_branch']",
            build_matrix["run"],
        )

        steps = workflow["jobs"]["test-container"]["steps"]
        notification = next(
            step for step in steps if step.get("name") == "Send Feishu notification"
        )
        self.assertEqual(
            notification["env"]["TARGET_REF"],
            "${{ needs.prepare.outputs.checkout_ref }}",
        )
        self.assertNotIn("OVERRIDE_REF", notification["env"])

    def test_workflow_run_name_identifies_target_repository(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.assertEqual(workflow["run-name"], "ops-test_${{ inputs.repo }}")

    def test_workflow_logs_effective_test_context_and_commands(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        job = workflow["jobs"]["test-container"]
        self.assertEqual(
            job["env"]["TARGET_REPOSITORY"],
            "${{ needs.prepare.outputs.target_repo }}",
        )
        self.assertEqual(
            job["env"]["TARGET_REF"],
            "${{ needs.prepare.outputs.checkout_ref }}",
        )
        self.assertEqual(job["env"]["TEST_OPS"], "${{ needs.prepare.outputs.ops }}")

        steps = job["steps"]
        run_tests = next(step for step in steps if step.get("name") == "Run ops tests")
        script = run_tests["run"]

        expected_log_entries = [
            "Target repository: ${TARGET_REPOSITORY}",
            "Target ref: ${TARGET_REF}",
            "Target commit: ${TARGET_COMMIT}",
            "Operator arguments: ${OP_ARGS}",
            "Base-data arguments: ${BASE_DATA_ARG:-<none>}",
            "Command: flagos-optest",
            "Command: flagos-add-labels",
            "Command: flagos-psum-text",
            "Command: flagos-psum-html",
        ]
        for entry in expected_log_entries:
            with self.subTest(entry=entry):
                self.assertIn(entry, script)

    def test_workflow_names_checkouts_and_install_commands(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        prepare_steps = workflow["jobs"]["prepare"]["steps"]
        self.assertTrue(
            any(step.get("name") == "Checkout optest workflow source" for step in prepare_steps)
        )

        steps = workflow["jobs"]["test-container"]["steps"]
        names = {step.get("name") for step in steps}
        self.assertIn("Checkout optest source", names)
        for attempt in range(1, 4):
            self.assertIn(f"Checkout target repository (attempt {attempt})", names)

        install_optest = next(
            step
            for step in steps
            if step.get("name") == "Configure package index and install optest"
        )
        install_target = next(
            step
            for step in steps
            if step.get("name") == "Install target repository and dependencies"
        )
        gpu_check = next(
            step for step in steps if step.get("name") == "Check GPU availability"
        )
        self.assertIn("Command: pip install .", install_optest["run"])
        self.assertNotIn("matrix.vendor", install_optest["run"])
        self.assertNotIn("https://pypi.org/simple", install_optest["run"])
        self.assertIn(
            "[Package index] PIP_INDEX_URL=${PIP_INDEX_URL_VALUE}",
            install_optest["run"],
        )
        self.assertIn(
            "PIP_INDEX_URL_VALUE=https://pypi.tuna.tsinghua.edu.cn/simple",
            install_optest["run"],
        )
        self.assertIn("PIP_EXTRA_INDEX_URL_VALUE=", install_optest["run"])
        self.assertIn("PIP_TRUSTED_HOST_VALUE=", install_optest["run"])
        self.assertIn("FLAGOS_GPU_VENDOR", gpu_check["run"])
        self.assertIn("FLAGOS_GPU_DRIVER", gpu_check["run"])
        self.assertIn("FLAGOS_GPU_RUNTIME", gpu_check["run"])
        self.assertIn("FLAGOS_GPU_RUNTIME_VERSION", gpu_check["run"])
        self.assertIn("Command: pip install ${EXTRA_DEPS}", install_target["run"])
        self.assertIn("Command: pip install ./target[test]", install_target["run"])
        self.assertIn(
            'flagos-install-target-deps ./target "${FLAGOS_GPU_VENDOR}" ${VENDOR_PIP_ARGS}',
            install_target["run"],
        )

    def test_workflow_retries_result_uploads(self):
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        steps = workflow["jobs"]["test-container"]["steps"]
        publish = next(
            step for step in steps if step.get("name") == "Publish results to op-monitor"
        )
        script = publish["run"]
        curl_command = script[
            script.index("curl -X POST") : script.index("--compressed")
        ]
        self.assertIn(
            "--max-time 300 --retry 3 --retry-delay 15 --retry-all-errors",
            curl_command,
        )
        retry_budget_seconds = 4 * 300 + 3 * 15
        self.assertGreaterEqual(publish["timeout-minutes"] * 60, retry_budget_seconds)


if __name__ == "__main__":
    unittest.main()
