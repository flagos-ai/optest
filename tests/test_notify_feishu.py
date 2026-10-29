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


if __name__ == "__main__":
    unittest.main()
