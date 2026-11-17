#!/usr/bin/env bash
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
#
# ASCEND_WORK_PATH must be pinned before anything sources ATB's set_env.sh.
# GitHub Actions job containers force HOME=/github/home; the Ascend/MKI
# libraries loaded by torch_npu derive their work path from HOME and print
# "path string is NULL" to stdout under that HOME. Huawei's ATB set_env.sh
# compares the whole stdout of a torch probe against "True" for ABI
# auto-detection, so the extra text makes it select cxx_abi_0.
set -eo pipefail

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  echo "[Ascend env] Configuring ASCEND_WORK_PATH before sourcing Ascend/ATB profiles"
  echo "[Ascend env] before rewrite: ASCEND_WORK_PATH=${ASCEND_WORK_PATH:-<unset>}"
  echo "[Ascend env] before rewrite: ATB_HOME_PATH=${ATB_HOME_PATH:-<unset>}"
  echo "[Ascend env] before rewrite: PATH=${PATH:-}"
  echo "[Ascend env] before rewrite: LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  echo "[Ascend env] .bash_profile before rewrite:"
  nl -ba "$HOME/.bash_profile" || true
  ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-/tmp/ascend_work/${GITHUB_RUN_ID:-manual}}"
  mkdir -p "$ASCEND_WORK_PATH"
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "ASCEND_WORK_PATH=$ASCEND_WORK_PATH" >> "$GITHUB_ENV"
  fi
  echo "[Ascend env] exported ASCEND_WORK_PATH=$ASCEND_WORK_PATH"
  {
    echo 'export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-/tmp/ascend_work/${GITHUB_RUN_ID:-manual}}"'
    echo 'mkdir -p "$ASCEND_WORK_PATH"'
    cat "$HOME/.bash_profile"
  } > "$HOME/.bash_profile.tmp"
  mv "$HOME/.bash_profile.tmp" "$HOME/.bash_profile"
  echo "[Ascend env] .bash_profile after rewrite:"
  nl -ba "$HOME/.bash_profile" || true
  echo "[Ascend env] verifying a fresh login shell after profile source"
  bash --login -c '
    set +e
    echo "[Ascend env] after login source: ASCEND_WORK_PATH=${ASCEND_WORK_PATH:-<unset>}"
    echo "[Ascend env] after login source: ATB_HOME_PATH=${ATB_HOME_PATH:-<unset>}"
    echo "[Ascend env] after login source: PATH=${PATH:-}"
    echo "[Ascend env] after login source: LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
    python3 -c "import torch; print(torch.compiled_with_cxx11_abi())" >/tmp/ascend_torch_abi_probe.out 2>/tmp/ascend_torch_abi_probe.err
    probe_rc=$?
    echo "[Ascend env] torch ABI probe rc=${probe_rc}"
    python3 -c "from pathlib import Path; print(\"[Ascend env] torch ABI probe stdout repr=\", repr(Path(\"/tmp/ascend_torch_abi_probe.out\").read_text())); print(\"[Ascend env] torch ABI probe stderr repr=\", repr(Path(\"/tmp/ascend_torch_abi_probe.err\").read_text()))"
  ' || true
fi
