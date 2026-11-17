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
# The Hygon torch wheel forces C10_USE_GFLAGS / C10_USE_GLOG in its CMake
# macros, so every translation unit that includes torch/extension.h needs
# gflags/gflags.h and glog/stl_logging.h. Neither ships with the image and
# neither exists on PyPI.
set -eo pipefail

# The DTK runtime only executes code objects older than v4, while aicc emits v4
# by default, which fails silently as a kernel that writes nothing.
export AITER_CODE_OBJECT_VERSION=5

if [ "$(id -u)" -ne 0 ]; then
  echo "[Install script] ERROR: must run as root to install system packages"
  exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
  echo "[Install script] ERROR: this image has no apt-get"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends libgflags-dev libgoogle-glog-dev

if [ -n "${GITHUB_ENV:-}" ]; then
  echo "AITER_CODE_OBJECT_VERSION=${AITER_CODE_OBJECT_VERSION}" >> "$GITHUB_ENV"
fi
