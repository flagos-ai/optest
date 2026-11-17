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
# The MUSA vendor build needs three patches on this image:
#   1. install the C++ toolchain target builds compile with and the MKL the
#      vendor torch was linked against;
#   2. put the MUSA runtime libraries in the dynamic loader cache, or
#      `import torch` fails;
#   3. prepend the MUSA clang device placement-new declarations CUB pulls in to
#      torchada's stable-ABI box header, or every including translation unit
#      fails with "no matching 'operator new'".
set -eo pipefail

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
apt-get install -y --no-install-recommends g++-12 libstdc++-12-dev intel-mkl

PYTHON=python3
MUSA=""
for candidate in \
  "${MUSA_HOME:+${MUSA_HOME}/lib}" \
  "${MUSA_PATH:+${MUSA_PATH}/lib}" \
  "${MUSA_ROOT:+${MUSA_ROOT}/lib}" \
  "$(dirname "$(dirname "$(command -v mcc || echo /nonexistent/bin/mcc)")")/lib"; do
  if [ -n "${candidate}" ] && ls "${candidate}"/libmufft.so* >/dev/null 2>&1; then
    MUSA="${candidate}"
    break
  fi
done
if [ -z "${MUSA}" ]; then
  echo "[Install script] ERROR: no MUSA runtime library directory found"
  exit 1
fi
echo "[Install script] MUSA runtime libraries from ${MUSA}"
echo "${MUSA}" > /etc/ld.so.conf.d/flagos-musa.conf
ldconfig

find_box() {
  "${PYTHON}" -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('torchada')
print(pathlib.Path(spec.origin).parent / 'csrc' / 'stable_compat' / 'torchada_stable_box.h' if spec and spec.origin else '')
"
}

BOX=$(find_box)
if [ -z "${BOX}" ]; then
  "${PYTHON}" -m pip install "torchada>=0.1.76"
  BOX=$(find_box)
fi
HEADER="$(mcc -print-resource-dir)/include/cuda_wrappers/new"
if [ ! -f "${BOX:-/nonexistent}" ] || [ ! -f "${HEADER}" ]; then
  echo "[Install script] ERROR: cannot patch torchada box (${BOX:-<missing>}) with ${HEADER}"
  exit 1
fi
LINE="#include \"${HEADER}\""
if ! grep -qxF "${LINE}" "${BOX}"; then
  echo "[Install script] prepending ${HEADER} to ${BOX}"
  {
    echo "${LINE}"
    sed -e '/^#include <new>$/d' -e '/cuda_wrappers\/new/d' "${BOX}"
  } > "${BOX}.new"
  mv "${BOX}.new" "${BOX}"
fi
