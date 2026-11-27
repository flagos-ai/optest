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

# The runtime image ships FlagCX for the old torch; suppress backend
# autoload until post_install rebuilds it against the final torch.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD}" >> "${GITHUB_ENV}"
fi

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
apt-get install -y --no-install-recommends g++-12 libstdc++-12-dev intel-mkl libz3-4 libz3-dev libgflags-dev libgoogle-glog-dev

PYTHON=python3

# The full MUSA 0.32 stack is needed only by the FlagGems-vllm target.
# Other MThreads repositories keep the torch/accelerator stack supplied by
# their own images.
TARGET_NAME="${TARGET_REPOSITORY:-}"
TARGET_NAME="${TARGET_NAME##*/}"
if [ -z "${TARGET_NAME}" ]; then
  TARGET_NAME="${TARGET_DIR:-}"
  TARGET_NAME="${TARGET_NAME##*/}"
fi
if [ -z "${TARGET_NAME}" ]; then
  TARGET_NAME="FlagGems-vllm"
fi
IS_FLAGGEMS_VLLM=0
if [ "${TARGET_NAME}" = "FlagGems-vllm" ]; then
  IS_FLAGGEMS_VLLM=1
  echo "[Install script] installing FlagGems-vllm MUSA 0.32 stack"
else
  echo "[Install script] target ${TARGET_NAME}; keeping its image-provided torch stack"
fi
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

# The vendor torch build also links MKL from /flagos/lib.
if [ -d /flagos/lib ]; then
  echo /flagos/lib > /etc/ld.so.conf.d/flagos-lib.conf
  ldconfig
fi

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

if [ "${IS_FLAGGEMS_VLLM}" = 1 ]; then
# The image ships only the base runtime; install the full MUSA 0.32 kernel
# stack here so the target repository and its tests see the same environment.
VENDOR_INDEX="https://dl.mthreads.com/repo/api/pypi/pypi/simple"
VENDOR_PACKAGES=(
  "torch==2.11.0.post1+musa5.2.0"
  "torch_musa==2.11.0.post1+musa5.2.0"
  "torchvision==0.26.0.post1+musa5.2.0"
  "torchaudio==2.11.0+musa5.2.0"
  "mate==0.2.8.post1"
  "mate-mubin==0.2.8.post1"
  "flash_attn_3==0.2.8.post1+musa"
  "flash_mla==0.2.8.post1+musa"
  "deep_gemm==0.2.8.post1+musa"
  "flashinfer-python==0.2.8.post1+musa"
  "sageattention==0.2.8.post1+musa"
  "deep_ep==1.1.0+musa5.2.0torch2.11.0.post2"
  "tilelang_musa==0.1.12+musa.5"
  "triton==3.2.0"
  "apache-tvm-ffi==0.1.11.post1+musa.1"
  "torch-c-dlpack-ext==0.1.5"
)
"${PYTHON}" -m pip install --disable-pip-version-check --no-deps \
  --index-url "${VENDOR_INDEX}" "${VENDOR_PACKAGES[@]}"
else
  echo "[Install script] skipping FlagGems-vllm MUSA 0.32 stack for ${TARGET_NAME}"
fi

