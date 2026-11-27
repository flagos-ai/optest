#!/usr/bin/env bash
# Build FlagCX against the final MThreads torch stack, then run the post-install
# profile selected by the target repository's test_env.<backend>.
#
# To add another stack:
#   1. add a setup_<profile>() function below;
#   2. add an arm to the PROFILE case statement;
#   3. set POST_INSTALL_PROFILE in the target repo's test_env.<backend>.
# No repository-name checks belong in this file.
set -euo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

TARGET="${TARGET_DIR:-}"
if [ -z "${TARGET}" ]; then
  echo "[Post-install script] TARGET_DIR is required" >&2
  exit 1
fi
ROOT="${GITHUB_WORKSPACE:-$PWD}"
WRAPPER="${ROOT}/${TARGET}/tools/flagcx_wrapper/build_flagcx_wheel.sh"
PROFILE="${POST_INSTALL_PROFILE:-musa-generic}"

build_flagcx() {
  if [ ! -x "${WRAPPER}" ]; then
    echo "[Post-install script] no FlagCX wrapper at ${WRAPPER}"
    return 0
  fi
  local wheel
  FLAGCX_OUT="$(mktemp -d)"
  trap 'rm -rf "${FLAGCX_OUT}"' EXIT
  FLAGCX_ADAPTOR=musa "${WRAPPER}" "${FLAGCX_OUT}"
  wheel="$(ls "${FLAGCX_OUT}"/flagcx-*.whl | head -1)"
  echo "[Post-install script] installing ${wheel}"
  uv pip install --python "$(command -v python3)" --force-reinstall --no-deps "${wheel}"
}

check_torch_musa() {
  local label="${1:-MThreads}"
  python3 - "${label}" <<'PY'
import sys
import torch_musa
import torch

assert torch.musa.is_available(), "torch.musa is not available"
print(f"[Post-install script] {sys.argv[1]}: MUSA torch runtime OK")
PY
}

setup_generic_musa() {
  check_torch_musa "musa-generic"
}

setup_vllm_musa() {
  check_torch_musa "musa-vllm"
  python3 - <<'PY'
import importlib.metadata as md
import torch_musa
import torch
import vllm
import vllm_musa
from vllm.platforms import current_platform

assert torch.musa.is_available(), "torch.musa is not available"
assert vllm.__version__.split("+")[0] == "0.32.0", vllm.__version__
assert vllm_musa.__version__ == "0.1.32", vllm_musa.__version__
assert current_platform.is_musa(), type(current_platform)

groups = {ep.group for ep in md.distribution("vllm-musa").entry_points}
assert "vllm.platform_plugins" in groups, groups
assert "vllm.general_plugins" in groups, groups
print("[Post-install script] MUSA vLLM platform OK")
PY
}

build_flagcx

case "${PROFILE}" in
  musa-generic)
    setup_generic_musa
    ;;
  musa-vllm)
    setup_vllm_musa
    ;;
  *)
    echo "[Post-install script] unknown POST_INSTALL_PROFILE=${PROFILE}" >&2
    exit 1
    ;;
esac
