#!/usr/bin/env bash
# Build FlagCX against the final Hygon torch stack, then run the post-install
# profile selected by the target repository's test_env.<backend>.
#
# To add another stack:
#   1. add a setup_<profile>() function below;
#   2. add an arm to the PROFILE case statement;
#   3. set POST_INSTALL_PROFILE in the target repo's test_env.<backend>.
# Keep repository-name special cases out of this file; use profiles instead.
set -euo pipefail

TARGET="${TARGET_DIR:-}"
if [ -z "${TARGET}" ]; then
  echo "[Post-install script] TARGET_DIR is required" >&2
  exit 1
fi
ROOT="${GITHUB_WORKSPACE:-$PWD}"
WRAPPER="${ROOT}/${TARGET}/tools/flagcx_wrapper/build_flagcx_wheel.sh"
PROFILE="${POST_INSTALL_PROFILE:-hygon-generic}"

build_flagcx() {
  if [ ! -x "${WRAPPER}" ]; then
    echo "[Post-install script] no FlagCX wrapper at ${WRAPPER}"
    return 0
  fi
  local wheel
  FLAGCX_OUT="$(mktemp -d)"
  trap 'rm -rf "${FLAGCX_OUT}"' EXIT
  FLAGCX_ADAPTOR=du "${WRAPPER}" "${FLAGCX_OUT}"
  wheel="$(ls "${FLAGCX_OUT}"/flagcx-*.whl | head -1)"
  echo "[Post-install script] installing ${wheel}"
  uv pip install --python "$(command -v python3)" --force-reinstall --no-deps "${wheel}"
}

check_torch_hygon() {
  local label="${1:-Hygon}"
  python3 - "${label}" <<'PY'
import sys
import torch

assert torch.cuda.is_available(), "torch.cuda is not available"
assert torch.cuda.device_count() > 0, "no Hygon device visible"
print(f"[Post-install script] {sys.argv[1]}: Hygon torch runtime OK")
PY
}

setup_generic_hygon() {
  check_torch_hygon "hygon-generic"
}

setup_vllm_hygon() {
  check_torch_hygon "hygon-vllm"
  python3 - <<'PY'
import flagcx
import torch
import vllm
import vllm_hcu
from vllm.platforms import current_platform

assert torch.cuda.is_available(), "torch.cuda is not available"
assert torch.cuda.device_count() > 0, "no Hygon device visible"
assert "hcu" in type(current_platform).__module__.lower(), type(current_platform)
assert "vllm_hcu" in type(current_platform).__module__, type(current_platform)
print(f"[Post-install script] Hygon vLLM platform OK: {type(current_platform).__module__}")
PY
}

build_flagcx

case "${PROFILE}" in
  hygon-generic)
    setup_generic_hygon
    ;;
  hygon-vllm)
    setup_vllm_hygon
    ;;
  *)
    echo "[Post-install script] unknown POST_INSTALL_PROFILE=${PROFILE}" >&2
    exit 1
    ;;
esac
