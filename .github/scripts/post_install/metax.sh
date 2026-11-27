#!/usr/bin/env bash
# Build FlagCX against the final MetaX torch stack, then run the post-install
# profile selected by the target repository's test_env.<backend>.
#
# To add another stack:
#   1. add a setup_<profile>() function below;
#   2. add an arm to the PROFILE case statement;
#   3. set POST_INSTALL_PROFILE in the target repo's test_env.<backend>.
# No repository-name checks belong in this file.
set -euo pipefail

TARGET="${TARGET_DIR:-}"
if [ -z "${TARGET}" ]; then
  echo "[Post-install script] TARGET_DIR is required" >&2
  exit 1
fi
ROOT="${GITHUB_WORKSPACE:-$PWD}"
WRAPPER="${ROOT}/${TARGET}/tools/flagcx_wrapper/build_flagcx_wheel.sh"
PROFILE="${POST_INSTALL_PROFILE:-metax-generic}"

check_torch_metax() {
  local label="${1:-MetaX}"
  python3 - "${label}" <<'PY'
import sys
import torch

assert torch.cuda.is_available(), "torch.cuda is not available"
assert torch.cuda.device_count() > 0, "no MetaX device visible"
print(f"[Post-install script] {sys.argv[1]}: MetaX torch runtime OK")
PY
}

build_flagcx() {
  if [ ! -x "${WRAPPER}" ]; then
    echo "[Post-install script] no FlagCX wrapper at ${WRAPPER}"
    return 0
  fi
  local wheel
  FLAGCX_OUT="$(mktemp -d)"
  trap 'rm -rf "${FLAGCX_OUT}"' EXIT
  FLAGCX_ADAPTOR=metax DEVICE_HOME=/opt/maca CCL_HOME=/opt/maca USE_METAX=1 \
    "${WRAPPER}" "${FLAGCX_OUT}"
  wheel="$(ls "${FLAGCX_OUT}"/flagcx-*.whl | head -1)"
  echo "[Post-install script] installing ${wheel}"
  uv pip install --python "$(command -v python3)" --force-reinstall --no-deps "${wheel}"
}

setup_generic_metax() {
  check_torch_metax "metax-generic"
}

setup_vllm_metax() {
  check_torch_metax "metax-vllm"
  python3 -m pip uninstall -y torchaudio || true
  python3 -c 'import vllm; assert vllm.__version__.split("+")[0] == "0.29.0"' 2>/dev/null \
    || python3 -m pip install --no-cache-dir --no-deps "vllm @ https://files.pythonhosted.org/packages/ca/09/7f79450e21bd1c2a0897ab946a544816a4f7e04c54f0ca49849b14b12d6a/vllm-0.29.0-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=09d48617fc2be9c6cdcd5db480651ab0d84817b257204f2cc2e3ecbb70bbb635"
  python3 -c 'import xgrammar' 2>/dev/null \
    || python3 -m pip install --no-cache-dir --no-deps "xgrammar==0.2.8"

  python3 - <<'PY'
import importlib.metadata as md
import torch
import vllm
import vllm_metax
import flagcx
from vllm.platforms import current_platform

assert torch.cuda.is_available(), "torch.cuda is not available"
assert torch.cuda.device_count() > 0, "no MetaX device visible"
assert vllm.__version__.split("+")[0] == "0.29.0", vllm.__version__
assert vllm_metax.__version__.startswith("0.29.0"), vllm_metax.__version__
assert type(current_platform).__module__.startswith("vllm_metax"), type(current_platform)
groups = {ep.group for ep in md.distribution("vllm-metax").entry_points}
assert "vllm.platform_plugins" in groups, groups
assert "vllm.general_plugins" in groups, groups
print("[Post-install script] MetaX vLLM platform OK")
PY
}

build_flagcx

case "${PROFILE}" in
  metax-generic)
    setup_generic_metax
    ;;
  metax-vllm)
    setup_vllm_metax
    ;;
  *)
    echo "[Post-install script] unknown POST_INSTALL_PROFILE=${PROFILE}" >&2
    exit 1
    ;;
esac
