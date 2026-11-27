#!/bin/bash
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
# Verify a vLLM app image (optest-managed).
#
# Ported from build-infra packaging/vllm/verify/verify-vllm-backend.sh.
# Adaptations for optest:
#   - Reads container/configs.yaml (not build-infra configs.yaml).
#   - Device flags read from .github/configs/backends/<BACKEND>.yaml.
#   - Registry host: harbor.baai.ac.cn (no separate build-config.yml).
#   - No from-scratch install path; --app-image mode is primary.
#
# Usage:
#   container/verify/verify-vllm-backend.sh \
#       --app-image <image> \
#       --vendor <vendor> \
#       --backend <backend> \
#       --stack <stack-key> \
#       [--model-path <path>] \
#       [--serve-timeout <seconds>] \
#       [--skip-serve] \
#       [--vllm-version <version>] \
#       [--plugin-fl-version <version>] \
#       [--flagcx-version <version>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

log_info()  { echo "[verify-vllm] INFO  $*"; }
log_step()  { echo ""; echo "[verify-vllm] ===== $* ====="; }
log_warn()  { echo "[verify-vllm] WARN  $*" >&2; }
log_error() { echo "[verify-vllm] ERROR $*" >&2; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
APP_IMAGE=""
VENDOR=""
BACKEND=""
STACK=""
MODEL_PATH="/data/models/Qwen3-4B"
SERVE_TIMEOUT=1800
SKIP_SERVE=false
VLLM_VERSION="0.32.0"
PLUGIN_FL_VERSION=""
FLAGCX_VERSION=""
EXPECT_NATIVE_PLUGIN=false
TP_SIZE=""

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --app-image)         APP_IMAGE="$2";         shift 2 ;;
        --vendor)            VENDOR="$2";             shift 2 ;;
        --backend)           BACKEND="$2";            shift 2 ;;
        --stack)             STACK="$2";              shift 2 ;;
        --model-path)        MODEL_PATH="$2";         shift 2 ;;
        --serve-timeout)     SERVE_TIMEOUT="$2";      shift 2 ;;
        --skip-serve)        SKIP_SERVE=true;         shift ;;
        --vllm-version)      VLLM_VERSION="$2";       shift 2 ;;
        --plugin-fl-version) PLUGIN_FL_VERSION="$2";  shift 2 ;;
        --flagcx-version)    FLAGCX_VERSION="$2";     shift 2 ;;
        --expect-native-plugin) EXPECT_NATIVE_PLUGIN=true; shift ;;
        --tp-size)           TP_SIZE="$2";            shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${APP_IMAGE}" ]]; then
    log_error "--app-image is required"
    exit 2
fi
if [[ -z "${VENDOR}" ]]; then
    log_error "--vendor is required"
    exit 2
fi
if [[ -z "${BACKEND}" ]]; then
    log_error "--backend is required"
    exit 2
fi

VENDOR_BACKEND="${VENDOR}-${BACKEND}"
CONTAINER="flagos-verify-vllm-runtime-$$"
APP_CONTAINER="flagos-verify-vllm-app-$$"
WORK_DIR="/tmp/flagos-verify-vllm-$$"
SERVE_CONTAINER="${APP_CONTAINER}"

# ---------------------------------------------------------------------------
# Determine runtime image from configs.yaml or backend YAML
# ---------------------------------------------------------------------------
RUNTIME_IMAGE=$(python3 -c "
import yaml, sys
with open('${REPO_ROOT}/container/configs.yaml') as f:
    cfg = yaml.safe_load(f)
stacks = cfg.get('apps', {}).get('vllm', {}).get('stacks', {})
stack = stacks.get('${STACK}', {})
print(stack.get('base_image', ''))
")

if [[ -z "${RUNTIME_IMAGE}" ]]; then
    log_error "Could not find base_image for stack '${STACK}' in container/configs.yaml"
    exit 2
fi

# Device run flags from backend YAML.
RUN_FLAGS=$(python3 -c "
import yaml, sys
backend_file = '${REPO_ROOT}/.github/configs/backends/${BACKEND}.yaml'
try:
    with open(backend_file) as f:
        cfg = yaml.safe_load(f)
    flags = cfg.get('docker_run_flags', '') or ''
    print(flags)
except FileNotFoundError:
    print('')
")

if [[ " ${RUN_FLAGS} " != *" --network "* ]]; then
    RUN_FLAGS="${RUN_FLAGS} --network host"
fi

# Proxy: relay name-only so values never appear in the log.
PROXY_ENV_ARGS=()
for _v in http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY; do
    if [[ -n "${!_v:-}" ]]; then
        PROXY_ENV_ARGS+=(-e "${_v}")
    fi
done
PROXY_RELAY_DESC="$( [[ ${#PROXY_ENV_ARGS[@]} -gt 0 ]] && echo "relaying ${#PROXY_ENV_ARGS[@]} proxy vars" || echo "none" )"

# ---------------------------------------------------------------------------
# Print header
# ---------------------------------------------------------------------------
echo "========================================"
echo "vLLM Backend Verification"
echo "========================================"
echo "Vendor-Backend:   ${VENDOR_BACKEND}"
echo "Runtime Image:    ${RUNTIME_IMAGE}"
echo "App Image:        ${APP_IMAGE}"
echo "vLLM Version:     ${VLLM_VERSION}+flagos"
echo "Plugin Version:   ${PLUGIN_FL_VERSION:-<none>}"
echo "Model Path:       ${MODEL_PATH}"
echo "Proxy relay:      ${PROXY_RELAY_DESC}"
echo ""

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    log_info "Cleaning up containers..."
    docker rm -f "${CONTAINER}" 2>/dev/null || true
    docker rm -f "${APP_CONTAINER}" 2>/dev/null || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Step 1: Start runtime container
# ---------------------------------------------------------------------------
log_step "Step 1: Starting runtime container"

mkdir -p "${WORK_DIR}"
docker rm -f "${CONTAINER}" 2>/dev/null || true

MODEL_MOUNT=""
[[ -d "${MODEL_PATH}" ]] && MODEL_MOUNT="-v ${MODEL_PATH}:${MODEL_PATH}:ro"

# shellcheck disable=SC2086
docker run -d --name "${CONTAINER}" \
    ${RUN_FLAGS} \
    -v "${WORK_DIR}:${WORK_DIR}" \
    ${MODEL_MOUNT} \
    "${RUNTIME_IMAGE}" \
    sleep infinity

log_info "Container started: ${CONTAINER}"

# ---------------------------------------------------------------------------
# App-image mode: matrix snapshot + import check, then serve
# ---------------------------------------------------------------------------
log_step "App-image mode: verifying ${APP_IMAGE}"

docker rm -f "${APP_CONTAINER}" 2>/dev/null || true

# torch_npu only on ascend; watching it on every vendor produces NOT_INSTALLED
# noise. Flag it exactly where it can exist.
WATCH_PKGS="torch triton flagtree flag_gems numpy"
[[ "${VENDOR}" == "ascend" ]] && WATCH_PKGS="torch torch_npu triton flagtree flag_gems numpy"

# flagtree and triton live in side dirs (/opt/flagtree, /opt/triton), not
# site-packages; importlib.metadata can't see them. Read their dist-info directly.
snapshot() {
    local cid="$1" pkg ver dir
    shift
    for pkg in "$@"; do
        case "$pkg" in
            flagtree) dir=/opt/flagtree ;;
            triton)   dir=/opt/triton ;;
            *)        dir="" ;;
        esac
        if [[ -n "$dir" ]]; then
            ver=$(docker exec "${cid}" bash -c \
                "ls -d ${dir}/*.dist-info 2>/dev/null | head -1 | xargs -r basename | sed 's/\.dist-info\$//; s/^[^-]*-//'" \
                2>/dev/null || true)
            [[ -n "$ver" ]] || ver="NOT_INSTALLED"
        else
            ver=$(docker exec "${cid}" python3 -c \
                "import importlib.metadata as m; print(m.version('${pkg}'))" \
                2>/dev/null || echo "NOT_INSTALLED")
        fi
        echo "${pkg}=${ver}"
    done
}

log_step "BEFORE snapshot (runtime image)"
# shellcheck disable=SC2086
snapshot "${CONTAINER}" ${WATCH_PKGS} | tee "${WORK_DIR}/before.txt"

# shellcheck disable=SC2086
docker run -d --name "${APP_CONTAINER}" \
    ${RUN_FLAGS} \
    ${MODEL_MOUNT} \
    "${APP_IMAGE}" \
    sleep infinity
log_info "App container started: ${APP_CONTAINER}"

log_step "AFTER snapshot (app image)"
# shellcheck disable=SC2086
snapshot "${APP_CONTAINER}" ${WATCH_PKGS} | tee "${WORK_DIR}/after.txt"
# flagcx is added by the app image — recorded for the log, never diffed.
snapshot "${APP_CONTAINER}" flagcx 2>/dev/null || true

log_step "Comparing critical-package matrix"
if diff -u "${WORK_DIR}/before.txt" "${WORK_DIR}/after.txt"; then
    log_info "✅ Matrix unchanged: the app image did not overwrite runtime packages"
else
    log_error "❌ Matrix changed: the app image overwrote a runtime package"
    exit 1
fi

log_step "Import check (vllm + vllm_fl)"
if docker exec "${APP_CONTAINER}" bash -c '
    python3 - <<PY
import importlib.metadata as m
def v(p):
    try: return m.version(p)
    except Exception: return "ABSENT"
import vllm, vllm_fl
print("vllm", vllm.__version__)
print("vllm-plugin-fl", v("vllm-plugin-fl"))
print("flaggems_vllm", v("flaggems_vllm"))
PY'; then
    log_info "✅ vllm + vllm_fl import OK"
else
    log_error "❌ import failed"
    exit 1
fi

if [[ "${EXPECT_NATIVE_PLUGIN}" == "true" ]]; then
    log_step "Native plugin check (vllm_fl._C weak_ref_tensor on CUDA)"
    if docker exec "${APP_CONTAINER}" bash -c '
        python3 -c "import vllm_fl._C, torch; d = torch._C._dispatch_dump(\"_C::weak_ref_tensor\"); print(d); import sys; sys.exit(0 if any(\"CUDA\" in line for line in d.splitlines()) else 1)"'; then
        log_info "✅ vllm_fl._C imports and registers weak_ref_tensor on CUDA"
    else
        log_error "❌ native plugin missing or weak_ref_tensor not CUDA-registered"
        exit 1
    fi
fi

log_step "Import check (flagcx)"
flagcx_ver=$(docker exec "${APP_CONTAINER}" python3 -c \
    'import importlib.metadata as m; print(m.version("flagcx"))' 2>/dev/null || true)
if [[ -n "${flagcx_ver}" ]]; then
    if docker exec "${APP_CONTAINER}" bash -c \
        'python3 -c "import flagcx.api; print(\"flagcx.api ok\")"'; then
        log_info "✅ flagcx ${flagcx_ver} present, flagcx.api imports OK"
    else
        log_error "❌ flagcx ${flagcx_ver} present but flagcx.api failed to import"
        exit 1
    fi
elif [[ -n "${FLAGCX_VERSION}" ]]; then
    log_error "❌ --flagcx-version ${FLAGCX_VERSION} requested but no flagcx distribution in image"
    exit 1
else
    log_info "flagcx not in this image — not applicable for this vendor"
fi

log_info "App-image import check PASSED — proceeding to serve test"

# ---------------------------------------------------------------------------
# Step 2: Serve test (Qwen3-4B, port 8031, 1× /v1/completions)
# ---------------------------------------------------------------------------
if [[ "${SKIP_SERVE}" == true ]]; then
    log_info "Serve test skipped (--skip-serve)"
else
    log_step "Step 2: Testing vllm serve (Qwen3-4B, /v1/completions)"

    if [[ ! -d "${MODEL_PATH}" ]]; then
        log_error "Model not found at: ${MODEL_PATH}"
        log_error "Pass --skip-serve for manual debugging only."
        exit 1
    fi

    EXPECT_ANCHOR=""
    case "${MODEL_PATH}" in *Qwen3-4B*) EXPECT_ANCHOR="Paris" ;; esac

    TP_ARG=""
    [[ -n "${TP_SIZE}" ]] && TP_ARG="--tensor-parallel-size ${TP_SIZE}"

    docker exec "${SERVE_CONTAINER}" bash -c "
        export VLLM_PLUGINS=fl
        export VLLM_FL_DISPATCH_DEBUG=1
        export SERVE_TIMEOUT=${SERVE_TIMEOUT}

        echo 'Starting serve...'
        vllm serve '${MODEL_PATH}' \
            --port 8031 \
            --gpu-memory-utilization 0.4 \
            --enforce-eager \
            --trust-remote-code \
            --max-model-len 2048 \
            ${TP_ARG} \
            > /tmp/vllm-serve.log 2>&1 &

        SERVE_PID=\$!
        echo 'Waiting for serve to become ready...'
        ready=0
        for i in \$(seq 1 \$((SERVE_TIMEOUT / 5))); do
            if ! kill -0 \${SERVE_PID} 2>/dev/null; then
                echo 'serve process exited during startup'
                break
            fi
            if grep -qE 'Application startup complete|Uvicorn running' /tmp/vllm-serve.log 2>/dev/null; then
                ready=1
                echo \"serve ready after ~\$((i*5))s\"
                break
            fi
            sleep 5
        done
        if [ \"\$ready\" != \"1\" ]; then
            echo 'serve did not become ready; last 60 lines of log:'
            tail -60 /tmp/vllm-serve.log 2>/dev/null || true
            kill \${SERVE_PID} 2>/dev/null || true
            exit 1
        fi

        echo ''
        echo 'Sending completion request...'
        python3 - <<'PY'
import json, sys, urllib.request

url = 'http://127.0.0.1:8031/v1/completions'
payload = json.dumps({
    'model': '${MODEL_PATH}',
    'prompt': 'The capital of France is',
    'max_tokens': 16,
    'temperature': 0,
}).encode()
req = urllib.request.Request(url, data=payload,
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=120) as resp:
    body = json.loads(resp.read())

text = body['choices'][0]['text']
print('Response text:', repr(text))
anchor = '${EXPECT_ANCHOR}'
if anchor and anchor.lower() not in text.lower():
    print('ERROR: expected anchor', repr(anchor), 'not found in response', file=sys.stderr)
    sys.exit(1)
print('SERVE_TEST_PASSED')
PY
    "
    log_info "✅ Serve test passed"
fi

log_info "✅ Verification complete: ${APP_IMAGE}"
