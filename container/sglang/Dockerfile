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

# ============================================================
# FlagOS sglang app image (optest-managed)
#
# Build args drive all per-vendor configuration; this single
# Dockerfile builds every sglang stack listed in configs.yaml.
#
# Two-layer model:
#   baked:    FlagTree / FlagGems / plugin-fl baked here.
#   hot-patch: ops-test.yaml upgrade step reinstalls a newer
#              daily at test time without a full image rebuild.
#
# Torch / runtime protection:
#   /tmp/protect.txt generated from base constraints + live freeze.
#   Every uv pip install passes -c /tmp/protect.txt.
#
# scipy guard:
#   sglang's unconstrained scipy dep resolves 1.18+ (numpy>=2.0.0)
#   since 2026-08. scipy<1.18 rides the same uv pip install line to
#   prevent the runtime numpy pin from being upgraded.
#
# PYTHONPATH:
#   /opt/triton prepended so xgrammar resolution sees the vendor
#   triton dist-info rather than pulling a generic triton wheel.
#
# Proxy:
#   http_proxy / https_proxy / no_proxy are ARG (not ENV).
# ============================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# --- Build arguments ------------------------------------------

ARG SGLANG_VERSION=0.5.18
ARG FLAGOS_PYPI=""
ARG FLAGOS_PYPI_DAILY="https://resource.flagos.net/repository/flagos-pypi-daily/simple"
ARG EXTRA_PYPI="https://mirrors.cloud.tencent.com/pypi/simple"
ARG SGLANG_FL_VERSION=""
ARG SGLANG_FL_REPO="https://github.com/flagos-ai/sglang-plugin-FL.git"
ARG SGL_KERNEL_SHIM_VERSION=""
ARG FLAGTREE_VERSION=""
ARG FLAGTREE_PACKAGE="flagtree"
ARG FLAGGEMS_VERSION=""
ARG APP_ENV=""
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=

# --- Runtime switches baked as ENV ----------------------------
# Must be exported before every sglang install and import to prevent
# flashinfer gate failures and version-check aborts.
ENV SGLANG_IS_FLASHINFER_AVAILABLE=false \
    TORCHINDUCTOR_COMPILE_THREADS=1 \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# --- Constraint file (torch / runtime protection) -------------

RUN set -e; \
    protect=/tmp/protect.txt; \
    if [ -f /etc/flagos/flaggems-constraints.txt ]; then \
        cp /etc/flagos/flaggems-constraints.txt "${protect}"; \
    else \
        : > "${protect}"; \
    fi; \
    uv pip freeze --python /flagos/bin/python >> "${protect}" 2>/dev/null || true

# --- FlagTree (staging-swap) ----------------------------------

RUN --mount=type=cache,id=flagos-uv-${FLAGTREE_PACKAGE},target=/root/.cache/uv \
    set -e; \
    if [ -n "${FLAGTREE_VERSION}" ] && [ -n "${FLAGTREE_PACKAGE}" ]; then \
        rm -rf /opt/flagtree.new; \
        uv pip install \
            --python /flagos/bin/python \
            --index-url "${FLAGOS_PYPI_DAILY}" \
            --extra-index-url "${EXTRA_PYPI}" \
            --no-deps \
            --target /opt/flagtree.new \
            -c /tmp/protect.txt \
            "${FLAGTREE_PACKAGE}==${FLAGTREE_VERSION}"; \
        rm -rf /opt/flagtree; \
        mv /opt/flagtree.new /opt/flagtree; \
    fi

ENV PYTHONPATH=/opt/flagtree:${PYTHONPATH}

# --- FlagGems --------------------------------------------------

RUN --mount=type=cache,id=flagos-uv-flaggems,target=/root/.cache/uv \
    if [ -n "${FLAGGEMS_VERSION}" ]; then \
        uv pip install \
            --python /flagos/bin/python \
            --index-url "${FLAGOS_PYPI_DAILY}" \
            --extra-index-url "${EXTRA_PYPI}" \
            --no-deps \
            -c /tmp/protect.txt \
            "flag-gems==${FLAGGEMS_VERSION}"; \
    fi

# --- sglang (repacked wheel + scipy guard) --------------------
#
# PYTHONPATH prepend ensures xgrammar's triton resolution sees the vendor
# triton dist-info in /opt/triton rather than pulling generic triton.
# scipy<1.18 rides the same install line: sglang's unconstrained scipy
# dep resolves 1.18+ (numpy>=2.0.0) since 2026-08 and would upgrade the
# runtime numpy pin when resolved separately.
RUN --mount=type=cache,id=flagos-uv-sglang,target=/root/.cache/uv \
    PYTHONPATH=/opt/triton${PYTHONPATH:+:${PYTHONPATH}} \
    uv pip install \
        --python /flagos/bin/python \
        --index-url "${FLAGOS_PYPI}" \
        --extra-index-url "${EXTRA_PYPI}" \
        -c /tmp/protect.txt \
        "sglang==${SGLANG_VERSION}+flagos" \
        "scipy<1.18"

# --- sgl-kernel-shim ------------------------------------------
#
# sglang imports sgl_kernel unconditionally; the repacked +flagos wheel
# ships no sgl-kernel dep. The shim provides the import-face; version
# tracks the sglang version on the vendor index.
RUN --mount=type=cache,id=flagos-uv-sgl-kernel-shim,target=/root/.cache/uv \
    if [ -n "${SGL_KERNEL_SHIM_VERSION}" ]; then \
        uv pip install \
            --python /flagos/bin/python \
            --index-url "${FLAGOS_PYPI}" \
            --extra-index-url "${EXTRA_PYPI}" \
            -c /tmp/protect.txt \
            "sgl-kernel-shim==${SGL_KERNEL_SHIM_VERSION}"; \
    fi

# --- sglang-plugin-fl (clone + build in-container) ------------
#
# Not on any index: clone at SGLANG_FL_VERSION and build pure-Python wheel
# inside the container the same way the from-scratch verify path does.
RUN --mount=type=cache,id=flagos-uv-sglang-fl,target=/root/.cache/uv \
    if [ -n "${SGLANG_FL_VERSION}" ]; then \
        rm -rf /tmp/sglang-plugin-FL /tmp/sglang-fl-wheels; \
        git clone --depth 1 --branch "${SGLANG_FL_VERSION}" \
            "${SGLANG_FL_REPO}" /tmp/sglang-plugin-FL; \
        mkdir -p /tmp/sglang-fl-wheels; \
        cd /tmp/sglang-plugin-FL && \
        pip wheel . --no-deps \
            --index-url "${FLAGOS_PYPI}" \
            --extra-index-url "${EXTRA_PYPI}" \
            -w /tmp/sglang-fl-wheels; \
        cd /tmp && \
        uv pip install \
            --python /flagos/bin/python \
            --index-url "${FLAGOS_PYPI}" \
            --extra-index-url "${EXTRA_PYPI}" \
            -c /tmp/protect.txt \
            /tmp/sglang-fl-wheels/sglang_fl-*.whl; \
        rm -rf /tmp/sglang-plugin-FL /tmp/sglang-fl-wheels; \
    fi

# --- App env ---------------------------------------------------

RUN if [ -n "${APP_ENV}" ]; then \
        printf '%s\n' "${APP_ENV}" | sed 's/^/export /' \
            > /etc/profile.d/app_env.sh; \
    fi

# --- Serve launcher -------------------------------------------

COPY sglang/sglang-serve /usr/local/bin/sglang-serve
RUN chmod +x /usr/local/bin/sglang-serve

WORKDIR /workspace
CMD ["sglang-serve", "--model-path", "/data/models/Qwen3-0.6B", "--port", "8032", \
    "--mem-fraction-static", "0.6", "--trust-remote-code", \
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
