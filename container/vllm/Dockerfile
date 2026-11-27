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
# FlagOS vLLM app image (optest-managed)
#
# Build args drive all per-vendor configuration; this single
# Dockerfile builds every vllm stack listed in configs.yaml.
#
# Two-layer model:
#   baked:    FlagTree / FlagGems / plugin-fl baked here.
#   hot-patch: ops-test.yaml upgrade step reinstalls a newer
#              daily at test time without a full image rebuild.
#
# Torch / runtime protection:
#   /etc/flagos/flaggems-constraints.txt written by the runtime
#   base pins torch, triton, and numpy. We generate
#   /tmp/protect.txt (constraints + live freeze) and pass
#   -c /tmp/protect.txt to every uv pip install so those
#   packages are never replaced.
#
# Proxy:
#   http_proxy / https_proxy / no_proxy are ARG (not ENV) so
#   they drive network access during build but are never baked
#   into the pushed image layer.
# ============================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# --- Build arguments ------------------------------------------

ARG VLLM_VERSION=0.32.0

# Vendor PyPI: vllm, plugin-fl, flagcx, flagtree, flaggems land here.
# Pattern: https://resource.flagos.net/repository/flagos-pypi-<vendor>/simple
ARG FLAGOS_PYPI=""

# Daily wheel index (flagtree daily builds).
ARG FLAGOS_PYPI_DAILY="https://resource.flagos.net/repository/flagos-pypi-daily/simple"

# Fallback index for packages not on the vendor index.
ARG EXTRA_PYPI="https://mirrors.cloud.tencent.com/pypi/simple"

# vllm-plugin-fl wheel version. Empty = skip.
ARG PLUGIN_FL_VERSION=""

# flagcx wheel version (full PEP 440 including local segment). Empty = skip.
ARG FLAGCX_VERSION=""

# FlagTree wheel version and package name (e.g. flagtree-nvidia).
ARG FLAGTREE_VERSION=""
ARG FLAGTREE_PACKAGE="flagtree"

# FlagGems wheel version.
ARG FLAGGEMS_VERSION=""

# Per-backend env vars written to /etc/profile.d/app_env.sh.
# Newline-separated KEY=VALUE pairs. Empty = skip.
ARG APP_ENV=""

# Proxy relayed as --build-arg from the workflow. ARG not ENV keeps
# the value out of pushed image metadata.
ARG http_proxy=
ARG https_proxy=
ARG no_proxy=

# --- Constraint file (torch / runtime protection) -------------

# Generate /tmp/protect.txt: constraints from the runtime base image
# (torch, triton, numpy) plus a live freeze snapshot of everything
# already installed. Every subsequent uv pip install passes
# -c /tmp/protect.txt so the runtime matrix is never disturbed.
RUN set -e; \
    protect=/tmp/protect.txt; \
    # Seed from the runtime base constraint file when present.
    if [ -f /etc/flagos/flaggems-constraints.txt ]; then \
        cp /etc/flagos/flaggems-constraints.txt "${protect}"; \
    else \
        : > "${protect}"; \
    fi; \
    # Append a full freeze so any unlisted runtime package is also pinned.
    uv pip freeze --python /flagos/bin/python >> "${protect}" 2>/dev/null || true

# --- FlagTree (staging-swap, mirrors ops-test pattern) --------

# Install to /opt/flagtree.new and replace /opt/flagtree only on success,
# matching the ops-test upgrade step behaviour. Prepend to PYTHONPATH so
# the baked version is active without further setup.
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

# --- vLLM (repacked wheel) ------------------------------------

# The vendor PyPI carries the repacked wheel as vllm=={version}+flagos.
# The +flagos local label is mandatory — a bare == would fall through to
# the upstream wheel whose torch constraint would disturb the pinned matrix.
RUN --mount=type=cache,id=flagos-uv-vllm,target=/root/.cache/uv \
    uv pip install \
        --python /flagos/bin/python \
        --index-url "${FLAGOS_PYPI}" \
        --extra-index-url "${EXTRA_PYPI}" \
        -c /tmp/protect.txt \
        "vllm==${VLLM_VERSION}+flagos"

# --- vllm-plugin-fl -------------------------------------------

RUN --mount=type=cache,id=flagos-uv-plugin-fl,target=/root/.cache/uv \
    if [ -n "${PLUGIN_FL_VERSION}" ]; then \
        uv pip install \
            --python /flagos/bin/python \
            --index-url "${FLAGOS_PYPI}" \
            --extra-index-url "${EXTRA_PYPI}" \
            -c /tmp/protect.txt \
            "vllm-plugin-fl==${PLUGIN_FL_VERSION}"; \
    fi

# --- FlagCX ----------------------------------------------------

# Full version pin including the local segment is required — PEP 440
# ignores the local label in a bare == comparison so a shortened pin
# would match another vendor's build of the same commit.
RUN --mount=type=cache,id=flagos-uv-flagcx,target=/root/.cache/uv \
    if [ -n "${FLAGCX_VERSION}" ]; then \
        uv pip install \
            --python /flagos/bin/python \
            --index-url "${FLAGOS_PYPI}" \
            --extra-index-url "${EXTRA_PYPI}" \
            -c /tmp/protect.txt \
            "flagcx==${FLAGCX_VERSION}"; \
    fi

# --- App env ---------------------------------------------------

# Bake per-backend env vars into /etc/profile.d/app_env.sh.
# Lines are 'export'-prefixed so they survive the launcher's exec
# (bare KEY=value assignments die with the sourcing shell).
RUN if [ -n "${APP_ENV}" ]; then \
        printf '%s\n' "${APP_ENV}" | sed 's/^/export /' \
            > /etc/profile.d/app_env.sh; \
    fi

# --- Serve launcher -------------------------------------------

COPY vllm/vllm-serve /usr/local/bin/vllm-serve
RUN chmod +x /usr/local/bin/vllm-serve

ENV VLLM_PLUGINS=fl
WORKDIR /workspace
CMD ["vllm-serve", "--model", "/data/models/Qwen3-4B", "--port", "8031", \
    "--gpu-memory-utilization", "0.6", "--enforce-eager", "--trust-remote-code", \
    "--max-model-len", "2048", "--dtype", "bfloat16"]
