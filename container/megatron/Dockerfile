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
# FlagOS Megatron app image (optest-managed)
#
# Single Dockerfile builds every megatron stack in configs.yaml.
# megatron-core provides the training framework; FlagTree/FlagGems
# provide vendor acceleration.
# ============================================================

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG MEGATRON_VERSION=0.18.2+0.3.0

ARG FLAGOS_PYPI=""
ARG FLAGOS_PYPI_DAILY="https://resource.flagos.net/repository/flagos-pypi-daily/simple"
ARG EXTRA_PYPI="https://mirrors.cloud.tencent.com/pypi/simple"

ARG FLAGTREE_VERSION=""
ARG FLAGTREE_PACKAGE="flagtree"

ARG FLAGGEMS_VERSION=""

ARG APP_ENV=""

ARG http_proxy=
ARG https_proxy=
ARG no_proxy=

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

# --- Megatron-Core --------------------------------------------

RUN --mount=type=cache,id=flagos-uv-megatron,target=/root/.cache/uv \
    uv pip install \
        --python /flagos/bin/python \
        --index-url "${FLAGOS_PYPI}" \
        --extra-index-url "${EXTRA_PYPI}" \
        -c /tmp/protect.txt \
        "megatron-core==${MEGATRON_VERSION}"

# --- App env ---------------------------------------------------

RUN if [ -n "${APP_ENV}" ]; then \
        printf '%s\n' "${APP_ENV}" | sed 's/^/export /' \
            > /etc/profile.d/app_env.sh; \
    fi

WORKDIR /workspace
