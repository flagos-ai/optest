# Repository Agent Guide

This is the canonical agent guide for the repository. The repository contains
the shared FlagOS operator-test scheduler and report tools. Read `README.md`
for the user-facing package and migration overview.

## Layout

- `src/flagos_optest/` is the Python package and contains the scheduler,
  report commands, HTML assets, and vendor GPU-check scripts.
- `.github/configs/repos.yaml` selects target repositories and their backends.
- `.github/configs/backends/` contains shared hardware settings.
- `.github/workflows/ops-test.yaml` runs the centralized matrix workflow.
- `pyproject.toml` defines the package, report/label commands, and vendor GPU
  check console scripts.

## Commands

Build and inspect artifacts with an isolated environment:

```bash
python -m build
python -m pip install dist/flagos_optest-*.whl
flagos-optest --help
flagos-add-labels --help
flagos-psum-text --help
flagos-psum-html --help
flagos-psum-web --help
```

For source-tree smoke tests, use `python src/flagos_optest/<script>`.
Full scheduler runs require the target repository's runtime stack and pytest
plugins; do not add those heavyweight target dependencies to this wheel.

## Incremental PR Process

Keep the packaging migration in small, independently reviewable PRs:

1. Migrate `tools/` to `src/flagos_optest/` and update package/config paths.
2. Install the optest checkout in the workflow and use installed console
   scripts for scheduling, labels, reports, and GPU checks.
3. Make the scheduler accept an optional target directory, default to the
   current directory, and update workflow working-directory handling.

Use PR #2 as the history for project-agnostic command behavior and PR #4 as
the history for wheel packaging. Do not mix a broad source-layout migration,
workflow behavior change, and CLI semantic change in one PR.

## Repository Onboarding

When adding a repository to `.github/configs/repos.yaml`, also add its Python
import name and repository display name to `_KNOWN_LIBS` in
`src/flagos_optest/run_tests.py`. The runtime probe uses this list to import the
library and record its name, version, vendor, and device metadata under `env`
in `summary.json`. Keep the repository configuration and probe list in sync so
new libraries are represented in test results.

## Vendor and Backend Onboarding

First determine whether the change adds hardware for an already-supported
vendor or introduces a genuinely new vendor. A new hardware backend for an
existing vendor normally reuses the vendor key, GPU probe, and metadata parser;
do not duplicate those entries. It may still have a distinct FlagTree
distribution when its backend or toolchain differs.

For every new hardware backend:

1. Add `.github/configs/backends/<hardware>.yaml` with the normalized `vendor`,
   runner label, container configuration, device access, `test_env`,
   `output_base_dir`, and any exceptional hardware-level dependencies. When
   FlagTree publishes a compatible backend wheel, add it under `deps` using the
   exact distribution name as the key, with its full default `version`,
   `index_url`, `priority: 100`, `version_input: flagtree_version`,
   `no_deps: true`, and `target: /opt/flagtree`. Omit that package when no
   compatible distribution exists; the workflow then retains the
   image-provided `/opt/flagtree` tree.
2. Add the backend under `backends` for each applicable repository in
   `.github/configs/repos.yaml`, plus repository-specific image overrides only
   when the shared backend image is unsuitable.
3. Ensure the target repository exposes an optional-dependency group whose name
   matches the lowercase vendor key reported by `flagos-gpu-check`. The workflow
   selects that group automatically.
4. Check FlagTree's `packaging/flagtree/backends.yaml` for a backend-specific
   wheel distribution. If it introduces a distribution name not already
   present, add that exact name to `_FLAGTREE_DISTRIBUTIONS` in
   `src/flagos_optest/run_tests.py`; backend names, vendor keys, and Python
   import names are not interchangeable here. The probe reports the wheel
   version and, when `Project-URL` contains a valid `Source-Commit` URL, appends
   the first eight commit characters as `<version>_<sha8>`. Release wheels named
   `flagtree`, or daily wheels without that metadata, retain the plain
   distribution version. Update `tests/test_run_tests_coverage.py` whenever the
   distribution list or metadata behavior changes. Confirm that the exact
   default version configured in the backend YAML exists in its configured
   index.
5. Update the backend inventory in `README.md` and `README.zh-CN.md`. If the
   hardware participates in NVIDIA baseline comparisons, also update
   `src/nv_baseline/hardware_specs.json` and its focused tests.

For a genuinely new vendor, also update all vendor registries and probes:

1. In `src/flagos_optest/gpu_check.py`, add the vendor-specific environment,
   detect and memory-sampling implementations, the `VENDORS` entry, a verified
   `/dev`-node rule in `_device_present`, and the correct position in
   `VENDOR_ORDER`. Check command and device-node collisions before choosing the
   order; several vendor stacks provide NVIDIA- or ROCm-compatible shims.
2. In `src/flagos_optest/gpu_metadata.py`, add the normalized display name,
   management command, runtime roots/environment variables, model pattern, and
   any vendor-specific driver or runtime-version parsing.
3. Add or update unit coverage in `tests/test_gpu_check.py`,
   `tests/test_gpu_check_coverage.py`, `tests/test_run_tests_coverage.py`, and
   metadata-focused tests as applicable. Capture sanitized output from real
   hardware in `tests/fixtures/gpu_check/<vendor>.json`, register it in the
   fixture README, and let `tests/test_gpu_check_live.py` replay it. If hardware
   access is not yet available, document that limitation and keep synthetic
   parser coverage until a live capture can be added.
4. Update the supported-vendor, runtime, metadata, and backend tables in both
   READMEs, plus any maintained vendor inventory comments in
   `.github/workflows/ops-test.yaml`.

The FlagGems runner-local operator-list paths and persistent package-cache paths
are derived from the normalized vendor key, so they normally require no code
branch for a new vendor. Verify that the runner supplies
`/data/flaggems/<vendor>/ops_include.txt` and/or `ops_exclude.txt` when those
lists are desired, and that its backend `output_base_dir` is durable. Run the
full test suite after onboarding; this repository requires 100% statement and
branch coverage.

## Configured Package Upgrades

The `Upgrade configured packages (FlagTree, FlagGems)` step runs after optest
installation and before GPU detection. Its generic `deps` configuration is
resolved independently for each exact package key, in this order from lowest
to highest precedence:

1. `defaults.deps` in `.github/configs/repos.yaml`, for all repositories and
   enabled backends.
2. `deps` in `.github/configs/backends/<backend>.yaml`, for that hardware
   backend in every repository where it is enabled.
3. `repos.<repo>.deps.default`, for every enabled backend of one repository.
4. `repos.<repo>.deps.<backend>`, for one repository/backend combination.

Merge the same package field-by-field; do not replace the whole package map.
A field omitted by a later layer retains its inherited value. The final
`enabled` value controls inclusion and has these exact rules:

- When no layer sets `enabled`, it defaults to `true`.
- A later explicit `enabled: false` disables an inherited package.
- An inherited `false` remains false through later layers that omit the field.
- Only a later explicit `enabled: true` re-enables an inherited disabled
  package. Therefore, the last explicit `enabled` value wins.

Package enablement is separate from `repos.<repo>.backends.<name>.enabled`.
The workflow removes a disabled backend from the matrix before resolving
packages, and a package-level `enabled: true` cannot re-enable that backend.
After disabled packages are removed, install order is ascending `priority`
(default `1000`) and then package name for equal priorities. Installation order
does not affect configuration precedence.

The package name is the mapping key. `version` produces `package==version`; an
omitted version installs the bare package. `display_name`, `enabled`,
`priority`, `version`, and `version_input` are workflow control fields. Every
other field maps from snake_case to a `uv pip install --kebab-case` option:
booleans add the flag only when true, scalar values add an option/value pair,
and lists repeat the option. An omitted `index_url` keeps the workflow-selected
package index; an explicit index remains command-local.

A package with `target` is installed into `<target>.new`; only a successful
install may replace the existing target. Targets must be normalized, specific
absolute paths. Keep removal failures fatal, update `PYTHONPATH` after a
successful replacement, and verify a pinned version through `ACTIVE_PYTHON`.
Never remove the old target before staging succeeds.

FlagTree remains a configured-package specialization. Its backend package uses
`priority: 100`, `version_input: flagtree_version`, and
`target: /opt/flagtree`. The optional dispatch input accepts a full daily or
release version, or `YYYYMMDD`; the date form keeps each backend's configured
version base. Before applying an override, the matrix builder checks the exact
version in that package and index. A missing version or index failure warns and
falls back only that backend. A date cannot be derived from a fixed release
default, but an available complete release version can override it.

FlagGems is the global default package with `priority: 200`, so FlagTree is
installed first. The `FlagGems` repository disables its inherited `flag-gems`
package under `deps.default` because its checkout is installed editable later.
Its 910B scope disables `flagtree-ascend-cann9.0.0` because that repository's
CANN 9.2.1 image is incompatible with the shared CANN 9.0.0 wheel. Other
repositories and FlagGems backends keep their inherited package behavior.
Cambricon retains image-provided FlagTree because its backend has no configured
FlagTree package.

When changing this flow, keep the resolution rules synchronized in
`.github/configs/repos.yaml`, both READMEs, and this guide. Tests must cover all
four layers, field inheritance, default enablement, disable inheritance,
explicit re-enablement at every later layer, final disablement at every later
layer, backend-level disablement, priority ordering, and the package-name
tie-breaker, plus FlagTree version overrides and safe target replacement.

## FlagGems Per-Vendor Operator Lists

The operator-test step in `.github/workflows/ops-test.yaml` probes runner-local
operator lists when, and only when, the target repository is `FlagGems`. The
probe is vendor-agnostic: it uses `FLAGOS_GPU_VENDOR` (the vendor reported by
`flagos-gpu-check`, already lowercase, e.g. `cambricon`) in the path, so each
vendor discovers its own files on its own runner with no hardcoded per-vendor
branch.

- Include list: `/data/flaggems/<vendor>/ops_include.txt` -> `--op-list-file`.
- Exclude list: `/data/flaggems/<vendor>/ops_exclude.txt` ->
  `--exclude-op-list-file`.
- Each file is optional; a missing file is skipped and never fails the step.
- A `ops` input other than `all` overrides the include probe, and an explicit
  `exclude_ops` input overrides the probed exclude file, matching the scheduler
  priority `--ops > --op-list-file` and `--exclude-ops > --exclude-op-list-file`.
- The step logs the probed paths, hit/miss for each, and the final operator and
  exclusion arguments for debugging.

When changing this logic, keep it FlagGems-only so other repositories keep the
default `--stages all` selection, and update the matching sections in
`README.md` and `README.zh-CN.md`.

## Persistent Package Cache

The `build-matrix` step in `.github/workflows/ops-test.yaml` derives a durable,
per-vendor package cache from each backend's `output_base_dir` and injects it
into the job `test_env`. This keeps large vendor wheels (for example `vllm`,
`flashinfer-cubin`) on the runner between runs so only version or URL changes
trigger a re-download. The cache is keyed by vendor only, not by repository:
different target repositories on the same vendor install nearly identical vendor
packages, so a single per-vendor cache lets them share one set of wheels.

- `UV_CACHE_DIR` = `<output_base_dir>/uv-cache/<vendor>`.
- `PIP_CACHE_DIR` = `<output_base_dir>/pip-cache/<vendor>`.
- `UV_LINK_MODE` defaults to `copy` so the cache and the venv can live on
  different filesystems without hardlink failures.

Notes for debugging and future changes:

- `output_base_dir` comes from the backend config under
  `.github/configs/backends/`. It is `/data/github-outputs` on most runners and
  `/home/github-outputs` on THead/PPU-ZW810E, which lack `/data`.
- The cache keys on package version plus wheel URL/hash. A cold (first) run
  downloads and populates the cache; warm (later) runs reuse it and skip the
  network. A version bump re-downloads only the changed package.
- The optest, FlagTree, and target-dependency install steps use normal
  `uv pip install` output. All three print a `[Cache debug]` report: the
  `UV_CACHE_DIR`/`PIP_CACHE_DIR`/
  `UV_LINK_MODE` values, the relevant `UV_`/`PIP_`/vendor env vars, and a
  COLD/WARM state with the cache size and file count before and after install.
  Grep the job log for `[Cache debug]` and `uv cache state:` to confirm a run
  hit the local cache instead of the network.
- Vendor-specific `test_env` variables in the backend config (for example
  `FLAGTREE_BACKEND`, `MACA_DIRECT_DISPATCH`, `GEMS_VENDOR`, `XPU_EVENT_KL3_ENABLE`)
  are preserved; the cache vars are merged on top without dropping them.

When changing the cache derivation, keep it keyed on `output_base_dir` and
`vendor` (not repository) so vendors share one cache across repositories, and
update the matching sections in `README.md` and `README.zh-CN.md`.

## Conventions

- Preserve UTF-8 without BOM and Unix line endings.
- Write code comments in English.
- Prefer `rg` for repository searches and `apply_patch` for focused edits.
- Keep network proxy variables scoped to the command that needs GitHub or
  package-index access.
- Never use destructive Git commands to discard unrelated user changes.
- Include focused build/smoke validation in every migration PR.
