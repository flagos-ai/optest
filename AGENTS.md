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
   `index_url`, `priority: 100`, `no_deps: true`, and
   `target: /opt/flagtree`. Omit that package when no
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
`priority`, and `version` are workflow control fields. Every other field maps
from snake_case to a `uv pip install --kebab-case` option:
booleans add the flag only when true, scalar values add an option/value pair,
and lists repeat the option. An omitted `index_url` keeps the workflow-selected
package index; an explicit index remains command-local.

Use the exact same package key in every layer that overrides it. The workflow
does not normalize hyphens and underscores while merging configuration, even
though Python package installers may treat those spellings as the same
distribution. Do not create two central entries for alternate spellings.

The control fields have narrow responsibilities:

- `display_name` is a log label only. It does not change package identity,
  imports, precedence, or install order. It defaults to the package key.
- `priority` is an integer install-order key. Lower values run first, the
  default is `1000`, and equal values sort by package key. It does not affect
  the four-layer configuration precedence.
- `enabled` defaults to true after merging and follows the last explicit value.
- `version` creates an exact `package==version` requirement. Omitting it uses
  the bare package requirement and may retain an already-satisfied install.
- `target` is both the uv `--target` value and the managed replacement path. It
  must be a normalized, specific absolute path such as `/opt/flagtree`.

A package with `target` is installed into `<target>.new`; only a successful
install may replace the existing target. Targets must be normalized, specific
absolute paths. Keep removal failures fatal, update `PYTHONPATH` after a
successful replacement, and verify a pinned version through `ACTIVE_PYTHON`.
Never remove the old target before staging succeeds.

### Dependency ownership and configuration recipes

Repository-owned dependencies normally belong in the target repository, not
in optest configuration:

- `[project].dependencies` owns requirements used on every backend.
- `[project.optional-dependencies].test` owns common test-only requirements.
- A normalized vendor extra such as `nvidia`, `ascend`, or `kunlunxin` owns
  backend/vendor-specific requirements. The workflow selects the detected
  vendor extra automatically.
- `[tool.uv].exclude-dependencies` is reserved for packages intentionally
  supplied by every applicable runtime image or by the shared configured
  environment. Exclusion prevents target resolution from changing that
  package, so verify that a compatible version is already present.

Central `deps` is for shared test-environment preparation: packages used by
multiple repositories, packages needed globally, hardware defaults shared by
repositories, and exceptional repository/backend policy that cannot live in a
target project. Do not duplicate a target repository's dependency contract in
optest merely to make CI pass; fix that repository's `pyproject.toml` instead.

Use these central locations when an exception is justified:

```yaml
# Every enabled backend of every repository.
defaults:
  deps:
    shared-helper:
      version: 1.2.3
      priority: 300

# Every repository that enables H800.
# .github/configs/backends/H800.yaml
deps:
  nvidia-shared-helper:
    version: 2.0.0

# One repository, or one repository/backend combination.
repos:
  SomeRepo:
    deps:
      default:
        repo-ci-exception:
          version: 3.0.0
        shared-helper:
          enabled: false      # exclude from every SomeRepo backend
      H800:
        h800-ci-exception:
          version: 4.0.0
        nvidia-shared-helper:
          enabled: false      # exclude only from SomeRepo/H800
```

An explicit `enabled: true` in a more specific layer can re-enable an inherited
disabled package. It cannot re-enable a backend disabled under
`repos.<repo>.backends.<name>.enabled`; backend filtering occurs first.
Use `repos.<repo>.backends.<name>.enabled: false` to remove the entire backend
job. Use `repos.<repo>.deps.<name>.<package>.enabled: false` to skip only one
package while keeping that repository/backend test job.

### Environment installation order

Keep dependency behavior aligned with the workflow's actual phases:

1. The backend container image supplies the initial environment.
2. The optest checkout is installed.
3. Merged central packages are installed in `priority`, package-key order.
4. GPU detection selects the target vendor extra.
5. The target project is installed editable with its base dependencies plus
   the `test` and detected-vendor extras. This is the last resolution phase.

For packages installed into the active Python environment, target-project
requirements have final authority but uv installs incrementally. An exact pin
or constraint replaces an earlier incompatible image/central version; a
currently installed version that already satisfies the target requirement may
be retained. If the target project itself has the same distribution name as an
installed wheel, its editable install becomes active; disable the redundant
central package, as the `FlagGems` repository does for `flag-gems`. A package
listed in `[tool.uv].exclude-dependencies` is omitted from target resolution,
so the image/central version remains active.

Do not treat `priority` as a dependency-conflict resolver. It orders central
install commands only; the target install always runs afterward. Also treat
`target` packages separately: they live outside the active environment and are
activated through a prepended `PYTHONPATH`, so a same-name active-environment
distribution can coexist and may not override the target directory. Avoid that
combination unless it is deliberate and tested.

FlagTree remains a configured-package specialization. Its backend package uses
`priority: 100` and `target: /opt/flagtree`. The backend YAML is the sole
version source: the workflow-dispatch form has no FlagTree version override and
the matrix builder does not query package indexes. When changing a pin, update
that backend's exact `version` and confirm it exists at `index_url`.

FlagGems is the global default package with `priority: 200`, so FlagTree is
installed first. The `FlagGems` repository disables its inherited `flag-gems`
package under `deps.default` because its checkout is installed editable later.
FlagGems inherits the shared 910B CANN 9.0.0 image and its configured
`flagtree-ascend-cann9.0.0` daily wheel. Cambricon retains image-provided
FlagTree because its backend has no configured FlagTree package.

When changing this flow, keep the resolution rules synchronized in
`.github/configs/repos.yaml`, both READMEs, and this guide. Tests must cover all
four layers, field inheritance, default enablement, disable inheritance,
explicit re-enablement at every later layer, final disablement at every later
layer, backend-level disablement, priority ordering, and the package-name
tie-breaker, fixed FlagTree version selection, and safe target replacement.

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
- The optest-install step must ensure `uv pip show --target` is supported
  (known baseline `uv 0.11.30`). For an older image-provided binary, try
  `uv self update 0.11.30` first so `UV_CACHE_DIR` is reused, then fall back to
  the existing pip index retry with `uv>=0.11.30` so `PIP_CACHE_DIR` is reused.
  After pip fallback, prepend the active Python scripts directory to `PATH` and
  persist it through `GITHUB_PATH`; do not leave an older `uv` first on `PATH`.

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
- Image preparation is a convention, not configuration:
  `.github/scripts/install/<vendor>.py` (or `.sh`), where `<vendor>` is the
  backend's vendor name in lower case, runs before the target install because
  native target extensions compile against it. Scripts must be idempotent, and
  vendor-specific patches belong there instead of a hardcoded `if vendor == ...`
  branch in the workflow.
- Vendor-specific `test_env` variables in the backend config (for example
  `FLAGTREE_BACKEND`, `MACA_DIRECT_DISPATCH`, `GEMS_VENDOR`, `XPU_EVENT_KL3_ENABLE`)
  are preserved; the cache vars are merged on top without dropping them.

When changing the cache derivation, keep it keyed on `output_base_dir` and
`vendor` (not repository) so vendors share one cache across repositories, and
update the matching sections in `README.md` and `README.zh-CN.md`.

## NV Baseline Incremental Refresh

`.github/workflows/nv_baseline.yaml` refreshes a published baseline
(`data/op_<short>_perf_baseline.json`) instead of rebuilding it. Its `mode`
dispatch input selects how:

- `incremental` (default) re-collects only operators that are absent from the
  committed baseline or whose stored `label` differs from the framework version
  installed in the collecting container, and carries every other published
  entry into the new output unchanged. The produced file is therefore the
  merged, complete baseline.
- `full` ignores the committed baseline and re-collects every selected
  operator, the behavior before the modes existed.

The decision lives in one place, `src/nv_baseline/baseline_refresh.py`
(`plan_refresh`), which both the single-card collector and the multi-GPU
scheduler call; neither re-implements the comparison. It also owns
`detect_label`, the probe for the framework version (`vllm`, `sglang`,
`deepspeed`) that each operator entry is labeled with. Add a framework to
`LABEL_FRAMEWORKS` there, not in the collectors.

`IGNORE_LOCAL_VERSION` in that module is a **temporary workaround**, currently
holding `deepspeed`: its entries are compared on the public version only, with
the PEP 440 local segment dropped (`0.20.1+b5e0ba6` -> `0.20.1`). FlagTrain
installs DeepSpeed from an unpinned `git clone` of main (both the workflow step
and FlagTrain's `baseline/README.md` prescribe that), and DeepSpeed appends the
checkout's commit hash to its version, so without this every FlagTrain refresh
re-collected all 32 operators. The cost is that a DeepSpeed main commit which
does change a baseline kernel no longer triggers a refresh. Each affected run
prints a `[refresh] WARNING` line, the workflow summary raises a `::warning::`
annotation, and the published `_incremental.local_version_ignored` names the
operators and both builds. Pin the install to a fixed ref or a PyPI release
(DeepSpeed publishes plain versions) and then drop `deepspeed` from the set;
no other code changes.

Behavior worth preserving when changing this flow:

- An operator queued for re-collection whose collection produces nothing
  (failed, skipped, timed out) keeps its previously published entry and its old
  label, so a refresh never deletes a known-good operator and the next run
  retries it. `restored_after_failure` in the output's `_incremental` block
  records that.
- An incremental run that finds nothing stale collects nothing and runs no
  operator subprocess. The classify step grades that as a pass (it tells it
  apart from a crash by the recorded plan).
- `_environment` and `_scaling_factors` describe the machine a run measured on,
  so only a collecting child produces them. When no child reports them -- an
  incremental run with nothing stale, or one whose every queued operator died
  before flushing its own metadata -- `carry_over_metadata` takes them from the
  previous baseline and names the keys in `status.json`'s
  `incremental.carried_over_metadata`; a value a child did report always wins.
  Without that, a run that still republishes operators (retained or restored)
  would publish them with no provenance at all. The single-card collector is
  not affected: it writes both keys in the parent before collecting.
- Every write of the baseline goes through `publishable()`, which puts
  `_environment` and `_scaling_factors` at the top of the file in that order
  (identically in both modes) and keeps run-time bookkeeping out of the
  published data. Add a new provenance key to `PUBLISHED_FIRST` and a new
  run-time key to `DROPPED_FROM_PUBLISHED`, rather than ordering at a call
  site.
- The `data/` update step refuses to commit an incremental result that holds
  fewer operators than the file it replaces. Full mode stays exempt, since
  there a failed operator is intentionally absent and the commit is marked
  forced.
- The refresh plan (mode, label, collected, retained, per-operator decisions,
  `local_version_ignored`, `carried_over_metadata`) is recorded in the sidecar
  `${OUTPUT}.work/status.json` under `incremental`, which the workflow grades
  and summarizes from and uploads as an artifact. It is deliberately NOT in the
  published baseline, which carries measurements and provenance only. Keep any
  new bookkeeping key `_`-prefixed; consumers identify operators by the absence
  of that prefix.

`tests/test_baseline_refresh.py` covers the planner, the scheduler path with a
stubbed collector subprocess, and the workflow's classify/update steps. The
published baselines are asserted to be fully labeled: an unlabeled operator
would make every refresh a full one.

## Conventions

- Preserve UTF-8 without BOM and Unix line endings.
- Write code comments in English.
- Prefer `rg` for repository searches and `apply_patch` for focused edits.
- Keep network proxy variables scoped to the command that needs GitHub or
  package-index access.
- Never use destructive Git commands to discard unrelated user changes.
- Include focused build/smoke validation in every migration PR.
