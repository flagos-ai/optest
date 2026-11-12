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
existing vendor normally reuses the vendor key, GPU probe, metadata parser, and
FlagTree distribution already registered for that vendor; do not duplicate
those entries.

For every new hardware backend:

1. Add `.github/configs/backends/<hardware>.yaml` with the normalized `vendor`,
   runner label, container configuration, device access, `test_env`,
   `output_base_dir`, and any exceptional hardware-level dependencies.
2. Add the backend under `backends` for each applicable repository in
   `.github/configs/repos.yaml`, plus repository-specific image overrides only
   when the shared backend image is unsuitable.
3. Ensure the target repository exposes an optional-dependency group whose name
   matches the lowercase vendor key reported by `flagos-gpu-check`. The workflow
   selects that group automatically.
4. Update the backend inventory in `README.md` and `README.zh-CN.md`. If the
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
3. In `src/flagos_optest/run_tests.py`, add the upstream FlagTree daily wheel
   distribution name to `_FLAGTREE_DISTRIBUTIONS`. Use the exact distribution
   name from FlagTree's `packaging/flagtree/backends.yaml`; backend names and
   Python import names are not interchangeable here. The probe reports the
   wheel version and, when `Project-URL` contains a valid `Source-Commit` URL,
   appends the first eight commit characters as `<version>_<sha8>`. Release
   wheels named `flagtree`, or daily wheels without that metadata, retain the
   plain distribution version.
4. Add or update unit coverage in `tests/test_gpu_check.py`,
   `tests/test_gpu_check_coverage.py`, `tests/test_run_tests_coverage.py`, and
   metadata-focused tests as applicable. Capture sanitized output from real
   hardware in `tests/fixtures/gpu_check/<vendor>.json`, register it in the
   fixture README, and let `tests/test_gpu_check_live.py` replay it. If hardware
   access is not yet available, document that limitation and keep synthetic
   parser coverage until a live capture can be added.
5. Update the supported-vendor, runtime, metadata, and backend tables in both
   READMEs, plus any maintained vendor inventory comments in
   `.github/workflows/ops-test.yaml`.

The FlagGems runner-local operator-list paths and persistent package-cache paths
are derived from the normalized vendor key, so they normally require no code
branch for a new vendor. Verify that the runner supplies
`/data/flaggems/<vendor>/ops_include.txt` and/or `ops_exclude.txt` when those
lists are desired, and that its backend `output_base_dir` is durable. Run the
full test suite after onboarding; this repository requires 100% statement and
branch coverage.

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
- Both install steps ("Configure package index and install optest" and
  "Install target repository and dependencies") run `uv pip install -v` and
  print a `[Cache debug]` report: the `UV_CACHE_DIR`/`PIP_CACHE_DIR`/
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
