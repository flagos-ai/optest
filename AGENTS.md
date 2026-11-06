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
