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
probe is vendor-agnostic: it uses `${{ matrix.vendor }}` in the path, so each
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

## Conventions

- Preserve UTF-8 without BOM and Unix line endings.
- Write code comments in English.
- Prefer `rg` for repository searches and `apply_patch` for focused edits.
- Keep network proxy variables scoped to the command that needs GitHub or
  package-index access.
- Never use destructive Git commands to discard unrelated user changes.
- Include focused build/smoke validation in every migration PR.
