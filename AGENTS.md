# Repository Agent Guide

This guide was initialized through Claude Code \`/init\` and is the single
canonical agent guide for this repository. This repository contains the shared
FlagOS operator-test scheduler and report tools. Read \`README.md\` for the
user-facing package and migration overview.

## Layout

- \`src/flagos_optest/\` is the Python package and contains the scheduler,
  report commands, HTML assets, and vendor GPU-check scripts.
- \`.github/configs/repos.yaml\` selects target repositories and their backends.
- \`.github/configs/backends/\` contains shared hardware settings.
- \`.github/workflows/ops-test.yaml\` runs the centralized matrix workflow.
- \`pyproject.toml\` defines the package and four console scripts.

## Commands

Build and inspect artifacts with an isolated environment:

\`\`\`bash
python -m build
python -m pip install dist/flagos_optest-*.whl
flagos-optest --help
flagos-psum-text --help
flagos-psum-html --help
flagos-psum-web --help
\`\`\`

For source-tree smoke tests, use \`python src/flagos_optest/<script>\`.
Full scheduler runs require the target repository's runtime stack and pytest
plugins; do not add those heavyweight target dependencies to this wheel.

## Incremental PR Process

Keep the packaging migration in small, independently reviewable PRs:

1. Migrate \`tools/\` to \`src/flagos_optest/\` and update package/config paths.
2. Install the optest checkout in the workflow and use the four console
   scripts (\`flagos-optest\`, \`flagos-psum-text\`, \`flagos-psum-html\`, and
   \`flagos-psum-web\`). Decide separately how \`add_labels\` is exposed.
3. Change the scheduler to use the current working directory as the target,
   then remove \`--project-root\` and update workflow working-directory handling.

Use PR #2 as the history for project-agnostic command behavior and PR #4 as
the history for wheel packaging. Do not mix a broad source-layout migration,
workflow behavior change, and CLI semantic change in one PR.

## Conventions

- Preserve UTF-8 without BOM and Unix line endings.
- Write code comments in English.
- Prefer \`rg\` for repository searches and \`apply_patch\` for focused edits.
- Keep network proxy variables scoped to the command that needs GitHub or
  package-index access.
- Never use destructive Git commands to discard unrelated user changes.
- Include focused build/smoke validation in every migration PR.
- Append an AI-agent/model signature to generated review or documentation
  summaries.

-- [AI] OpenAI Codex / GPT-5
