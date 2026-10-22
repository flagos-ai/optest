# FlagOS Operator Test Tools

`flagos-optest` is the shared scheduler and result-reporting toolkit used by
FlagOS operator repositories. It runs accuracy and performance tests against a
target checkout, then produces JSON, CSV/Markdown, HTML, and web-report data.

## Install

Install the local package into the target test environment:

```bash
python -m pip install .
```

The package provides these console commands:

```text
flagos-optest       # schedule accuracy and performance tests
flagos-psum-text    # render summary.json as CSV or Markdown
flagos-psum-html    # render summary.json as HTML
flagos-psum-web     # convert summaries to web data JSON
```

The target repository supplies its own test stack, including PyTorch, the
FlagOS library, pytest, and the plugins that provide options such as
`--record` and `--collect-marks`.

## Current Usage

The workflow installs this package and uses the same console commands shown
below. The scheduler still targets another checkout explicitly until the next
cwd-based CLI migration:

```bash
flagos-optest --project-root /path/to/target \
  --gpus 0 --stages all --output-dir /tmp/optest-results
flagos-psum-text -o /tmp/optest-results/result.csv /tmp/optest-results
flagos-psum-html -o /tmp/optest-results/result.html /tmp/optest-results
```

The source checkout remains runnable without installation:

```bash
python src/flagos_optest/run_tests.py --project-root /path/to/target --help
```

## Repository Layout

```text
src/flagos_optest/       Python package, report scripts, and GPU checks
.github/configs/          Target-repository and backend configuration
.github/workflows/        Centralized GitHub Actions workflow
pyproject.toml            Build metadata and console-script definitions
```

## Migration Roadmap

The packaging migration is intentionally split into small pull requests:

1. Move the source package into `src/flagos_optest/` and keep direct workflow
   execution working.
2. Install this repository with `pip install .` in the workflow and invoke the
   installed console scripts instead of source paths.
3. Make `flagos-optest` discover the target project from the current working
   directory, then remove `--project-root` from `run_tests.py` and update the
   workflow to run from the target checkout like pytest.

Each step should build a wheel/sdist and smoke-test all four console commands
before it is submitted as a separate PR.

## Development

Use Python 3.9 or newer. Keep files UTF-8 without a BOM and use Unix line
endings. Code comments are written in English. Changes should include focused
validation and should preserve the target repository's test-environment
contract.

-- [AI] OpenAI Codex / GPT-5
