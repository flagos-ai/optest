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
flagos-add-labels   # add operator labels to summary.json
flagos-psum-text    # render summary.json as CSV or Markdown
flagos-psum-html    # render summary.json as HTML
flagos-psum-web     # convert summaries to web data JSON
flagos-gpu-check-*  # check vendor GPU availability (one command per backend)
```

The target repository supplies its own test stack, including PyTorch, the
FlagOS library, pytest, and the plugins that provide options such as
`--record` and `--collect-marks`.

## Current Usage

Like `pytest`, `flagos-optest` accepts an optional project directory. It uses
the current directory when none is supplied. The selected directory becomes
the scheduler's working directory and the base for its test processes, so
relative configuration, input, and output paths are resolved from there:

```bash
cd /path/to/target
flagos-optest \
  --gpus 0 --stages all --output-dir /tmp/optest-results

# Equivalent explicit-directory invocation from another directory:
flagos-optest /path/to/target \
  --gpus 0 --stages all --output-dir /tmp/optest-results

cd /path/to/target
flagos-add-labels --project-root . /tmp/optest-results
flagos-psum-text -o /tmp/optest-results/result.csv /tmp/optest-results
flagos-psum-html -o /tmp/optest-results/result.html /tmp/optest-results
```

GPU checks are installed as commands such as `flagos-gpu-check-nvidia` and
`flagos-gpu-check-amd`. Backend configuration refers to these command names,
so checks continue to work when the package is installed from a wheel without
an optest source checkout.

The scheduler source remains runnable without installation from the target
repository directory:

```bash
python /path/to/optest/src/flagos_optest/run_tests.py --help
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
3. Make `flagos-optest` accept an optional target directory, default to the
   current directory, and update the workflow to run it like pytest.

Each step should build a wheel/sdist and smoke-test the scheduler, report,
label, and GPU-check console commands before it is submitted as a separate PR.

## Development

Use Python 3.9 or newer. Keep files UTF-8 without a BOM and use Unix line
endings. Code comments are written in English. Changes should include focused
validation and should preserve the target repository's test-environment
contract.

-- [AI] OpenAI Codex / GPT-5
