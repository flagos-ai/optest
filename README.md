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

## Usage

Like `pytest`, the commands accept an optional directory and use the current
directory when it is omitted. `flagos-optest` and `flagos-add-labels` select a
target project directory containing `conf/operators.yaml`; the two report
commands select a result directory containing `summary.json`.

Run from the target and result directories without directory arguments:

```bash
cd /path/to/target
flagos-optest \
  --gpus 0 --stages all --output-dir /tmp/optest-results
flagos-add-labels /tmp/optest-results

cd /tmp/optest-results
flagos-psum-text -o result.csv
flagos-psum-html -o result.html
```

Or pass the directories explicitly from another working directory:

```bash
flagos-optest /path/to/target \
  --gpus 0 --stages all --output-dir /tmp/optest-results
flagos-add-labels /path/to/target /tmp/optest-results
flagos-psum-text /tmp/optest-results -o /tmp/optest-results/result.csv
flagos-psum-html /tmp/optest-results -o /tmp/optest-results/result.html
```

For `flagos-add-labels`, the optional first directory is the target project;
the required `RESULT_DIR` remains the location of `summary.json`.

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
