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

"""Console-script entry points for the ``flagos_optest`` package.

Some scripts in the package are flat, extensionless files that predate
packaging (``psum_html``, ``psum_text``): they ``import consts`` and resolve
``html_assets/`` relative to ``__file__``. Python cannot import an extensionless
file as a module, so setuptools console scripts must not target them directly.

These wrappers instead run those files with ``runpy.run_path`` using a ``run_name``
of ``__main__``. runpy then sets the same globals the scripts already rely on
(``__file__``, ``__name__ == "__main__"``) and, when ``__spec__`` is absent, leaves
the file's own directory on ``sys.path[0]`` -- so the flat ``import consts`` and
``Path(__file__)``-relative asset lookups keep working when installed from a wheel.

A function wrapper doubles as the console-script target (setuptools needs a
plain callable, not a module path); it forwards ``sys.argv`` so argparse sees the
command line unchanged.
"""

import sys
from pathlib import Path

import runpy


def _run_flat(script):
    """Execute an extensionless package script with its legacy import context.

    ``psum_html``/``psum_text``/``psum_web`` are flat scripts that do
    ``import consts`` instead of a package-relative import. Running one by path
    puts the script's own directory on ``sys.path``. runpy.run_path does not,
    so restore
    that behaviour explicitly, then run with ``run_name="__main__"`` so the
    scripts see the same ``__file__``/``__name__``/``__package__`` globals they
    already rely on (``__package__`` stays empty, keeping their top-level
    ``import consts`` and ``Path(__file__)``-relative asset lookups valid).
    """
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir))
    try:
        runpy.run_path(str(package_dir / script), run_name="__main__")
    finally:
        sys.path.remove(str(package_dir))


def psum_html():
    _run_flat("psum_html")


def psum_text():
    _run_flat("psum_text")


def psum_web():
    _run_flat("psum_web")
