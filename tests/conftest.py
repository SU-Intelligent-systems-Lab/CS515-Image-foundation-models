"""
pytest configuration: ensures the repo's ``src/`` directory is importable
under the package name ``dinovpr``.

This avoids requiring ``pip install -e .`` before running the tests.
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

if "dinovpr" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "dinovpr", _REPO_ROOT / "src" / "__init__.py",
        submodule_search_locations=[str(_REPO_ROOT / "src")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["dinovpr"] = module
    spec.loader.exec_module(module)
