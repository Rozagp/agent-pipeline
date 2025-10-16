# agents/cli.py
"""
CLI launcher that runs the project-level main.py even when installed as a package.
"""

from __future__ import annotations
import pathlib
import sys
import runpy


def main() -> None:
    # Project root = agents
    root = pathlib.Path(__file__).resolve().parents[1]
    # Ensure project root is importable 
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Execute main.py as if run directly
    runpy.run_module("main", run_name="__main__")
