"""Tests for Jupyter Notebook execution, validity, and outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).parent.parent
NOTEBOOK_PATH = ROOT_DIR / "notebooks" / "exploratory_data_analysis.ipynb"


def test_notebook_json_validity_and_outputs():
    """Verify notebook is valid JSON, all code cells have execution_count and non-empty outputs."""
    assert NOTEBOOK_PATH.exists(), f"Notebook not found at: {NOTEBOOK_PATH}"

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) > 0

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) >= 3, "Expected at least 3 code cells in EDA notebook"

    has_stdout = False
    has_image = False

    for idx, cell in enumerate(code_cells):
        exec_count = cell.get("execution_count")
        assert exec_count is not None and exec_count > 0, f"Cell {idx} has invalid execution_count: {exec_count}"

        outputs = cell.get("outputs", [])
        assert len(outputs) > 0, f"Cell {idx} has empty outputs"

        for out in outputs:
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                text = "".join(out.get("text", []))
                if text.strip():
                    has_stdout = True
            elif out.get("output_type") == "display_data":
                data = out.get("data", {})
                if "image/png" in data and len(data["image/png"]) > 100:
                    has_image = True

    assert has_stdout, "Expected at least one code cell to have stdout output"
    assert has_image, "Expected at least one code cell to have image display output"


def test_notebook_code_cells_execute_without_error_from_root():
    """Verify that each code cell can be executed from the project root without errors."""
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    scope = {"__name__": "__main__"}

    for idx, cell in enumerate(code_cells):
        code = "".join(cell.get("source", []))
        try:
            exec(code, scope)
        except Exception as e:
            pytest.fail(f"Execution of cell {idx} failed from root directory: {e}")


def test_notebook_code_cells_execute_without_error_from_notebooks_dir(monkeypatch):
    """Verify that notebook paths work when running with cwd set to notebooks/."""
    monkeypatch.chdir(ROOT_DIR / "notebooks")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    scope = {"__name__": "__main__"}

    for idx, cell in enumerate(code_cells):
        code = "".join(cell.get("source", []))
        try:
            exec(code, scope)
        except Exception as e:
            pytest.fail(f"Execution of cell {idx} failed from notebooks/ directory: {e}")
