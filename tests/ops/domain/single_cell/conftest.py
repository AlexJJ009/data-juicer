"""Shared synthetic sparse cells for conversion and fitted-operator tests."""

from typing import Any

import pytest


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    "Return three sparse cells with a distinct stimulated fitting holdout."
    return [
        dict(cell_id="a", cell_type="T", label="ctrl", replicate="d1", genes=["A", "B"], counts=[1, 3]),
        dict(cell_id="b", cell_type="T", label="ctrl", replicate="d1", genes=["A"], counts=[4]),
        dict(cell_id="c", cell_type="T", label="stim", replicate="d1", genes=["C"], counts=[100]),
    ]
