"""Synthetic tests for single-cell qc filter."""

import json
from pathlib import Path
from typing import Any

from data_juicer.core.data import NestedDataset
from data_juicer.ops.domain.single_cell import SingleCellQCFilter


def test_qc_grouped_mad_and_report(rows: list[dict[str, Any]], tmp_path: Path) -> None:
    "Verify qc grouped mad and report."
    op = SingleCellQCFilter(exclude={"label": "stim"}, report_path=str(tmp_path / "qc.json"), num_proc=1)
    result = op.run(NestedDataset.from_list(rows))
    assert len(result) == 2
    assert result[0]["__dj__stats__"]["total_counts"] == 4
    report = json.loads((tmp_path / "qc.json").read_text())
    assert report["removed_by_cell_type"] == {"T": 1}


def test_qc_donor_groups_prevent_cross_donor_outliers(rows: list[dict[str, Any]]) -> None:
    "Verify qc donor groups prevent cross donor outliers."
    population = [dict(rows[0], cell_id=f"a{i}", genes=["A"], counts=[4]) for i in range(6)]
    population += [dict(rows[0], cell_id=f"b{i}", replicate="d2", genes=["A"], counts=[400]) for i in range(6)]
    result = SingleCellQCFilter(num_proc=1).run(NestedDataset.from_list(population))
    assert len(result) == 12


def test_qc_mitochondrial_cutoff_requires_absolute_percent(rows: list[dict[str, Any]]) -> None:
    "Verify qc mitochondrial cutoff requires absolute percent."
    population = [dict(rows[0], cell_id=f"a{i}", genes=["A", "MT-X"], counts=[100, 1]) for i in range(8)]
    population += [
        dict(rows[0], cell_id="low", genes=["A", "MT-X"], counts=[98, 3]),
        dict(rows[0], cell_id="high", genes=["A", "MT-X"], counts=[90, 11]),
    ]
    op = SingleCellQCFilter(num_proc=1)
    result = op.run(NestedDataset.from_list(population))
    assert "low" in result["cell_id"]
    assert "high" not in result["cell_id"]
