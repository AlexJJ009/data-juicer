"""Synthetic tests for single-cell hvg mapper."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data_juicer.core.data import NestedDataset
from data_juicer.ops.domain.single_cell import (
    SingleCellHvgMapper,
    SingleCellNormalizeMapper,
)


def test_hvg_fits_raw_counts_and_excludes_holdout(
    rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    "Verify hvg fits raw counts and excludes holdout."
    from data_juicer.ops.domain.single_cell import single_cell_hvg_mapper as module

    def fake_hvg(adata: Any, **kwargs: Any) -> None:
        "Verify the raw fitting matrix before emulating Scanpy HVG selection."
        assert adata.n_obs == 2
        assert kwargs["flavor"] == "seurat_v3"
        assert kwargs["batch_key"] == "replicate"
        assert kwargs["span"] == 0.3
        np.testing.assert_array_equal(adata.X.toarray(), [[1, 3], [4, 0]])
        adata.var["highly_variable"] = [True, False]

    monkeypatch.setattr(module.sc.pp, "highly_variable_genes", fake_hvg)
    op = SingleCellHvgMapper(
        n_top_genes=1, exclude={"label": "stim"}, report_path=str(tmp_path / "hvg.json"), num_proc=1
    )
    result = op.run(NestedDataset.from_list(rows))
    assert result[0]["genes"] == ["A"]
    assert result[2]["genes"] == []
    assert json.loads((tmp_path / "hvg.json").read_text())["genes"] == ["A"]


def test_real_scanpy_hvg_preserves_raw_counts() -> None:
    "Verify real scanpy hvg preserves raw counts."
    rng = np.random.default_rng(4)
    matrix = rng.poisson(rng.gamma(2, 2, size=(120, 100)))
    rows = [
        dict(
            cell_id=str(i),
            replicate=str(i % 2),
            cell_type="T",
            label="ctrl",
            genes=[f"g{j:03}" for j in np.flatnonzero(counts)],
            counts=counts[counts > 0].tolist(),
        )
        for i, counts in enumerate(matrix)
    ]
    dataset = NestedDataset.from_list(rows)
    normalized = SingleCellNormalizeMapper(num_proc=1).run(dataset)
    op = SingleCellHvgMapper(n_top_genes=20, span=1.0, num_proc=1)
    result = op.run(normalized)
    assert len(op.selected_genes_) == 20
    for original, refined in zip(rows, result):
        lookup = dict(zip(original["genes"], original["counts"]))
        assert refined["counts"] == [lookup[gene] for gene in refined["genes"]]
        assert len(refined["genes"]) == len(refined["log_norm"])
