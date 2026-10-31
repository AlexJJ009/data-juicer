"""Synthetic tests for single-cell doublet filter."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data_juicer.core.data import NestedDataset


def test_doublet_grouping_seed_and_keep_mode(
    rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    "Verify doublet grouping seed and keep mode."
    from data_juicer.ops.domain.single_cell import SingleCellDoubletFilter
    from data_juicer.ops.domain.single_cell import single_cell_doublet_filter as module

    rows = [dict(row) for row in rows]
    rows[2]["cell_type"] = "CD4 T cells"
    calls = []

    def fake_scrublet(adata: Any, **kwargs: Any) -> None:
        "Capture donor size and seed, then return deterministic doublet labels."
        calls.append((adata.n_obs, kwargs["random_state"]))
        adata.obs["doublet_score"] = [0.1, 0.9, 0.2]
        adata.obs["predicted_doublet"] = [False, True, False]

    monkeypatch.setattr(module.sc.pp, "scrublet", fake_scrublet)
    op = SingleCellDoubletFilter(seed=17, num_proc=1, report_path=str(tmp_path / "doublets.json"))
    result = op.run(NestedDataset.from_list(rows))
    assert calls == [(3, 17)]
    assert [row["cell_id"] for row in result] == ["a", "c"]
    assert result[0]["__dj__stats__"]["doublet_score"] == 0.1
    assert result[1]["label"] == "stim"
    assert result[1]["__dj__stats__"]["doublet_score"] == 0.2
    kept = SingleCellDoubletFilter(keep_doublets=True, num_proc=1).run(NestedDataset.from_list(rows))
    assert len(kept) == 3
    assert kept[1]["__dj__stats__"]["predicted_doublet"]


@pytest.mark.parametrize("exclude", [{"label": "stim"}, {"missing_field": "value"}])
def test_doublet_rejects_nonempty_exclude_at_construction(exclude: dict[str, str]) -> None:
    "Verify doublet rejects nonempty exclude at construction."
    from data_juicer.ops.domain.single_cell import SingleCellDoubletFilter

    with pytest.raises(ValueError, match="all cells"):
        SingleCellDoubletFilter(exclude=exclude)


def test_real_scanpy_scrublet_uses_raw_counts_and_scores_all_cells() -> None:
    "Verify real scanpy scrublet uses raw counts and scores all cells."
    from data_juicer.ops.domain.single_cell import SingleCellDoubletFilter

    rng = np.random.default_rng(7)
    matrix = rng.poisson(rng.gamma(1.5, 2, size=(160, 400)))
    rows = [
        dict(
            cell_id=str(i),
            replicate="d1",
            cell_type="T",
            label="ctrl",
            genes=[f"g{j:03}" for j in np.flatnonzero(counts)],
            counts=counts[counts > 0].tolist(),
        )
        for i, counts in enumerate(matrix)
    ]
    op = SingleCellDoubletFilter(keep_doublets=True, seed=3, num_proc=1)
    result = op.run(NestedDataset.from_list(rows))
    assert len(result) == len(rows)
    scores = [row["__dj__stats__"]["doublet_score"] for row in result]
    assert np.isfinite(scores).all()
    assert all(isinstance(row["__dj__stats__"]["predicted_doublet"], bool) for row in result)


def test_scanpy_approximate_neighbors_dependency_contract() -> None:
    """Exercise the approximate-neighbor branch used by full donor Scrublet."""
    import anndata as ad
    import scanpy as sc

    data = ad.AnnData(np.random.default_rng(3).normal(size=(40, 12)))
    sc.pp.neighbors(data, n_neighbors=5, transformer="pynndescent", use_rep="X", random_state=0)
    assert data.obsp["distances"].shape == (40, 40)
