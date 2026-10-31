"""Synthetic tests for single-cell normalize mapper."""

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pytest

from data_juicer.core.data import NestedDataset
from data_juicer.ops.domain.single_cell import SingleCellNormalizeMapper
from data_juicer.ops.domain.single_cell.conversion import samples_to_anndata
from tools.fmt_conversion.single_cell.dj_to_h5ad import dj_to_h5ad


def test_normalized_export(rows: list[dict[str, Any]], tmp_path: Path) -> None:
    "Verify normalized export."
    result = SingleCellNormalizeMapper(target_sum=10, num_proc=1).run(NestedDataset.from_list(rows))
    path = tmp_path / "normalized.jsonl"
    path.write_text("".join(json.dumps(dict(row)) + "\n" for row in result))
    dj_to_h5ad(path, tmp_path / "normalized.h5ad")
    data = ad.read_h5ad(tmp_path / "normalized.h5ad")
    assert data.X.dtype == np.float32
    assert data.X.shape == data.layers["counts"].shape == (3, 3)
    np.testing.assert_allclose(data.X.toarray()[0, :2], np.log1p([2.5, 7.5]))


def test_invalid_counts_and_missing_fit_population(rows: list[dict[str, Any]]) -> None:
    "Verify invalid counts and missing fit population."
    with pytest.raises(ValueError, match="counts"):
        samples_to_anndata([dict(rows[0], counts=[-1, 2])])
    with pytest.raises(ValueError, match="fitting"):
        SingleCellNormalizeMapper(exclude={"cell_type": "T"}).fit(rows)


def test_normalization_excludes_holdout(rows: list[dict[str, Any]]) -> None:
    "Verify normalization excludes holdout."
    op = SingleCellNormalizeMapper(exclude={"label": "stim"}, num_proc=1)
    result = op.run(NestedDataset.from_list(rows))
    assert op.target_sum_ == 4
    np.testing.assert_allclose(result[2]["log_norm"], [np.log1p(4)])
