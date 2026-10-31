"""Synthetic tests for single-cell conversion."""

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from tools.fmt_conversion.single_cell.dj_to_h5ad import dj_to_h5ad
from tools.fmt_conversion.single_cell.h5ad_to_dj import h5ad_to_dj


def test_roundtrip_preserves_obs_and_counts(tmp_path: Path) -> None:
    "Verify roundtrip preserves obs and counts."
    obs = pd.DataFrame(
        {
            "kind": pd.Categorical(["T", "B"], categories=["B", "T", "NK"], ordered=True),
            "number": np.array([2, 3], dtype=np.int32),
            "flag": [True, False],
        },
        index=pd.Index(["c2", "c1"], name="barcode"),
    )
    data = ad.AnnData(sparse.csc_matrix([[0, 2, 0], [3, 0, 0]]), obs=obs, var=pd.DataFrame(index=["B", "A", "ZERO"]))
    src, mid, dst = tmp_path / "in.h5ad", tmp_path / "cells.jsonl", tmp_path / "out.h5ad"
    data.write_h5ad(src)
    h5ad_to_dj(src, mid)
    dj_to_h5ad(mid, dst)
    result = ad.read_h5ad(dst)
    # JSONL contains only nonzero genes; the all-zero column is not in the union.
    assert list(result.var_names) == ["A", "B"]
    np.testing.assert_array_equal(result.layers["counts"].toarray(), data[:, ["A", "B"]].X.toarray())
    pd.testing.assert_frame_equal(result.obs, obs)
    assert json.loads(mid.read_text().splitlines()[0])["genes"] == ["A"]


def test_roundtrip_full_matrix_and_nullable_obs(tmp_path: Path) -> None:
    "Verify roundtrip full matrix and nullable obs."
    obs = pd.DataFrame({"value": [1.5, np.nan], "nullable": pd.array([2, None], dtype="Int64")}, index=["a", "b"])
    data = ad.AnnData(sparse.csc_matrix([[0, 2], [3, 0]]), obs=obs, var=pd.DataFrame(index=["A", "B"]))
    data.write_h5ad(tmp_path / "in.h5ad")
    h5ad_to_dj(tmp_path / "in.h5ad", tmp_path / "cells.jsonl")
    dj_to_h5ad(tmp_path / "cells.jsonl", tmp_path / "out.h5ad")
    result = ad.read_h5ad(tmp_path / "out.h5ad")
    np.testing.assert_array_equal(result.X.toarray(), data.X.toarray())
    pd.testing.assert_frame_equal(result.obs, obs)
