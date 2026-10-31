"""Check exported matrix contracts and exact observation preservation."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tools.fmt_conversion.single_cell.check_refined import check_refined


def test_check_refined_detects_invalid_exports(tmp_path: Path) -> None:
    "Verify check refined detects invalid exports."
    obs = pd.DataFrame({"cell_type": ["CD4 T cells"], "label": ["stim"]}, index=["a"])
    original = ad.AnnData(sparse.csr_matrix([[2, 3]]), obs=obs, var=pd.DataFrame(index=["A", "B"]))
    original.write_h5ad(tmp_path / "original.h5ad")
    refined = original.copy()
    refined.layers["counts"] = refined.X.copy()
    refined.X = sparse.csr_matrix(np.log1p(refined.X.toarray()).astype(np.float32))
    refined.write_h5ad(tmp_path / "refined.h5ad")
    result = check_refined(tmp_path / "refined.h5ad", tmp_path / "original.h5ad", expected_genes=2)
    assert result["holdout_cells"] == 1
    with pytest.raises(ValueError, match="genes"):
        check_refined(tmp_path / "refined.h5ad", tmp_path / "original.h5ad", expected_genes=3)
    refined.obs["label"] = "ctrl"
    refined.write_h5ad(tmp_path / "refined.h5ad")
    with pytest.raises(ValueError, match="obs"):
        check_refined(tmp_path / "refined.h5ad", tmp_path / "original.h5ad", expected_genes=2)
