"""Validate exported refinement contracts against the original h5ad observations."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_juicer.ops.domain.single_cell.conversion import ad


def check_refined(
    refined_path: str | Path,
    original_path: str | Path,
    expected_genes: int = 2000,
    cell_type_key: str = "cell_type",
    condition_key: str = "label",
    holdout_cell_type: str = "CD4 T cells",
    holdout_condition: str = "stim",
) -> dict[str, Any]:
    """Check refined_path against original_path; return sizes and condition counts.

    expected_genes sets the required feature count. Field and holdout arguments
    select observation columns and equality values. Invalid exports raise ValueError.
    """
    refined, original = ad.read_h5ad(refined_path), ad.read_h5ad(original_path)
    if refined.n_vars != expected_genes:
        raise ValueError(f"Expected {expected_genes} genes, found {refined.n_vars}")
    if refined.X.dtype != np.float32:
        raise ValueError(f"Expected float32 X, found {refined.X.dtype}")
    if "counts" not in refined.layers or refined.X.shape != refined.layers["counts"].shape:
        raise ValueError("X and layers['counts'] must exist with identical shapes")
    if not refined.obs_names.is_unique or not refined.var_names.is_unique:
        raise ValueError("Cell and gene names must be unique")
    if not refined.var_names.isin(original.var_names).all():
        raise ValueError("Exported genes are absent from original var index")
    if not refined.obs_names.isin(original.obs_names).all():
        raise ValueError("Exported cells are absent from original obs index")
    try:
        pd.testing.assert_frame_equal(refined.obs, original.obs.loc[refined.obs_names], check_exact=True)
    except AssertionError as exc:
        raise ValueError(f"Exported obs differ from original observations: {exc}") from exc
    holdout = (refined.obs[cell_type_key] == holdout_cell_type) & (refined.obs[condition_key] == holdout_condition)
    if not holdout.any():
        raise ValueError("No holdout cells remain in the exported dataset")
    table = refined.obs.groupby([cell_type_key, condition_key], observed=True).size()
    return {
        "shape": list(refined.shape),
        "holdout_cells": int(holdout.sum()),
        "cells_by_type_condition": [
            {"cell_type": str(kind), "condition": str(condition), "cells": int(count)}
            for (kind, condition), count in table.items()
        ],
    }


def main() -> None:
    """Parse h5ad paths and field selectors; print evidence or exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("refined_path", type=Path)
    parser.add_argument("original_path", type=Path)
    parser.add_argument("--expected-genes", type=int, default=2000)
    parser.add_argument("--cell-type-key", default="cell_type")
    parser.add_argument("--condition-key", default="label")
    parser.add_argument("--holdout-cell-type", default="CD4 T cells")
    parser.add_argument("--holdout-condition", default="stim")
    print(json.dumps(check_refined(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
