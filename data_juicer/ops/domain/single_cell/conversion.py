"""Convert sparse cell records to and from AnnData before and after DJ refinement."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

ad = LazyLoader("anndata")
OBS_SCHEMA = "_single_cell_obs_schema"
Sample = dict[str, Any]


def write_samples_jsonl(path: str | Path, samples: Iterable[Sample]) -> None:
    """Write samples as JSONL to path without rounding obs floats; return None."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(dict(sample), ensure_ascii=False, allow_nan=False) + "\n")


def validate_sample(
    sample: Mapping[str, Any], genes_key: str = "genes", counts_key: str = "counts", log_norm_key: str = "log_norm"
) -> None:
    """Validate unique genes and aligned finite nonnegative integral counts; return None."""
    genes, counts = sample[genes_key], np.asarray(sample[counts_key], dtype=np.float64)
    if len(genes) != len(counts) or counts.ndim != 1:
        raise ValueError("genes and counts must be aligned one-dimensional lists")
    if any(not isinstance(gene, str) or not gene for gene in genes) or len(set(genes)) != len(genes):
        raise ValueError("genes must contain unique nonempty strings")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be finite nonnegative integers")
    if log_norm_key in sample:
        values = np.asarray(sample[log_norm_key], dtype=np.float64)
        if values.shape != counts.shape or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("log_norm must contain aligned finite nonnegative values")


def anndata_to_samples(
    adata: Any,
    genes_key: str = "genes",
    counts_key: str = "counts",
    log_norm_key: str = "log_norm",
    cell_id_key: str = "cell_id",
) -> Iterable[Sample]:
    """Yield sparse DJ rows from AnnData raw X, retaining obs types in a JSON schema field.

    Field arguments configure expression and identifier keys. Returns an iterator
    of samples. Raw counts are validated; reserved obs names raise ValueError.
    """
    reserved = {genes_key, counts_key, log_norm_key, cell_id_key, OBS_SCHEMA, Fields.stats}
    if reserved.intersection(adata.obs.columns):
        raise ValueError(f"obs columns collide with reserved fields: {reserved.intersection(adata.obs.columns)}")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and gene names must be unique")
    schema = {"index_name": adata.obs.index.name, "columns": {}}
    for key, series in adata.obs.items():
        spec = {"dtype": str(series.dtype)}
        if isinstance(series.dtype, pd.CategoricalDtype):
            spec.update(categories=series.cat.categories.tolist(), ordered=series.cat.ordered)
        schema["columns"][key] = spec
    encoded = json.dumps(schema)
    matrix = sparse.csr_matrix(adata.X)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    names = np.asarray(adata.var_names)
    for i, (cell_id, obs) in enumerate(adata.obs.iterrows()):
        vector = matrix.getrow(i)
        row = {
            key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
            for key, value in obs.items()
        }
        row.update(
            {
                cell_id_key: cell_id,
                genes_key: names[vector.indices].tolist(),
                counts_key: vector.data.tolist(),
                OBS_SCHEMA: encoded,
            }
        )
        validate_sample(row, genes_key, counts_key, log_norm_key)
        yield row


def samples_to_anndata(
    samples: Iterable[Sample],
    genes_key: str = "genes",
    counts_key: str = "counts",
    log_norm_key: str = "log_norm",
    cell_id_key: str = "cell_id",
) -> Any:
    """Return AnnData with alphabetically ordered union genes and original obs.

    samples supplies sparse cell dictionaries; field arguments select their keys.
    X is float32 log_norm when present, otherwise raw counts. layers['counts']
    always contains raw counts. Inconsistent schemas and duplicate IDs raise.
    """
    rows = list(samples)
    if not rows:
        raise ValueError("Cannot convert an empty cell dataset")
    genes = sorted({gene for row in rows for gene in row[genes_key]})
    lookup = {gene: i for i, gene in enumerate(genes)}
    indices, indptr, counts, normalized = [], [0], [], []
    has_norm = log_norm_key in rows[0]
    obs_rows, ids = [], []
    excluded = {
        genes_key,
        counts_key,
        log_norm_key,
        cell_id_key,
        OBS_SCHEMA,
        Fields.stats,
        Fields.meta,
        Fields.context,
        Fields.source_file,
    }
    schema_text = rows[0].get(OBS_SCHEMA)
    schema = json.loads(schema_text) if schema_text else None
    for row in rows:
        validate_sample(row, genes_key, counts_key, log_norm_key)
        if (log_norm_key in row) != has_norm or row.get(OBS_SCHEMA) != schema_text:
            raise ValueError("All cells must have consistent normalization and obs schemas")
        indices.extend(lookup[gene] for gene in row[genes_key])
        counts.extend(row[counts_key])
        if has_norm:
            normalized.extend(row[log_norm_key])
        indptr.append(len(indices))
        ids.append(row[cell_id_key])
        obs_rows.append(
            {key: row[key] for key in schema["columns"]}
            if schema
            else {key: val for key, val in row.items() if key not in excluded}
        )
    if len(set(ids)) != len(ids):
        raise ValueError("Cell IDs must be unique")
    shape = (len(rows), len(genes))
    raw = sparse.csr_matrix((np.asarray(counts, dtype=np.int64), indices, indptr), shape=shape)
    x = (
        sparse.csr_matrix((np.asarray(normalized, dtype=np.float32), indices, indptr), shape=shape)
        if has_norm
        else raw.copy()
    )
    obs = pd.DataFrame(obs_rows, index=pd.Index(ids, name=schema["index_name"] if schema else None))
    if schema:
        for key, spec in schema["columns"].items():
            obs[key] = (
                pd.Categorical(obs[key], categories=spec["categories"], ordered=spec["ordered"])
                if spec["dtype"] == "category"
                else obs[key].astype(spec["dtype"])
            )
    result = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=genes))
    result.layers["counts"] = raw
    return result
