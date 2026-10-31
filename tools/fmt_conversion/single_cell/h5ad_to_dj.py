"""Convert raw-count h5ad into one sparse Data-Juicer JSONL record per cell."""

import argparse
from pathlib import Path

from data_juicer.ops.domain.single_cell.conversion import (
    ad,
    anndata_to_samples,
    write_samples_jsonl,
)


def h5ad_to_dj(
    input_path: str | Path,
    output_path: str | Path,
    genes_key: str = "genes",
    counts_key: str = "counts",
    log_norm_key: str = "log_norm",
    cell_id_key: str = "cell_id",
) -> None:
    """Read input h5ad, validate raw X counts, and write sparse cell JSONL to output.

    Field arguments configure expression and identifier names. Returns None.
    """
    data = ad.read_h5ad(input_path)
    write_samples_jsonl(output_path, anndata_to_samples(data, genes_key, counts_key, log_norm_key, cell_id_key))


def main() -> None:
    """Parse input/output and configurable field names; run conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    for key in ("genes", "counts", "log_norm", "cell_id"):
        parser.add_argument("--" + key.replace("_", "-") + "-key", default=key)
    h5ad_to_dj(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
