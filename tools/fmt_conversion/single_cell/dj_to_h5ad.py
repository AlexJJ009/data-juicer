"""Export sparse cell JSONL as h5ad with log expression X and raw-count layer."""

import argparse
import json
from pathlib import Path

from data_juicer.ops.domain.single_cell.conversion import samples_to_anndata


def dj_to_h5ad(
    input_path: str | Path,
    output_path: str | Path,
    genes_key: str = "genes",
    counts_key: str = "counts",
    log_norm_key: str = "log_norm",
    cell_id_key: str = "cell_id",
) -> None:
    """Read cell JSONL and write h5ad with sorted union genes and preserved obs.

    Field arguments configure expression and identifier keys. Returns None.
    """
    with Path(input_path).open() as stream:
        data = samples_to_anndata(
            (json.loads(line) for line in stream if line.strip()), genes_key, counts_key, log_norm_key, cell_id_key
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(output)


def main() -> None:
    """Parse input/output and configurable field names; run conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    for key in ("genes", "counts", "log_norm", "cell_id"):
        parser.add_argument("--" + key.replace("_", "-") + "-key", default=key)
    dj_to_h5ad(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
