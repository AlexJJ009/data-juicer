"""Fit gene detection frequencies and remove rarely detected genes from every cell."""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from data_juicer.ops.base_op import OPERATORS

from .base import FittedMapper
from .conversion import Sample, validate_sample


@OPERATORS.register_module("single_cell_gene_mapper")
class SingleCellGeneMapper(FittedMapper):
    """Keep genes detected in at least min_cells fitting cells.

    The name avoids *_filter_mapper, which DJ configuration registers twice.
    Default min_cells=20 follows
    https://www.sc-best-practices.org/preprocessing-visualization/quality-control/
    """

    def __init__(self, min_cells: int = 20, **kwargs: Any) -> None:
        """Set minimum detected cells and shared field kwargs; return None."""
        super().__init__(**kwargs)
        if not isinstance(min_cells, int) or min_cells < 1:
            raise ValueError("min_cells must be a positive integer")
        self.min_cells = min_cells

    def fit(self, dataset: Iterable[Sample]) -> None:
        """Count nonzero detections in nonexcluded cells and store selected genes."""
        detected = Counter()
        for row in self.fitting_rows(dataset):
            validate_sample(row, self.genes_key, self.counts_key, self.log_norm_key)
            detected.update(gene for gene, count in zip(row[self.genes_key], row[self.counts_key]) if count > 0)
        self.selected_genes_ = {gene for gene, count in detected.items() if count >= self.min_cells}

    def process_single(self, sample: Sample) -> Sample:
        """Return sample with genes, counts and optional log_norm aligned and filtered."""
        return self.subset_genes(sample, self.selected_genes_)
