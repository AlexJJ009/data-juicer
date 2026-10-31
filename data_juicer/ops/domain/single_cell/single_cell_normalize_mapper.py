"""Fit a library-size target, then compute shifted-log expression per cell."""

from collections.abc import Iterable
from typing import Any

import numpy as np

from data_juicer.ops.base_op import OPERATORS

from .base import FittedMapper
from .conversion import Sample, validate_sample


@OPERATORS.register_module("single_cell_normalize_mapper")
class SingleCellNormalizeMapper(FittedMapper):
    """Normalize to a fixed total and log1p, retaining raw counts.

    Default target is the fitting-cell median library size, following
    https://www.sc-best-practices.org/preprocessing-visualization/normalization/
    """

    def __init__(self, target_sum: float | None = None, **kwargs: Any) -> None:
        """Set optional positive target_sum and shared field kwargs; return None."""
        super().__init__(**kwargs)
        if target_sum is not None and (not np.isfinite(target_sum) or target_sum <= 0):
            raise ValueError("target_sum must be finite and positive")
        self.target_sum = target_sum

    def fit(self, dataset: Iterable[Sample]) -> None:
        """Store explicit target or median fitting-cell count total; return None."""
        rows = self.fitting_rows(dataset)
        totals = []
        for row in rows:
            validate_sample(row, self.genes_key, self.counts_key, self.log_norm_key)
            totals.append(sum(row[self.counts_key]))
        self.target_sum_ = float(self.target_sum if self.target_sum is not None else np.median(totals))
        if self.target_sum_ <= 0:
            raise ValueError("Median fitting-cell library size must be positive")

    def process_single(self, sample: Sample) -> Sample:
        """Return sample with log_norm aligned to genes; zero-count cells stay zero."""
        validate_sample(sample, self.genes_key, self.counts_key, self.log_norm_key)
        counts = np.asarray(sample[self.counts_key], dtype=np.float64)
        total = counts.sum()
        sample[self.log_norm_key] = np.log1p(counts * (self.target_sum_ / total if total else 0)).tolist()
        return sample
