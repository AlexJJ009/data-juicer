"""Fit donor-specific MAD thresholds, then flag and filter low-quality cells."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields

from .base import FittedFilter
from .conversion import Sample, validate_sample

TOP_GENES = 20
QC_METRICS = ("log1p_total_counts", "log1p_n_genes_by_counts", "pct_counts_in_top_20_genes", "pct_counts_mt")


@OPERATORS.register_module("single_cell_qc_filter")
class SingleCellQCFilter(FittedFilter):
    """Filter donor-wise MAD outliers using sc-best-practices QC defaults.

    Defaults: https://www.sc-best-practices.org/preprocessing-visualization/quality-control/
    """

    def __init__(
        self, nmads: float = 5, mito_nmads: float = 3, mito_max_pct: float = 8, mito_prefix: str = "MT-", **kwargs: Any
    ) -> None:
        """Configure MAD cutoffs, mitochondrial prefix, and shared field kwargs."""
        super().__init__(**kwargs)
        if not np.isfinite([nmads, mito_nmads, mito_max_pct]).all() or min(nmads, mito_nmads, mito_max_pct) < 0:
            raise ValueError("QC cutoffs must be finite and nonnegative")
        if not mito_prefix:
            raise ValueError("mito_prefix must be nonempty")
        self.nmads, self.mito_nmads, self.mito_max_pct = nmads, mito_nmads, mito_max_pct
        self.mito_prefix = mito_prefix

    def metrics(self, sample: Sample) -> dict[str, float | int]:
        """Return UMI, detected-gene, top-20 and mitochondrial metrics for sample."""
        validate_sample(sample, self.genes_key, self.counts_key, self.log_norm_key)
        counts = np.asarray(sample[self.counts_key], dtype=np.float64)
        total, detected = float(counts.sum()), int(np.count_nonzero(counts))
        mito = sum(value for gene, value in zip(sample[self.genes_key], counts) if gene.startswith(self.mito_prefix))
        top = np.sort(counts)[-TOP_GENES:].sum()
        return dict(
            total_counts=total,
            n_genes_by_counts=detected,
            log1p_total_counts=float(np.log1p(total)),
            log1p_n_genes_by_counts=float(np.log1p(detected)),
            pct_counts_in_top_20_genes=float(100 * top / total) if total else 0.0,
            pct_counts_mt=float(100 * mito / total) if total else 0.0,
        )

    def fit(self, dataset: Iterable[Sample]) -> None:
        """Fit per-donor median and unscaled MAD from nonexcluded cells; return None."""
        rows = list(dataset)
        self.has_mito_ = any(gene.startswith(self.mito_prefix) for row in rows for gene in row[self.genes_key])
        if not self.has_mito_:
            logger.warning("No genes with mitochondrial prefix {!r}; skipping mitochondrial QC", self.mito_prefix)
        groups = defaultdict(list)
        for row in self.fitting_rows(rows):
            metrics = self.metrics(row)
            groups[row[self.donor_key]].append([metrics[key] for key in QC_METRICS])
        self.thresholds_ = {}
        for donor, values in groups.items():
            matrix = np.asarray(values)
            center = np.median(matrix, axis=0)
            self.thresholds_[donor] = (center, np.median(np.abs(matrix - center), axis=0))
        missing = {row[self.donor_key] for row in rows} - self.thresholds_.keys()
        if missing:
            raise ValueError(f"Donors without fitting cells: {missing}")

    def compute_stats_single(self, sample: Sample, context: bool = False) -> Sample:
        """Return sample with QC metrics and a donor-wise outlier flag in DJ stats."""
        metrics = self.metrics(sample)
        center, mad = self.thresholds_[sample[self.donor_key]]
        values = np.asarray([metrics[key] for key in QC_METRICS])
        outliers = np.abs(values - center) > self.nmads * mad
        mito_outlier = self.has_mito_ and abs(values[-1] - center[-1]) > self.mito_nmads * mad[-1]
        metrics["qc_outlier"] = bool(outliers[:3].any() or (mito_outlier and values[-1] > self.mito_max_pct))
        sample.setdefault(Fields.stats, {}).update(metrics)
        return sample

    def process_single(self, sample: Sample) -> bool:
        """Return True when sample is not a QC outlier."""
        return not sample[Fields.stats]["qc_outlier"]
