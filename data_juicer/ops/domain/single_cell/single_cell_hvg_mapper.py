"""Select donor-aware Seurat-v3 HVGs on fitting raw counts, then subset every cell."""

from collections.abc import Iterable
from typing import Any

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.lazy_loader import LazyLoader

from .base import FittedMapper, write_json
from .conversion import Sample

sc = LazyLoader("scanpy")


@OPERATORS.register_module("single_cell_hvg_mapper")
class SingleCellHvgMapper(FittedMapper):
    """Keep raw-count Seurat-v3 HVGs with donor-aware ranking.

    Feature selection reference:
    https://www.sc-best-practices.org/preprocessing-visualization/feature-selection/
    span=0.3 is Scanpy's default; LOESS errors propagate without retry.
    """

    def __init__(self, n_top_genes: int = 2000, span: float = 0.3, **kwargs: Any) -> None:
        """Configure positive n_top_genes, LOESS span in (0, 1], and shared kwargs."""
        super().__init__(**kwargs)
        if not isinstance(n_top_genes, int) or n_top_genes < 1:
            raise ValueError("n_top_genes must be a positive integer")
        if not 0 < span <= 1:
            raise ValueError("span must lie in (0, 1]")
        self.n_top_genes, self.span = n_top_genes, span

    def fit(self, dataset: Iterable[Sample]) -> None:
        """Fit Seurat-v3 HVGs on nonexcluded raw counts and optionally write genes."""
        data = self.to_anndata(self.fitting_rows(dataset))
        if data.n_vars < self.n_top_genes:
            raise ValueError(f"Only {data.n_vars} fitting genes available for {self.n_top_genes} HVGs")
        # Normalization runs before this mapper; selecting on its X would use logs.
        data.X = data.layers["counts"].copy()
        sc.pp.highly_variable_genes(
            data, flavor="seurat_v3", batch_key=self.donor_key, n_top_genes=self.n_top_genes, span=self.span
        )
        self.selected_genes_ = set(data.var_names[data.var["highly_variable"]])
        if len(self.selected_genes_) != self.n_top_genes:
            raise ValueError(f"Scanpy selected {len(self.selected_genes_)} genes, expected {self.n_top_genes}")
        if self.report_path:
            write_json(self.report_path, {"genes": sorted(self.selected_genes_), "span": self.span})

    def process_single(self, sample: Sample) -> Sample:
        """Return sample subset to fitted HVGs with all expression fields aligned."""
        return self.subset_genes(sample, self.selected_genes_)
