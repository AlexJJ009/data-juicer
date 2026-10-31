"""Fit single-cell operators globally before Data-Juicer maps or filters rows."""

import json
from abc import abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from data_juicer.ops.base_op import Filter, Mapper

from .conversion import Sample, samples_to_anndata, validate_sample


class CellFields:
    """Configure cell fields and an AND selector used only during fitting."""

    _supported_exec_modes = ("default",)

    def __init__(
        self,
        cell_type_key: str = "cell_type",
        condition_key: str = "label",
        donor_key: str = "replicate",
        genes_key: str = "genes",
        counts_key: str = "counts",
        log_norm_key: str = "log_norm",
        cell_id_key: str = "cell_id",
        exclude: dict[str, Any] | None = None,
        report_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize shared configuration; return None.

        :param cell_type_key: Observation column used for deletion summaries.
        :param condition_key: Observation column identifying experimental conditions.
        :param donor_key: Observation column used for donor-specific fitting.
        :param genes_key: Field containing nonzero gene names.
        :param counts_key: Field containing aligned raw UMI counts.
        :param log_norm_key: Field containing aligned shifted-log expression.
        :param cell_id_key: Field containing unique cell identifiers.
        :param exclude: Equality conditions combined with AND to exclude fitting rows.
        :param report_path: Optional JSON output path for operator evidence.
        :param kwargs: Arguments forwarded to the Data-Juicer Mapper or Filter.
        """
        super().__init__(**kwargs)
        keys = [cell_type_key, condition_key, donor_key, genes_key, counts_key, log_norm_key, cell_id_key]
        if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Cell field names must be nonempty and distinct")
        self.cell_type_key, self.condition_key, self.donor_key = keys[:3]
        self.genes_key, self.counts_key, self.log_norm_key, self.cell_id_key = keys[3:]
        self.exclude = exclude or {}
        self.report_path = report_path

    def is_excluded(self, sample: Mapping[str, Any]) -> bool:
        """Return whether sample matches every configured exclusion equality."""
        for key in self.exclude:
            if key not in sample:
                raise ValueError(f"Exclusion field {key!r} is missing")
        return bool(self.exclude) and all(sample[k] == v for k, v in self.exclude.items())

    def fitting_rows(self, dataset: Iterable[Sample]) -> list[Sample]:
        """Return nonexcluded rows; raise if no fitting population remains."""
        rows = [row for row in dataset if not self.is_excluded(row)]
        if not rows:
            raise ValueError("No cells remain in the fitting population")
        return rows

    def subset_genes(self, sample: Sample, selected: set[str]) -> Sample:
        """Subset all aligned expression fields to selected genes; return sample."""
        validate_sample(sample, self.genes_key, self.counts_key, self.log_norm_key)
        indices = [i for i, gene in enumerate(sample[self.genes_key]) if gene in selected]
        for key in (self.genes_key, self.counts_key, self.log_norm_key):
            if key in sample:
                sample[key] = [sample[key][i] for i in indices]
        return sample

    def to_anndata(self, rows: Iterable[Sample]) -> Any:
        """Return a sparse AnnData using configured expression and identifier fields."""
        return samples_to_anndata(rows, self.genes_key, self.counts_key, self.log_norm_key, self.cell_id_key)

    @abstractmethod
    def fit(self, dataset: Iterable[Sample]) -> None:
        """Fit parameters from dataset before row processing; return None."""
        raise NotImplementedError


class FittedMapper(CellFields, Mapper):
    """A DJ Mapper that fits on the complete dataset before parent run."""

    def run(self, dataset: Any, *, exporter: Any = None, tracer: Any = None) -> Any:
        """Fit dataset, then return the parent mapper's processed dataset."""
        start = perf_counter()
        before = _dimensions(dataset, self.genes_key)
        self.fit(dataset)
        result = super().run(dataset, exporter=exporter, tracer=tracer)
        logger.info(
            "Single-cell stage: {}",
            dict(
                operator=self._name,
                before=before,
                after=_dimensions(result, self.genes_key),
                seconds=perf_counter() - start,
                target_sum=getattr(self, "target_sum_", None),
                span=getattr(self, "span", None),
            ),
        )
        return result


class FittedFilter(CellFields, Filter):
    """A DJ Filter that fits globally and reports actual deletions by cell type."""

    def run(self, dataset: Any, *, exporter: Any = None, tracer: Any = None, reduce: bool = True) -> Any:
        """Fit dataset, run the parent filter, and return results with a deletion report."""
        start = perf_counter()
        dimensions = _dimensions(dataset, self.genes_key)
        self.fit(dataset)
        before_types = dataset[self.cell_type_key]
        before = Counter(before_types)
        before_groups = Counter(zip(before_types, dataset[self.condition_key]))
        result = super().run(dataset, exporter=exporter, tracer=tracer, reduce=reduce)
        after_types = result[self.cell_type_key]
        after = Counter(after_types)
        after_groups = Counter(zip(after_types, result[self.condition_key]))
        removed = {str(key): value - after[key] for key, value in before.items()}
        report = dict(
            input_cells=len(dataset),
            output_cells=len(result),
            removed_by_cell_type=removed,
            removed_by_cell_type_condition=[
                dict(cell_type=str(kind), condition=str(condition), removed=count - after_groups[kind, condition])
                for (kind, condition), count in before_groups.items()
            ],
            before=dimensions,
            after=_dimensions(result, self.genes_key),
            seconds=perf_counter() - start,
        )
        logger.info("{}: {}", self._name, report)
        if self.report_path:
            write_json(self.report_path, report)
        return result


def write_json(path: str | Path, value: Any) -> None:
    """Write a JSON value to path, creating its parent directories; return None."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _dimensions(dataset: Any, genes_key: str) -> dict[str, int]:
    """Return cell and nonzero-union gene counts using only the gene column."""
    return {"cells": len(dataset), "genes": len({gene for genes in dataset[genes_key] for gene in genes})}
