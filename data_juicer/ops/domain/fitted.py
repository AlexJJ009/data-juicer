"""Fit dataset-wide state before the standard Data-Juicer row processing pass."""

from abc import abstractmethod
from collections.abc import Iterable
from typing import Any

from data_juicer.ops.base_op import Filter, Mapper


class _Fitted:
    """Require global fitting; support the default executor only."""

    _supported_exec_modes = ("default",)

    @abstractmethod
    def fit(self, dataset: Iterable[dict[str, Any]]) -> None:
        """Fit state from dataset before processing rows; return None."""
        raise NotImplementedError


class FittedMapper(_Fitted, Mapper):
    """Fit once on every run, then map through the parent implementation."""

    def run(self, dataset: Any, *, exporter: Any = None, tracer: Any = None) -> Any:
        """Fit dataset and return the mapped dataset, forwarding exporter and tracer."""
        self.fit(dataset)
        return super().run(dataset, exporter=exporter, tracer=tracer)


class FittedFilter(_Fitted, Filter):
    """Fit once on every run, then filter through the parent implementation."""

    def run(self, dataset: Any, *, exporter: Any = None, tracer: Any = None, reduce: bool = True) -> Any:
        """Fit dataset and return parent results; reduce controls actual deletion."""
        self.fit(dataset)
        return super().run(dataset, exporter=exporter, tracer=tracer, reduce=reduce)
