"""Score raw counts with Scanpy Scrublet separately for each donor and filter doublets."""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.utils.constant import Fields
from data_juicer.utils.lazy_loader import LazyLoader

from .base import FittedFilter
from .conversion import Sample

sc = LazyLoader("scanpy")


@OPERATORS.register_module("single_cell_doublet_filter")
class SingleCellDoubletFilter(FittedFilter):
    """Run donor-wise Scrublet with Scanpy defaults and reproducible randomness.

    Batch-wise detection follows
    https://www.sc-best-practices.org/preprocessing-visualization/quality-control/
    All cells, including holdouts, are scored: this is per-cell quality control,
    and holdout doublets must also be removed. Simulated pairs stay within donor;
    scores do not become other cells' downstream representation. Scanpy has no
    fitted-model external-cell scoring API; see scanpy/preprocessing/_scrublet/
    __init__.py:scrublet and _scrublet_call_doublets.
    """

    def __init__(self, keep_doublets: bool = False, seed: int = 0, **kwargs: Any) -> None:
        """Configure keep_doublets (mark only), seed, and shared field/report kwargs."""
        super().__init__(**kwargs)
        if self.exclude:
            raise ValueError(
                "Scrublet scores all cells within each donor, including holdouts, "
                "because holdouts also need doublet QC; exclude must be empty"
            )
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        self.keep_doublets, self.seed = keep_doublets, seed

    def fit(self, dataset: Iterable[Sample]) -> None:
        """Run Scrublet per donor and store score/prediction by unique cell ID."""
        rows = list(dataset)
        if not rows:
            raise ValueError("Cannot fit Scrublet on an empty dataset")
        groups = defaultdict(list)
        identifiers = [row[self.cell_id_key] for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Scrublet requires unique cell IDs")
        self.scores_ = {}
        for row in rows:
            groups[row[self.donor_key]].append(row)
        for donor, cells in groups.items():
            logger.info("Scrublet donor {!r}: {} cells (all conditions)", donor, len(cells))
            data = self.to_anndata(cells)
            data.X = data.layers["counts"].copy()
            try:
                sc.pp.scrublet(data, random_state=self.seed)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"Scrublet failed for donor {donor!r}: {exc}") from exc
            scores = data.obs["doublet_score"].to_numpy()
            predicted = data.obs["predicted_doublet"].to_numpy()
            if not np.isfinite(scores).all() or data.obs["predicted_doublet"].isna().any():
                raise ValueError(f"Scrublet produced invalid scores or predictions for donor {donor!r}")
            for cell_id, score, prediction in zip(data.obs_names, scores, predicted):
                self.scores_[cell_id] = (float(score), bool(prediction))

    def compute_stats_single(self, sample: Sample, context: bool = False) -> Sample:
        """Return sample with fitted doublet_score and predicted_doublet in DJ stats."""
        score, prediction = self.scores_[sample[self.cell_id_key]]
        sample.setdefault(Fields.stats, {}).update(doublet_score=score, predicted_doublet=prediction)
        return sample

    def process_single(self, sample: Sample) -> bool:
        """Return whether sample survives doublet filtering or mark-only mode."""
        return self.keep_doublets or not sample[Fields.stats]["predicted_doublet"]
