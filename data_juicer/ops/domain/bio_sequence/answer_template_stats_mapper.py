"""Fit global answer-template frequencies and annotate QA rows without deletion."""

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.domain.fitted import FittedMapper


@OPERATORS.register_module("answer_template_stats_mapper")
class AnswerTemplateStatsMapper(FittedMapper):
    """Normalize configured identifiers into stable templates and dataset frequencies."""

    _batched_op = True

    def __init__(
        self,
        patterns: list[str] | None = None,
        answer_key: str = "output",
        placeholder: str = "<id>",
        template_id_key: str = "answer_template_id",
        ratio_key: str = "answer_template_ratio",
        **kwargs: Any,
    ) -> None:
        """Configure ordered substitution patterns, input and output fields; return None.

        :param patterns: Ordered identifier patterns; defaults to KO and EC numbers.
        :param answer_key: Answer text field.
        :param placeholder: Literal replacement for every matched identifier.
        :param template_id_key: Destination for the stable template hash.
        :param ratio_key: Destination for the global template frequency.
        :param kwargs: Standard Mapper arguments.

        SHA256 IDs depend only on normalized template content, not dataset row order.
        Ratios describe the population passed to fit, after preceding filters.
        """
        super().__init__(**kwargs)
        self.patterns = [
            re.compile(pattern)
            for pattern in (patterns if patterns is not None else [r"K\d{5}", r"EC\s*:?\s*\d+\.\d+\.\d+\.(?:\d+|-)"])
        ]
        self.answer_key, self.placeholder = answer_key, placeholder
        self.template_id_key, self.ratio_key = template_id_key, ratio_key
        self.counts: Counter[str] = Counter()
        self.total = 0

    def _template(self, answer: str) -> str:
        for pattern in self.patterns:
            answer = pattern.sub(lambda _: self.placeholder, answer)
        return answer

    def fit(self, dataset: Iterable[dict[str, Any]]) -> None:
        """Reset and count normalized answers from dataset; return None."""
        self.counts = Counter(self._template(row[self.answer_key]) for row in dataset)
        self.total = sum(self.counts.values())

    def process_single(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return sample with stable template ID and fitted-population share."""
        template = self._template(sample[self.answer_key])
        if not self.total or template not in self.counts:
            raise ValueError("Answer template was not present during fit")
        sample[self.template_id_key] = hashlib.sha256(template.encode()).hexdigest()
        sample[self.ratio_key] = self.counts[template] / self.total
        return sample
