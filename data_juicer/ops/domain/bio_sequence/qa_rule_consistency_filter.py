"""Apply recipe-defined question/answer consistency rules as tags or deletions."""

import re
from typing import Any

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields


@OPERATORS.register_module("qa_rule_consistency_filter")
class QARuleConsistencyFilter(Filter):
    """Record named rules whose question matches but required answer pattern does not."""

    _batched_op = True

    def __init__(
        self,
        rules: list[dict[str, str]] | None = None,
        mode: str = "tag",
        question_key: str = "input",
        answer_key: str = "output",
        **kwargs: Any,
    ) -> None:
        """Configure named question_regex/answer_regex rules, fields and tag/drop mode.

        :param rules: Dictionaries with name, question_regex and answer_regex.
        :param mode: tag preserves all rows; drop deletes rule violations.
        :param question_key: Question field.
        :param answer_key: Answer field.
        :param kwargs: Standard Filter arguments.

        Patterns use case-insensitive matching. Tags indicate lexical rule violations,
        not a scientific determination of answer correctness.
        """
        super().__init__(**kwargs)
        if mode not in ("tag", "drop"):
            raise ValueError("mode must be 'tag' or 'drop'")
        rules = rules or []
        names = [rule["name"] for rule in rules]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Rule names must be nonempty and unique")
        self.rules = [
            (rule["name"], re.compile(rule["question_regex"], re.I), re.compile(rule["answer_regex"], re.I))
            for rule in rules
        ]
        self.mode, self.question_key, self.answer_key = mode, question_key, answer_key

    def compute_stats_single(self, sample: dict[str, Any], context: bool = False) -> dict[str, Any]:
        """Return sample with a list of violated rule names in stats."""
        sample.setdefault(Fields.stats, {})["qa_rule_violations"] = [
            name
            for name, question, answer in self.rules
            if question.search(sample[self.question_key]) and not answer.search(sample[self.answer_key])
        ]
        return sample

    def process_single(self, sample: dict[str, Any]) -> bool:
        """Return True for tag mode or a row without violations in drop mode."""
        return self.mode == "tag" or not sample[Fields.stats]["qa_rule_violations"]
