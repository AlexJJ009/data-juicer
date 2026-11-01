"""Extract configurable provenance or labels from fields without external models."""

import re
from typing import Any

from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("regex_field_extract_mapper")
class RegexFieldExtractMapper(Mapper):
    """Write the first matching regular-expression capture to a new field."""

    _batched_op = True

    def __init__(
        self,
        source_key: str = "text",
        target_key: str = "extracted",
        patterns: list[str] | None = None,
        group: int = 1,
        **kwargs: Any,
    ) -> None:
        """Configure input/output fields, ordered fallback patterns and capture group.

        :param source_key: String field searched by the patterns.
        :param target_key: Destination field for the capture.
        :param patterns: Ordered regular expressions; required when using this operator.
        :param group: Capture index, zero for the entire match.
        :param kwargs: Standard Mapper arguments.

        At least one pattern must match every row; unmatched rows raise ValueError.
        """
        super().__init__(**kwargs)
        if not source_key or not target_key or source_key == target_key or not patterns:
            raise ValueError("Distinct nonempty fields and at least one pattern are required")
        self.source_key, self.target_key, self.group = source_key, target_key, group
        self.patterns = [re.compile(pattern) for pattern in patterns]
        if group < 0 or any(group > pattern.groups for pattern in self.patterns):
            raise ValueError("Capture group does not exist in every pattern")

    def process_single(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return sample with the capture; raise on absent or unmatched input."""
        for pattern in self.patterns:
            match = pattern.search(sample[self.source_key])
            if match and match.group(self.group) is not None:
                sample[self.target_key] = match.group(self.group)
                return sample
        raise ValueError(f"No extraction pattern matched field {self.source_key!r}")
