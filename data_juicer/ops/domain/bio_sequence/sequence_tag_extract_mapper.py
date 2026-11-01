"""Extract tagged sequences and question templates before QA sequence refinement."""

import re
from typing import Any

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.constant import Fields


@OPERATORS.register_module("sequence_tag_extract_mapper")
class SequenceTagExtractMapper(Mapper):
    """Extract one sequence between identical opening tags, recording parse status."""

    _batched_op = True

    def __init__(
        self,
        tag: str = "protein",
        sequence_key: str = "sequence",
        template_key: str = "question_template",
        placeholder: str = "<sequence>",
        **kwargs: Any,
    ) -> None:
        """Configure tag, destination keys and replacement; text_key selects input.

        :param tag: Delimiter name; both delimiters are <tag>, without a slash.
        :param sequence_key: Destination sequence field.
        :param template_key: Destination question-template field.
        :param placeholder: Literal replacement for the tagged sequence.
        :param kwargs: Standard Mapper arguments, including text_key.

        Missing/multiple/malformed pairs produce an empty sequence with explicit stats;
        a downstream alphabet or length filter can reject these rows.
        """
        super().__init__(**kwargs)
        if not tag or any(c in tag for c in "<>/"):
            raise ValueError("tag must be a nonempty tag name without <, > or /")
        if not sequence_key or not template_key or sequence_key == template_key:
            raise ValueError("Sequence and template destination fields must be distinct and nonempty")
        self.delimiter = f"<{tag}>"
        self.pattern = re.compile(re.escape(self.delimiter) + r"(.*?)" + re.escape(self.delimiter), re.DOTALL)
        self.sequence_key, self.template_key, self.placeholder = sequence_key, template_key, placeholder

    def process_single(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return sample with sequence, template, match count and parse status."""
        text = sample[self.text_key]
        matches = list(self.pattern.finditer(text))
        delimiters = text.count(self.delimiter)
        status = "ok" if len(matches) == 1 and delimiters == 2 else "missing"
        if delimiters % 2:
            status = "malformed"
        elif len(matches) > 1:
            status = "multiple"
        sample[self.sequence_key] = matches[0].group(1) if status == "ok" else ""
        sample[self.template_key] = self.pattern.sub(lambda _: self.placeholder, text)
        sample.setdefault(Fields.stats, {}).update(sequence_tag_status=status, sequence_tag_matches=len(matches))
        return sample
