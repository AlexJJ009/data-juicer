"""Measure ambiguity and reject illegal alphabet characters before sequence clustering."""

import math
from collections import Counter
from typing import Any

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields


@OPERATORS.register_module("sequence_alphabet_filter")
class SequenceAlphabetFilter(Filter):
    """Keep nonempty sequences with legal characters and bounded ambiguity."""

    _batched_op = True

    def __init__(
        self,
        seq_type: str = "protein",
        sequence_key: str = "sequence",
        ambiguous_chars: str | None = None,
        max_ambiguous_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Set alphabet, field and ambiguity tolerance; return None.

        :param seq_type: protein or dna selects the canonical alphabet.
        :param sequence_key: Field containing the unnormalized sequence.
        :param ambiguous_chars: Allowed ambiguity symbols, default X/protein or N/dna.
        :param max_ambiguous_ratio: Maximum fraction of ambiguity symbols, inclusive.
        :param kwargs: Standard Filter arguments.

        protein accepts the 20 canonical amino acids, dna accepts ACGT (uppercase).
        ambiguous_chars defaults to X or N respectively. The default zero tolerance
        is a conservative demo policy, not a biological quality standard; callers
        may relax it. Whitespace, lowercase and stop symbols are not normalized.
        """
        super().__init__(**kwargs)
        alphabets = {"protein": "ACDEFGHIKLMNPQRSTVWY", "dna": "ACGT"}
        if seq_type not in alphabets:
            raise ValueError("seq_type must be 'protein' or 'dna'")
        if not math.isfinite(max_ambiguous_ratio) or not 0 <= max_ambiguous_ratio <= 1:
            raise ValueError("max_ambiguous_ratio must be finite and between zero and one")
        self.standard = set(alphabets[seq_type])
        self.ambiguous = set(
            ambiguous_chars if ambiguous_chars is not None else ("X" if seq_type == "protein" else "N")
        )
        if self.standard & self.ambiguous:
            raise ValueError("Ambiguous and standard alphabets must be disjoint")
        self.sequence_key, self.max_ambiguous_ratio = sequence_key, max_ambiguous_ratio

    def compute_stats_single(self, sample: dict[str, Any], context: bool = False) -> dict[str, Any]:
        """Return sample with length, ambiguity ratio and illegal-character count."""
        sequence = sample[self.sequence_key]
        counts = Counter(sequence)
        ambiguous = sum(n for char, n in counts.items() if char in self.ambiguous)
        invalid = sum(n for char, n in counts.items() if char not in self.standard | self.ambiguous)
        sample.setdefault(Fields.stats, {}).update(
            sequence_length=len(sequence),
            sequence_ambiguous_ratio=ambiguous / len(sequence) if sequence else 0.0,
            sequence_invalid_count=invalid,
        )
        return sample

    def process_single(self, sample: dict[str, Any]) -> bool:
        """Return whether a nonempty sequence satisfies the alphabet policy."""
        stats = sample[Fields.stats]
        return bool(
            stats["sequence_length"]
            and stats["sequence_invalid_count"] == 0
            and stats["sequence_ambiguous_ratio"] <= self.max_ambiguous_ratio
        )
