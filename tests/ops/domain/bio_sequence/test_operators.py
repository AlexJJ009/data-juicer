"""Synthetic contracts for sequence refinement and field-configurable upstream ops."""

from pathlib import Path
from unittest.mock import patch

import pytest

from data_juicer.core.data import NestedDataset
from data_juicer.ops.base_op import Mapper
from data_juicer.ops.deduplicator.document_deduplicator import DocumentDeduplicator
from data_juicer.ops.domain.bio_sequence import (
    AnswerTemplateStatsMapper,
    QARuleConsistencyFilter,
    RegexFieldExtractMapper,
    SequenceAlphabetFilter,
    SequenceClusterMapper,
    SequenceTagExtractMapper,
)
from data_juicer.ops.domain.fitted import FittedMapper
from data_juicer.ops.filter.general_field_filter import GeneralFieldFilter
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.utils.constant import Fields


def test_fit_once_before_parent() -> None:
    """Fitting precedes parent mapping and needs no cell fields."""

    class Probe(FittedMapper):
        def fit(self, dataset):
            self.values = list(dataset)

    op = Probe()
    with patch.object(Mapper, "run", side_effect=lambda dataset, **kw: op.values) as parent:
        assert op.run([1, 2]) == [1, 2]
        parent.assert_called_once()


def test_tag_extraction_and_malformed_stats() -> None:
    """Same opening delimiters work; absent, repeated and dangling tags are explicit."""
    op = SequenceTagExtractMapper(text_key="prompt", tag="aa", sequence_key="s", template_key="q")
    row = op.process_single({"prompt": "Is <aa>ACD<aa> active?"})
    assert (row["s"], row["q"]) == ("ACD", "Is <sequence> active?")
    for text, status in [("none", "missing"), ("<aa>A<aa><aa>C<aa>", "multiple"), ("<aa>A", "malformed")]:
        result = op.process_single({"prompt": text})
        assert result[Fields.stats]["sequence_tag_status"] == status
        assert result["s"] == ""


def test_alphabet_threshold_and_invalid() -> None:
    """Ambiguity uses the full length denominator and empty sequences fail."""
    op = SequenceAlphabetFilter(seq_type="protein", max_ambiguous_ratio=0.25)
    for sequence, ratio, invalid, keep in [
        ("ACDX", 0.25, 0, True),
        ("AXXX", 0.75, 0, False),
        ("ACB", 0, 1, False),
        ("", 0, 0, False),
    ]:
        row = op.compute_stats_single({"sequence": sequence, Fields.stats: {}})
        assert row[Fields.stats]["sequence_ambiguous_ratio"] == ratio
        assert row[Fields.stats]["sequence_invalid_count"] == invalid
        assert op.process_single(row) is keep
    dna = SequenceAlphabetFilter(seq_type="dna")
    assert dna.process_single(dna.compute_stats_single({"sequence": "ACGTN", Fields.stats: {}})) is False
    with pytest.raises(ValueError):
        SequenceAlphabetFilter(seq_type="rna")


def test_regex_fallback_and_failure() -> None:
    """First matching capture is used; missing matches raise rather than disappear."""
    op = RegexFieldExtractMapper(source_key="a", target_key="b", patterns=[r"(K\d{5})", r"system (\w+)"])
    assert op.process_single({"a": "system AbiD"})["b"] == "AbiD"
    with pytest.raises(ValueError, match="a"):
        op.process_single({"a": "absent"})


def test_rule_tag_and_drop() -> None:
    """Violations are named and tag mode preserves flagged rows."""
    rules = [{"name": "enzyme_ec", "question_regex": "enzyme", "answer_regex": r"EC:\s*\d"}]
    for mode, keep in [("tag", True), ("drop", False)]:
        op = QARuleConsistencyFilter(rules=rules, mode=mode)
        row = op.compute_stats_single({"input": "enzyme?", "output": "yes", Fields.stats: {}})
        assert row[Fields.stats]["qa_rule_violations"] == ["enzyme_ec"]
        assert op.process_single(row) is keep
    with pytest.raises(ValueError):
        QARuleConsistencyFilter(rules=rules, mode="unknown")


def test_answer_fit_refits_and_stable_ids() -> None:
    """Template frequencies are global, deterministic, and reset on repeated fit."""
    op = AnswerTemplateStatsMapper(patterns=[r"K\d{5}"])
    op.fit([{"output": "KO K00001"}, {"output": "KO K00002"}, {"output": "other"}])
    a = op.process_single({"output": "KO K00003"})
    b = op.process_single({"output": "KO K00004"})
    assert a["answer_template_id"] == b["answer_template_id"]
    assert a["answer_template_ratio"] == 2 / 3
    op.fit([{"output": "KO K00001"}])
    assert op.process_single({"output": "KO K00001"})["answer_template_ratio"] == 1


def test_upstream_field_contracts() -> None:
    """Length, exact deduplication and representative selection use configured fields."""
    ds = NestedDataset.from_list(
        [
            {"sequence": "AC", "representative": True},
            {"sequence": "AC", "representative": False},
            {"sequence": "ACD", "representative": True},
        ]
    )
    lengths = TextLengthFilter(text_key="sequence", min_len=3, num_proc=1).run(ds)
    assert len(lengths) == 1
    assert len(DocumentDeduplicator(text_key="sequence", num_proc=1).run(ds)) == 2
    assert len(GeneralFieldFilter(filter_condition="representative == True", num_proc=1).run(ds)) == 2


def test_cluster_external_contract_and_cleanup(tmp_path: Path) -> None:
    """Numeric surrogate IDs isolate arbitrary original IDs and temporary files vanish."""
    executable = tmp_path / "mmseqs"
    executable.write_text('#!/bin/sh\nprintf "0\\t0\\n0\\t1\\n" > "$3"_cluster.tsv\n')
    executable.chmod(0o755)
    op = SequenceClusterMapper(mmseqs_path=str(executable), temp_dir=str(tmp_path), id_key="id")
    rows = [{"id": "a with spaces", "sequence": "ACDE"}, {"id": "b", "sequence": "ACDE"}]
    op.fit(rows)
    assert op.process_single(rows[1])["cluster_id"] == "a with spaces"
    assert op.process_single(rows[0])["is_representative"] is True
    assert op.process_single(rows[1])["is_representative"] is False
    assert list(tmp_path.iterdir()) == [executable]
    with pytest.raises(ValueError, match="duplicate"):
        op.fit([rows[0], rows[0]])
    with pytest.raises(FileNotFoundError, match="MMseqs2"):
        SequenceClusterMapper(mmseqs_path=str(tmp_path / "missing")).fit(rows)


def test_fitted_filter_forwards_reduce_without_cell_fields() -> None:
    """The generic fitted filter fits once and forwards the no-deletion flag."""
    from data_juicer.ops.base_op import Filter
    from data_juicer.ops.domain.fitted import FittedFilter

    class Probe(FittedFilter):
        def fit(self, dataset):
            self.fitted = len(dataset)

    op = Probe()
    with patch.object(Filter, "run", side_effect=lambda dataset, **kwargs: dataset) as parent:
        assert op.run([1], reduce=False) == [1]
        assert op.fitted == 1
        assert parent.call_args.kwargs["reduce"] is False


@pytest.mark.parametrize(
    "body, error",
    [
        ("exit 2\n", RuntimeError),
        ('printf "0\\t0\\n" > "$3"_cluster.tsv\n', ValueError),
        ('printf "0\\t0\\n0\\t0\\n" > "$3"_cluster.tsv\n', ValueError),
    ],
)
def test_cluster_failure_cleans_scratch(tmp_path: Path, body: str, error: type[Exception]) -> None:
    """External failures, missing members and duplicate members leave no scratch."""
    executable = tmp_path / "mmseqs"
    executable.write_text("#!/bin/sh\n" + body)
    executable.chmod(0o755)
    op = SequenceClusterMapper(mmseqs_path=str(executable), temp_dir=str(tmp_path))
    with pytest.raises(error):
        op.fit([{"seq_id": "a", "sequence": "ACD"}, {"seq_id": "b", "sequence": "ACD"}])
    assert list(tmp_path.iterdir()) == [executable]


def test_column_batches_preserve_annotations() -> None:
    """Sequence operators accept DJ column batches without wrapping every annotation cell."""
    alphabet = SequenceAlphabetFilter()
    batch = {"sequence": ["ACD", "AXD"], "gtdb.F_name": ["f1", "f2"], Fields.stats: [{}, {}]}
    result = alphabet.compute_stats(batch)
    assert list(alphabet.process(result)) == [True, False]
    assert result["gtdb.F_name"] == ["f1", "f2"]
    mapper = SequenceTagExtractMapper(text_key="input")
    extracted = mapper.process(
        {"input": ["<protein>ACD<protein>", "<protein>EFG<protein>"], Fields.stats: [{}, {}], "annotation": ["a", "b"]}
    )
    assert extracted["sequence"] == ["ACD", "EFG"]
    assert extracted["annotation"] == ["a", "b"]
