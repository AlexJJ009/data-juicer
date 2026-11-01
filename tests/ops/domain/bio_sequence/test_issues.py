"""Test compact issue aggregation with synthetic stage evidence."""

from demos.process_topic2.summarize_issues import IssueCounter


def test_issue_counts_keep_three_examples_and_denominator() -> None:
    """Every occurrence counts, while examples remain bounded and source-specific."""
    counter = IssueCounter()
    for index in range(5):
        counter.add("illegal_characters", "source.jsonl", str(index), denominator=10)
    result = counter.result()["illegal_characters"]["source.jsonl"]
    assert result == {"count": 5, "denominator": 10, "ratio": 0.5, "examples": ["0", "1", "2"]}
