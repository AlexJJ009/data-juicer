"""Synthetic tests for single-cell gene mapper."""

from typing import Any

from data_juicer.core.data import NestedDataset
from data_juicer.ops.domain.single_cell import (
    SingleCellGeneMapper,
    SingleCellNormalizeMapper,
)


def test_operator_names_are_unique_in_upstream_config_sorter() -> None:
    "Verify operator names are unique in upstream config sorter."
    from data_juicer.config.config import sort_op_by_types_and_names
    from data_juicer.ops.base_op import OPERATORS

    operators = [(name, cls) for name, cls in OPERATORS.modules.items() if name.startswith("single_cell_")]
    sorted_ops = sort_op_by_types_and_names(operators)
    names = [name for name, cls in sorted_ops]
    assert len(names) == len(set(names)) == 5


def test_gene_selection_excludes_holdout_and_aligns_expression(rows: list[dict[str, Any]]) -> None:
    "Verify gene selection excludes holdout and aligns expression."
    normalized = SingleCellNormalizeMapper(target_sum=4, num_proc=1).run(NestedDataset.from_list(rows))
    op = SingleCellGeneMapper(min_cells=2, exclude={"label": "stim"}, num_proc=1)
    result = op.run(normalized)
    assert result[0]["genes"] == ["A"]
    assert len(result[0]["counts"]) == len(result[0]["log_norm"]) == 1
    assert result[2]["genes"] == []
