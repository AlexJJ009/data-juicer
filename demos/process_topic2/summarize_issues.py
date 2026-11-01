"""Compact DJ stage exports into source-level issue evidence and per-stage counts."""

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

STATS = "__dj__stats__"
QA_SOURCES = {
    "gcmetaNCYC": "qa_pairs_1.jsonl",
    "gcmeta_αamylase": "qa_pairs_2.jsonl",
    "gcmeta_defense": "qa_pairs_3.jsonl",
}


class IssueCounter:
    """Count issues per source and retain at most three identifier examples."""

    def __init__(self) -> None:
        """Initialize an empty issue table; return None."""
        self.table: dict[str, dict[str, dict[str, Any]]] = {}

    def add(self, category: str, source: str, identifier: str | None, denominator: int) -> None:
        """Count an occurrence; identifier=None initializes a zero-count category."""
        row = self.table.setdefault(category, {}).setdefault(
            source, dict(count=0, denominator=denominator, ratio=0.0, examples=[])
        )
        if row["denominator"] != denominator:
            raise ValueError(f"Inconsistent denominator for {category}/{source}")
        if identifier is not None:
            row["count"] += 1
            if len(row["examples"]) < 3:
                row["examples"].append(identifier)
        row["ratio"] = row["count"] / denominator if denominator else 0.0

    def result(self) -> dict[str, Any]:
        """Return the counted categories, sources, denominators and examples."""
        return self.table


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL records from path; malformed or missing files raise."""
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def _group(row: dict[str, Any], kind: str) -> tuple[str, str]:
    if kind != "qa":
        return f"Elephant_Gut_{'faa' if kind == 'protein' else 'fna'}_data_example.fasta", kind
    task = row.get("task") or re.split(r"_\d{14}_", row["data_no"], maxsplit=1)[0]
    return QA_SOURCES[task], task


def compact_run(directory: Path, kind: str, issues: IssueCounter) -> dict[str, Any]:
    """Summarize one completed run and save identifier-bearing audit stats.

    Filter snapshot populations are the actual stage inputs; issues from those
    stats describe that population, after earlier deletions. Exact duplicates
    are reported from the input hash or ID difference across the dedup stage.
    Return per-step counts grouped by task plus answer-template distributions.
    """
    summary_path = directory / "stage_summary.json"
    if not (directory / "alphabet_rows.jsonl").exists() and summary_path.exists():
        if not (directory / "audit_stats.jsonl").exists():
            raise ValueError("Cached stage summary requires its identifier-bearing audit statistics")
        cached = json.loads(summary_path.read_text())
        for category, sources in cached["issues"].items():
            issues.table.setdefault(category, {}).update(sources)
        return cached
    recipe = yaml.safe_load((directory / "recipe.yaml").read_text())
    snapshots: dict[str, Counter[str]] = {}
    kept: dict[str, Counter[str]] = {}
    flagged: dict[str, Counter[str]] = {}
    identifiers: dict[str, dict[str, tuple[str, str]]] = {}
    templates: dict[str, dict[str, Any]] = {}
    audit_path = directory / "audit_stats.jsonl"
    with audit_path.open("w") as audit:
        for stage in ("alphabet", "length", "rule", "cluster", "final"):
            path = directory / ("refined.jsonl" if stage == "final" else f"{stage}_rows.jsonl")
            if stage not in ("alphabet", "length", "final") and not path.exists():
                continue
            totals: Counter[str] = Counter()
            survivors: Counter[str] = Counter()
            tags: Counter[str] = Counter()
            ids: dict[str, tuple[str, str]] = {}
            # A first streaming pass establishes the population denominators.
            for row in read_jsonl(path):
                totals[_group(row, kind)[1]] += 1
            for row in read_jsonl(path):
                source, task = _group(row, kind)
                identifier = row["data_no" if kind == "qa" else "seq_id"]
                if identifier in ids:
                    raise ValueError(f"Duplicate record ID in {path}: {identifier}")
                ids[identifier] = (source, task)
                stats = row.get(STATS, {})
                keep = True
                categories: list[str] = []
                if stage == "alphabet":
                    keep = bool(
                        stats["sequence_length"]
                        and stats["sequence_invalid_count"] == 0
                        and stats["sequence_ambiguous_ratio"]
                        <= next(
                            s["sequence_alphabet_filter"]["max_ambiguous_ratio"]
                            for s in recipe["process"]
                            if "sequence_alphabet_filter" in s
                        )
                    )
                    categories = [
                        category
                        for category, bad in (
                            ("illegal_characters", stats["sequence_invalid_count"] > 0),
                            ("ambiguous_characters", stats["sequence_ambiguous_ratio"] > 0),
                            ("empty_sequence", stats["sequence_length"] == 0),
                            ("tag_extraction", stats.get("sequence_tag_status", "ok") != "ok"),
                        )
                        if bad
                    ]
                    for category in ("illegal_characters", "ambiguous_characters", "empty_sequence", "tag_extraction"):
                        issues.add(category, source, identifier if category in categories else None, totals[task])
                elif stage == "length":
                    bounds = next(s["text_length_filter"] for s in recipe["process"] if "text_length_filter" in s)
                    keep = bounds["min_len"] <= stats["text_len"] <= bounds["max_len"]
                    issues.add("length_anomaly", source, identifier if not keep else None, totals[task])
                elif stage == "rule":
                    categories = [f"rule:{name}" for name in stats["qa_rule_violations"]]
                    rules = next(
                        s["qa_rule_consistency_filter"]["rules"]
                        for s in recipe["process"]
                        if "qa_rule_consistency_filter" in s
                    )
                    for rule in rules:
                        category = f"rule:{rule['name']}"
                        issues.add(category, source, identifier if category in categories else None, totals[task])
                elif stage == "cluster" or (stage == "final" and kind == "qa"):
                    redundant = not row["is_representative"]
                    issues.add("cluster_redundancy", source, identifier if redundant else None, totals[task])
                    if stage == "cluster":
                        keep = not redundant
                if stage == "final" and "answer_template_id" in row:
                    entry = templates.setdefault(source, {}).setdefault(
                        row["answer_template_id"], dict(count=0, global_ratio=row["answer_template_ratio"], examples=[])
                    )
                    entry["count"] += 1
                    if len(entry["examples"]) < 3:
                        entry["examples"].append(identifier)
                survivors[task] += int(keep)
                tags[task] += int(bool(categories) or not keep)
                compact = dict(stage=stage, source_file=source, task=task, record_id=identifier, kept=keep, stats=stats)
                for key in ("label", "cluster_id", "is_representative", "answer_template_id", "answer_template_ratio"):
                    if key in row:
                        compact[key] = row[key]
                audit.write(json.dumps(compact, ensure_ascii=False) + "\n")
            snapshots[stage], kept[stage], flagged[stage], identifiers[stage] = totals, survivors, tags, ids
        input_totals: Counter[str] = Counter()
        duplicates: Counter[str] = Counter()
        if kind == "qa":
            seen: set[str] = set()
            paths = sorted((directory / "input").glob("*.jsonl"))
            for path in paths:
                for row in read_jsonl(path):
                    input_totals[_group(row, kind)[1]] += 1
            for path in paths:
                for row in read_jsonl(path):
                    source, task = _group(row, kind)
                    digest = hashlib.md5(row["input"].strip().encode()).hexdigest()
                    duplicate = digest in seen
                    seen.add(digest)
                    issues.add("exact_duplicate", source, row["data_no"] if duplicate else None, input_totals[task])
                    duplicates[task] += int(duplicate)
                    if duplicate:
                        audit.write(
                            json.dumps(
                                dict(
                                    stage="exact_duplicate",
                                    source_file=source,
                                    task=task,
                                    record_id=row["data_no"],
                                    kept=False,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
        else:
            input_totals = snapshots["alphabet"]
            if kind == "protein":
                for row in read_jsonl(directory / "length_rows.jsonl"):
                    source, task = _group(row, kind)
                    length = row[STATS]["text_len"]
                    bounds = next(s["text_length_filter"] for s in recipe["process"] if "text_length_filter" in s)
                    if not bounds["min_len"] <= length <= bounds["max_len"]:
                        continue
                    identifier = row["seq_id"]
                    duplicate = identifier not in identifiers["cluster"]
                    duplicates[task] += int(duplicate)
                    issues.add("exact_duplicate", source, identifier if duplicate else None, kept["length"][task])
                    if duplicate:
                        audit.write(
                            json.dumps(
                                dict(
                                    stage="exact_duplicate",
                                    source_file=source,
                                    task=task,
                                    record_id=identifier,
                                    kept=False,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    log = (directory / "process.log").read_text()
    elapsed = [
        (name, float(seconds), int(count))
        for name, seconds, count in re.findall(r"OP \[([^\]]+)\] Done in ([\d.]+)s\. Left (\d+) samples", log)
    ]
    steps = []
    current = input_totals.copy()
    for index, stage in enumerate(recipe["process"]):
        name = next(iter(stage))
        after, tags = current.copy(), Counter()
        if name == "document_deduplicator":
            after = Counter({task: current[task] - duplicates[task] for task in current})
        elif name in ("sequence_alphabet_filter", "text_length_filter"):
            key = "alphabet" if name == "sequence_alphabet_filter" else "length"
            after, tags = kept[key], flagged[key]
        elif name == "qa_rule_consistency_filter":
            tags = flagged["rule"]
        elif name == "general_field_filter":
            after, tags = kept["cluster"], flagged["cluster"]
        elif name in ("answer_template_stats_mapper", "sequence_cluster_mapper"):
            tags = current.copy()
        if index >= len(elapsed) or elapsed[index][0] != name or sum(after.values()) != elapsed[index][2]:
            raise ValueError(f"Stage summary does not agree with actual CLI log: {directory}/{name}")
        steps.append(
            dict(
                operator=name,
                before=sum(current.values()),
                after=sum(after.values()),
                removed=sum(current.values()) - sum(after.values()),
                tagged=sum(tags.values()),
                seconds=elapsed[index][1],
                by_task={
                    task: dict(
                        before=current[task], after=after[task], removed=current[task] - after[task], tagged=tags[task]
                    )
                    for task in current
                },
            )
        )
        current = after.copy()
    for source, entries in templates.items():
        denominator = sum(entry["count"] for entry in entries.values())
        for entry in entries.values():
            entry["denominator"] = denominator
            entry["source_ratio"] = entry["count"] / denominator
    result = dict(
        kind=kind,
        steps=steps,
        answer_templates=templates,
        metrics=json.loads((directory / "metrics.json").read_text()),
        issues=issues.result(),
        issue_population="Actual stage input after preceding deletions; exact duplicates use deduplicator input.",
    )
    (directory / "stage_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    logger.info("Summarized {}: {} -> {}", kind, sum(input_totals.values()), sum(current.values()))
    return result


def summarize(root: Path, data_dir: Path, output_path: Path) -> None:
    """Summarize the three demos, full annotation depth and full MMseqs2 membership."""
    issues = IssueCounter()
    runs = {}
    for kind in ("qa", "protein", "dna"):
        local = IssueCounter()
        runs[kind] = compact_run(root / kind, kind, local)
        for category, sources in local.result().items():
            issues.table.setdefault(category, {}).update(sources)
    annotations = {}
    for suffix in ("faa", "fna"):
        path = data_dir / f"Elephant_Gut_{suffix}_anno_example.jsonl"
        total = 0
        family_ids = []
        count = 0
        for row in read_jsonl(path):
            total += 1
            if row["biorank"] == "family" and not row["gtdb.G_name"] and not row["gtdb.S_name"]:
                count += 1
                if len(family_ids) < 3:
                    family_ids.append(row["seq_id"])
        annotations[path.name] = dict(count=count, denominator=total, ratio=count / total, examples=family_ids)
    full = json.loads((root / "full_protein/metrics.json").read_text())
    full["source_file"] = "Elephant_Gut_faa_data_example.fasta"
    full["redundant_records"] = full["input_records"] - full["clusters"]
    full["examples"] = []
    with (root / "full_protein/clusters_cluster.tsv").open() as handle:
        for line in handle:
            rep, member = line.rstrip("\n").split("\t")
            if rep != member and len(full["examples"]) < 3:
                full["examples"].append(member)
    result = dict(
        issues=issues.result(),
        answer_template_shares=runs["qa"]["answer_templates"],
        annotation_family_only=annotations,
        full_protein_clustering=full,
        populations={kind: run["issue_population"] for kind, run in runs.items()},
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    """Parse run root and original data paths, then write topic2_issues.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()
    summarize(args.root, args.data_dir, args.output_path or args.root / "topic2_issues.json")


if __name__ == "__main__":
    main()
