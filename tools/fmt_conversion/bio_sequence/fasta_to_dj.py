"""Join FASTA records to arbitrary-order JSONL annotations for DJ sequence recipes."""

import argparse
import json
import random
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

from loguru import logger


def read_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (first header token, concatenated sequence); reject malformed FASTA."""
    identifier: str | None = None
    pieces: list[str] = []
    with path.open() as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if identifier is not None:
                    if not pieces:
                        raise ValueError(f"Empty FASTA sequence for {identifier!r}")
                    yield identifier, "".join(pieces)
                tokens = line[1:].split()
                if not tokens:
                    raise ValueError(f"Empty FASTA identifier at line {number}")
                identifier, pieces = tokens[0], []
            else:
                if identifier is None or any(char.isspace() for char in line):
                    raise ValueError(f"Malformed FASTA sequence at line {number}")
                pieces.append(line)
    if identifier is not None:
        if not pieces:
            raise ValueError(f"Empty FASTA sequence for {identifier!r}")
        yield identifier, "".join(pieces)


def fasta_to_dj(
    fasta_path: Path,
    annotation_path: Path,
    output_path: Path,
    seq_type: str,
    sample_size: int | None = None,
    seed: int = 42,
    temp_dir: Path | None = None,
    min_free_gib: float = 5.0,
) -> dict[str, Any]:
    """Stream an ID join into output_path, preserving every annotation field.

    A temporary SQLite index supports unordered annotations with bounded Python
    memory. A seeded reservoir samples FASTA positions without replacement when
    sample_size is set. All IDs and lengths are still checked, including records
    outside the sample. A temporary output is published only after validation.
    min_free_gib reserves disk space (5 GiB by default), checked every 1000 records.
    Return counts; raise ValueError on missing, extra, duplicate or conflicting IDs.
    """
    fasta_path, annotation_path, output_path = map(Path, (fasta_path, annotation_path, output_path))
    if seq_type not in ("protein", "dna") or (sample_size is not None and sample_size < 1):
        raise ValueError("seq_type must be protein/dna and sample_size must be positive")
    if output_path.resolve() in (fasta_path.resolve(), annotation_path.resolve()):
        raise ValueError("Output must not overwrite either input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if min_free_gib < 0:
        raise ValueError("min_free_gib must be nonnegative")

    def check_space() -> None:
        if shutil.disk_usage(output_path.parent).free < min_free_gib * 1024**3:
            raise RuntimeError(f"Insufficient disk reserve: require {min_free_gib} GiB free")

    check_space()
    report = dict(
        fasta_records=0,
        annotation_records=0,
        missing_annotations=0,
        extra_annotations=0,
        length_mismatches=0,
        written=0,
        seed=seed,
        sample_size=sample_size,
    )
    with tempfile.TemporaryDirectory(prefix="dj-fasta-", dir=temp_dir) as directory:
        with closing(sqlite3.connect(str(Path(directory) / "annotations.sqlite"))) as db:
            # This disposable index is rebuilt on failure; rollback copies would
            # duplicate gigabytes without protecting any durable user data.
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("CREATE TABLE annotations (id TEXT PRIMARY KEY, payload TEXT, seen INTEGER DEFAULT 0)")
            with annotation_path.open() as handle:
                for number, line in enumerate(handle, 1):
                    if number % 1000 == 0:
                        check_space()
                    row = json.loads(line)
                    identifier = row.get("seq_id")
                    if not isinstance(identifier, str) or not identifier:
                        raise ValueError(f"Invalid annotation seq_id at line {number}")
                    if any(key in row for key in ("sequence", "seq_type")):
                        raise ValueError(f"Annotation collides with generated fields: {identifier!r}")
                    try:
                        db.execute("INSERT INTO annotations(id, payload) VALUES (?, ?)", (identifier, line))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"Duplicate annotation ID {identifier!r}") from error
                    report["annotation_records"] += 1
            db.commit()
            selected: set[int] | None = None
            if sample_size is not None:
                # Sampling positions needs only O(sample_size) memory, not all sequences.
                rng = random.Random(seed)
                reservoir: list[int] = []
                for index, _ in enumerate(read_fasta(fasta_path)):
                    if index < sample_size:
                        reservoir.append(index)
                    else:
                        replacement = rng.randrange(index + 1)
                        if replacement < sample_size:
                            reservoir[replacement] = index
                if len(reservoir) < sample_size:
                    raise ValueError("sample_size exceeds the number of FASTA records")
                selected = set(reservoir)
            # Keep staging beside its destination for an atomic rename.
            with tempfile.NamedTemporaryFile(
                mode="w", prefix=".dj-fasta-", dir=output_path.parent, delete=False
            ) as handle:
                staging = Path(handle.name)
                try:
                    db.execute("CREATE TABLE fasta_ids (id TEXT PRIMARY KEY)")
                    for index, (identifier, sequence) in enumerate(read_fasta(fasta_path)):
                        if index % 1000 == 0:
                            check_space()
                        report["fasta_records"] += 1
                        try:
                            db.execute("INSERT INTO fasta_ids VALUES (?)", (identifier,))
                        except sqlite3.IntegrityError as error:
                            raise ValueError(f"Duplicate FASTA ID {identifier!r}") from error
                        match = db.execute("SELECT payload FROM annotations WHERE id = ?", (identifier,)).fetchone()
                        if match is None:
                            report["missing_annotations"] += 1
                            continue
                        row = json.loads(match[0])
                        db.execute("UPDATE annotations SET seen = 1 WHERE id = ?", (identifier,))
                        if "length" in row and row["length"] != len(sequence):
                            report["length_mismatches"] += 1
                        if selected is None or index in selected:
                            row.update(sequence=sequence, seq_type=seq_type)
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            report["written"] += 1
                    report["extra_annotations"] = db.execute(
                        "SELECT COUNT(*) FROM annotations WHERE seen=0"
                    ).fetchone()[0]
                    logger.info("FASTA annotation join: {}", report)
                    if any(report[key] for key in ("missing_annotations", "extra_annotations", "length_mismatches")):
                        raise ValueError(f"FASTA/annotation mismatch: {report}")
                    handle.close()
                    staging.replace(output_path)
                finally:
                    staging.unlink(missing_ok=True)
    return report


def main() -> None:
    """Parse conversion paths, sequence type and optional sampling; log counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_path", type=Path)
    parser.add_argument("annotation_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--seq-type", choices=("protein", "dna"), required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp-dir", type=Path)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    fasta_to_dj(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
