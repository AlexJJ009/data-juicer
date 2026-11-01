"""Fit MMseqs2 clusters globally, then attach representative IDs to sequence rows."""

import math
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from loguru import logger

from data_juicer.ops.base_op import OPERATORS
from data_juicer.ops.domain.fitted import FittedMapper


@OPERATORS.register_module("sequence_cluster_mapper")
class SequenceClusterMapper(FittedMapper):
    """Run easy-linclust or easy-cluster without deleting any input records."""

    _batched_op = True

    def __init__(
        self,
        mmseqs_path: str = "mmseqs",
        identity: float = 0.95,
        coverage: float = 0.8,
        coverage_mode: int = 0,
        threads: int = 4,
        method: str = "easy-linclust",
        sequence_key: str = "sequence",
        id_key: str = "seq_id",
        cluster_key: str = "cluster_id",
        representative_key: str = "is_representative",
        temp_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure executable, similarity, coverage, CPU count, method and fields.

        :param mmseqs_path: Executable name on PATH or explicit executable path.
        :param identity: Minimum alignment identity fraction.
        :param coverage: Required sequence coverage fraction.
        :param coverage_mode: MMseqs2 coverage mode, default bidirectional (0).
        :param threads: Number of MMseqs2 CPU threads.
        :param method: easy-linclust or easy-cluster.
        :param sequence_key: Sequence input field.
        :param id_key: Unique string identifier field.
        :param cluster_key: Destination for the original representative ID.
        :param representative_key: Destination for the representative Boolean.
        :param temp_dir: Parent directory for disposable MMseqs2 files.
        :param kwargs: Standard Mapper arguments.

        coverage_mode=0 requires bidirectional coverage; 0.8 is a demo policy that
        avoids merging short fragments based on a shared local motif. identity is
        passed as --min-seq-id, with MMseqs2 alignment identity (--seq-id-mode 0).
        Temporary FASTA, databases and logs are removed even on subprocess failure.
        temp_dir selects the parent of the isolated temporary directory.
        """
        super().__init__(**kwargs)
        if any(not math.isfinite(x) or not 0 < x <= 1 for x in (identity, coverage)):
            raise ValueError("identity and coverage must be finite in (0, 1]")
        if threads < 1 or coverage_mode not in range(6) or method not in ("easy-linclust", "easy-cluster"):
            raise ValueError("Invalid threads, coverage_mode or clustering method")
        self.mmseqs_path, self.identity, self.coverage = mmseqs_path, identity, coverage
        self.coverage_mode, self.threads, self.method = coverage_mode, threads, method
        self.sequence_key, self.id_key = sequence_key, id_key
        self.cluster_key, self.representative_key, self.temp_dir = cluster_key, representative_key, temp_dir
        self.clusters: dict[str, str] = {}

    def fit(self, dataset: Iterable[dict[str, Any]]) -> None:
        """Cluster every row, validate total membership, and retain only ID assignments."""
        executable = shutil.which(self.mmseqs_path)
        if executable is None:
            raise FileNotFoundError(f"MMseqs2 executable not found or not executable: {self.mmseqs_path}")
        self.clusters = {}
        ids: list[str] = []
        seen: set[str] = set()
        with tempfile.TemporaryDirectory(prefix="dj-mmseqs-", dir=self.temp_dir) as directory:
            root = Path(directory)
            fasta = root / "input.fasta"
            with fasta.open("w") as handle:
                for row in dataset:
                    identifier, sequence = row[self.id_key], row[self.sequence_key]
                    if not isinstance(identifier, str) or not identifier:
                        raise ValueError("Sequence IDs must be nonempty strings")
                    if identifier in seen:
                        raise ValueError(f"duplicate sequence ID: {identifier!r}")
                    if not isinstance(sequence, str) or not sequence or any(c.isspace() for c in sequence):
                        raise ValueError(f"Empty or whitespace-containing sequence for {identifier!r}")
                    seen.add(identifier)
                    # Surrogates protect FASTA parsing from spaces and punctuation in source IDs.
                    handle.write(f">{len(ids)}\n{sequence}\n")
                    ids.append(identifier)
            if not ids:
                return
            prefix = root / "result"
            command = [
                executable,
                self.method,
                str(fasta),
                str(prefix),
                str(root / "tmp"),
                "--min-seq-id",
                str(self.identity),
                "-c",
                str(self.coverage),
                "--cov-mode",
                str(self.coverage_mode),
                "--seq-id-mode",
                "0",
                "--threads",
                str(self.threads),
                "--remove-tmp-files",
                "1",
            ]
            logger.info("MMseqs2 command: {}", command)
            log = root / "mmseqs.log"
            with log.open("w") as handle:
                result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                # Preserve a bounded diagnostic in the raised error before temporary cleanup.
                with log.open("rb") as handle:
                    handle.seek(max(0, log.stat().st_size - 8192))
                    tail = handle.read().decode(errors="replace")
                raise RuntimeError(f"MMseqs2 failed with exit {result.returncode}: {tail}")
            memberships: dict[int, int] = {}
            with Path(f"{prefix}_cluster.tsv").open() as handle:
                for line in handle:
                    columns = line.rstrip("\n").split("\t")
                    if len(columns) != 2:
                        raise ValueError(f"Malformed MMseqs2 membership row: {line!r}")
                    representative, member = map(int, columns)
                    if not (0 <= representative < len(ids) and 0 <= member < len(ids)) or member in memberships:
                        raise ValueError("MMseqs2 returned an unknown or duplicate membership")
                    memberships[member] = representative
            if len(memberships) != len(ids) or any(memberships.get(rep) != rep for rep in memberships.values()):
                raise ValueError("MMseqs2 membership is incomplete or representatives are inconsistent")
            self.clusters = {ids[member]: ids[rep] for member, rep in memberships.items()}
            logger.info("MMseqs2 input rows={}, clusters={}", len(ids), len(set(self.clusters.values())))

    def process_single(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Return row with fitted representative ID and representative Boolean."""
        identifier = sample[self.id_key]
        if identifier not in self.clusters:
            raise ValueError(f"Sequence ID absent from fitted clusters: {identifier!r}")
        sample[self.cluster_key] = self.clusters[identifier]
        sample[self.representative_key] = identifier == self.clusters[identifier]
        return sample
