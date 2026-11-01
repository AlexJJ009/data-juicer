"""Prepare topic-two inputs, invoke the standard DJ CLI, and record execution metrics."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from loguru import logger

from tools.fmt_conversion.bio_sequence.fasta_to_dj import fasta_to_dj

REPO = Path(__file__).resolve().parents[2]


def cap_debug_logs(directory: Path, limit_bytes: int) -> int:
    """Cap upstream DEBUG files and retain a small diagnostic tail; return discarded bytes."""
    if limit_bytes <= 0:
        raise ValueError("DEBUG log size limit must be positive")
    discarded = 0
    for path in directory.rglob("*_DEBUG.txt"):
        try:
            size = path.stat().st_size
            if size <= limit_bytes:
                continue
            with path.open("rb+") as handle:
                handle.seek(max(0, size - min(limit_bytes, 64 * 1024)))
                tail = handle.read(min(limit_bytes, 64 * 1024))
                handle.truncate(0)
            (directory / "debug_tail.log").write_bytes(tail)
            discarded += size
        except FileNotFoundError:
            continue
    return discarded


def run(
    kind: str,
    data_dir: Path,
    output_dir: Path,
    mmseqs_path: str = "mmseqs",
    seed: int = 42,
    sample_size: int | None = None,
    min_free_gib: float = 5.0,
    debug_log_limit_mib: int = 32,
) -> None:
    """Prepare inputs, write the recipe, and record CLI command, time and return code."""
    data_dir, output_dir = data_dir.resolve(), output_dir.resolve()
    if kind not in ("qa", "protein", "dna") or min_free_gib < 5 or debug_log_limit_mib < 1:
        raise ValueError("kind must be qa/protein/dna; reserve >= 5 GiB; DEBUG limit >= 1 MiB")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists():
        raise ValueError(f"Existing run metrics in {output_dir}; choose a fresh output directory")
    if shutil.disk_usage(output_dir).free < min_free_gib * 1024**3:
        raise RuntimeError("Insufficient free disk before starting conversion")
    for name in ("tmp", "datasets-cache"):
        (output_dir / name).mkdir(exist_ok=True)
    start = perf_counter()
    metrics: dict[str, Any] = dict(kind=kind, seed=seed)
    try:
        if kind == "qa":
            (output_dir / "input").mkdir(exist_ok=True)
            for index in (1, 2, 3):
                source = data_dir / f"qa_pairs_{index}.jsonl"
                link = output_dir / "input" / source.name
                if not link.exists():
                    link.symlink_to(source)
        else:
            suffix = "faa" if kind == "protein" else "fna"
            metrics["conversion"] = fasta_to_dj(
                data_dir / f"Elephant_Gut_{suffix}_data_example.fasta",
                data_dir / f"Elephant_Gut_{suffix}_anno_example.jsonl",
                output_dir / "input.jsonl",
                kind,
                sample_size if sample_size is not None else (300000 if kind == "protein" else 20000),
                seed,
                output_dir / "tmp",
                min_free_gib,
            )
        metrics["preparation_seconds"] = perf_counter() - start
        config = yaml.safe_load(Path(__file__).with_name(f"{kind}.yaml").read_text())
        for stage in config["process"]:
            if "sequence_cluster_mapper" in stage:
                stage["sequence_cluster_mapper"]["mmseqs_path"] = (
                    str(Path(mmseqs_path).resolve()) if "/" in mmseqs_path else mmseqs_path
                )
        recipe = output_dir / "recipe.yaml"
        recipe.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])),
        }
        command = [sys.executable, "-m", "tools.process_data", "--config", str(recipe)]
        metrics["command"] = command
        logger.info("DJ command: {} (cwd={})", command, output_dir)
        with (output_dir / "process.log").open("w") as handle:
            with subprocess.Popen(
                command, cwd=output_dir, env=environment, stdout=handle, stderr=subprocess.STDOUT
            ) as process:
                # Upstream always creates a DEBUG sink, even with debug=false.
                while True:
                    try:
                        metrics["returncode"] = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        cap_debug_logs(output_dir, debug_log_limit_mib * 1024**2)
        if metrics["returncode"]:
            raise RuntimeError(f"DJ failed; inspect {output_dir / 'process.log'}")
    finally:
        metrics["total_seconds"] = perf_counter() - start
        (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        for name in ("tmp", "datasets-cache"):
            shutil.rmtree(output_dir / name)


def main() -> None:
    """Parse dataset, output and clustering configuration and run one recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("qa", "protein", "dna"))
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--mmseqs-path", default="mmseqs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--min-free-gib", type=float, default=5.0)
    parser.add_argument("--debug-log-limit-mib", type=int, default=32)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
