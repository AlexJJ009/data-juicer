"""Call h5ad conversion, the standard DJ CLI, then h5ad export sequentially."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from loguru import logger

REPO = Path(__file__).resolve().parents[2]


def run(input_path: Path, output_dir: Path, config: Path) -> None:
    """Convert input_path through DJ using config into output_dir; return None.

    Each subprocess is an existing CLI. Failures propagate without retries.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / "kang2018.jsonl"
    refined = output_dir / "refined.jsonl"
    commands = [
        ["tools.fmt_conversion.single_cell.h5ad_to_dj", str(input_path.resolve()), str(source)],
        [
            "tools.process_data",
            "--config",
            str(config.resolve()),
            "--dataset_path",
            str(source),
            "--export_path",
            str(refined),
        ],
        ["tools.fmt_conversion.single_cell.dj_to_h5ad", str(refined), str(output_dir / "refined.h5ad")],
    ]
    # Run inside output_dir so the recipe's relative report and cache paths land next to
    # the data; PYTHONPATH keeps tools/ and data_juicer/ resolving to this checkout.
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))}
    for arguments in commands:
        start = perf_counter()
        subprocess.run([sys.executable, "-m", *arguments], check=True, cwd=output_dir, env=environment)
        logger.info("{} total seconds: {:.3f}", arguments[0], perf_counter() - start)


def main() -> None:
    """Parse source, destination and recipe paths; run the three CLI steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("kang2018.yaml"))
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
