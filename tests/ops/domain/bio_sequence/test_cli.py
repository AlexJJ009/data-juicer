"""Exercise all demo recipes through the standard DJ CLI with synthetic inputs."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("recipe", ["qa", "protein", "dna"])
def test_recipe_cli_execution(recipe: str, tmp_path: Path) -> None:
    """Tiny recipes execute without global required-argument or Arrow schema errors."""
    repo = Path(__file__).resolve().parents[4]
    source = tmp_path / "input.jsonl"
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 3 if recipe != "dna" else "ACGT" * 40
    if recipe == "qa":
        rows = [
            {
                "data_no": f"gcmetaNCYC_20261027150104_{index}",
                "input": f"Does <protein>{sequence}<protein> correspond to a functional enzyme?",
                "output": "KO K00001 EC:1.1.1.1",
            }
            for index in range(2)
        ]
    else:
        rows = [
            {"seq_id": str(index), "sequence": sequence + "A" * index, "gtdb.D_name": "Bacteria", "gtdb.G_name": None}
            for index in range(2)
        ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    executable = tmp_path / "mmseqs"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nfrom pathlib import Path\n"
        "ids = [line[1:].strip() for line in Path(sys.argv[2]).read_text().splitlines() if line.startswith('>')]\n"
        "Path(sys.argv[3] + '_cluster.tsv').write_text(''.join(f'{ids[0]}\\t{i}\\n' for i in ids))\n"
    )
    executable.chmod(0o755)
    config = yaml.safe_load((repo / f"demos/process_topic2/{recipe}.yaml").read_text())
    config.update(
        dataset_path=str(source),
        export_path=str(tmp_path / "refined.jsonl"),
        temp_dir=str(tmp_path / "temp"),
        ds_cache_dir=str(tmp_path / "cache"),
        work_dir=str(tmp_path / "work"),
    )
    for stage in config["process"]:
        arguments = next(iter(stage.values()))
        if "stats_export_path" in arguments:
            arguments["stats_export_path"] = str(tmp_path / Path(arguments["stats_export_path"]).name)
        if "sequence_cluster_mapper" in stage:
            arguments.update(mmseqs_path=str(executable), temp_dir=str(tmp_path))
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [sys.executable, "-m", "tools.process_data", "--config", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": str(repo), "HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
    output = [json.loads(line) for line in (tmp_path / "refined.jsonl").read_text().splitlines()]
    assert len(output) == (1 if recipe == "qa" else 2)
    if recipe != "dna":
        assert output[0]["cluster_id"] == (rows[0].get("data_no") or "0")
    if recipe == "protein":
        assert output[1]["cluster_id"] == "0"
        assert output[1]["is_representative"] is False
    if recipe != "qa":
        assert output[0]["gtdb.D_name"] == "Bacteria"
        assert output[0]["gtdb.G_name"] is None

    (tmp_path / "process.log").write_text(result.stdout + result.stderr)
    from demos.process_topic2.summarize_issues import IssueCounter, compact_run

    (tmp_path / "metrics.json").write_text('{"total_seconds": 0}')
    if recipe == "qa":
        (tmp_path / "input").mkdir()
        (tmp_path / "input/qa_pairs_1.jsonl").symlink_to(source)
    summary = compact_run(tmp_path, recipe, IssueCounter())
    assert summary["steps"][-1]["after"] == len(output)
    assert sum(stage["removed"] for stage in summary["steps"]) == 2 - len(output)
    # Final audit statistics remain usable after large intermediate row exports are removed.
    for snapshot in tmp_path.glob("*_rows.jsonl"):
        snapshot.unlink()
    replayed = compact_run(tmp_path, recipe, IssueCounter())
    assert replayed["steps"] == summary["steps"]
