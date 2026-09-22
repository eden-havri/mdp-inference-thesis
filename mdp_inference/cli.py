from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import write_grid_spec
from .chain_diagnostics import audit_tempering_chains
from .experiment import ExperimentConfig, run_experiment
from .gridworld import GRID_DIFFICULTIES, generate_gridworld


def generate_maps(output_dir: Path, tiers: list[str], seeds: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for tier in tiers:
        if tier not in GRID_DIFFICULTIES:
            raise ValueError(f"unknown tier {tier!r}")
        for seed in seeds:
            spec = generate_gridworld(tier, seed)
            path = output_dir / f"{tier}-seed{seed}.json"
            digest = write_grid_spec(path, spec)
            manifest.append({"tier": tier, "seed": seed, "path": str(path), "sha256": digest})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_manifest_row(manifest_path: Path, index: int) -> Path:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if index < 0 or index >= len(rows):
        raise IndexError(f"manifest index {index} outside 0..{len(rows)-1}")
    return run_experiment(ExperimentConfig(**rows[index]))


def audit_results(manifest_path: Path) -> dict[str, list[str]]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = {"complete": [], "missing": [], "partial": [], "failed": []}
    for row in rows:
        run_dir = Path(row.get("output_root", "results")) / row["run_id"]
        if (run_dir / "DONE").exists():
            audit["complete"].append(row["run_id"])
        elif (run_dir / "failure.json").exists():
            audit["failed"].append(row["run_id"])
        elif run_dir.exists():
            audit["partial"].append(row["run_id"])
        else:
            audit["missing"].append(row["run_id"])
    return audit


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mdpi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    maps_parser = subparsers.add_parser("generate-maps")
    maps_parser.add_argument("--output-dir", type=Path, default=Path("experiments/maps"))
    maps_parser.add_argument("--tiers", nargs="+", default=["smoke", "easy", "medium", "hard"])
    maps_parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config", type=Path)

    row_parser = subparsers.add_parser("run-manifest-row")
    row_parser.add_argument("manifest", type=Path)
    row_parser.add_argument("index", type=int)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("manifest", type=Path)

    chain_audit_parser = subparsers.add_parser("audit-tempering-chains")
    chain_audit_parser.add_argument("output_root", type=Path)

    args = parser.parse_args(argv)
    if args.command == "generate-maps":
        generate_maps(args.output_dir, args.tiers, args.seeds)
    elif args.command == "run":
        print(run_experiment(ExperimentConfig.from_json(args.config)))
    elif args.command == "run-manifest-row":
        print(run_manifest_row(args.manifest, args.index))
    elif args.command == "audit":
        print(json.dumps(audit_results(args.manifest), indent=2))
    elif args.command == "audit-tempering-chains":
        print(json.dumps(audit_tempering_chains(args.output_root), indent=2))


if __name__ == "__main__":
    main()
