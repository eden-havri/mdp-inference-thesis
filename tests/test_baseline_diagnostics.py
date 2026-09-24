from __future__ import annotations

import json
from pathlib import Path

import pytest

from mdp_inference.baseline_diagnostics import audit_baseline_gate


def _write_run(
    root: Path,
    run_id: str,
    *,
    method: str = "ppo",
    seed: int = 12,
    committed: float = 2.0,
    random_value: float = -1.0,
    oracle: float = 3.0,
    simulator_steps: int = 100,
    done: bool = True,
) -> dict[str, object]:
    row: dict[str, object] = {
        "run_id": run_id,
        "method": method,
        "training_seed": seed,
        "transition_budget": 100,
        "output_root": str(root),
    }
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    config = dict(row)
    result = {
        "method": method,
        "committed_policy_value": committed,
        "random_committed_value": random_value,
        "oracle_value": oracle,
        "simulator_steps": simulator_steps,
        "transition_budget": 100,
    }
    for name, value in (
        ("config.json", config),
        ("environment.json", {"source_snapshot_sha256": "same"}),
        ("map.json", {"name": "test"}),
        ("result.json", result),
    ):
        (run_dir / name).write_text(json.dumps(value), encoding="utf-8")
    if done:
        (run_dir / "DONE").write_text("ok\n", encoding="utf-8")
    return row


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_baseline_gate_passes_complete_finite_improving_runs(tmp_path: Path) -> None:
    rows = [
        _write_run(tmp_path / "results", "ppo-seed12", method="ppo", seed=12),
        _write_run(
            tmp_path / "results",
            "ppo-seed13",
            method="ppo",
            seed=13,
            committed=1.5,
        ),
        _write_run(
            tmp_path / "results",
            "cem-seed12",
            method="cem",
            seed=12,
            committed=0.0,
        ),
    ]
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, rows)

    audit = audit_baseline_gate(manifest)

    assert audit["passed"] is True
    assert audit["methods"]["ppo"]["passed"] is True
    assert audit["methods"]["ppo"]["runs"] == 2
    assert audit["methods"]["cem"]["minimum_improvement_over_random"] == 1.0


@pytest.mark.parametrize(
    ("changes", "failed_check"),
    [
        ({"done": False}, "complete"),
        ({"committed": float("nan")}, "finite"),
        ({"committed": 3.1}, "oracle_bounded"),
        ({"committed": -1.0}, "beats_matched_random"),
        ({"simulator_steps": 99}, "budget_reached"),
    ],
)
def test_baseline_gate_rejects_each_failure_mode(
    tmp_path: Path, changes: dict[str, object], failed_check: str
) -> None:
    row = _write_run(tmp_path / "results", "run", **changes)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [row])

    audit = audit_baseline_gate(manifest)

    assert audit["passed"] is False
    assert audit["runs"]["run"]["checks"][failed_check] is False
    assert audit["methods"]["ppo"]["passed"] is False


def test_baseline_gate_rejects_duplicate_manifest_ids(tmp_path: Path) -> None:
    row = _write_run(tmp_path / "results", "duplicate")
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [row, row])

    with pytest.raises(ValueError, match="duplicate run_id"):
        audit_baseline_gate(manifest)


def test_baseline_gate_rejects_method_hyperparameter_mismatch(tmp_path: Path) -> None:
    row = _write_run(tmp_path / "results", "run")
    row["learning_rate"] = 0.01
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [row])

    audit = audit_baseline_gate(manifest)

    assert audit["passed"] is False
    assert audit["runs"]["run"]["checks"]["config_matches_manifest"] is False
    assert "learning_rate" in audit["runs"]["run"]["failures"][0]
