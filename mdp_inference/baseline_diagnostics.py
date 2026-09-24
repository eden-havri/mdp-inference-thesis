from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


_REQUIRED_ARTIFACTS = ("config.json", "environment.json", "map.json", "result.json", "DONE")
_MISSING = object()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _nonfinite_numeric_paths(value: Any, prefix: str = "result") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, item in value.items():
            paths.extend(_nonfinite_numeric_paths(item, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_nonfinite_numeric_paths(item, f"{prefix}[{index}]"))
        return paths
    if isinstance(value, float) and not math.isfinite(value):
        return [prefix]
    return []


def _finite_number(result: dict[str, Any], field: str) -> float | None:
    value = result.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def audit_baseline_gate(manifest_path: Path) -> dict[str, Any]:
    """Audit the fail-fast baseline gate without changing any run artifacts.

    A run passes only when its declared artifacts are complete, all numeric result
    fields are finite, its transition budget was reached, its committed-policy
    value does not exceed the exact oracle, and it strictly beats the matched
    random committed-policy value stored in the same result.
    """

    try:
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSONL from {manifest_path}") from exc
    if not rows:
        raise ValueError("baseline-gate manifest is empty")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("every baseline-gate manifest row must be a JSON object")

    run_ids = [row.get("run_id") for row in rows]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise ValueError("every baseline-gate manifest row needs a nonempty run_id")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("baseline-gate manifest contains duplicate run_id values")

    run_reports: dict[str, dict[str, Any]] = {}
    for row in rows:
        run_id = row["run_id"]
        output_root = Path(row.get("output_root", "results"))
        run_dir = output_root / run_id
        missing_artifacts = [name for name in _REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
        failures: list[str] = []
        if missing_artifacts:
            failures.append("missing artifacts: " + ", ".join(missing_artifacts))
        if (run_dir / "failure.json").exists():
            failures.append("failure.json is present")

        result: dict[str, Any] = {}
        config: dict[str, Any] = {}
        if not missing_artifacts:
            try:
                result = _load_json(run_dir / "result.json")
                config = _load_json(run_dir / "config.json")
            except ValueError as exc:
                failures.append(str(exc))

        config_mismatches: list[str] = []
        if result or config:
            for field, expected in row.items():
                actual = config.get(field, _MISSING)
                if actual != expected:
                    rendered_actual = "<missing>" if actual is _MISSING else repr(actual)
                    config_mismatches.append(
                        f"{field}: expected {expected!r}, found {rendered_actual}"
                    )
            if config_mismatches:
                failures.append(
                    "config does not reproduce manifest fields: "
                    + "; ".join(config_mismatches)
                )
        if result and result.get("method") != row.get("method"):
            failures.append(
                f"result method mismatch: expected {row.get('method')!r}, "
                f"found {result.get('method')!r}"
            )

        nonfinite = _nonfinite_numeric_paths(result)
        if nonfinite:
            failures.append("non-finite numeric result fields: " + ", ".join(nonfinite))

        committed = _finite_number(result, "committed_policy_value")
        random_value = _finite_number(result, "random_committed_value")
        oracle = _finite_number(result, "oracle_value")
        simulator_steps = _finite_number(result, "simulator_steps")
        transition_budget = _finite_number(result, "transition_budget")
        required_values = {
            "committed_policy_value": committed,
            "random_committed_value": random_value,
            "oracle_value": oracle,
            "simulator_steps": simulator_steps,
            "transition_budget": transition_budget,
        }
        missing_values = [name for name, value in required_values.items() if value is None]
        if missing_values:
            failures.append("missing or non-finite required values: " + ", ".join(missing_values))

        complete = not missing_artifacts and not (run_dir / "failure.json").exists()
        finite = not nonfinite and not missing_values
        manifest_budget = row.get("transition_budget")
        result_budget_matches_manifest = (
            isinstance(manifest_budget, int)
            and transition_budget is not None
            and transition_budget == manifest_budget
        )
        budget_reached = (
            simulator_steps is not None
            and transition_budget is not None
            and simulator_steps >= transition_budget
        )
        oracle_bounded = committed is not None and oracle is not None and committed <= oracle + 1e-10
        beats_random = (
            committed is not None and random_value is not None and committed > random_value
        )
        if not budget_reached and not missing_values:
            failures.append("training transition budget was not reached")
        if not result_budget_matches_manifest and transition_budget is not None:
            failures.append("result transition budget does not match the manifest")
        if not oracle_bounded and committed is not None and oracle is not None:
            failures.append("committed-policy value exceeds the exact oracle")
        if not beats_random and committed is not None and random_value is not None:
            failures.append("committed-policy value does not beat matched random")

        method = row.get("method")
        run_reports[run_id] = {
            "method": method,
            "passed": not failures,
            "checks": {
                "complete": complete,
                "finite": finite,
                "config_matches_manifest": not config_mismatches,
                "result_budget_matches_manifest": result_budget_matches_manifest,
                "budget_reached": budget_reached,
                "oracle_bounded": oracle_bounded,
                "beats_matched_random": beats_random,
            },
            "committed_policy_value": committed,
            "random_committed_value": random_value,
            "oracle_value": oracle,
            "improvement_over_random": (
                committed - random_value
                if committed is not None and random_value is not None
                else None
            ),
            "oracle_margin": oracle - committed if oracle is not None and committed is not None else None,
            "failures": failures,
        }

    method_reports: dict[str, dict[str, Any]] = {}
    methods = sorted({str(report["method"]) for report in run_reports.values()})
    for method in methods:
        reports = [report for report in run_reports.values() if report["method"] == method]
        improvements = [
            report["improvement_over_random"]
            for report in reports
            if report["improvement_over_random"] is not None
        ]
        oracle_margins = [
            report["oracle_margin"]
            for report in reports
            if report["oracle_margin"] is not None
        ]
        method_reports[method] = {
            "passed": all(report["passed"] for report in reports),
            "runs": len(reports),
            "passed_runs": sum(bool(report["passed"]) for report in reports),
            "minimum_improvement_over_random": min(improvements) if improvements else None,
            "minimum_oracle_margin": min(oracle_margins) if oracle_margins else None,
        }

    return {
        "passed": all(report["passed"] for report in run_reports.values()),
        "scope": "baseline_stability_gate",
        "criteria": {
            "complete_artifacts_and_budget": True,
            "all_numeric_result_fields_finite": True,
            "committed_policy_value_at_most_oracle": True,
            "committed_policy_value_strictly_above_matched_random": True,
        },
        "methods": method_reports,
        "runs": run_reports,
    }
