from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .gridworld import GridWorldSpec


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def source_tree_sha256(root: Path) -> tuple[str, int]:
    """Hash the executable Python snapshot even when Git metadata is absent."""

    root = root.resolve()
    candidates = [root / "pyproject.toml"]
    for package in ("mdp_inference", "gocai"):
        candidates.extend((root / package).rglob("*.py"))
    files = sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest(), len(files)


def grid_spec_to_dict(spec: GridWorldSpec) -> dict[str, Any]:
    value = asdict(spec)
    for key in ("start",):
        value[key] = list(value[key])
    for key in ("goals", "walls", "hazards"):
        value[key] = [list(cell) for cell in value[key]]
    return value


def grid_spec_from_dict(value: dict[str, Any]) -> GridWorldSpec:
    return GridWorldSpec(
        rows=int(value["rows"]),
        cols=int(value["cols"]),
        start=tuple(value["start"]),
        goals=tuple(tuple(cell) for cell in value["goals"]),
        walls=tuple(tuple(cell) for cell in value.get("walls", [])),
        hazards=tuple(tuple(cell) for cell in value.get("hazards", [])),
        slip_probability=float(value.get("slip_probability", 0.2)),
        horizon=int(value.get("horizon", 40)),
        step_reward=float(value.get("step_reward", -0.1)),
        goal_reward=float(value.get("goal_reward", 5.0)),
        hazard_reward=float(value.get("hazard_reward", -5.0)),
        gamma=float(value.get("gamma", 1.0)),
    )


def load_grid_spec(path: Path) -> tuple[GridWorldSpec, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    declared_hash = value.pop("sha256", None)
    actual_hash = sha256_json(value)
    if declared_hash is not None and declared_hash != actual_hash:
        raise ValueError(f"map hash mismatch for {path}")
    return grid_spec_from_dict(value), actual_hash


def write_grid_spec(path: Path, spec: GridWorldSpec) -> str:
    value = grid_spec_to_dict(spec)
    digest = sha256_json(value)
    value_with_hash = {**value, "sha256": digest}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value_with_hash, indent=2) + "\n", encoding="utf-8")
    return digest


def software_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pid": os.getpid(),
    }
