from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdp_inference.artifacts import load_grid_spec  # noqa: E402


def main() -> None:
    maps = [
        ("Easy", ROOT / "experiments/maps/easy-seed0.json"),
        ("Medium", ROOT / "experiments/maps/medium-seed0.json"),
        ("Hard", ROOT / "experiments/maps/hard-seed0.json"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.55), constrained_layout=True)
    cmap = ListedColormap(["#ffffff", "#232323", "#2c7fb8", "#f4b942"])
    for axis, (tier, path) in zip(axes, maps):
        spec, _ = load_grid_spec(path)
        image = np.zeros((spec.rows, spec.cols), dtype=np.int64)
        for row, col in spec.walls:
            image[row, col] = 1
        image[spec.start] = 2
        for goal in spec.goals:
            image[goal] = 3
        axis.imshow(image, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(
            f"{tier}: {spec.rows}×{spec.cols}\n"
            f"walls {len(spec.walls)/(spec.rows*spec.cols):.1%}, "
            f"success {1.0-spec.slip_probability:.2f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#555555")
    output = ROOT / "latex/figures"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "gridworld-tiers.pdf", bbox_inches="tight")
    figure.savefig(output / "gridworld-tiers.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
