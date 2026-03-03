from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw

from gridworld import PAVEMENT, MEADOW, DIRT, WALL, SWAMP, GOAL, RIGHT, UP, LEFT, DOWN

CELL_SIZE = 72
CELL_MARGIN = CELL_SIZE // 6

CELL_COLOR = {
    PAVEMENT: (128, 126, 120, 255),
    MEADOW: (52, 124, 56, 255),
    DIRT: (114, 93, 76, 255),
    WALL: (196, 88, 36, 255),
    SWAMP: (54, 84, 68, 255),
    GOAL: (255, 215, 0, 255),
}


def grid_image(grid: List[List[int]]) -> Image.Image:
    w = len(grid)
    h = len(grid[0])
    img = Image.new("RGBA", (w * CELL_SIZE, h * CELL_SIZE), (255, 255, 255, 255))
    dr = ImageDraw.Draw(img)
    for i in range(w):
        for j in range(h):
            x0 = i * CELL_SIZE
            y0 = (h - j - 1) * CELL_SIZE
            dr.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], fill=CELL_COLOR[grid[i][j]])
            dr.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], outline=(0, 0, 0, 255), width=2)
    return img


def overlay_visits(img: Image.Image, visits: List[List[int]], i_start: int, j_start: int) -> Image.Image:
    w = len(visits)
    h = len(visits[0])
    mx = max(max(row) for row in visits) if visits else 1
    dr = ImageDraw.Draw(img, "RGBA")
    for i in range(w):
        for j in range(h):
            v = visits[i][j]
            if v <= 0:
                continue
            a = int(180 * (v / mx))
            x0 = i * CELL_SIZE + CELL_MARGIN
            y0 = (h - j - 1) * CELL_SIZE + CELL_MARGIN
            x1 = (i + 1) * CELL_SIZE - CELL_MARGIN
            y1 = (h - j) * CELL_SIZE - CELL_MARGIN
            dr.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, a))
    # start marker
    x0 = i_start * CELL_SIZE + CELL_SIZE // 3
    y0 = (h - j_start - 1) * CELL_SIZE + CELL_SIZE // 3
    x1 = x0 + CELL_SIZE // 3
    y1 = y0 + CELL_SIZE // 3
    dr.ellipse([x0, y0, x1, y1], fill=(0, 0, 0, 255))
    return img


def overlay_guide_arrows(img: Image.Image, probs: List[List[List[float]]]) -> Image.Image:
    """
    Draw 4 straight guide lines per cell (RIGHT/UP/LEFT/DOWN):

    - All start at the exact same cell center.
    - Length is proportional to action probability.
    - If p == 1.0, the line reaches the inner edge of the cell (not outside).
    - Lines are thin and white.
    """
    import math

    w = len(probs)
    h = len(probs[0])
    dr = ImageDraw.Draw(img, "RGBA")

    line_w = 1  # thinner

    # We stop at the inner edge (inside the black border), so it looks clean.
    # From center to inner edge: half cell minus border margin.
    max_len = (CELL_SIZE / 2) - 3  # tweak 3px so it doesn't touch the border

    def seg(cx: float, cy: float, dx: float, dy: float, p: float) -> None:
        if p <= 0.0:
            return
        p = float(p)
        # p=1 => reaches edge; also clamp for safety
        if p > 1.0:
            p = 1.0

        x1 = cx + dx * (max_len * p)
        y1 = cy + dy * (max_len * p)

        # always white; alpha slightly scaled so small probs still visible
        alpha = int(60 + 195 * p)  # p=0 => invisible (we return), p=1 => 255
        dr.line([cx, cy, x1, y1], fill=(255, 255, 255), width=line_w)

    for i in range(w):
        for j in range(h):
            p = probs[i][j]  # [RIGHT, UP, LEFT, DOWN]

            # normalize defensively (keeps behavior stable if float drift)
            s = float(sum(p))
            if not (s > 0.0) or not math.isfinite(s):
                p = [0.25, 0.25, 0.25, 0.25]
            else:
                p = [max(0.0, float(x)) / s for x in p]

            cx = i * CELL_SIZE + CELL_SIZE * 0.5
            cy = (h - j - 1) * CELL_SIZE + CELL_SIZE * 0.5

            seg(cx, cy, +1.0, 0.0, p[RIGHT])
            seg(cx, cy, 0.0, -1.0, p[UP])
            seg(cx, cy, -1.0, 0.0, p[LEFT])
            seg(cx, cy, 0.0, +1.0, p[DOWN])

    return img


def save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
