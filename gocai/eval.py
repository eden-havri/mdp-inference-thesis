from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from gridworld import Gridworld, SWAMP, GOAL, NACTIONS
from guide import Guide
from render import grid_image, overlay_visits, overlay_guide_arrows, save_png


def evaluate(gw: Gridworld, guide: Guide, fname_base: Path, mparticles: int, msteps: int, mbins: int, rscale: float,
             startloc: Tuple[int, int], showguide: bool) -> float:
    # precompute probs per cell to speed up action sampling (mirrors Go)
    probs = [[[None for _ in range(NACTIONS)] for __ in range(gw.height)] for ___ in range(gw.width)]
    for i in range(gw.width):
        for j in range(gw.height):
            state = gw.loc_to_state(i, j)
            p = guide.probs(state).detach().cpu().numpy().tolist()
            probs[i][j] = p

    visits = [[0 for _ in range(gw.height)] for __ in range(gw.width)]
    returns: List[float] = []
    er = 0.0

    rng = np.random.default_rng()
    for _ in range(mparticles):
        state = gw.s0
        pvis = [[0 for _ in range(gw.height)] for __ in range(gw.width)]
        i, j = gw.state_to_loc(state)
        pvis[i][j] = 1
        r = 0.0
        for __ in range(msteps):
            aprobs = probs[i][j]

            p = np.asarray(aprobs, dtype=np.float64)

            # Softmax in float32 can drift so p may not sum to 1 exactly.
            # Go’s categorical sampling doesn’t choke on tiny drift, so we normalize here.
            p = np.clip(p, 0.0, None)
            s = p.sum()

            if not np.isfinite(s) or s <= 0.0:
                p = np.full(NACTIONS, 1.0 / NACTIONS, dtype=np.float64)
            else:
                p /= s

            action = int(rng.choice(NACTIONS, p=p))

            state2 = gw.step(state, action)
            r += gw.reward(state, action, state2)
            state = state2
            i, j = gw.state_to_loc(state)
            pvis[i][j] = 1
        for i2 in range(gw.width):
            for j2 in range(gw.height):
                visits[i2][j2] += pvis[i2][j2]
        er += r
        returns.append(r)

    er /= float(mparticles)
    if rscale != 0.0:
        er /= rscale

    # render visits + optional guide
    img = grid_image(gw.grid)
    img = overlay_visits(img, visits, startloc[0], startloc[1])

    if showguide:
        # Go code makes guide uniform in swamp/goal
        probs_for_draw = [[[p for p in probs[i][j]] for j in range(gw.height)] for i in range(gw.width)]
        for i in range(gw.width):
            for j in range(gw.height):
                if gw.grid[i][j] in (SWAMP, GOAL):
                    probs_for_draw[i][j] = [1.0 / NACTIONS] * NACTIONS
        img = overlay_guide_arrows(img, probs_for_draw)

    path = fname_base.with_suffix('').as_posix() + "-visits.png" if isinstance(fname_base, Path) else str(
        fname_base) + "-visits.png"
    save_png(img, Path(path))

    # histogram
    plt.figure()
    plt.xlabel("return")
    plt.ylabel("density")
    plt.hist(np.array(returns), bins=mbins, density=True)
    out_png = fname_base.with_suffix('').as_posix() + "-returns.png"
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

    # returns json
    out_json = fname_base.with_suffix('').as_posix() + "-returns.json"
    Path(out_json).write_text(json.dumps(returns))

    return float(er)
