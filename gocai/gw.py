from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple

import torch

from config import GridworldInstance
from eval import evaluate
from gridworld import Gridworld, MEADOW, GOAL, WALL
from guide import MLPPolicy, Guide
from objective import LtEObjective
from render import grid_image, save_png
from smc import SMC
from trace import Trace


def _parse_loc(s: str) -> Tuple[int, int]:
    # user passes "i,j" in 0-based coords (we keep it simple)
    i_str, j_str = s.split(",")
    return int(i_str), int(j_str)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gw", description="Gridworld variational SMC training (Python port)")
    # optimization
    p.add_argument("--niter", type=int, default=100_000)
    p.add_argument("--nplateau", type=int, default=0)
    p.add_argument("--nwarmup", type=int, default=1000)
    p.add_argument("--rate", type=float, default=2e-5)
    p.add_argument("--minrate", type=float, default=0.0)
    p.add_argument("--decay", type=float, default=0.0)
    p.add_argument("--cosinelr", action="store_true")
    # smc/eval
    p.add_argument("--nsteps", type=int, default=20)
    p.add_argument("--msteps", type=int, default=0)
    p.add_argument("--nparticles", type=int, default=10)
    p.add_argument("--mparticles", type=int, default=10_000)
    p.add_argument("--mbins", type=int, default=0)
    # annealing
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--annealing", type=float, default=1.0)
    p.add_argument("--miness", type=float, default=0.2)
    p.add_argument("--no-deterministic-policy", action="store_true")
    p.add_argument("--no-shared-dynamics", action="store_true")
    # policy net
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--lrwidth", type=int, default=64)
    # domain
    p.add_argument("--rscale", type=float, default=1.0)
    p.add_argument("--psucc", type=float, default=0.0)
    p.add_argument("--startloc", type=_parse_loc, default=None)
    # output
    p.add_argument("--imgpath", type=Path, default=Path("."))
    p.add_argument("--savegrid", action="store_true")
    p.add_argument("--showguide", action="store_true", default=True)
    p.add_argument("instance", nargs="?", default=None, help="path to instance JSON (optional)")
    return p


def load_env(args: argparse.Namespace) -> Tuple[str, Gridworld, Tuple[int, int]]:
    if args.instance is None:
        name = "builtin"
        gw = Gridworld.new_builtin(
            width=3,
            height=6,
            landscape={
                MEADOW: [(1, 1), (2, 3), (2, 4)],
                GOAL: [(2, 5)],
                WALL: [(0, 4), (1, 4)],
            },
            psucc=0.8,
            rscale=args.rscale,
        )
    else:
        path = Path(args.instance)
        name = path.stem
        inst = GridworldInstance.load(path)
        # rebuild as Gridworld
        gw = Gridworld(
            grid=inst.Grid,
            psucc=inst.PSucc,
            rscale=args.rscale,
            s0=(inst.S0[0], inst.S0[1]),
            xmin=inst.Xmin, xmax=inst.Xmax, dx=inst.Dx,
            ymin=inst.Ymin, ymax=inst.Ymax, dy=inst.Dy,
        )

    # startloc override
    if args.startloc is None:
        i0, j0 = gw.state_to_loc(gw.s0)
    else:
        i0, j0 = args.startloc
        gw.s0 = gw.loc_to_state(i0, j0)

    # psucc override
    if args.psucc != 0.0:
        gw.psucc = args.psucc

    return name, gw, (i0, j0)


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    # defaults mirroring Go's init() logic
    if args.minrate == 0.0:
        if args.decay == 0.0:
            args.minrate = 0.1 * args.rate
        else:
            args.minrate = args.rate * (args.decay ** args.niter)
    if args.decay == 0.0:
        args.decay = (args.minrate / args.rate) ** (1.0 / args.niter)
    if args.msteps == 0:
        args.msteps = args.nsteps
    if args.nplateau == 0:
        args.nplateau = args.niter
    if args.mbins == 0:
        args.mbins = max(1, args.mparticles // 1000)

    name, gw, startloc = load_env(args)

    args.imgpath.mkdir(parents=True, exist_ok=True)

    if args.savegrid:
        save_png(grid_image(gw.grid), args.imgpath / f"{name}-grid.png")

    # guide
    policy = MLPPolicy(in_dim=2, out_dim=4, hidden_layers=args.nlayers, hidden_width=args.lrwidth)
    guide = Guide(policy=policy)

    # initial eval
    er0 = evaluate(gw, guide, args.imgpath / f"{name}-initial", args.mparticles, args.msteps, args.mbins, args.rscale,
                   startloc, args.showguide)
    print(f"initial expected return: {er0:.3f}")

    # training setup
    smc = SMC(mdp=gw, guide=guide, nsteps=args.nsteps, min_ess=args.miness,
              no_deterministic_policy=args.no_deterministic_policy,
              no_shared_dynamics=args.no_shared_dynamics)
    tr = Trace.new(args.nsteps, args.nparticles)
    objective = LtEObjective(n_actions=4)
    opt = torch.optim.Adam(policy.parameters(), lr=args.rate)

    # progress smoothing factor (same formula)
    gamma = 1.0 - 1.0 / max(2.0, float(min(100, args.niter // 10)))
    elogpy = None
    elogpy_best = -1e30
    theta_best = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    plateau = 0
    temp = args.temp

    particles = [gw.s0 for _ in range(args.nparticles)]

    report_every = max(1, min(100, args.niter // 10))
    for it in range(args.niter):
        particles = [gw.s0 for _ in range(args.nparticles)]
        logpy = smc.sweep(particles, temp, tr)

        # build flat state tensor [nsteps*nparticles, 2]
        states_flat = torch.tensor([s for step in tr.states for s in step], dtype=torch.float32)
        logits = guide.logits(states_flat)
        loss = -objective(tr, logits)  # maximize objective

        opt.zero_grad()
        loss.backward()
        opt.step()

        logpy_f = float(logpy)
        if it == 0:
            elogpy = logpy_f
            elogpy_best = logpy_f
        else:
            elogpy = gamma * elogpy + (1.0 - gamma) * logpy_f

        if (it + 1) % report_every == 0:
            er = evaluate(gw, guide, args.imgpath / f"{name}", args.mparticles, args.msteps, args.mbins, args.rscale,
                          startloc, args.showguide)
            print(f"iter {it}: plateau {plateau}: E(logpy): {elogpy:.3f} Er: {er:.3f}")

        if (elogpy > elogpy_best) or (it < args.nwarmup):
            elogpy_best = elogpy
            theta_best = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            plateau = 0
        else:
            plateau += 1
            if plateau == args.nplateau:
                break

        # lr schedule
        if args.cosinelr:
            lr = args.minrate + 0.5 * (args.rate - args.minrate) * (
                    1.0 + math.cos(float(it + 1) / float(args.niter) * math.pi))
            for pg in opt.param_groups:
                pg['lr'] = lr
        else:
            for pg in opt.param_groups:
                pg['lr'] *= args.decay

        temp *= args.annealing

    policy.load_state_dict(theta_best)

    erf = evaluate(gw, guide, args.imgpath / f"{name}-final", args.mparticles, args.msteps, args.mbins, args.rscale,
                   startloc, args.showguide)
    print(f"final: E(logpy): {elogpy_best:.3f} Er: {erf:.3f}")


if __name__ == "__main__":
    main()
