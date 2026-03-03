from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from gridworld import Gridworld
from guide import Guide
from trace import Trace


def _sample_categorical(ps: List[float]) -> int:
    u = random.random()
    cum = 0.0
    for i, p in enumerate(ps):
        cum += p
        if u < cum:
            return i
    return len(ps) - 1


def _systematic_resample(probs: List[float]) -> List[int]:
    # mirrors Go's resampleU
    n = len(probs)
    step = 1.0 / n
    u = random.random()
    idxs = [0] * n
    i = -1
    cum = 0.0
    k = 0
    while k != n:
        if u < cum:
            idxs[k] = i
            k += 1
            u += step
        else:
            i = (i + 1) % n
            cum += probs[i]
    return idxs


@dataclass
class SMC:
    mdp: Gridworld
    guide: Guide
    nsteps: int
    min_ess: float = 0.2
    no_deterministic_policy: bool = False
    no_shared_dynamics: bool = False

    def sweep(self, particles: List[Tuple[float, float]], temp: float, tr: Trace) -> float:
        tr.temp = temp
        n_particles = len(particles)
        prior_logprob = -math.log(self.mdp.NACTIONS if hasattr(self.mdp, "NACTIONS") else 4)

        # policy memories: per particle, deterministic action per state key
        p_mem: List[Dict[int, int]] = [dict() for _ in range(n_particles)]

        # transition count: per particle counts of (skey,action)->ct
        t_cnt: List[Dict[Tuple[int, int], int]] = [dict() for _ in range(n_particles)]

        # transition memory shared across particles unless disabled
        t_mem: Dict[Tuple[int, int, int], Tuple[float, float]] = {}

        plogws = [0.0] * n_particles
        logpy = 0.0

        for is_ in range(self.nsteps):
            # advance all particles one step
            for ip in range(n_particles):
                state = particles[ip]
                tr.states[is_][ip] = state
                skey = self.mdp.skey(state)

                # choose action
                proposed = False
                if (skey not in p_mem[ip]) or self.no_deterministic_policy:
                    proposed = True
                    aprobs_t = self.guide.probs(state)
                    aprobs = aprobs_t.detach().cpu().numpy().tolist()
                    action = _sample_categorical(aprobs)
                    logprob = math.log(max(aprobs[action], 1e-30))
                    p_mem[ip][skey] = action
                else:
                    action = p_mem[ip][skey]
                    logprob = 0.0  # unused when not proposed

                tr.proposed[is_][ip] = proposed
                tr.actions[is_][ip] = action

                # transition with shared dynamics keyed by (skey, action, ct)
                key_sa = (skey, action)
                ct = t_cnt[ip].get(key_sa, 0)
                t_cnt[ip][key_sa] = ct + 1

                tmkey = (skey, action, ct)
                if (tmkey not in t_mem) or self.no_shared_dynamics:
                    state2 = self.mdp.step(state, action)
                    t_mem[tmkey] = state2
                else:
                    state2 = t_mem[tmkey]

                reward = self.mdp.reward(state, action, state2)
                tr.rewards[is_][ip] = reward

                plogws[ip] += reward
                if proposed:
                    plogws[ip] += temp * (prior_logprob - logprob)

                particles[ip] = state2

            # compute normalized weights
            maxlogw = max(plogws)
            w = [math.exp(lw - maxlogw) for lw in plogws]
            wsum = sum(w)
            probs = [wi / wsum for wi in w]

            # ESS (normalized by n)
            p2sum = sum(pi * pi for pi in probs)
            ess = (1.0 / p2sum) / n_particles
            resampled = (ess < self.min_ess) or (is_ == self.nsteps - 1)
            tr.resampled[is_] = resampled

            if resampled:
                # log evidence increment
                logpy += maxlogw + math.log(wsum) - math.log(n_particles)

                idxs = _systematic_resample(probs)
                particles = [particles[i] for i in idxs]
                p_mem = [p_mem[i].copy() for i in idxs]
                t_cnt = [t_cnt[i].copy() for i in idxs]
                plogws = [0.0] * n_particles

        # write back particles list (caller typically reuses container)
        for i in range(n_particles):
            particles[i] = particles[i]  # no-op, kept for symmetry
        return logpy
