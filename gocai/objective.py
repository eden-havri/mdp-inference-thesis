from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class LtEObjective:
    """
    Autograd-safe port of Go's LtE.Observe.

    Computes the LtE objective from:
      - trace.rewards[step][particle]
      - trace.actions[step][particle]
      - trace.proposed[step][particle]
      - trace.resampled[step]
      - logits[step, particle, action]
    """
    n_actions: int

    def __call__(self, trace, all_logits: torch.Tensor) -> torch.Tensor:
        nsteps = trace.nsteps
        nparticles = trace.nparticles

        logits = all_logits.view(nsteps, nparticles, self.n_actions)

        prior_logprob = -math.log(self.n_actions)
        temp = float(trace.temp)

        # logpys[step] and logqs[step] are scalars (no in-place)
        logpys = []
        logqs = []

        # logweights is a differentiable tensor because it contains logprob terms
        logweights = torch.zeros(nparticles, dtype=logits.dtype, device=logits.device)

        for is_ in range(nsteps):
            # constants from trace (no grad)
            rewards = torch.tensor(
                trace.rewards[is_], dtype=logits.dtype, device=logits.device
            )

            actions = torch.tensor(
                trace.actions[is_], dtype=torch.long, device=logits.device
            )

            proposed = torch.tensor(
                trace.proposed[is_], dtype=logits.dtype, device=logits.device
            )  # 1.0 if proposed else 0.0

            # logprob of chosen action under current logits
            l = logits[is_]  # [P, A]
            logZ = torch.logsumexp(l, dim=-1)  # [P]
            chosen = l.gather(1, actions.view(-1, 1)).squeeze(1)  # [P]
            logprob = chosen - logZ  # [P]

            # same update as Go:
            # logweights += reward
            # if proposed: logweights += temp*(prior - logprob)
            logweights = logweights + rewards + proposed * (temp * (prior_logprob - logprob))

            # logq at this step is sum(logprob) over proposed particles
            logqs.append((proposed * logprob).sum())

            if trace.resampled[is_]:
                # logpy = logsumexp(logweights) - log(P)
                logpys.append(torch.logsumexp(logweights, dim=-1) - math.log(nparticles))
                # IMPORTANT: reset by creating a NEW tensor (no in-place)
                logweights = torch.zeros_like(logweights)
            else:
                logpys.append(torch.tensor(0.0, dtype=logits.dtype, device=logits.device))

        # Backward accumulation (same as Go code)
        pd = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
        sf = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

        for is_rev in range(nsteps):
            is_ = nsteps - is_rev - 1
            pd = pd + logpys[is_]
            sf = sf + logqs[is_] * pd

        return sf + pd
