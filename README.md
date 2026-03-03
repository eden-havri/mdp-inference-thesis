# gocai_py — Python 3 port of `gw.go`

This is a faithful, readable Python port of the Go `gw` experiment:
- Gridworld environment
- MLP policy (guide)
- SMC sweep with deterministic-per-particle policies + optional shared dynamics
- LtE objective (same math as Go's `vsmc/model/ad/model.go`)
- Training loop with LR decay / cosine schedule + temperature annealing
- Evaluation that produces:
  - `*-visits.png`
  - `*-returns.png`
  - `*-returns.json`

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic example run (fast sanity check)
Built-in instance:
```bash
python -m gocai.gw --niter 2000 --mparticles 500 --nparticles 20 --imgpath out
```

Run one of the provided JSON instances:
```bash
python -m gocai.gw --niter 2000 --mparticles 500 --nparticles 20 --imgpath out instances/0.json
```

Outputs will be written under `out/`:
- `<name>-initial-visits.png`, `<name>-initial-returns.png`, `<name>-initial-returns.json`
- periodic `<name>-visits.png`, `<name>-returns.png`, `<name>-returns.json` (overwritten)
- `<name>-final-visits.png`, `<name>-final-returns.png`, `<name>-final-returns.json`
