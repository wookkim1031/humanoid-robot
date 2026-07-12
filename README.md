# humanoid-robot

Zero shot humanoid motion control for G1
Pipeline: video/text -> GEM (SMPL-X) -> GMR (G1 retargeting) -> RL tracking
policy (PPO + MVAE + AMP, HIRO), simulated in mjlab (MuJuCo-Wrap) or Isaac Lab

## Setup

Requires Python 3.10 and [Poetry](https://python-poetry.org/).

```bash
poetry env use python3.10
poetry install
```

Run commands inside the environment with `poetry run <cmd>`, or activate a shell with `poetry shell`.


**GPU box (RunPod / L40S / H20 — training):**
```bash
pip install -e ".[mjlab,dev]"
python scripts/train/train_mjlab.py data/g1_motions/walk_01.npz --num-envs 8 --iters 5   # smoke
python scripts/train/train_mjlab.py data/g1_motions/walk_01.npz --num-envs 2048          # real
```

eval $(poetry env activate) 