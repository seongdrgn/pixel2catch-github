# pixel2catch

**Learning a Dynamic Object-Catching Policy from Pixels — Joint MAPPO Training in Isaac Lab.**

`pixel2catch` is an [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) *direct* RL task in
which a UR5e arm equipped with an Allegro hand learns to **catch a thrown object using
on-board camera (pixel) observations**. The arm (approach) and the hand (grasp) are treated
as two cooperating agents and trained jointly with **MAPPO** (Multi-Agent PPO).

> This repository contains **our method only** — the `pixel2catch` environment, its
> configuration, and the agent (MAPPO) config. Baseline variants and development
> iterations are intentionally excluded.

---

## Training algorithm

Training uses the **MAPPO algorithm from Isaac Lab's built-in [skrl](https://skrl.readthedocs.io)
integration, without any modification.** We do not ship a custom RL library — the policy is
trained by Isaac Lab's standard skrl runner driven by the agent config in
[`catchpolicy/agents/pixel2catch.yaml`](catchpolicy/agents/pixel2catch.yaml)
(`agent.class: MAPPO`, `trainer.class: SequentialTrainer`). Verified identical to upstream
`skrl==1.4.3`.

---

## Requirements

- Ubuntu 22.04
- NVIDIA GPU + recent driver (RTX-class recommended for camera/tiled rendering)
- Python 3.10 (Conda recommended)
- **Isaac Sim + Isaac Lab** — install first via the official guide:
  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- `skrl==1.4.3` (see [`requirements.txt`](requirements.txt))

```bash
# after Isaac Lab is installed and its conda env is active
pip install -r requirements.txt
```

---

## Installation (register the task into Isaac Lab)

This repo is an Isaac Lab *direct* task package. Copy (or symlink) the `catchpolicy/`
folder into Isaac Lab's direct-tasks directory so it is auto-discovered and registered:

```bash
# from the root of this repository
cp -r catchpolicy \
  /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/

# or symlink (keeps a single source of truth)
ln -s "$(pwd)/catchpolicy" \
  /path/to/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/catchpolicy
```

The task registers the Gym id **`pixel2catch`** (see `catchpolicy/__init__.py`).

### Assets

The robot/table USD assets needed by the task are bundled under
`catchpolicy/assets/` and are resolved automatically relative to the package
(no absolute paths). Specifically:

- `catchpolicy/assets/allegroUR5e/...` — UR5e + Allegro hand robot
- `catchpolicy/assets/table.usd` — table

Thrown objects are generated procedurally (cones/cylinders), so no object USD library
is required.

---

## Training

Train the joint arm+hand catching policy with MAPPO using Isaac Lab's built-in skrl trainer:

```bash
# run from the IsaacLab root
python scripts/reinforcement_learning/skrl/train.py \
    --task pixel2catch \
    --algorithm MAPPO \
    --num_envs 4096 \
    --headless
```

Logs and checkpoints are written under `logs/skrl/pixel2catch/` (configured by the
`experiment` block in `pixel2catch.yaml`).

---

## Evaluation / Play

Roll out a trained checkpoint:

```bash
python scripts/reinforcement_learning/skrl/play.py \
    --task pixel2catch \
    --algorithm MAPPO \
    --num_envs 16 \
    --checkpoint /path/to/logs/skrl/pixel2catch/<run>/checkpoints/best_agent.pt
```

---

## Monitoring (TensorBoard)

```bash
# from the IsaacLab root
./isaaclab.sh -p -m tensorboard.main --logdir=logs/skrl/pixel2catch
```

---

## Repository layout

```
pixel2catch-github/
├── README.md
├── requirements.txt
├── .gitignore
└── catchpolicy/
    ├── __init__.py            # registers the `pixel2catch` Gym task
    ├── pixel2catch.py         # DynamicCatchEnv (environment logic)
    ├── pixel2catch_cfg.py     # DynamicCatchEnvCfg (scene / sensors / rewards)
    ├── agents/
    │   ├── __init__.py
    │   └── pixel2catch.yaml   # MAPPO agent config (skrl)
    └── assets/
        ├── allegroUR5e/       # UR5e + Allegro hand USD
        └── table.usd
```

---

## Troubleshooting

For Anaconda users, if you hit shared-library errors, complete the Isaac Lab setup and
source the Isaac Sim conda env:

```bash
cd /path/to/IsaacLab/_isaac_sim && source setup_conda_env.sh
```
