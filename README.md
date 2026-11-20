# How Much Should a World Model Reconstruct?

**Reconstruction Weight versus Task-Irrelevant Observations in DreamerV3**

Research project for **INF8250AE — Reinforcement Learning**, Polytechnique Montréal (Fall 2026).

**Authors:** Antoine Deckers, Romain Haessler, Nathan Lewy, and Lucas Silva.

## About the project

World models allow reinforcement learning agents to simulate future trajectories using learned representations of their environments. However, reconstructing observations may require the model to learn task-irrelevant information, potentially affecting its performance and sample efficiency.

This project investigates **how observation reconstruction influences DreamerV3's learning performance when observations contain irrelevant information**.

## Quickstart

To get started from inside a **Google Colab notebook** (runtime `2026.07`,
Python 3.12: `dm_control`/`labmaze` do not build on Python 3.13), run:

```
!git clone https://github.com/NathanLeonidas/RL_reconstruction_abstraction.git
%cd /content/RL_reconstruction_abstraction
!grep -v -E '^(ale_py|autorom)' requirements.txt > requirements_colab.txt
!pip install -U -r requirements_colab.txt
!pip install dm_control
!apt-get install -y -qq libosmesa6 libgl1-mesa-glx
```

## Approach

We will modify DreamerV3's prediction loss by introducing a weight $w$ on its observation reconstruction term. We will then compare different values of $w$ while adding varying numbers of **random, predictable, or constant distractors** to the agent's observations.

Our primary environment will be **Cartpole Swingup** from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), with additional environments considered if resources allow. We plan to train lightweight DreamerV3 models from scratch and evaluate performance and sample efficiency across multiple random seeds.

## Experiment parameters

| Parameter | Flag | Default |
|---|---|---|
| Reconstruction weight `w` | `--agent.loss_scales.rec` | `1.0` (standard DreamerV3) |
| Distractor type | `--distractors.type` | `none` (`random`, `predictable`, `constant`) |
| Number of distractors `k` | `--distractors.k` | `0` |
| Sinusoid periods (predictable) | `--distractors.period_min/max` | `10` / `100` steps |
| Seed | `--seed` | `0` |

`w` multiplies the decoder loss of every observation key, including the
`distractors` key. The distractors are a separate observation key, so their
reconstruction loss is logged on its own as `train/loss/distractors`. See
[embodied/envs/distractors.py](embodied/envs/distractors.py) for the exact
definitions and seeding.

## Tests

```
python -m pytest tests/ -q
```

- `tests/test_distractors.py`: the distractor wrapper (needs only `numpy`,
  `elements`, `pytest`).
- `tests/check_env.py` (not collected by pytest): the wrapper on the real DMC
  environment, `MUJOCO_GL=osmesa python tests/check_env.py --k 20`.

Smoke test on Colab (a few minutes, NOT a valid experiment: reduced
`batch_size`, `batch_length` and `train_ratio`). Check that the run ends with
exit code 0 and that `train/loss/distractors` appears in the logs:

```
!MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python dreamerv3/main.py \
  --configs dmc_proprio --task dmc_cartpole_swingup \
  --distractors.type random --distractors.k 20 \
  --run.envs 1 --batch_size 4 --batch_length 32 --run.train_ratio 8 \
  --run.steps 2000 --run.log_every 10 \
  --logdir /content/runs/smoke_random_k20
```

Pilot / timing run with the real hyperparameters (`dmc_proprio` config,
`train_eval` script, which also runs evaluation episodes):

```
!MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python dreamerv3/main.py \
  --configs dmc_proprio --task dmc_cartpole_swingup --script train_eval \
  --agent.loss_scales.rec 1.0 --distractors.type none --distractors.k 0 \
  --seed 0 --run.steps 20000 \
  --logdir /content/runs/pilot_none_k0_w1_s0
```

Relaunching the same command with the same `--logdir` resumes from the last
checkpoint.

## References

- [DreamerV3 paper](https://arxiv.org/abs/2301.04104) and [official implementation](https://github.com/danijar/dreamerv3)
- [TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html)
- [Policy-Shaped Prediction](https://arxiv.org/abs/2412.05766)

*Status: research project in development.*
