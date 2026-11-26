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

## Distractors

| Parameter | Flag | Default |
|---|---|---|
| Distractor type | `--distractors.type` | `none` (`random`, `predictable`, `constant`) |
| Number of distractors `k` | `--distractors.k` | `0` |
| Sinusoid periods (predictable) | `--distractors.period_min/max` | `10` / `100` steps |

The distractors are a separate observation key, so their reconstruction loss
is logged on its own as `train/loss/distractors`. They are seeded by `--seed`.
See [embodied/envs/distractors.py](embodied/envs/distractors.py) for the exact
definitions and seeding.

## Running experiments

Two scripts wrap `dreamerv3/main.py`, so that everyone runs with exactly the
same settings. Run them from the repository root.

### Installation

- **Colab (GPU, for the real runs):** follow the [Quickstart](#quickstart).
- **Mac (CPU, to try things):** Python 3.11 or 3.12 (`dm_control` and
  `jax==0.4.33` do not install on newer versions), without the CUDA packages:

  ```
  uv venv --python 3.11 ~/venvs/dreamerv3
  grep -v -E '^(ale_py|autorom|jax\[cuda12\]|nvidia-cuda-nvcc)' requirements.txt > requirements_mac.txt
  uv pip install --python ~/venvs/dreamerv3/bin/python -r requirements_mac.txt jax==0.4.33 jaxlib==0.4.33 dm_control pytest
  source ~/venvs/dreamerv3/bin/activate
  ```

### One run: `scripts/run_experiment.py`

```
python scripts/run_experiment.py --w 0.5 --distractors random --k 64 --seed 0 --profile gpu
```

| Option | Meaning | Default |
|---|---|---|
| `--w` | reconstruction weight (`agent.loss_scales.rec`) | `1.0` |
| `--distractors` | `none`, `random`, `predictable`, `constant` | `none` |
| `--k` | number of distractors | `0` without distractors, `20` otherwise |
| `--seed` | random seed | `0` |
| `--profile` | training budget, see below | `laptop` |
| `--steps` | override the number of steps of the profile | |
| `--outdir` | folder for the run folders (on Colab, put it in Google Drive) | `runs/` |
| `--dry-run` | print the DreamerV3 command without running it | |

| Profile | Steps | `train_ratio` | Use |
|---|---|---|---|
| `smoke` | 2K | 8 | checks that everything runs (~30 s on a laptop) |
| `laptop` | 50K | 64 | a real training loop on a CPU (~20 min), too short to learn |
| `gpu` | 100K | 1024 | official DreamerV3 budget for DMC proprio, needs a GPU (provisional values) |

The script picks the MuJoCo rendering backend (`glfw` on macOS, `osmesa` on
Linux) and the device (CPU on macOS, GPU elsewhere). It uses DreamerV3's
`train` script, because `train_eval` logs evaluation episodes under the same
name as training episodes. Running the same command again resumes the run
from its last checkpoint.

Each run gets its own folder, named after its settings, e.g.
`runs/w0.5_random_k64_s0_gpu/`. Besides DreamerV3's own files
(`config.yaml`, `metrics.jsonl`, `scores.jsonl`, `ckpt/`), it contains:

- `scores.csv`: `step, score` of every finished episode;
- `losses.csv`: every loss term during training, including the reconstruction
  loss of each observation key (`position`, `velocity`, `distractors`);
- `summary.json`: the settings, the final score (mean of the last 5
  episodes) and the duration.

### Many runs: `scripts/run_grid.py`

Runs a grid of experiments one after the other. Without options, it runs the
proposal grid: w in {0, 0.1, 0.5, 1} x {none, random, predictable, constant}
distractors (k = 64) x seeds 0-9, i.e. 160 runs.

```
# List the runs without launching anything
python scripts/run_grid.py --seeds 0-1 --profile gpu --dry-run

# Run seeds 0 and 1 of the grid on Colab, results in Google Drive
python scripts/run_grid.py --seeds 0-1 --profile gpu \
    --outdir /content/drive/MyDrive/INF8250AE_research_project/runs
```

- `--seeds 0-1`, `--seeds 2-3`, ... lets each member run a different share.
- Finished runs (their `summary.json` says so) are skipped, and unfinished
  runs resume, so the same command can be launched again after a Colab
  disconnection.
- All settings of seed 0 run first, then seed 1, and so on: if time runs out,
  every setting has the same number of seeds.
- `--w`, `--distractors`, `--k`, `--profile` and `--steps` change the grid,
  e.g. `--w 0 1 --distractors none random --k 20 --seeds 0 --profile smoke`.

## Tests

Unit tests of the distractor wrapper and of the run scripts (need only
`numpy`, `elements`, `pytest`):

```
python -m pytest tests/ -q
```

## References

- [DreamerV3 paper](https://arxiv.org/abs/2301.04104) and [official implementation](https://github.com/danijar/dreamerv3)
- [TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html)
- [Policy-Shaped Prediction](https://arxiv.org/abs/2412.05766)

*Status: research project in development.*
