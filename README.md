# How Much Should a World Model Reconstruct?

**Reconstruction Weight versus Task-Irrelevant Observations in DreamerV3**

Research project for **INF8250AE — Reinforcement Learning**, Polytechnique Montréal (Fall 2026).

**Authors:** Antoine Deckers, Romain Haessler, Nathan Lewy, and Lucas Silva.

## About the project

World models allow reinforcement learning agents to simulate future trajectories using learned representations of their environments. However, reconstructing observations may require the model to learn task-irrelevant information, potentially affecting its performance and sample efficiency.

This project investigates **how observation reconstruction influences DreamerV3's learning performance when observations contain irrelevant information**.

## Quickstart

To get started from inside a **google collab notebook** run the follwing commands:

'''
!git clone git@github.com:NathanLeonidas/RL_reconstruction_abstraction.git
%cd /content/RL_reconstruction_abstraction.git
!grep -v -E '^(ale_py|autorom)' requirements.txt > requirements_colab.txt
!pip install -U -r requirements_colab.txt
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
'''

## Approach

We will modify DreamerV3's prediction loss by introducing a weight $w$ on its observation reconstruction term. We will then compare different values of $w$ while adding varying numbers of **random, predictable, or constant distractors** to the agent's observations.

Our primary environment will be **Cartpole Swingup** from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), with additional environments considered if resources allow. We plan to train lightweight DreamerV3 models from scratch and evaluate performance and sample efficiency across multiple random seeds.

## References

- [DreamerV3 paper](https://arxiv.org/abs/2301.04104) and [official implementation](https://github.com/danijar/dreamerv3)
- [TD-MPC](https://proceedings.mlr.press/v162/hansen22a.html)
- [Policy-Shaped Prediction](https://arxiv.org/abs/2412.05766)

*Status: research project in development.*
