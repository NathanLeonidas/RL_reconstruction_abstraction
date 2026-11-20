"""Task-irrelevant observation dimensions ("distractors").

Appends a vector n_t of k extra dimensions to every observation, under its own
key (default 'distractors'), so that x_t = [s_t ; n_t]. The distractors are
independent of the state, the actions and the reward: the task and the optimal
return are unchanged.

Types:
  random       n_t ~ N(0, 1), redrawn at every step.
  predictable  n_t^i = sqrt(2) * sin(omega_i * t + phi_i). The angular
               frequencies omega_i are drawn once per run (periods log-uniform
               in [period_min, period_max] steps); the phases phi_i are redrawn
               at the start of every episode; t is the step index within the
               episode. The sqrt(2) amplitude gives unit variance, like random.
  constant     Values drawn once per run from N(0, 1), fixed afterwards.

Seeding: the run-level quantities (omega for predictable, the values for
constant) depend only on `seed`, so every environment instance of a run (train
and eval) sees the same ones. The per-step randomness (random noise, episode
phases) also depends on `index`, so parallel environments get different
streams. Note that embodied creates train and eval envs with the same indices
0..n-1, so train env i and eval env i share that stream; this is harmless since
the distractors are independent of everything else.

This module deliberately imports only numpy and elements (not embodied, which
pulls in JAX), so it can be unit-tested without a GPU or dm_control.
"""

import functools

import elements
import numpy as np

KINDS = ('random', 'predictable', 'constant')

# Fixed salt so the distractor RNG streams never coincide with other users of
# the run seed.
_SALT = 0xD157


class Distractors:

  def __init__(
      self, env, kind, k, seed, index=0, key='distractors',
      period_min=10.0, period_max=100.0):
    if kind not in KINDS:
      raise ValueError(f'Unknown distractor type {kind!r}, expected {KINDS}.')
    if int(k) != k or k < 1:
      raise ValueError(f'Number of distractors must be a positive int: {k}')
    if key in env.obs_space:
      raise ValueError(f'Observation key {key!r} already exists.')
    if not 0 < period_min <= period_max:
      raise ValueError((period_min, period_max))
    self.env = env
    self.kind = kind
    self.k = int(k)
    self.key = key
    run_rng = np.random.default_rng([int(seed), _SALT])
    periods = np.exp(run_rng.uniform(
        np.log(period_min), np.log(period_max), self.k))
    self.omega = 2 * np.pi / periods
    self.constant = run_rng.standard_normal(self.k)
    self._rng = np.random.default_rng([int(seed), _SALT, int(index) + 1])
    self._t = 0
    self._phi = np.zeros(self.k)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self.env, name)

  @functools.cached_property
  def obs_space(self):
    return {
        **self.env.obs_space,
        self.key: elements.Space(np.float32, (self.k,)),
    }

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    if obs['is_first']:
      self._t = 0
      self._phi = self._rng.uniform(0, 2 * np.pi, self.k)
    else:
      self._t += 1
    return {**obs, self.key: self._sample()}

  def _sample(self):
    if self.kind == 'random':
      value = self._rng.standard_normal(self.k)
    elif self.kind == 'predictable':
      value = np.sqrt(2) * np.sin(self.omega * self._t + self._phi)
    else:
      value = self.constant
    return value.astype(np.float32)

  def close(self):
    return self.env.close()
