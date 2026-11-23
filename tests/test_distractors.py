"""Unit tests for embodied/envs/distractors.py.

Run from the repository root (needs only numpy, elements and pytest):
  python -m pytest tests/test_distractors.py -q
"""

import importlib.util
import pathlib

import elements
import numpy as np
import pytest

# Load the module by path: importing the embodied package would pull in JAX.
_PATH = pathlib.Path(__file__).parents[1] / 'embodied/envs/distractors.py'
_spec = importlib.util.spec_from_file_location('distractors', _PATH)
distractors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distractors)
Distractors = distractors.Distractors

LENGTH = 50  # Episode length of the fake env.


class FakeEnv:
  """Minimal embodied-style env with fixed-length episodes."""

  obs_space = {
      'reward': elements.Space(np.float32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
      'position': elements.Space(np.float32, (3,)),
  }
  act_space = {
      'reset': elements.Space(bool),
      'action': elements.Space(np.float32, (1,), -1, 1),
  }

  def __init__(self):
    self.t = -1

  def step(self, action):
    self.t = 0 if action['reset'] or self.t >= LENGTH - 1 else self.t + 1
    return {
        'reward': np.float32(self.t), 'is_first': self.t == 0,
        'is_last': self.t == LENGTH - 1, 'is_terminal': False,
        'position': np.full(3, self.t, np.float32),
    }


def rollout(env, steps, action=0.0):
  acts = [{'reset': i == 0, 'action': np.full(1, action, np.float32)}
          for i in range(steps)]
  return [env.step(a) for a in acts]


def values(kind, steps=200, k=8, seed=0, index=0, action=0.0, **kw):
  env = Distractors(FakeEnv(), kind, k, seed, index, **kw)
  obs = rollout(env, steps, action)
  return env, obs, np.stack([o['distractors'] for o in obs])


@pytest.mark.parametrize('kind', distractors.KINDS)
def test_adds_key_and_keeps_observation(kind):
  env, obs, _ = values(kind, k=7)
  assert env.obs_space.keys() == FakeEnv.obs_space.keys() | {'distractors'}
  assert env.act_space == FakeEnv.act_space
  for o, ref in zip(obs, rollout(FakeEnv(), len(obs))):
    assert o['distractors'] in env.obs_space['distractors']  # float32, (7,)
    for key in ref:
      np.testing.assert_array_equal(o[key], ref[key])


@pytest.mark.parametrize('kind', distractors.KINDS)
def test_reproducible_and_independent_of_actions(kind):
  a = values(kind, seed=3, index=1, action=-1.0)[2]
  b = values(kind, seed=3, index=1, action=1.0)[2]
  np.testing.assert_array_equal(a, b)


def test_random():
  v = values('random', steps=5000, k=20)[2]
  assert abs(v.mean()) < 0.02 and abs(v.std() - 1) < 0.02
  assert not np.any(np.all(v[1:] == v[:-1], axis=1))  # Redrawn every step.
  assert not np.allclose(v[:50], values('random', k=20, index=1)[2][:50])
  assert not np.allclose(v[:50], values('random', k=20, seed=1)[2][:50])


def test_predictable():
  env, obs, v = values('predictable', steps=3 * LENGTH, k=1000)
  periods = 2 * np.pi / env.omega
  assert periods.min() >= 10 and periods.max() <= 100
  # Within the last episode: the sinusoid of the episode step t.
  t = np.arange(LENGTH)[:, None]
  np.testing.assert_allclose(
      v[-LENGTH:], np.sqrt(2) * np.sin(env.omega * t + env._phi), atol=1e-5)
  # New phase each episode, same frequencies for every env of the run.
  assert [i for i, o in enumerate(obs) if o['is_first']] == [0, 50, 100]
  assert not np.allclose(v[0], v[LENGTH])
  other = values('predictable', k=1000, index=3)[0]
  np.testing.assert_array_equal(env.omega, other.omega)
  assert abs(v.var() - 1) < 0.05


def test_constant():
  v = values('constant', steps=3 * LENGTH, k=20000)[2]
  assert np.all(v == v[0])  # Over steps and episodes.
  assert abs(v[0].mean()) < 0.03 and abs(v[0].std() - 1) < 0.03
  np.testing.assert_array_equal(v[0], values('constant', k=20000, index=2)[2][0])
  assert not np.allclose(v[0], values('constant', k=20000, seed=1)[2][0])


@pytest.mark.parametrize('kwargs', [
    dict(kind='noise'), dict(k=0), dict(k=2.5), dict(key='position'),
    dict(kind='predictable', period_min=50, period_max=10),
])
def test_invalid_arguments(kwargs):
  args = dict(kind='random', k=3, seed=0) | kwargs
  with pytest.raises(ValueError):
    Distractors(FakeEnv(), **args)
