"""Unit tests for embodied/envs/distractors.py.

Run from the repository root (needs only numpy, elements and pytest):
  python -m pytest tests/test_distractors.py -q
"""

import importlib.util
import pathlib

import elements
import numpy as np
import pytest

# Load the module by path so the test does not import the embodied package
# (which pulls in JAX).
_PATH = (pathlib.Path(__file__).parents[1] / 'embodied' / 'envs' /
         'distractors.py')
_spec = importlib.util.spec_from_file_location('distractors', _PATH)
distractors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(distractors)
Distractors = distractors.Distractors


class FakeEnv:
  """Minimal embodied-style env with fixed-length episodes."""

  def __init__(self, length=50):
    self.length = length
    self.t = 0
    self.done = True
    self.closed = False

  @property
  def obs_space(self):
    return {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'position': elements.Space(np.float32, (3,)),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (1,), -1, 1),
    }

  def step(self, action):
    if action['reset'] or self.done:
      self.t = 0
    else:
      self.t += 1
    self.done = self.t >= self.length - 1
    return {
        'reward': np.float32(self.t),
        'is_first': self.t == 0,
        'is_last': self.done,
        'is_terminal': False,
        'position': np.full(3, self.t, np.float32),
    }

  def close(self):
    self.closed = True


def rollout(env, steps, action_value=0.0):
  obs = []
  for i in range(steps):
    act = {'reset': i == 0, 'action': np.full(1, action_value, np.float32)}
    obs.append(env.step(act))
  return obs


def make(kind, k=8, seed=0, index=0, length=50, **kw):
  return Distractors(FakeEnv(length), kind, k, seed, index, **kw)


def stack(obs, key='distractors'):
  return np.stack([o[key] for o in obs])


@pytest.mark.parametrize('kind', distractors.KINDS)
class TestCommon:

  def test_space_and_dtype(self, kind):
    env = make(kind, k=7)
    space = env.obs_space['distractors']
    assert space.shape == (7,) and space.dtype == np.float32
    for obs in rollout(env, 20):
      value = obs['distractors']
      assert value.shape == (7,) and value.dtype == np.float32
      assert np.isfinite(value).all()
      assert value in space  # Same check as embodied.wrappers.CheckSpaces.

  def test_original_obs_unchanged(self, kind):
    base, wrapped = FakeEnv(), make(kind)
    for a, b in zip(rollout(base, 120), rollout(wrapped, 120)):
      assert set(b) == set(a) | {'distractors'}
      for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert {k: v for k, v in wrapped.obs_space.items()
            if k != 'distractors'}.keys() == base.obs_space.keys()
    assert wrapped.act_space.keys() == base.act_space.keys()

  def test_reproducible(self, kind):
    a = stack(rollout(make(kind, seed=3, index=1), 200))
    b = stack(rollout(make(kind, seed=3, index=1), 200))
    np.testing.assert_array_equal(a, b)

  def test_independent_of_actions(self, kind):
    a = stack(rollout(make(kind, seed=5), 200, action_value=-1.0))
    b = stack(rollout(make(kind, seed=5), 200, action_value=1.0))
    np.testing.assert_array_equal(a, b)

  def test_close_forwarded(self, kind):
    env = make(kind)
    env.close()
    assert env.env.closed


class TestRandom:

  def test_unit_gaussian(self):
    values = stack(rollout(make('random', k=20, length=1000), 5000))
    assert abs(values.mean()) < 0.02
    assert abs(values.std() - 1) < 0.02

  def test_redrawn_every_step(self):
    values = stack(rollout(make('random'), 100))
    assert not np.any(np.all(values[1:] == values[:-1], axis=1))
    # No temporal correlation.
    corr = np.corrcoef(values[1:].ravel(), values[:-1].ravel())[0, 1]
    assert abs(corr) < 0.1

  def test_index_changes_stream(self):
    a = stack(rollout(make('random', seed=0, index=0), 50))
    b = stack(rollout(make('random', seed=0, index=1), 50))
    assert not np.allclose(a, b)

  def test_seed_changes_stream(self):
    a = stack(rollout(make('random', seed=0), 50))
    b = stack(rollout(make('random', seed=1), 50))
    assert not np.allclose(a, b)


class TestPredictable:

  def test_follows_sinusoid_within_episode(self):
    env = make('predictable', k=6, length=200)
    obs = rollout(env, 200)
    values = stack(obs)
    t = np.arange(200)[:, None]
    expected = np.sqrt(2) * np.sin(env.omega * t + env._phi)
    np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-5)

  def test_periods_in_range(self):
    env = make('predictable', k=1000, period_min=10, period_max=100)
    periods = 2 * np.pi / env.omega
    assert periods.min() >= 10 and periods.max() <= 100
    # Log-uniform: roughly half the periods below the geometric mean.
    assert 0.4 < np.mean(periods < np.sqrt(10 * 100)) < 0.6

  def test_frequencies_shared_across_instances(self):
    # Same run seed -> same omega for every env (train and eval).
    a = make('predictable', seed=2, index=0)
    b = make('predictable', seed=2, index=3)
    np.testing.assert_array_equal(a.omega, b.omega)
    assert not np.array_equal(a.omega, make('predictable', seed=4).omega)

  def test_phase_resampled_each_episode(self):
    env = make('predictable', k=8, length=30)
    obs = rollout(env, 90)
    firsts = [i for i, o in enumerate(obs) if o['is_first']]
    assert firsts == [0, 30, 60]
    starts = stack([obs[i] for i in firsts])
    assert not np.allclose(starts[0], starts[1])
    assert not np.allclose(starts[1], starts[2])

  def test_time_resets_each_episode(self):
    env = make('predictable', k=4, length=30)
    rollout(env, 45)
    assert env._t == 14

  def test_unit_variance(self):
    # Over many episodes (random phases) each dimension has variance ~1.
    values = stack(rollout(make('predictable', k=10, length=100), 20000))
    assert abs(values.mean()) < 0.05
    assert abs(values.var() - 1) < 0.05


class TestConstant:

  def test_fixed_across_steps_and_episodes(self):
    values = stack(rollout(make('constant', k=5, length=20), 100))
    assert np.all(values == values[0])

  def test_shared_across_instances(self):
    a = stack(rollout(make('constant', seed=1, index=0), 10))
    b = stack(rollout(make('constant', seed=1, index=2), 10))
    np.testing.assert_array_equal(a, b)

  def test_seed_changes_values(self):
    a = make('constant', seed=0).constant
    b = make('constant', seed=1).constant
    assert not np.allclose(a, b)

  def test_drawn_from_unit_gaussian(self):
    values = make('constant', k=20000).constant
    assert abs(values.mean()) < 0.03
    assert abs(values.std() - 1) < 0.03


class TestValidation:

  def test_bad_kind(self):
    with pytest.raises(ValueError):
      make('noise')

  @pytest.mark.parametrize('k', [0, -1, 2.5])
  def test_bad_k(self, k):
    with pytest.raises(ValueError):
      make('random', k=k)

  def test_key_collision(self):
    with pytest.raises(ValueError):
      Distractors(FakeEnv(), 'random', 3, 0, key='position')

  def test_bad_periods(self):
    with pytest.raises(ValueError):
      make('predictable', period_min=50, period_max=10)
