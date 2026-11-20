"""Integration check of the distractor wrapper on the real DMC environment.

Builds the environment exactly like dreamerv3/main.py (same config parsing,
same wrappers including CheckSpaces) for each distractor type, plays about two
episodes with random actions and checks keys, shapes and statistics. No
training, no GPU needed. Needs the full dependencies (dm_control, JAX, ...).

Run it in its own process (not inside a notebook kernel that imported numpy
before `pip install`, since requirements.txt downgrades numpy):
  MUJOCO_GL=osmesa python tests/check_env.py --k 20
"""

import argparse
import os
import pathlib
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import elements  # noqa: E402
import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402

from dreamerv3 import main as dmain  # noqa: E402


def make_config(argv):
  configs = yaml.YAML(typ='safe').load(
      elements.Path(REPO / 'dreamerv3' / 'configs.yaml').read())
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  return elements.Flags(config).parse(other)


def rollout(task, kind, k, steps, seed):
  config = make_config([
      '--configs', 'dmc_proprio', '--task', task,
      '--distractors.type', kind, '--distractors.k', str(k),
      '--seed', str(seed)])
  env = dmain.make_env(config, 0)
  space = dict(env.obs_space)
  obs = []
  for i in range(steps):
    act = {key: s.sample() for key, s in env.act_space.items() if key != 'reset'}
    act['reset'] = (i == 0)
    obs.append(env.step(act))
  env.close()
  return space, obs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--task', default='dmc_cartpole_swingup')
  parser.add_argument('--k', type=int, default=20)
  parser.add_argument('--steps', type=int, default=2100)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()

  base_space, _ = rollout(args.task, 'none', 0, 5, args.seed)
  base_keys = set(base_space)
  print('Base observation:', {key: s.shape for key, s in base_space.items()})
  assert 'distractors' not in base_keys

  for kind in ('random', 'predictable', 'constant'):
    space, obs = rollout(args.task, kind, args.k, args.steps, args.seed)
    assert set(space) == base_keys | {'distractors'}, set(space)
    assert space['distractors'].shape == (args.k,)
    values = np.stack([o['distractors'] for o in obs])
    assert values.shape == (len(obs), args.k), values.shape
    assert values.dtype == np.float32, values.dtype
    for key in base_keys:
      assert np.shape(obs[1][key]) == base_space[key].shape, key
    firsts = [i for i, o in enumerate(obs) if o['is_first']]
    assert len(firsts) >= 2, f'Expected several episodes, got {firsts}'
    line = (f'{kind:12s} episodes start at {firsts}  '
            f'mean {values.mean():+.3f}  var {values.var():.3f}')
    if kind == 'random':
      assert abs(values.mean()) < 0.05 and abs(values.var() - 1) < 0.05
    elif kind == 'constant':
      assert np.all(values == values[0])
      line += '  (fixed over the whole run)'
    else:
      step = np.abs(np.diff(values[firsts[0]:firsts[1]], axis=0)).mean()
      line += f'  mean |change| per step {step:.3f} (smooth)'
      # Max change per step is bounded by sqrt(2) * omega_max.
      assert step < 1.0, step
    print(line)
  print('\nOK: the wrapper works in the real environment pipeline.')


if __name__ == '__main__':
  main()
