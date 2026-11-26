"""Tests for scripts/run_grid.py (nothing is trained).

Run from the repository root: python -m pytest tests/test_run_grid.py -q
"""

import importlib.util
import json
import pathlib

_PATH = pathlib.Path(__file__).parents[1] / 'scripts' / 'run_grid.py'
_spec = importlib.util.spec_from_file_location('run_grid', _PATH)
run_grid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_grid)


def test_parse_seeds():
  assert run_grid.parse_seeds('0-3') == [0, 1, 2, 3]
  assert run_grid.parse_seeds('0,5,7-8') == [0, 5, 7, 8]


def test_default_grid_is_the_proposal_grid_seed_first():
  runs = run_grid.plan(run_grid.parse_args([]))
  assert len(runs) == 4 * 4 * 10
  names = [run_grid.run_experiment.run_name(
      run_grid.run_experiment.parse_args(flags)) for flags in runs]
  assert names[0] == 'w0_none_k0_s0_laptop'
  assert names[15] == 'w1_constant_k64_s0_laptop'  # all of seed 0 comes first
  assert names[16] == 'w0_none_k0_s1_laptop'


def test_finished_runs_are_skipped(tmp_path):
  args = run_grid.parse_args(['--w', '1', '--distractors', 'none',
                              '--seeds', '0-1', '--outdir', str(tmp_path)])
  done, todo = run_grid.plan(args)
  (tmp_path / 'w1_none_k0_s0_laptop').mkdir()
  (tmp_path / 'w1_none_k0_s0_laptop' / 'summary.json').write_text(
      json.dumps({'finished': True}))
  assert run_grid.is_finished(done)
  assert not run_grid.is_finished(todo)
