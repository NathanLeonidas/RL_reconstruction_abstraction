"""Tests for scripts/run_experiment.py (no training, no GPU needed).

Run from the repository root: python -m pytest tests/test_run_experiment.py -q
"""

import importlib.util
import json
import pathlib

import pytest

# Load the script by path: scripts/ is not a Python package.
_PATH = pathlib.Path(__file__).parents[1] / 'scripts' / 'run_experiment.py'
_spec = importlib.util.spec_from_file_location('run_experiment', _PATH)
run_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiment)


def test_command_contains_our_settings(tmp_path):
  args = run_experiment.parse_args([
      '--w', '0.5', '--distractors', 'random', '--k', '64', '--seed', '3',
      '--profile', 'gpu', '--device', 'cuda'])
  command = ' '.join(run_experiment.build_command(args, tmp_path))
  assert run_experiment.run_name(args) == 'w0.5_random_k64_s3_gpu'
  for flag in ['--agent.loss_scales.rec 0.5', '--distractors.type random',
               '--distractors.k 64', '--seed 3', '--run.train_ratio 1024',
               '--script train']:
    assert flag in command
  assert '--jax.platform cpu' not in command


@pytest.mark.parametrize('flags', [
    ['--distractors', 'random', '--k', '0'],
    ['--distractors', 'none', '--k', '5'],
])
def test_rejects_inconsistent_k(flags):
  with pytest.raises(SystemExit):
    run_experiment.parse_args(flags)


def test_export_results(tmp_path):
  lines = [{'step': 1000 * i, 'episode/score': 10.0 * i} for i in range(1, 8)]
  (tmp_path / 'scores.jsonl').write_text(
      ''.join(json.dumps(line) + '\n' for line in lines) + '{"step": 80')
  (tmp_path / 'metrics.jsonl').write_text(
      json.dumps({'step': 500, 'train/loss/rew': 2.0, 'fps/train': 9.0}) + '\n')
  args = run_experiment.parse_args(['--w', '0.1'])
  summary = run_experiment.export_results(tmp_path, args, 1.0, 0)
  assert summary['episodes'] == 7
  assert summary['final_score'] == 50.0  # mean of the last 5: 30..70
  assert (tmp_path / 'losses.csv').read_text().splitlines() == [
      'step,rew', '500,2.0']


def test_profiles_get_different_folders():
  # A smoke test must not share its folder (and checkpoint) with a gpu run.
  smoke = run_experiment.parse_args(['--profile', 'smoke'])
  gpu = run_experiment.parse_args(['--profile', 'gpu'])
  assert run_experiment.run_name(smoke) != run_experiment.run_name(gpu)
