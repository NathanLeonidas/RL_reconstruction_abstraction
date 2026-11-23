"""Run MANY DreamerV3 experiments, one after the other.

Why this file exists
--------------------
Our research question needs a grid of runs: several reconstruction weights w,
several distractor types, and 10 seeds for each combination (the course asks
for at least 10 runs per setting to report a mean with an error bar).
That is 4 w x 4 distractor types x 10 seeds = 160 runs. This script launches
them for you, by calling scripts/run_experiment.py once per run.

How it helps the team
---------------------
- Split the work: each member runs a different set of seeds with --seeds,
  e.g. one person runs `--seeds 0-1`, another `--seeds 2-3`, and so on.
- Safe to stop and restart: a run that already finished (its summary.json
  says so) is skipped, and an unfinished run resumes from its checkpoint.
  This matters on Colab, which stops sessions after a few hours.
- Balanced order: all settings of seed 0 run first, then seed 1, etc. If we
  run out of time, every setting has the same number of seeds.

It works the same on a Mac and on Colab. On Colab, put --outdir in Google
Drive so the results survive the end of the session.

Usage (from the repository root)
--------------------------------
  # See the list of runs without launching anything
  python scripts/run_grid.py --seeds 0-1 --dry-run

  # The full proposal grid, seeds 0 and 1, on a GPU, results on Google Drive
  python scripts/run_grid.py --seeds 0-1 --profile gpu \\
      --outdir /content/drive/MyDrive/INF8250AE_research_project/runs

  # A small grid to try things on a laptop
  python scripts/run_grid.py --w 0 1 --distractors none random --k 20 \\
      --seeds 0 --profile laptop
"""

import argparse 
import importlib.util
import json
import pathlib
import subprocess 
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
REPO = SCRIPTS.parent

# Reuse run_experiment.py so that run names (folder names) are always the same.
_spec = importlib.util.spec_from_file_location(
    'run_experiment', SCRIPTS / 'run_experiment.py')
run_experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_experiment)


def parse_seeds(text):
  """'0-9' -> [0, 1, ..., 9]   '0,3,5' -> [0, 3, 5]   '2' -> [2]"""
  seeds = []
  for part in text.split(','):
    if '-' in part:
      first, last = part.split('-')
      seeds += range(int(first), int(last) + 1)
    else:
      seeds.append(int(part))
  return seeds


def parse_args(argv=None):
  """Read the different settings to try from the command line."""
  parser = argparse.ArgumentParser(
      description='Run a grid of DreamerV3 experiments, one after the other.')
  # Defaults = the grid of the proposal.
  parser.add_argument('--w', type=float, nargs='+', default=[0, 0.1, 0.5, 1],
                      help='reconstruction weights to try')
  parser.add_argument('--distractors', nargs='+',
                      choices=run_experiment.DISTRACTOR_TYPES,
                      default=['none', 'random', 'predictable', 'constant'],
                      help='distractor types to try')
  parser.add_argument('--k', type=int, default=64,
                      help='number of distractors (for types other than none)')
  parser.add_argument('--seeds', type=parse_seeds, default='0-9',
                      help="seeds to run, e.g. '0-9', '0-1' or '0,3,5'")
  parser.add_argument('--profile', choices=sorted(run_experiment.PROFILES),
                      default='laptop', help='training budget of each run')
  parser.add_argument('--steps', type=int, default=None,
                      help='override the number of steps of the profile')
  parser.add_argument('--outdir', default=str(REPO / 'runs'),
                      help='folder that will contain all the run folders')
  parser.add_argument('--dry-run', action='store_true',
                      help='only print the list of runs')
  return parser.parse_args(argv)


def plan(args):
  """The list of runs, as argument lists for run_experiment.py.
  Order: seed first, so the grid fills up evenly."""
  runs = []
  for seed in args.seeds:
    for distractors in args.distractors:
      k = 0 if distractors == 'none' else args.k
      for w in args.w:
        flags = ['--w', f'{w:g}', '--distractors', distractors,
                 '--k', str(k), '--seed', str(seed),
                 '--profile', args.profile, '--outdir', args.outdir]
        if args.steps is not None:
          flags += ['--steps', str(args.steps)]
        runs.append(flags)
  return runs


def is_finished(flags):
  """True if this run already ended without error (see summary.json)."""
  run_args = run_experiment.parse_args(flags)
  summary = (pathlib.Path(run_args.outdir) / run_experiment.run_name(run_args)
             / 'summary.json')
  return summary.exists() and json.loads(summary.read_text())['finished']


def main(argv=None):
  args = parse_args(argv)
  runs = plan(args)
  print(f'{len(runs)} runs planned.')
  failed = []
  for index, flags in enumerate(runs, 1):
    name = run_experiment.run_name(run_experiment.parse_args(flags))
    if is_finished(flags):
      print(f'[{index}/{len(runs)}] {name}: already finished, skipped')
      continue
    print(f'[{index}/{len(runs)}] {name}')
    if args.dry_run:
      continue
    # One process per run: if a run crashes, the next ones still start.
    command = [sys.executable, str(SCRIPTS / 'run_experiment.py'), *flags]
    if subprocess.run(command, cwd=REPO).returncode != 0:
      failed.append(name)
  if failed:
    print(f'{len(failed)} runs stopped with an error: {", ".join(failed)}')
    print('Run the same command again to resume them.')
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main())
