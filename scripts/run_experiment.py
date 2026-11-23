"""Run ONE RUN of DreamerV3 experiment on Cartpole Swingup, with simple options.

Why this file exists
--------------------
DreamerV3 is started with dreamerv3/main.py and a long list of flags
(config, task, w, distractors, seed, training budget, rendering, logdir...).
Typing them by hand for every run is slow and easy to get wrong, and every
team member must use exactly the same settings for the results to be
comparable. This script is a thin wrapper around dreamerv3/main.py:

  - you give it only what our experiments change (w, distractors, k, seed)
    and a "profile" for the training budget;
  - it builds the full DreamerV3 command, with the right settings for your
    machine (Mac or Colab), and runs it;
  - when the run ends, it writes small, easy-to-read files for the analysis
    (scores.csv, losses.csv, summary.json).

It does not change DreamerV3 itself.

Usage (from the repository root)
--------------------------------
  # 30-second check that everything works. NOT a real experiment.
  python scripts/run_experiment.py --profile smoke --distractors random --k 20

  # A short real run on a laptop CPU (about 20 minutes, too short to learn)
  python scripts/run_experiment.py --profile laptop --w 0.5 --distractors random --k 20

  # The official DreamerV3 training budget (needs a GPU, e.g. Colab)
  python scripts/run_experiment.py --profile gpu --w 0.5 --distractors random --k 64 --seed 3

  # Show the DreamerV3 command without running it
  python scripts/run_experiment.py --w 0.1 --dry-run

Running the same command again resumes the run from its last checkpoint.

What you get, in runs/<run name>/ (e.g. runs/w0.5_random_k64_s3/)
------------------------------------------------------------------
  config.yaml, metrics.jsonl, scores.jsonl, ckpt/   written by DreamerV3
  scores.csv     one row per finished episode: step, score
  losses.csv     the loss terms during training: reconstruction of each
                 observation key (position, velocity, distractors), reward,
                 continue, KL (dyn, rep)...
  summary.json   the settings, the final score and the run duration
"""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1] # get to the repository root
TASK = 'dmc_cartpole_swingup'

# Training budgets. Only these settings differ between profiles; everything
# else is the official DreamerV3 `dmc_proprio` config (dreamerv3/configs.yaml).
#   steps        number of environment steps (the x-axis of our curves)
#   train_ratio  replayed steps trained on per environment step. This is what
#                makes DreamerV3 slow: 1024 is the official value.
#   envs         environments that collect data in parallel
#   log_every    seconds between two writes of the training metrics (losses)

# Using profiles, everyone will use the same training budget. Please copy it 
PROFILES = {
    # Only checks that the code runs end to end (about 30 s on a laptop).
    'smoke': dict(steps=2_000, train_ratio=8, envs=1, log_every=5,
                  batch_size=4, batch_length=32),
    # \------------\
    # A real training loop on a laptop CPU (about 20 min on a Mac M4 Pro).
    # Too small a budget to learn Cartpole Swingup: on 2026-11-23, w = 1
    # without distractors ended at a score of 25 (random level). Use it to
    # look at the losses and the pipeline, not for results.
    'laptop': dict(steps=50_000, train_ratio=64, envs=4, log_every=30),
    # \------------\
    # Official DreamerV3 training budget for DMC proprio. Needs a GPU: on a
    # laptop CPU it runs at about 3 environment steps per second.
    'gpu': dict(steps=100_000, train_ratio=1024, envs=16, log_every=60),
}

DISTRACTOR_TYPES = ('none', 'random', 'predictable', 'constant')


def parse_args(argv=None):
  """
  Read the different settings to try from the command line.
  choices define the allowed values for each argument. 
  default defines the default value if the argument is not provided.
  :return: the parsed arguments as a Namespace object. -> args.w ; args.distractors ; args.k ; args.seed ; args.profile ; args.steps ; args.device ; args.outdir ; args.dry_run
  """
  parser = argparse.ArgumentParser(
      description='Run one DreamerV3 experiment on Cartpole Swingup.')
  parser.add_argument('--w', type=float, default=1.0,
                      help='reconstruction weight (1 = standard DreamerV3)')
  parser.add_argument('--distractors', choices=DISTRACTOR_TYPES,
                      default='none', help='type of distractors')
  parser.add_argument('--k', type=int, default=None,
                      help='number of distractors (0 if --distractors none)')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--profile', choices=sorted(PROFILES), default='laptop',
                      help='training budget (see PROFILES in this file)')
  parser.add_argument('--steps', type=int, default=None,
                      help='override the number of steps of the profile')
  parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'),
                      default='auto',
                      help='auto = cpu on macOS, cuda (GPU) elsewhere')
  parser.add_argument('--outdir', default=str(REPO / 'runs'),
                      help='folder that will contain the run folder')
  parser.add_argument('--dry-run', action='store_true',
                      help='print the DreamerV3 command and stop')
  args = parser.parse_args(argv)

  # k defaults to 0 without distractors, and must be > 0 with them.
  if args.k is None:
    args.k = 0 if args.distractors == 'none' else 20
  if (args.distractors == 'none') != (args.k == 0):
    parser.error('use --k 0 with --distractors none, and --k > 0 otherwise')
  return args


def run_name(args):
  """Folder name of a run, e.g. w0.5_random_k64_s3.
  Build the name from the arguments, so that the same settings always produce the same folder name.
  """
  return f'w{args.w:g}_{args.distractors}_k{args.k}_s{args.seed}'


def build_command(args, logdir):
  """The full dreamerv3/main.py command for these settings, as a list."""
  budget = dict(PROFILES[args.profile])
  if args.steps is not None:
    budget['steps'] = args.steps
  device = args.device
  if device == 'auto':
    device = 'cpu' if sys.platform == 'darwin' else 'cuda' #darwin for macoS

  command = [
      sys.executable, str(REPO / 'dreamerv3' / 'main.py'), # Python interpreter and DreamerV3 main.py
      '--configs', 'dmc_proprio',     # official config, state observations
      '--task', TASK,
      '--script', 'train',            # see the note below
      '--agent.loss_scales.rec', str(args.w),
      '--distractors.type', args.distractors,
      '--distractors.k', str(args.k),
      '--seed', str(args.seed),
      '--run.steps', str(budget['steps']),
      '--run.train_ratio', str(budget['train_ratio']),
      '--run.envs', str(budget['envs']),
      '--run.log_every', str(budget['log_every']),
      '--logger.outputs', 'jsonl',    # only the metrics files we read
      '--logdir', str(logdir),
  ]
  # Note: we use the `train` script rather than `train_eval` because
  # `train_eval` logs evaluation episodes under the same name as training
  # episodes (`episode/score`), so the two cannot be told apart afterwards.
  if 'batch_size' in budget:
    command += ['--batch_size', str(budget['batch_size']),
                '--batch_length', str(budget['batch_length'])]
  if device == 'cpu':
    command += ['--jax.platform', 'cpu', '--jax.prealloc', 'False']
  return command


def rendering_env():
  """Environment variables for MuJoCo rendering (DreamerV3 renders frames
  even when the agent only sees the state). Macs use glfw, Linux machines
  without a screen (Colab) use osmesa."""
  env = dict(os.environ)
  if sys.platform == 'darwin':
    env.setdefault('MUJOCO_GL', 'glfw')
  else:
    env.setdefault('MUJOCO_GL', 'osmesa')
    env.setdefault('PYOPENGL_PLATFORM', 'osmesa')
  return env


def read_jsonl(path):
  """Lines of a .jsonl file as dicts. A line cut in the middle (run stopped
  while writing) is skipped."""
  rows = []
  if path.exists():
    for line in path.read_text().splitlines():
      try:
        rows.append(json.loads(line)) #json.loads() parses a JSON string and returns a Python object (dict here).
      except json.JSONDecodeError:
        pass
  return rows


def export_results(logdir, args, duration_min, returncode):
  """Write scores.csv, losses.csv and summary.json in the run folder."""
  # Episode scores: one row per finished episode.
  scores = [(row['step'], row['episode/score'])
            for row in read_jsonl(logdir / 'scores.jsonl')
            if 'episode/score' in row]
  with (logdir / 'scores.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['step', 'score'])
    writer.writerows(scores)

  # Loss terms. metrics.jsonl has one line per logging moment, with dozens of
  # metrics (speed, memory, losses...). Example of one line:
  #   {"step": 840, "fps/train": 1091.9,
  #    "train/loss/position": 3.02, "train/loss/distractors": 13.9, ...}
  # We keep only the losses (names starting with 'train/loss/'), without
  # that prefix. The line above becomes:
  #   {"step": 840, "position": 3.02, "distractors": 13.9, ...}
  prefix = 'train/loss/'
  losses = []  # one dict per logging moment
  for row in read_jsonl(logdir / 'metrics.jsonl'):
    terms = {}
    for key, value in row.items():
      if key.startswith(prefix):
        short_name = key[len(prefix):]  # 'train/loss/position' -> 'position'
        terms[short_name] = value
    if terms:  # some lines contain no loss at all (e.g. only episode scores)
      losses.append({'step': row['step'], **terms})  # ** copies all of terms

  # Columns of the CSV file: 'step' first, then every loss name that appears
  # in at least one line, in alphabetical order. A set keeps each name once.
  names = set()
  for row in losses:
    names.update(row.keys())  # add the names of this line to the set
  names.discard('step')       # 'step' is added separately, as first column
  columns = ['step'] + sorted(names)
  # e.g. ['step', 'con', 'distractors', 'dyn', 'position', 'rep', 'rew', ...]

  # Write the file: a header line with the column names, then one line per
  # logging moment. DictWriter matches each value to its column by name.
  with (logdir / 'losses.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, columns)
    writer.writeheader()
    writer.writerows(losses)

  # Summary. The final score is the mean of the last 5 episodes, which is
  # less noisy than the very last one.
  last = [score for _, score in scores[-5:]]
  summary = dict(
      run=run_name(args), task=TASK, w=args.w,
      distractors=args.distractors, k=args.k, seed=args.seed,
      profile=args.profile, steps=scores[-1][0] if scores else 0,
      episodes=len(scores),
      final_score=sum(last) / len(last) if last else None,
      duration_min=round(duration_min, 1),
      finished=returncode == 0)
  (logdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
  return summary


def main(argv=None):
  args = parse_args(argv)
  logdir = pathlib.Path(args.outdir).resolve() / run_name(args)
  command = build_command(args, logdir)
  print('Run folder:', logdir)
  print('Command:', ' '.join(command))
  if args.dry_run:
    return 0

  start = time.time()
  returncode = subprocess.run(command, env=rendering_env(), cwd=REPO).returncode # Handle possible errors in the DreamerV3 run. If the run crashes, the returncode will be non-zero.
  summary = export_results(logdir, args, (time.time() - start) / 60,
                           returncode)
  print(json.dumps(summary, indent=2))
  if returncode != 0:
    print(f'DreamerV3 stopped with an error (code {returncode}). '
          'Run the same command again to resume.')
  return returncode


if __name__ == '__main__':
  sys.exit(main())
