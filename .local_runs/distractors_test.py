# /// script
# dependencies = ["dm-control", "pytest", "requirements-colab-txt"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import subprocess
    import os
    import jax, numpy as np
    import json, re, sys, time
    from pathlib import Path
    import elements
    import imageio.v2 as imageio


    return Path, jax, json, mo, os, re, subprocess, sys, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tests on local compute (distractors + reconstruction weight `w`)

    Assumes repo from GitHub is already cloned and has installed the dependencies in an environnement, that is **sourced before launching this notebook*
    checks the distractor wrapper on the real `cartpole_swingup` environment, then
    runs a short training smoke test and an optional timed pilot run.

    ```
    cd RL_reconstruction_abstraction
    source .env/bin/activate
    marimo edit .local_runs/distractors_test.py
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Parameters
    """)
    return


@app.cell
def _(Path, sys):
    REPO_URL = 'https://github.com/NathanLeonidas/RL_reconstruction_abstraction.git'
    BRANCH = 'feature/distractors'
    REPO = '.'
    RUNS = './.local_runs/runs'      
    RESULTS_DIR = './.local_runs/results'

    sys.path.insert(0, str(Path(REPO).resolve()))
    import dreamerv3.main as main
    import embodied

    return BRANCH, REPO, RESULTS_DIR, RUNS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Sanity check
    """)
    return


@app.cell
def _(jax):
    print("JAX version:", jax.__version__)
    print("Default backend:", jax.default_backend())
    print("Available devices:", jax.devices())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Launch utils

    Runs `dreamerv3/main.py` in a subprocess, streams its output (without color
    codes), measures the time, and at the end writes `run_info.json` (flags,
    duration, commit) in the logdir and copies the logdir to `RESULTS_DIR`.
    """)
    return


@app.cell
def _(
    BRANCH,
    Path,
    REPO,
    RESULTS_DIR,
    RUNS,
    json,
    os,
    re,
    subprocess,
    sys,
    time,
):

    ANSI = re.compile('\\x1b\\[[0-9;]*m')

    def run_dreamer(name, flags, copy_to_drive=True, quiet_lines=('Space(',)):
        logdir = Path(RUNS) / name
        cmd = [sys.executable,
               '-u', 'dreamerv3/main.py',
               *flags,
               '--logdir', str(logdir)]
        env = {**os.environ,
               'MUJOCO_GL': 'osmesa',
               'PYOPENGL_PLATFORM': 'osmesa'}
        print(' '.join(cmd), '\n', flush=True)
        start = time.time()
        proc = subprocess.Popen(cmd,
                                cwd=REPO,
                                env=env,
                                text=True,
                                bufsize=1,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        last_step = 0
        try:
            for line in proc.stdout:
                line = ANSI.sub('', line)
                match = re.search('\\[Agent Step\\s+([\\d_]+)\\]', line)
                if match:
                    last_step = int(match.group(1).replace('_', ''))
                    line = line.rstrip() + f'   ({(time.time() - start) / 60:.1f} min)\n'
                if not any((q in line for q in quiet_lines)):
                    print(line, end='', flush=True)
        except KeyboardInterrupt:
            print('\nInterrupted: stopping the run (the checkpoint allows resuming).')
            proc.terminate()
        proc.wait()
        elapsed = time.time() - start
        git = lambda *a: subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
        info = dict(name=name, flags=flags, returncode=proc.returncode, elapsed_min=round(elapsed / 60, 2), last_logged_step=last_step, branch=BRANCH, commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '--untracked-files=no')))
        if logdir.exists():
            (logdir / 'run_info.json').write_text(json.dumps(info, indent=2))
            if copy_to_drive:
                dest = Path(RESULTS_DIR) / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(['rsync', '-a', f'{logdir}/', f'{dest}/'], check=False)
                print('Copied to Drive:', dest)
        print(f'\nExit code {proc.returncode} - {elapsed / 60:.2f} min - last logged step {last_step}')
        return (logdir, info)

    def read_jsonl(path):
        path = Path(path)
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def read_metrics(logdir):
        return read_jsonl(Path(logdir) / 'metrics.jsonl')

    def read_scores(logdir):
        return read_jsonl(Path(logdir) / 'scores.jsonl') 

    return read_metrics, read_scores, run_dreamer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Training smoke test

    **Short experiment**: `batch_size`, `batch_length` and `train_ratio` are
    reduced so that it runs in a few minutes. It only checks that full training
    works with distractors and that their reconstruction loss is logged
    (`train/loss/distractors`).
    """)
    return


@app.cell
def _(read_metrics, run_dreamer):
    SMOKE_TYPE, SMOKE_K, SMOKE_W = ('random', 22, 1.0)
    smoke_name = f'smoke_{SMOKE_TYPE}_k{SMOKE_K}_w{SMOKE_W}'
    logdir, info = run_dreamer(smoke_name, [
        '--configs', 'dmc_proprio',
        '--jax.compute_dtype', 'float32',
        '--task', 'dmc_cartpole_swingup',
        '--distractors.type', SMOKE_TYPE,
        '--distractors.k', str(SMOKE_K),
        '--agent.loss_scales.rec', str(SMOKE_W),
        '--seed', '0',
        '--run.envs', '1',
        '--batch_size', '1',
        '--batch_length', '32',
        '--run.train_ratio', '8',
        '--run.steps', '2000',
        '--run.log_every', '10'],
        copy_to_drive=False)

    assert info['returncode'] == 0, 'The run failed, see the output above.'
    _metrics = read_metrics(logdir)
    keys = sorted({k for m in _metrics for k in m if k.startswith('train/loss/')})
    print('Logged losses:', keys)

    assert 'train/loss/distractors' in keys, 'The distractor loss is not logged!'
    last = {k: m[k] for m in _metrics for k in keys if k in m}
    print({k.split('/')[-1]: round(v, 3) for k, v in last.items()})
    print('\nOK: training works with distractors.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Timed pilot run

    Real hyperparameters (`dmc_proprio`, `train_ratio` 1024), `train_eval` script
    (upstream DreamerV3: evaluation every `report_every` = 300 seconds of wall
    clock, about 3 min per evaluation on a T4). Used to measure the duration of a
    run before sizing the grid.

    Measured on a T4: about 660 env steps per minute of training, so roughly
    1 h for 20k steps including the evaluations.
    """)
    return


@app.cell
def _():
    PILOT_TYPE, PILOT_K, PILOT_W, PILOT_SEED = ('none', 0, 1.0, 0)
    PILOT_STEPS = 2000
    pilot_name = f'pilot_{PILOT_TYPE}_k{PILOT_K}_w{PILOT_W}_s{PILOT_SEED}'
    return PILOT_K, PILOT_SEED, PILOT_STEPS, PILOT_TYPE, PILOT_W, pilot_name


@app.cell
def _(
    PILOT_K,
    PILOT_SEED,
    PILOT_STEPS,
    PILOT_TYPE,
    PILOT_W,
    pilot_name,
    run_dreamer,
):

    logdir_1, info_1 = run_dreamer(pilot_name, [
        '--configs', 'dmc_proprio',
        '--task', 'dmc_cartpole_swingup',
        '--jax.compute_dtype', 'float32',
        '--script', 'train_eval',
        '--distractors.type', PILOT_TYPE,
        '--distractors.k', str(PILOT_K),
        '--agent.loss_scales.rec', str(PILOT_W),
        '--seed', str(PILOT_SEED),
        '--run.steps', str(PILOT_STEPS)])
    return info_1, logdir_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pilot summary

    The time per step includes the JAX compilation at the start and the
    evaluations: the estimate for a longer run is therefore **an
    extrapolation**, to be confirmed by a measurement.
    """)
    return


@app.cell
def _(info_1, logdir_1, read_metrics, read_scores):
    _metrics = read_metrics(logdir_1)
    scores = read_scores(logdir_1)
    steps = max((m.get('step', 0) for m in _metrics), default=0)
    print(f"Duration: {info_1['elapsed_min']:.1f} min for ~{steps} env steps")
    fps = [m['fps/policy'] for m in _metrics if 'fps/policy' in m]
    if fps:
        print(f'fps/policy (last): {fps[-1]:.1f} steps/s')
    if steps:
        per_step = info_1['elapsed_min'] / steps
        for target in (50000, 100000):
            print(f'Extrapolation (not measured) for {target:,} steps: {per_step * target / 60:.1f} h')
    print('\nLogged episodes (training and evaluation mixed):', len(scores))
    for s in scores[-5:]:
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in s.items()})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Vison based experiments

    Grid search for different distractors/reconstuction settings but with an actual image.
    Note that for this, we'll need noise on the image instead of additional discrete vector.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **epstats-policy_image.mp4**	The agent interacting with the real environment using its learned policy.

    **epstats-policy_recon-image.mp4**	The world model reconstructing the observations encountered during policy execution.

    **report-openloop-image.mp4**	The world model reconstructing/predicting observations over an open-loop sequence.

    **eval-openloop-image.mp4**	Open-loop world-model predictions generated during evaluation.
    """)
    return


@app.cell
def _(PILOT_K, PILOT_SEED, PILOT_TYPE, PILOT_W, run_dreamer):


    pilot_names = f'pilot_vision_{PILOT_TYPE}_k{PILOT_K}_w{PILOT_W}_s{PILOT_SEED}'

    logdir_2, info_2 = run_dreamer(pilot_names, [
        '--configs', 'dmc_vision',
        '--task', 'dmc_cartpole_swingup',
        '--jax.compute_dtype', 'float32',
        '--script', 'train_eval',
        '--distractors.type', PILOT_TYPE,
        '--distractors.k', str(PILOT_K),
        '--agent.loss_scales.rec', str(PILOT_W),
        '--seed', str(PILOT_SEED),
        '--run.steps', str(100000),
        '--seed', '0',
        '--run.envs', '1',
        '--batch_size', '4',
        '--batch_length', '32',
        '--run.train_ratio', '64',
        '--run.log_every', '10'])
    return (logdir_2,)


@app.cell
def _(Path, logdir_2):
    from IPython.display import Video, display, HTML

    scope = Path(logdir_2) / 'scope'

    video_types = {
        'Policy (actual environment)': 'epstats-policy_image.mp4',
        'Posterior reconstruction': 'epstats-policy_recon-image.mp4',
        'Open-loop imagination': 'report-openloop-image.mp4',
    }

    videos = {}

    for title, folder in video_types.items():
        files = list((scope / folder).glob('*.mp4'))
        if files:
            videos[title] = max(files, key=lambda p: p.stat().st_mtime)

    # Display side by side.
    columns = []

    for title, path in videos.items():
        video = Video(str(path), embed=True, width=320)
        columns.append((title, video))

    display(HTML("""
    <div style="display:flex; gap:15px;">
      <div id="video-container"></div>
    </div>
    """))

    for title, video in columns:
        display(HTML(f"<h4>{title}</h4>"))
        display(video)
    return


@app.cell
def _(logdir_2, read_metrics):
    import pandas as pd
    import matplotlib.pyplot as plt

    metrics = pd.DataFrame(read_metrics(logdir_2))

    scores = (
        metrics[['step', 'episode/score']]
        .dropna()
        .groupby('step')['episode/score']
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        scores['step'],
        scores['episode/score'],
        alpha=0.8,
        label='Mean episode reward'
    )

    plt.xlabel('Training steps')
    plt.ylabel('Episode return')
    plt.title('DreamerV3 - Training reward')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
