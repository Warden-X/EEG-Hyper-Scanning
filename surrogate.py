"""Generate leader-fixed cross-team surrogate IBC within behavioral team strata."""
# Original author: Min

import os

# Limit library threads before importing numerical packages.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import itertools
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hypyp
import mne
import numpy as np

from hyperEEG import data_split, find_file_dir

STAGES = ("pre", "exp", "post")
PAIR_ORDER = list(itertools.combinations(range(4), 2))


def tum_standard(group_name="good", raw_root="data", results_dir="results"):
    """Select teams by mean MTD; these strata are not dyad-level ML targets."""
    raw_root, results_dir = Path(raw_root), Path(results_dir)
    teams = find_file_dir(raw_root / "correct") + find_file_dir(raw_root / "error")
    with (results_dir / "time_group.pkl").open("rb") as stream:
        times = pickle.load(stream)
    with (results_dir / "tum_time_diff_mean.pkl").open("rb") as stream:
        mean_mtd = np.asarray(pickle.load(stream)).mean(axis=1)
    groups = {
        "good": np.flatnonzero(mean_mtd <= .4),
        "middle": np.flatnonzero((mean_mtd > .4) & (mean_mtd < .55)),
        "bad": np.flatnonzero(mean_mtd > .55),
    }
    if len(teams) != len(mean_mtd) or len(times) != len(teams):
        raise ValueError("Recording folders, drumbeat times and MTD must have the same team order")
    if group_name.lower() not in groups:
        raise ValueError("Group must be good, middle or bad")
    selected = groups[group_name.lower()]
    return [teams[i] for i in selected], [times[i] for i in selected]


def get_epochs(group_name="good", raw_root="data", results_dir="results"):
    """Epoch each source team using its own leader's drumbeat times."""
    teams, times = tum_standard(group_name, raw_root, results_dir)
    epochs = [[], [], []]
    for team, team_times in zip(teams, times):
        for output, stage in zip(epochs, data_split(find_file_dir(team), team_times)):
            output.append(stage)
    return tuple(epochs)


def build_surrogate_teams(pre_all, exp_all, post_all):
    """Keep each leader and draw followers from three distinct other teams."""
    n_teams = len(pre_all)
    if n_teams < 4 or any(len(stage) != n_teams for stage in (exp_all, post_all)):
        raise ValueError("Require at least four teams and matching stage lengths")
    stages = (pre_all, exp_all, post_all)
    output = [[], [], []]
    for focal_team in range(n_teams):
        donors = np.random.choice(
            [i for i in range(n_teams) if i != focal_team], size=3, replace=False
        )
        members = [np.random.randint(1, 4) for _ in donors]
        # Reuse the same donor members across stages, retaining source epoch alignment.
        for stage, result in zip(stages, output):
            team = [stage[focal_team][0]]
            team.extend(stage[donor][member] for donor, member in zip(donors, members))
            result.append(team)
    return tuple(output)


def compute_team_ibc_mean(s_pre, s_exp, s_post, delta_low=1):
    """Return teams x 3 stages x 6 pairs x 6 metrics x 5 bands x 30 x 30."""
    bands = {
        "delta": [delta_low, 4], "theta": [4, 8], "alpha": [8, 13],
        "beta": [13, 30], "gamma": [30, 49],
    }
    modes = ("ccorr", "coh", "pli", "wpli", "pow_corr", "plv")
    result = []
    for team_stages in zip(s_pre, s_exp, s_post):
        stage_results = []
        for team in team_stages:
            pair_results = []
            for first, second in PAIR_ORDER:
                epochs = [team[first].copy(), team[second].copy()]
                mne.epochs.equalize_epoch_counts(epochs)
                analytic = hypyp.analyses.compute_freq_bands(
                    data=np.array(epochs), sampling_rate=200, freq_bands=bands
                )
                metrics = []
                for mode in modes:
                    synchrony = hypyp.analyses.compute_sync(
                        complex_signal=analytic, mode=mode, epochs_average=True
                    )
                    metrics.append(synchrony[:, :30, 30:60])
                pair_results.append(metrics)
            stage_results.append(pair_results)
        result.append(stage_results)
    return np.asarray(result)


def split_iterations(n_surrogate, n_workers):
    """Assign alternate iteration numbers to workers."""
    return [list(range(worker, n_surrogate, n_workers)) for worker in range(n_workers)]


def surrogate_worker(
    group_name, iteration_list, save_dir, base_seed=20260508,
    skip_existing=True, delta_low=1,
):
    """Read cached epochs inside each worker to avoid transferring large objects."""
    save_dir = Path(save_dir)
    suffix = "" if group_name == "good" else f"_{group_name}"
    pending = [
        i for i in iteration_list
        if not (skip_existing and (save_dir / f"C_surrogate{suffix}_{i}.pkl").exists())
    ]
    if pending:
        stages = []
        for stage in STAGES:
            with (save_dir / f"{stage}_all{suffix}.pkl").open("rb") as stream:
                stages.append(pickle.load(stream))
        for iteration in pending:
            np.random.seed(base_seed + iteration)
            surrogate = build_surrogate_teams(*stages)
            ibc = compute_team_ibc_mean(*surrogate, delta_low=delta_low)
            path = save_dir / f"C_surrogate{suffix}_{iteration}.pkl"
            with path.open("wb") as stream:
                pickle.dump(ibc, stream, protocol=pickle.HIGHEST_PROTOCOL)
            metadata = {
                "seed": base_seed + iteration,
                "delta_hz": [delta_low, 4],
                "pair_order": PAIR_ORDER,
                "alignment": "Donor epochs retain their own source-team leader alignment",
            }
            Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2))
            print(f"{group_name}: saved surrogate {iteration}", flush=True)
            del surrogate, ibc
    return {"group_name": group_name, "finished_iterations": list(iteration_list)}


def main(raw_root="data", results_dir="results", iterations=20, workers=2, delta_low=1):
    """Generate missing files; leave existing caches and their metadata unchanged."""
    if iterations < 1 or workers < 1:
        raise ValueError("Iterations and workers must be positive")
    for group in ("good", "middle", "bad"):
        folder = Path(results_dir) / "surrogate" / ("Good" if group == "good" else group)
        folder.mkdir(parents=True, exist_ok=True)
        suffix = "" if group == "good" else f"_{group}"
        paths = [folder / f"{stage}_all{suffix}.pkl" for stage in STAGES]
        if not all(path.exists() for path in paths):
            for path, epochs in zip(paths, get_epochs(group, raw_root, results_dir)):
                with path.open("wb") as stream:
                    pickle.dump(epochs, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(surrogate_worker, group, chunk, folder, 20260508, True, delta_low)
                for chunk in split_iterations(iterations, workers) if chunk
            ]
            for future in as_completed(futures):
                print(future.result(), flush=True)

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--delta-low", type=float, default=1)
    args = parser.parse_args()
    main(args.raw_root, args.results_dir, args.iterations, args.workers, args.delta_low)
