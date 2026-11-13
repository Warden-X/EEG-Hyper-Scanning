"""Prepare grouped train/test data and select features on training teams."""
# Original author: Lenovo

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pickle
from sklearn.model_selection import StratifiedGroupKFold

from feat_selection_methods import feat_selection_F, feat_selection_SFS

from feature_names import feature_names


def data_import_mean(
    real_path="results/C_all_mean.pkl",
    mtd_path="results/tum_time_diff_mean.pkl",
    return_pair_ids=False,
):
    """Flatten LF features in Good, Middle, Bad order; retain original pair IDs."""
    with open(real_path, "rb") as stream:
        ibc = np.asarray(pickle.load(stream))
    if ibc.ndim != 7 or ibc.shape[1:] != (3, 3, 6, 5, 30, 30):
        raise ValueError("IBC axes must be teams x stages x LF pairs x metrics x bands x channels x channels")
    if not np.isfinite(ibc).all():
        raise ValueError("IBC contains nonfinite values")
    with open(mtd_path, "rb") as stream:
        mtd = np.asarray(pickle.load(stream))
    if mtd.shape != (len(ibc), 3) or not np.isfinite(mtd).all() or np.any(mtd < 0):
        raise ValueError("MTD must be finite, nonnegative, and aligned with original LF pairs")
    if np.any((mtd == .4) | (mtd == .55)):
        raise ValueError("Original strict MTD thresholds do not define exact boundary labels")

    # Keep the original class order and the team-major order within each class.
    labels = np.where(mtd < .4, 2., np.where(mtd > .55, 0., 1.)).ravel()
    pair_ids = np.concatenate([np.flatnonzero(labels == value) for value in (2, 1, 0)])
    stage_features = ibc.swapaxes(0, 1).reshape(3, len(labels), -1)
    result = (stage_features[:, pair_ids], labels[pair_ids], pair_ids // 3)
    return (*result, pair_ids) if return_pair_ids else result


def grouped_stratified_split(y, groups, random_state=2):
    """Create an approximately 7:3 split without sharing teams."""

    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape or len(np.unique(groups)) < 10:
        raise ValueError("Require aligned labels and at least ten teams")
    splitter = StratifiedGroupKFold(
        n_splits=10, shuffle=True, random_state=random_state
    )
    placeholder_X = np.zeros((len(y), 1))
    test_folds = []

    for fold_number, (_, fold_test_index) in enumerate(
        splitter.split(placeholder_X, y, groups=groups)
    ):
        if fold_number < 3:
            test_folds.append(fold_test_index)

    test_index = np.sort(np.concatenate(test_folds))
    train_index = np.setdiff1d(np.arange(len(y)), test_index)
    if np.intersect1d(groups[train_index], groups[test_index]).size:
        raise ValueError("A team crossed the train/test boundary")
    if any(not np.array_equal(np.unique(y[ids]), [0, 1, 2]) for ids in [train_index, test_index]):
        raise ValueError("Each predefined partition must contain all three coordination classes")
    return train_index, test_index


def feat_selection_mean(
    real_path="results/C_all_mean.pkl", mtd_path="results/tum_time_diff_mean.pkl",
    output_dir="results/results/mean", top_k=200, final_k=100, cv_folds=10,
    random_state=2, n_jobs=1, input_delta_low=2,
):
    """Select F-screened and forward-selected features using training teams only."""

    dataset, y, team_ids, pair_ids = data_import_mean(real_path, mtd_path, return_pair_ids=True)
    if not 0 < final_k < top_k <= dataset.shape[-1]:
        raise ValueError("Require 0 < final_k < top_k <= stage feature count")
    stages = ["pre", "exp", "post"]

    dataset_mean = {}
    feat_index_mean = {}
    feat_names_mean = {}

    train_index, test_index = grouped_stratified_split(y, team_ids, random_state)
    y_train = y[train_index]
    y_test = y[test_index]
    groups_train = team_ids[train_index]
    groups_test = team_ids[test_index]

    all_names = feature_names()
    for stage, X in zip(stages, dataset):
        X_train = X[train_index]
        X_test = X[test_index]

        X_train, y_train, feat_index = feat_selection_F(X_train, y_train, top_k)
        X_test = X_test[:, feat_index]
        feat_names = [all_names[i] for i in feat_index]

        # Forward selection uses grouped folds within the training partition.
        X_train, feat_index2 = feat_selection_SFS(
            X_train, y_train, groups_train, direction="forward", n_features=final_k,
            n_splits=cv_folds, random_state=random_state, n_jobs=n_jobs,
        )

        feat_index = [feat_index[i] for i in feat_index2]
        feat_names = [feat_names[i] for i in feat_index2]

        X_test = X_test[:, feat_index2]

        dataset_one = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "groups_train": groups_train,
            "groups_test": groups_test,
            "pair_ids_train": pair_ids[train_index],
            "pair_ids_test": pair_ids[test_index],
            "selection_scope": "outer_training_teams_only",
        }

        dataset_mean[stage] = dataset_one
        feat_index_mean[stage] = feat_index
        feat_names_mean[stage] = feat_names

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in [
        ("dataset_mean.pkl", dataset_mean),
        ("feat_index_mean.pkl", feat_index_mean),
        ("feat_names_mean.pkl", feat_names_mean),
    ]:
        with (output_dir / filename).open("wb") as stream:
            pickle.dump(value, stream)

    metadata = {
        "pipeline": "team_grouped_training_only_selection",
        "seed": random_state,
        "top_k": top_k, "final_k": final_k, "sfs_cv_folds": cv_folds,
        "train_pairs": len(train_index), "test_pairs": len(test_index),
        "train_teams": len(np.unique(groups_train)),
        "test_teams": len(np.unique(groups_test)),
        "shared_teams": 0, "cached_input_delta_hz": [input_delta_low, 4],
        "source_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (real_path, mtd_path)
        },
        "cv_qualification": (
            "F screening precedes training-side SFS and tuning CV; these scores "
            "are conditional model-selection criteria, not unbiased nested-CV "
            "estimates. Final held-out teams are excluded from all selection and tuning."
        ),
    }
    (output_dir / "selection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return dataset_mean

# %% main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-only F/SFS selection with a fixed team-grouped holdout")
    parser.add_argument("--real", default="results/C_all_mean.pkl")
    parser.add_argument("--mtd", default="results/tum_time_diff_mean.pkl")
    parser.add_argument("--output", default="results/results/mean")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--input-delta-low", type=float, default=2, help="Lower Delta bound used to generate the supplied cached IBC")
    args = parser.parse_args()
    feat_selection_mean(args.real, args.mtd, args.output, n_jobs=args.jobs, input_delta_low=args.input_delta_low)
