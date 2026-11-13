"""Reselect features and retrain All-SVM independently for cross-team surrogates."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from build_dataset import grouped_stratified_split
from classifier_compare import make_classifier
from feat_selection_methods import feat_selection_F
from feature_names import feature_names

STAGES = ("pre", "exp", "post")
C_VALUES = np.arange(100, 1000, 50)
GAMMA_VALUES = np.linspace(.01, .5, 50)


def load(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def make_cv(y, focal, donor, n_splits=10, repeats=10):
    """Exclude both focal and donor source teams across each internal split."""
    folds = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=2 + repeat)
        for train, validation in splitter.split(np.zeros((len(y), 1)), y, focal):
            validation_teams = np.union1d(focal[validation], donor[validation])
            keep = ~np.isin(focal[train], validation_teams) & ~np.isin(donor[train], validation_teams)
            train = train[keep]
            if not np.array_equal(np.unique(y[train]), [0, 1, 2]):
                raise ValueError("An internal training fold is missing a class")
            folds.append((train, validation))
    return folds


def read_surrogate(rows, root, pair_ids):
    """Reconstruct LF inputs from the recovered pointers to recomputed IBC."""
    pointers = ["original_pair_id", "source_file", "source_focal_local_team", "source_lf_slot"]
    if rows[pointers].isna().any().any():
        raise ValueError("Recovered source pointers contain missing values")
    X = np.full((3, len(pair_ids), 27000), np.nan)
    positions = {pair: position for position, pair in enumerate(pair_ids)}
    for source_file, source_rows in rows.groupby("source_file"):
        cache = np.asarray(load(Path(root) / source_file))
        if cache.ndim != 7 or cache.shape[1:] != (3, 6, 6, 5, 30, 30):
            raise ValueError(f"Invalid surrogate IBC shape: {source_file}")
        for row in source_rows.itertuples(index=False):
            if not (0 <= row.source_focal_local_team < len(cache) and 0 <= row.source_lf_slot < 3):
                raise ValueError("Recovered source indices are outside the LF cache")
            X[:, positions[row.original_pair_id]] = cache[
                row.source_focal_local_team, :, row.source_lf_slot].reshape(3, -1)
    if not np.isfinite(X).all():
        raise ValueError("Surrogate features contain missing or nonfinite values")
    return X


def select_features(X, y_train, train, test, folds, n_jobs=1, top_k=200, final_k=100):
    """Fit F screening and LDA forward selection on outer training data only."""
    selected_train, selected_test, indices, pools = [], [], [], []
    for stage_X in X:
        screened, _, pool = feat_selection_F(stage_X[train], y_train, top_k)
        selector = SequentialFeatureSelector(
            LinearDiscriminantAnalysis(solver="eigen", shrinkage=.2),
            n_features_to_select=final_k, direction="forward", scoring="accuracy",
            cv=folds, n_jobs=n_jobs,
        ).fit(screened, y_train)
        selected = pool[selector.get_support(indices=True)]
        selected_train.append(stage_X[np.ix_(train, selected)])
        selected_test.append(stage_X[np.ix_(test, selected)])
        indices.append(selected)
        pools.append(pool)
    return {"X_train": np.concatenate(selected_train, axis=1),
            "X_test": np.concatenate(selected_test, axis=1),
            "selected_indices": np.array(indices), "screened_indices": np.array(pools)}


def parameter_scores(X, y, folds, C, gamma):
    """Fit a fresh RBF SVM on every internal training fold."""
    result = []
    for train, validation in folds:
        model = make_classifier("SVM", [C, gamma]).fit(X[train], y[train])
        result.append(model.score(X[validation], y[validation]))
    return result


def fit_svm(X, y, folds, n_jobs=1, C_values=C_VALUES, gamma_values=GAMMA_VALUES, repeats=10):
    """Search the original grid and refit on all outer training observations."""
    parameters = [(C, gamma) for gamma in gamma_values for C in C_values]
    fold_scores = np.asarray(Parallel(n_jobs=n_jobs)(
        delayed(parameter_scores)(X, y, folds, C, gamma) for C, gamma in parameters))
    # Preserve the original mean-of-repeat-means and gamma-first tie order.
    means = fold_scores.reshape(len(parameters), repeats, -1).mean(axis=2).mean(axis=1)
    best = int(np.argmax(means))
    model = make_classifier("SVM", parameters[best]).fit(X, y)
    return model, {"parameters": parameters, "fold_scores": fold_scores,
                   "mean_scores": means, "best_index": best}


def scores(target, prediction):
    return {"accuracy": accuracy_score(target, prediction),
            "weighted_f1": f1_score(target, prediction, average="weighted", zero_division=0),
            "macro_f1": f1_score(target, prediction, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(target, prediction)}


def evaluate(root, mtd_path, plan_path, output_dir, iterations=20, n_jobs=1):
    """Retrain each surrogate without loading original features or fitted models."""
    if iterations < 1 or n_jobs == 0:
        raise ValueError("Iterations must be positive and jobs must be nonzero")
    root = Path(root)
    mtd_path, plan_path, output_dir = root / mtd_path, root / plan_path, root / output_dir
    mtd = np.asarray(load(mtd_path))
    if mtd.ndim != 2 or mtd.shape[1] != 3 or not np.isfinite(mtd).all() or np.any(mtd < 0):
        raise ValueError("MTD must be finite and nonnegative: teams x three LF dyads")
    if np.any((mtd == .4) | (mtd == .55)):
        raise ValueError("Exact MTD boundaries have no original dyad label")
    labels = np.where(mtd < .4, 2, np.where(mtd > .55, 0, 1)).ravel()
    pair_ids = np.concatenate([np.flatnonzero(labels == value) for value in (2, 1, 0)])
    y, focal = labels[pair_ids], pair_ids // 3
    train, test = grouped_stratified_split(y, focal, random_state=2)
    partition = np.full(len(y), "test", dtype="U5")
    partition[train] = "train"
    mean_mtd = mtd.mean(axis=1)
    strata = np.where(mean_mtd <= .4, 2, np.where(mean_mtd > .55, 0, 1))
    plan = pd.read_csv(plan_path)
    if plan.duplicated(["reassignment_iteration", "original_pair_id"]).any():
        raise ValueError("The plan contains duplicate realization/dyad identities")
    if not set(range(iterations)).issubset(plan.reassignment_iteration):
        raise ValueError("The plan does not contain every requested realization")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_names = feature_names()
    performance, predictions, feature_rows, fold_rows = [], [], [], []
    for iteration in range(iterations):
        rows = plan[plan.reassignment_iteration == iteration]
        if not np.array_equal(np.sort(rows.original_pair_id), np.arange(len(labels))):
            raise ValueError("Each realization must contain every original LF dyad exactly once")
        rows = rows.set_index("original_pair_id", drop=False).loc[pair_ids]
        donor = rows.donor_original_team.to_numpy()
        if (not np.array_equal(rows.inherited_original_label, y)
                or not np.array_equal(rows.focal_original_team, focal)
                or not np.array_equal(rows.partition, partition)):
            raise ValueError("The plan disagrees with original labels, focal teams or the fixed split")
        if np.any((donor < 0) | (donor >= len(mtd))) or np.any(donor == focal):
            raise ValueError("Donors must belong to other valid source teams")
        if (not np.array_equal(strata[donor], strata[focal])
                or not (rows.groupby("focal_original_team").donor_original_team.nunique() == 3).all()):
            raise ValueError("Require three distinct donor teams within each focal team's MTD stratum")
        if np.intersect1d(np.union1d(focal[train], donor[train]), np.union1d(focal[test], donor[test])).size:
            raise ValueError("Training and test sets share a leader or donor source team")
        X = read_surrogate(rows, root, pair_ids)
        folds = make_cv(y[train], focal[train], donor[train])
        print(f"Surrogate {iteration + 1}/{iterations}: selecting features", flush=True)
        selected = select_features(X, y[train], train, test, folds[:10], n_jobs)
        del X
        print(f"Surrogate {iteration + 1}/{iterations}: tuning All-SVM", flush=True)
        model, tuning = fit_svm(selected["X_train"], y[train], folds, n_jobs)
        prediction = model.predict(selected["X_test"])
        result = {"iteration": iteration, "C": float(model.C), "gamma": float(model.gamma),
                  "conditional_cv_accuracy": float(tuning["mean_scores"][tuning["best_index"]]),
                  **scores(y[test], prediction)}
        performance.append(result)
        bundle = {**selected, "model": model, "tuning": tuning, "cv_folds": folds,
                  "y_train": y[train], "y_test": y[test], "prediction": prediction,
                  "pair_ids_train": pair_ids[train], "pair_ids_test": pair_ids[test],
                  "focal_train": focal[train], "focal_test": focal[test],
                  "donor_train": donor[train], "donor_test": donor[test], "scores": result,
                  "selected_feature_names": [[all_names[i] for i in indices] for indices in selected["selected_indices"]]}
        with (output_dir / f"surrogate_{iteration:02d}.pkl").open("wb") as stream:
            pickle.dump(bundle, stream, protocol=4)
        predictions.extend({"iteration": iteration, "original_pair_id": int(pair), "donor_team": int(team),
                            "true_label": int(target), "prediction": int(value)}
                           for pair, team, target, value in zip(pair_ids[test], donor[test], y[test], prediction))
        feature_rows.extend({"iteration": iteration, "stage": stage, "feature_index": int(index),
                             "feature_name": all_names[index]}
                            for stage, indices in zip(STAGES, selected["selected_indices"]) for index in indices)
        fold_rows.extend({"iteration": iteration, "repeat": k // 10, "fold": k % 10,
                          "train_pairs": len(tr), "validation_pairs": len(va), "shared_source_teams": 0}
                         for k, (tr, va) in enumerate(folds))
        print(f"Surrogate {iteration + 1}/{iterations}: accuracy={result['accuracy']:.4f}", flush=True)
    frame = pd.DataFrame(performance)
    summary = frame[list(scores(y[test], prediction))].agg(["mean", "std", "min", "max"]).T
    summary.index.name = "metric"
    with pd.ExcelWriter(output_dir / "svm_surrogate_retrained.xlsx") as writer:
        frame.to_excel(writer, sheet_name="classification_scores", index=False)
        summary.to_excel(writer, sheet_name="performance_summary")
        pd.DataFrame(predictions).to_excel(writer, sheet_name="test_predictions", index=False)
        pd.DataFrame(feature_rows).to_excel(writer, sheet_name="selected_features", index=False)
        pd.DataFrame(fold_rows).to_excel(writer, sheet_name="inner_fold_audit", index=False)
    metadata = {"pipeline": "independent_surrogate_retraining_All_SVM", "iterations": iterations,
                "seed": 2, "train_pairs": len(train), "test_pairs": len(test),
                "train_teams": len(np.unique(focal[train])), "test_teams": len(np.unique(focal[test])),
                "train_class_counts": np.bincount(y[train], minlength=3).tolist(),
                "test_class_counts": np.bincount(y[test], minlength=3).tolist(),
                "F_features_per_stage": 200, "SFS_features_per_stage": 100,
                "SVM_grid": {"C": C_VALUES.tolist(), "gamma": GAMMA_VALUES.tolist()},
                "SFS_folds": 10, "tuning_repeats": 10, "tuning_folds": 10,
                "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (mtd_path, plan_path)},
                "cv_qualification": "F screening precedes SFS CV and SFS precedes tuning CV. Internal CV scores are conditional selection criteria, not unbiased nested-CV estimates. Held-out source teams are excluded throughout.",
                "sampling_qualification": "Conditional reassignment from the recovered, already recomputed cross-team LF cache pool; original dyad labels are retained."}
    (output_dir / "retraining_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)
    return frame, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--mtd", type=Path, default=Path("results/tum_time_diff_mean.pkl"))
    parser.add_argument("--plan", type=Path, default=Path("results/surrogate_recovered/reassignment_plan.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/statis_abalysis/surrogate_ml_retrained"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    evaluate(args.root, args.mtd, args.plan, args.output, args.iterations, args.jobs)
