"""Tune manuscript classifiers on training teams and test on held-out teams."""
# Original author: comeo

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

N_JOBS = int(os.environ.get("EEG_ML_JOBS", "1"))
STAGES = ("pre", "exp", "post")


def repeated_group_cv_score(
    clf, X_train, y_train, groups_train, n_splits=10, repeats=10, random_state=2
):
    """Average grouped CV scores after training-only feature selection."""
    groups_train = np.asarray(groups_train)
    if groups_train.shape != np.asarray(y_train).shape:
        raise ValueError("Training labels and team IDs must be aligned")
    if len(np.unique(groups_train)) < n_splits:
        raise ValueError("Too few training teams for grouped CV")

    repeated_scores = []
    for repeat in range(repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state + repeat
        )
        folds = list(cv.split(X_train, y_train, groups=groups_train))
        scores = cross_val_score(
            clf, X_train, y_train, cv=folds, n_jobs=N_JOBS, error_score="raise"
        )
        repeated_scores.append(scores.mean())
    return np.mean(repeated_scores)


def make_classifier(name, parameters):
    """Construct a classifier with the original parameter settings."""
    if name == "SVM":
        return SVC(C=parameters[0], gamma=parameters[1])
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=parameters[0], max_depth=parameters[1], random_state=2
        )
    if name == "KNN":
        return KNeighborsClassifier(
            algorithm="kd_tree", n_neighbors=parameters[0], leaf_size=parameters[1]
        )
    if name == "LDA":
        return LinearDiscriminantAnalysis(solver="eigen", shrinkage=parameters)
    if name == "NB":
        return GaussianNB(var_smoothing=parameters)
    if name == "LR":
        return LogisticRegression(max_iter=500, random_state=2)
    raise ValueError(f"Unknown classifier: {name}")


def grid_result(scores, parameters):
    """Store grid scores and the selected parameter values."""
    scores = np.asarray(scores)
    return {
        "acc": scores,
        "acc_max": scores.max(),
        "index_optimal": parameters,
    }


def SVM_results(X_train, y_train, groups_train):
    C_values = np.arange(100, 1000, 50)
    gamma_values = np.linspace(0.01, 0.5, 50)
    scores = np.empty((len(gamma_values), len(C_values)))
    for i, gamma in enumerate(gamma_values):
        for j, C in enumerate(C_values):
            scores[i, j] = repeated_group_cv_score(
                make_classifier("SVM", [C, gamma]), X_train, y_train, groups_train
            )
            print(f"SVM C={C}, gamma={gamma}: {scores[i, j]}")
    row, column = np.unravel_index(np.argmax(scores), scores.shape)
    return grid_result(scores, [C_values[column], gamma_values[row]])


def RF_results(X_train, y_train, groups_train):
    trees = np.arange(50, 250, 10)
    depths = np.arange(5, 30, 2)
    scores = np.empty((len(trees), len(depths)))
    for i, n in enumerate(trees):
        for j, depth in enumerate(depths):
            scores[i, j] = repeated_group_cv_score(
                make_classifier("RF", [n, depth]), X_train, y_train, groups_train
            )
            print(f"RF trees={n}, depth={depth}: {scores[i, j]}")
    row, column = np.unravel_index(np.argmax(scores), scores.shape)
    return grid_result(scores, [trees[row], depths[column]])


def KNN_results(X_train, y_train, groups_train):
    neighbors = np.arange(5, 100, 5)
    leaf_sizes = np.arange(10, 300, 10)
    scores = np.empty((len(neighbors), len(leaf_sizes)))
    for i, n in enumerate(neighbors):
        for j, leaf_size in enumerate(leaf_sizes):
            scores[i, j] = repeated_group_cv_score(
                make_classifier("KNN", [n, leaf_size]), X_train, y_train, groups_train
            )
            print(f"KNN neighbors={n}, leaf_size={leaf_size}: {scores[i, j]}")
    row, column = np.unravel_index(np.argmax(scores), scores.shape)
    return grid_result(scores, [neighbors[row], leaf_sizes[column]])


def LDA_results(X_train, y_train, groups_train):
    parameters = np.linspace(0.01, 1, 100)
    scores = [
        repeated_group_cv_score(
            make_classifier("LDA", value), X_train, y_train, groups_train
        )
        for value in parameters
    ]
    return grid_result(scores, parameters[np.argmax(scores)])


def NB_results(X_train, y_train, groups_train):
    parameters = np.array(
        [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1, 5]
    )
    scores = [
        repeated_group_cv_score(
            make_classifier("NB", value), X_train, y_train, groups_train
        )
        for value in parameters
    ]
    return grid_result(scores, parameters[np.argmax(scores)])


def LR_results(X_train, y_train, groups_train):
    return repeated_group_cv_score(
        make_classifier("LR", None), X_train, y_train, groups_train
    )


def ml_test(X_train, y_train, X_test, y_test, key_word, index, return_model=False):
    """Fit on training teams and report test accuracy and weighted F1."""
    model = make_classifier(key_word, index).fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    f1 = f1_score(y_test, model.predict(X_test), average="weighted")
    return (accuracy, f1, model) if return_model else (accuracy, f1)


def save_pickle(value, path):
    with Path(path).open("wb") as stream:
        pickle.dump(value, stream)


def fit_dataset(data):
    """Apply the six manuscript classifiers to one feature set."""
    tuners = {
        "SVM": SVM_results, "RF": RF_results, "KNN": KNN_results,
        "LDA": LDA_results, "NB": NB_results, "LR": LR_results,
    }
    tuning, performance, models = {}, {}, {}
    for name, tune in tuners.items():
        tuning[name] = tune(data["X_train"], data["y_train"], data["groups_train"])
        parameters = None if name == "LR" else tuning[name]["index_optimal"]
        accuracy, f1, model = ml_test(
            data["X_train"], data["y_train"], data["X_test"], data["y_test"],
            name, parameters, return_model=True,
        )
        performance[name] = (accuracy, f1)
        models[name] = model
    return tuning, performance, models


def main(analysis_dir="results/results/mean"):
    analysis_dir = Path(analysis_dir)
    all_dir = analysis_dir / "All_Stages"
    all_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / "dataset_mean.pkl").open("rb") as stream:
        datasets = pickle.load(stream)
    with (analysis_dir / "feat_names_mean.pkl").open("rb") as stream:
        names = pickle.load(stream)

    # All stages must retain the same original pairs and held-out teams.
    shared_keys = (
        "y_train", "y_test", "groups_train", "groups_test",
        "pair_ids_train", "pair_ids_test",
    )
    for stage in STAGES:
        data = datasets[stage]
        if data.get("selection_scope") != "outer_training_teams_only":
            raise ValueError("Generate training-only features before classifier tuning")
        if np.intersect1d(data["groups_train"], data["groups_test"]).size:
            raise ValueError("Training and test sets share a team")
        for key in shared_keys:
            if not np.array_equal(data[key], datasets["pre"][key]):
                raise ValueError(f"Stage {stage} has a different sample order: {key}")

    tuning = {name: {} for name in ("SVM", "RF", "KNN", "LDA", "NB", "LR")}
    performance = {name: {} for name in tuning}
    for stage in STAGES:
        stage_tuning, stage_scores, models = fit_dataset(datasets[stage])
        for name in tuning:
            tuning[name][stage] = stage_tuning[name]
            performance[name][stage] = stage_scores[name]
        save_pickle(models["SVM"], analysis_dir / f"{stage}_SVM.pkl")
    save_pickle(tuning, analysis_dir / "results_algorithms_tuning.pkl")
    save_pickle(performance, analysis_dir / "results_algorithms_test.pkl")

    combined = {key: datasets["pre"][key] for key in shared_keys}
    combined["selection_scope"] = "outer_training_teams_only"
    for part in ("train", "test"):
        combined[f"X_{part}"] = np.concatenate(
            [datasets[stage][f"X_{part}"] for stage in STAGES], axis=1
        )
    combined_names = [f"{name}_{stage}" for stage in STAGES for name in names[stage]]
    save_pickle(combined, all_dir / "dataset_mean_all_stages.pkl")
    save_pickle(combined_names, all_dir / "feat_names_300.pkl")

    tuning, performance, models = fit_dataset(combined)
    save_pickle(tuning, all_dir / "results_all_stages_tuning.pkl")
    save_pickle(performance, all_dir / "results_all_stages_test.pkl")
    save_pickle(models["SVM"], all_dir / "clf_SVM.pkl")
    save_pickle(models["LDA"], all_dir / "clf_LDA.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="results/results/mean")
    main(parser.parse_args().analysis_dir)
