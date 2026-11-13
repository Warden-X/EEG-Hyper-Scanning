"""Explain stage and combined-stage SVM predictions using training backgrounds."""
# Original author: Chen Min

import os
import pickle
from pathlib import Path

import hypyp
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
from sklearn.svm import SVC

from NDFSysMNE import mneNDF
from feature_names import feature_names

STAGES = ("pre", "exp", "post")
CHANNELS = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
)


def load_pickle(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


def bar_color(index):
    return ("#f59311", "#63b2ee", "#1F77B4")[index]


def explain_svm(data, names, parameters, output_path):
    """Refit the selected SVM with probability calibration for SHAP."""
    train = pd.DataFrame(data["X_train"], columns=names)
    test = pd.DataFrame(data["X_test"], columns=names)
    model = SVC(
        C=parameters[0], gamma=parameters[1], probability=True, random_state=2
    ).fit(train, data["y_train"])
    explainer = shap.KernelExplainer(
        model.predict_proba, shap.kmeans(train, 5), link="logit"
    )
    values = explainer.shap_values(test)
    shap.summary_plot(
        values, test, plot_type="bar", class_names=["Bad", "Middle", "Good"],
        max_display=20, show=False, class_inds=[2, 1, 0], color=bar_color,
    )
    plt.xlabel("Mean SHAP Value")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    return values, test


def ch_inf(file_path):
    """Load EEG channel locations for the interbrain topographic plot."""
    data = mneNDF(file_path).read2MneRaw().pick(list(CHANNELS))
    data.set_montage(montage="standard_1020")
    return data


def main(analysis_dir="results/results/mean", output_dir="results/figure/SHAP"):
    np.random.seed(2)
    plt.rcParams.update({
        "savefig.dpi": 300, "font.size": 18, "font.family": "Times New Roman",
        "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    })
    analysis_dir, output_dir = Path(analysis_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = load_pickle(analysis_dir / "feat_index_mean.pkl")
    datasets = load_pickle(analysis_dir / "dataset_mean.pkl")
    tuning = load_pickle(analysis_dir / "results_algorithms_tuning.pkl")
    all_names = feature_names()
    stage_names = {stage: [all_names[i] for i in indices[stage]] for stage in STAGES}

    for stage in STAGES:
        explain_svm(
            datasets[stage], stage_names[stage],
            tuning["SVM"][stage]["index_optimal"], output_dir / f"{stage}.png",
        )

    combined_names = [
        f"{name}_{label}"
        for stage, label in zip(STAGES, ("Pre", "Exe", "Rec"))
        for name in stage_names[stage]
    ]
    all_dir = analysis_dir / "All_Stages"
    combined_data = load_pickle(all_dir / "dataset_mean_all_stages.pkl")
    combined_tuning = load_pickle(all_dir / "results_all_stages_tuning.pkl")
    values, test = explain_svm(
        combined_data, combined_names,
        combined_tuning["SVM"]["index_optimal"], output_dir / "ALL.png",
    )

    shap.summary_plot(values[:, :, 2], test, max_display=30, show=False)
    plt.gcf().set_size_inches(5, 10)
    plt.gcf().set_dpi(300)
    plt.tight_layout(pad=1)
    plt.savefig(output_dir / "good.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    # Map the same top 20 combined-stage features to channel pairs.
    importance = np.sum(np.abs(values), axis=(0, 2))
    top20 = np.argsort(importance)[-20:][::-1]
    channel_index = {channel: i for i, channel in enumerate(CHANNELS)}
    matrix = np.zeros((30, 30))
    for index in top20:
        first, second = combined_names[index].split("_")[-3:-1]
        matrix[channel_index[first], channel_index[second]] = importance[index]
    recording = os.environ.get("EEG_EXAMPLE_RECORDING", "data/example/1")
    channels = ch_inf(recording)
    channels.set_montage("biosemi32")
    hypyp.viz.viz_2D_topomap_inter(channels, channels, matrix, lab=True)


if __name__ == "__main__":
    main()
