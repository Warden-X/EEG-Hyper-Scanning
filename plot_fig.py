"""Plot supplementary IBC group differences and export exploratory graph comparisons."""
# Original author: Chen Min

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

GROUP_RESULT = Path("results/statis_abalysis/group/Supplementary_statistical_resutls.py.pkl")
CHANNELS = (
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
    "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
)


def plot_ibc_groups(result_path=GROUP_RESULT, output_dir="results/figure/significant difference"):
    """Plot the original between-group ANOVA masks for Supplementary Figs. S2-S4."""
    with Path(result_path).open("rb") as stream:
        result = pickle.load(stream)
    p_values = np.asarray(result["p_value"], dtype=float)
    if p_values.shape != (3, 6, 5, 30, 30) or not np.isfinite(p_values).all():
        raise ValueError("Group p values must be finite with shape (3, 6, 5, 30, 30)")
    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("Group p values must lie between zero and one")

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 16, "savefig.dpi": 300})
    stage_names = ("Pre", "Exp", "Post")
    metrics = ("CCorr", "Coherence", "PLI", "WPLI", "Power Corr", "PLV")
    band_values = (.1, .2, .3, .4, .5)
    output_paths = []
    for stage, stage_name in enumerate(stage_names):
        folder = Path(output_dir) / stage_name
        folder.mkdir(parents=True, exist_ok=True)
        for metric, metric_name in enumerate(metrics):
            figure, ax = plt.subplots(figsize=(6, 5))
            for band, value in enumerate(band_values):
                layer = np.where(p_values[stage, metric, band] <= .05, value, np.nan)
                sns.heatmap(
                    layer, vmin=0, vmax=1, cmap="Set1", cbar=False, center=.5,
                    xticklabels=CHANNELS, yticklabels=CHANNELS, alpha=.7, ax=ax,
                )
            path = folder / f"{metric_name}.png"
            figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.02)
            plt.close(figure)
            output_paths.append(path)
    return output_paths


def anova_analysis(data):

    df_data = pd.DataFrame(data).T
    df_data.columns = ['G', 'M', 'B']
    df_data_melt = df_data.melt()
    df_data_melt = df_data_melt.dropna()
    df_data_melt.columns = ['Class','Index']

    model = ols('Index~C(Class)', data=df_data_melt).fit()
    anova_table = anova_lm(model)

    f_value = anova_table['F'].iloc[0]
    p_value = anova_table['PR(>F)'].iloc[0]

    if p_value<0.05:
        tukey_result = pairwise_tukeyhsd(df_data_melt['Index'], df_data_melt['Class'], alpha = 0.05)
        tukey_result = tukey_result.reject

        tukey_result = tukey_result.astype(np.int32)
    else:
        tukey_result = [0,0,0]

    return p_value, f_value, tukey_result


def split_group(name):
    """
    return: row 0: good     row 1: middle    row 2: bad
    """

    with open(f'./results/{name}.pkl', 'rb') as file_obj:
        index_all = pickle.load(file_obj)

    index_all = np.transpose(index_all, (0, 2, 3, 1, 4))

    with open('./results/tum_time_diff_mean.pkl', 'rb') as file_obj:
        tum_time_diff = pickle.load(file_obj)

    tum_time_diff = np.mean(tum_time_diff, axis=1)

    good_p = np.where(tum_time_diff<=0.4)[0] #  T<0.4s
    middle_p = np.where((tum_time_diff>0.4) & (tum_time_diff<0.55))[0] # 0.4s<T<0.55
    bad_p = np.where(tum_time_diff>=0.55)[0]  # T>0.55s

    good_group = index_all[:,:,:,good_p,:]
    middle_group = index_all[:,:,:,middle_p,:]
    bad_group = index_all[:,:,:,bad_p,:]

    return good_group, middle_group, bad_group


def statis_analysis_group_GT(name):
    """Statistical analysis of each group at different stages """

    good_group, middle_group, bad_group =  split_group(name)

    p_value_all = np.zeros((3, 6, 5, 4))
    f_value_all = np.zeros((3, 6, 5, 4))
    tukey_result_all = np.zeros((3, 6, 5, 4, 3))

    for s in range(3):
        for i in range(6):
            for j in range(5):
                for m in range(4):
                    data = [good_group[s,i,j,:,m], middle_group[s,i,j,:,m],
                                            bad_group[s,i,j,:,m]]

                    p_value, f_value, tukey_result = anova_analysis(data)

                    p_value_all[s,i,j,m] = p_value
                    f_value_all[s,i,j,m] = f_value
                    tukey_result_all[s,i,j,m] = tukey_result

    return p_value_all, f_value_all, tukey_result_all


def export_graph_groups(graph_name="index_all_95", output_dir="results/statis_abalysis/graph_theory"):
    """Export original group comparisons for the exploratory graph figure."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = graph_name
    p_value_all, _, tukey_result_all = statis_analysis_group_GT(name)

    num = np.argwhere(p_value_all<0.05)
    i0 = num[:, 0]
    i1 = num[:, 1]
    i2 = num[:, 2]
    i3 = num[:, 3]

    mask = np.any(tukey_result_all[i0, i1, i2, i3] == 1, axis=1)
    i0, i1, i2, i3 = i0[mask], i1[mask], i2[mask], i3[mask]

    good_group, middle_group, bad_group =  split_group(name)

    good_group = np.transpose(good_group, (0,1,2,4,3))
    good_group = good_group[i0,i1,i2,i3,:]

    middle_group = np.transpose(middle_group, (0,1,2,4,3))
    middle_group = middle_group[i0,i1,i2,i3,:]

    bad_group = np.transpose(bad_group, (0,1,2,4,3))
    bad_group = bad_group[i0,i1,i2,i3,:]

    stage = ['Pre', 'Exp', 'Post']
    index1 = ['CCorr', 'Coh', 'PLI', 'WPLI', 'PowerCorr', 'PLV']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    index2 = ['GE', 'Sigma', 'LC', 'CC']

    name = []
    for i in range(len(i0)):
        name_ = stage[i0[i]] + '_' + index1[i1[i]] + '_' + bands[i2[i]] + '_' + index2[i3[i]]
        name.append(name_)

    df_good = pd.DataFrame(good_group, index=name)
    df_mid = pd.DataFrame(middle_group, index=name)
    df_bad = pd.DataFrame(bad_group, index=name)

    df_good.to_excel(output_dir / "good_group.xlsx", index=True)
    df_mid.to_excel(output_dir / "middle_group.xlsx", index=True)
    df_bad.to_excel(output_dir / "bad_group.xlsx", index=True)
    return df_good, df_mid, df_bad


def plot_behavior(mtd_path="results/tum_time_diff_mean.pkl", output_path="results/figure/MTD_histogram.png"):
    """Plot the original mean drumbeat time-difference histogram."""
    with Path(mtd_path).open("rb") as stream:
        mtd = np.asarray(pickle.load(stream))
    if mtd.ndim != 2 or mtd.shape[1] != 3 or not np.isfinite(mtd).all() or np.any(mtd < 0):
        raise ValueError("MTD must be finite and nonnegative: teams x three LF pairs")
    data = mtd.ravel()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 18, "savefig.dpi": 300})
    figure, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(data, bins=50, density=True, alpha=.6, color="#1F77B4", edgecolor="black")
    ax.set(xlabel="Mean Time Difference", ylabel="Probability Density", title="Normalized Histogram")
    figure.subplots_adjust(bottom=.2)
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("groups", "graph", "behavior", "all"), default="all")
    args = parser.parse_args()
    if args.part in ("groups", "all"):
        plot_ibc_groups()
    if args.part in ("graph", "all"):
        export_graph_groups()
    if args.part in ("behavior", "all"):
        plot_behavior()
