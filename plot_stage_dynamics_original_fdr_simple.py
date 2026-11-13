import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Inputs, outputs and plotting style
MASK_PATH = Path(
    "results/statis_abalysis/stage_dynamics_original_fdr_simple/"
    "original_stage_fdr_masks.pkl"
)
OUTPUT_DIR = Path("results/figure/stage_dynamics_original_fdr_simple")
COUNT_FILENAME = "Fig6_IBC_metric_band_counts.xlsx"

BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
CHANNELS = (
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "Cz",
    "C3",
    "C4",
    "T7",
    "T8",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
)

# Keep data order and figure order explicit to avoid metric misalignment.
SOURCE_METRICS = ("CCorr", "Coh", "PLI", "WPLI", "PowerCorr", "PLV")
FIGURE_METRICS = ("CCorr", "Coh", "PLI", "PLV", "Pow_Corr", "WPLI")
SOURCE_INDEX = {
    "CCorr": 0,
    "Coh": 1,
    "PLI": 2,
    "WPLI": 3,
    "Pow_Corr": 4,
    "PLV": 5,
}
GRAPH_Y_LABELS = ("CCorr", "Coh", "PLI", "WPLI", "Pow_Corr", "PLV")

BAND_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)
GRAPH_VALUES = (0.1, 0.2, 0.3, 0.4)
IBC_TICK_SIZE = 10
GRAPH_TICK_SIZE = 16

plt.rcParams.update(
    {
        "font.family": "Nimbus Roman",
        "font.size": 16,
        "savefig.dpi": 300,
    }
)


def load_masks(path=MASK_PATH):
    """Map statistical masks to Fig. 6 panels A-E."""

    with Path(path).open("rb") as stream:
        result = pickle.load(stream)

    ibc_final = np.asarray(result["ibc"]["final_pair_reject"], dtype=bool)
    graph_final = np.asarray(result["graph"]["final_pair_reject"], dtype=bool)
    ibc_masks = {
        "A": ibc_final[0],
        "B": ibc_final[1],
        "C": ibc_final[2],
        "D": np.asarray(result["ibc"]["all_three_pair_reject"], dtype=bool),
    }
    graph_masks = {
        "Pre-Exe": graph_final[0],
        "Exe-Rec": graph_final[1],
        "Pre-Rec": graph_final[2],
        "All-three": np.asarray(result["graph"]["all_three_pair_reject"], dtype=bool),
    }
    omnibus_mask = np.asarray(result["ibc"]["reject_fdr"], dtype=bool)

    if (
        ibc_final.shape != (3, 6, 5, 30, 30)
        or any(mask.shape != (6, 5, 30, 30) for mask in ibc_masks.values())
        or omnibus_mask.shape != (6, 5, 30, 30)
    ):
        raise ValueError("IBC masks must have shape (6, 5, 30, 30)")
    if graph_final.shape != (3, 6, 5, 4) or any(
        mask.shape != (6, 5, 4) for mask in graph_masks.values()
    ):
        raise ValueError("Graph masks must have shape (6, 5, 4)")

    return ibc_masks, graph_masks, omnibus_mask


def plot_ibc(mask, metric_index, output_path):
    """Plot 30 x 30 channel-pair masks for one IBC metric."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(6, 5))

    for band_index, value in enumerate(BAND_VALUES):
        layer = np.where(mask[metric_index, band_index], value, np.nan)
        sns.heatmap(
            layer,
            vmin=0,
            vmax=1,
            cmap="Set1",
            center=0.5,
            cbar=False,
            alpha=0.7,
            xticklabels=CHANNELS,
            yticklabels=CHANNELS,
            ax=ax,
        )

    ax.tick_params(axis="both", labelsize=IBC_TICK_SIZE)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    _show_border(ax)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return ax


def plot_graph(mask, output_path):
    """Plot graph-feature masks for one stage comparison."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(4, 3.5))

    for graph_index, value in enumerate(GRAPH_VALUES):
        layer = np.where(mask[:, :, graph_index], value, np.nan)
        sns.heatmap(
            layer,
            vmin=0,
            vmax=1,
            cmap="Set1",
            center=0.5,
            cbar=False,
            alpha=0.7,
            xticklabels=BANDS,
            yticklabels=GRAPH_Y_LABELS,
            ax=ax,
        )

    ax.tick_params(axis="both", labelsize=GRAPH_TICK_SIZE)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    _show_border(ax)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return ax


def _show_border(ax):
    """Show all four plot borders."""

    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color("black")


def _make_count_table(counts):
    """Arrange metric-band counts in figure order and add totals."""

    table = pd.DataFrame(counts, index=SOURCE_METRICS, columns=BANDS)
    table = table.rename(index={"PowerCorr": "Pow_Corr"}).loc[list(FIGURE_METRICS)]
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table


def _ibc_count_table(mask):
    """Count significant channel pairs by metric and band."""

    counts = np.asarray(mask, dtype=bool).sum(axis=(2, 3))
    return _make_count_table(counts)


def _graph_count_table(mask):
    """Count significant graph features by metric and band."""

    counts = np.asarray(mask, dtype=bool).sum(axis=2)
    return _make_count_table(counts)


def make_count_workbook(ibc_masks, graph_masks, omnibus_mask, output_path):
    """Save detailed metric-band counts for IBC and graph features."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ibc_categories = {
        "Omnibus": omnibus_mask,
        "Pre-Exe": ibc_masks["A"],
        "Exe-Rec": ibc_masks["B"],
        "Pre-Rec": ibc_masks["C"],
        "All-three": ibc_masks["D"],
    }
    graph_categories = graph_masks
    ibc_tables = {
        f"IBC_{name}": _ibc_count_table(mask) for name, mask in ibc_categories.items()
    }
    graph_tables = {
        f"GT_{name}": _graph_count_table(mask)
        for name, mask in graph_categories.items()
    }

    summary_rows = []
    for family, tables in (("IBC", ibc_tables), ("Graph", graph_tables)):
        for name, table in tables.items():
            summary_rows.append(
                {
                    "Family": family,
                    "Category": name.split("_", 1)[1],
                    "Significant_features": int(table.loc["Total", "Total"]),
                }
            )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        for name, table in ibc_tables.items():
            table.to_excel(writer, sheet_name=name)
        for name, table in graph_tables.items():
            table.to_excel(writer, sheet_name=name)
    return output_path


def main(mask_path=MASK_PATH, output_dir=OUTPUT_DIR):
    """Generate 28 separate panels and save the count workbook."""

    ibc_masks, graph_masks, omnibus_mask = load_masks(mask_path)
    output_dir = Path(output_dir)
    output_paths = []

    for panel in ("A", "B", "C", "D"):
        for metric_name in FIGURE_METRICS:
            path = output_dir / panel / f"{panel}_{metric_name}.png"
            plot_ibc(ibc_masks[panel], SOURCE_INDEX[metric_name], path)
            output_paths.append(path)

    for comparison, mask in graph_masks.items():
        path = output_dir / "E" / f"E_{comparison}_GT.png"
        plot_graph(mask, path)
        output_paths.append(path)

    workbook_path = Path(mask_path).parent / COUNT_FILENAME
    make_count_workbook(ibc_masks, graph_masks, omnibus_mask, workbook_path)
    return output_paths


if __name__ == "__main__":
    files = main()
    print(f"Generated {len(files)} subfigures in {OUTPUT_DIR}")
    print(f"Saved count table: {MASK_PATH.parent / COUNT_FILENAME}")
