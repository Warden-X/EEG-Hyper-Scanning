import pickle
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Inputs, outputs and comparison order
IBC_PATH = Path("results/C_all_mean.pkl")
GRAPH_PATH = Path("results/index_all_95.pkl")
OUTPUT_DIR = Path("results/statis_abalysis/stage_dynamics_original_fdr_simple")

EXPECTED_IBC_SHAPE = (96, 3, 3, 6, 5, 30, 30)
EXPECTED_GRAPH_SHAPE = (3, 96, 6, 5, 4)
PAIRS = ((0, 1, "Pre-Exe"), (1, 2, "Exe-Rec"), (0, 2, "Pre-Rec"))


def prepare_ibc(ibc, expected_shape=EXPECTED_IBC_SHAPE):
    """Reshape IBC into three stages x 288 LF pairs x features."""

    ibc = np.asarray(ibc, dtype=float)
    if ibc.shape != expected_shape or not np.isfinite(ibc).all():
        raise ValueError(f"IBC data must be finite with shape {expected_shape}")

    # Input: teams x stages x three LF pairs x features.
    # Output: stages x 288 LF pairs x features.
    stage_first = np.transpose(ibc, (1, 0, 2, *range(3, ibc.ndim)))
    return stage_first.reshape(3, expected_shape[0] * expected_shape[2], *ibc.shape[3:])


def analyze_stages(data, alpha=0.05):
    """Compare three stages using one-way ANOVA, BH-FDR and Tukey HSD."""

    data = np.asarray(data, dtype=float)
    if data.ndim < 3 or data.shape[0] != 3 or not np.isfinite(data).all():
        raise ValueError("data must be finite: 3 stages × samples × features")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    n_stages, n_samples = data.shape[:2]
    feature_shape = data.shape[2:]

    # Compute one-way ANOVA across three stages for each feature.
    stage_means = data.mean(axis=1)
    grand_mean = data.mean(axis=(0, 1))
    ss_between = n_samples * np.sum((stage_means - grand_mean) ** 2, axis=0)
    ss_within = np.sum((data - stage_means[:, None]) ** 2, axis=(0, 1))
    df_between = n_stages - 1
    df_within = n_stages * (n_samples - 1)
    mse = ss_within / df_within
    f_value = (ss_between / df_between) / mse
    p_raw = stats.f.sf(f_value, df_between, df_within)

    # Apply BH-FDR to all omnibus p values within this feature family.
    reject, p_fdr, _, _ = multipletests(p_raw.reshape(-1), alpha=alpha, method="fdr_bh")
    reject_fdr = reject.reshape(feature_shape)
    p_fdr = p_fdr.reshape(feature_shape)

    # Compare Pre-Exe, Exe-Rec and Pre-Rec using Tukey HSD.
    q_critical = stats.studentized_range.ppf(1 - alpha, n_stages, df_within)
    standard_error = np.sqrt(mse / n_samples)
    differences = np.stack([stage_means[a] - stage_means[b] for a, b, _ in PAIRS])
    q_value = np.divide(
        np.abs(differences),
        standard_error,
        out=np.full((len(PAIRS), *feature_shape), np.nan),
        where=standard_error > 0,
    )
    tukey_reject = q_value > q_critical

    # Keep pairwise rejections only after a significant FDR-corrected omnibus test.
    final_pair_reject = tukey_reject & reject_fdr[None]
    all_three = np.logical_and.reduce(final_pair_reject, axis=0)

    return {
        "p_raw": p_raw,
        "p_fdr": p_fdr,
        "reject_fdr": reject_fdr,
        "tukey_reject": tukey_reject,
        "final_pair_reject": final_pair_reject,
        "all_three_pair_reject": all_three,
    }


def main(
    ibc_path=IBC_PATH,
    graph_path=GRAPH_PATH,
    output_dir=OUTPUT_DIR,
    alpha=0.05,
):
    """Analyze real IBC and graph features and save plotting masks."""

    # Load the original inputs.
    with Path(ibc_path).open("rb") as stream:
        ibc_raw = pickle.load(stream)
    with Path(graph_path).open("rb") as stream:
        graph = np.asarray(pickle.load(stream), dtype=float)

    # Check dimensions and reshape IBC into 288 LF pairs.
    ibc = prepare_ibc(ibc_raw)
    if graph.shape != EXPECTED_GRAPH_SHAPE or not np.isfinite(graph).all():
        raise ValueError(f"Graph data must be finite with shape {EXPECTED_GRAPH_SHAPE}")

    # Apply the same statistical procedure to IBC and graph features.
    ibc_result = analyze_stages(ibc, alpha=alpha)
    graph_result = analyze_stages(graph, alpha=alpha)

    # Save significance masks used by the Fig. 6 plotting script.
    payload = {
        "alpha": alpha,
        "pairs": [name for _, _, name in PAIRS],
        "ibc": ibc_result,
        "graph": graph_result,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "original_stage_fdr_masks.pkl"
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream)

    # Print significant feature counts.
    for name, result in (("IBC", ibc_result), ("Graph", graph_result)):
        pair_counts = result["final_pair_reject"].reshape(3, -1).sum(axis=1)
        all_three_count = int(result["all_three_pair_reject"].sum())
        print(
            f"{name}: Pre-Exe={pair_counts[0]}, "
            f"Exe-Rec={pair_counts[1]}, Pre-Rec={pair_counts[2]}, "
            f"all three={all_three_count}"
        )
    print(f"Saved: {output_path}")
    return payload, output_path


if __name__ == "__main__":
    main()
