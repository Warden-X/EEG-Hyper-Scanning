"""Feature names in the same metric-band-channel order as flattened IBC."""
# Original author: Chen Min


def feature_names():
    metrics = ("CCorr", "Coh", "PLI", "WPLI", "Pow_Corr", "PLV")
    bands = ("Delta", "Theta", "Alpha", "Beta", "Gamma")
    channels = (
        "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
        "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5", "CP6",
        "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2",
    )
    return [
        f"{metric}_{band}_{source}_{target}"
        for metric in metrics
        for band in bands
        for source in channels
        for target in channels
    ]
