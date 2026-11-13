"""F screening and LDA sequential feature selection."""
# Original author: Lenovo

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold


def feat_selection_F(X, y, k):
    """Screen features using an F test on training observations only."""
    selector = SelectKBest(f_classif, k=k)
    selected = selector.fit_transform(X, y)
    return selected, y, selector.get_support(indices=True)


def feat_selection_SFS(
    X, y, groups, direction='forward', n_features=100,
    n_splits=10, random_state=2, n_jobs=1,
):
    """Select features with LDA using grouped folds within the training set."""
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.2)

    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    if X.ndim != 2 or len(X) != len(y) or groups.shape != y.shape or not np.isfinite(X).all():
        raise ValueError("SFS requires finite, aligned training features, labels and team IDs")
    if len(np.unique(groups)) < n_splits or not 0 < n_features < X.shape[1]:
        raise ValueError("Too few training teams or invalid SFS feature count")
    # No training team crosses an internal SFS boundary.
    group_cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    cv_splits = list(group_cv.split(X, y, groups=groups))

    selector = SequentialFeatureSelector(
        clf, n_features_to_select=n_features, cv=cv_splits,
        direction=direction, scoring="accuracy", n_jobs=n_jobs,
    )
    selected = selector.fit_transform(X, y)
    return selected, selector.get_support(indices=True)
