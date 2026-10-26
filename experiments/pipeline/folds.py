"""Subject-grouped folds with explicit tie ordering for historical reproduction.

Stable ordering of equal-size groups matches all committed CT organ-rate OOF
predictions and the MRI baseline summary. Do not rely on NumPy's default sort
to resolve equal group sizes: that behavior is not a recorded fold assignment.
"""
import numpy as np
from sklearn.model_selection import GroupKFold


class StableGroupKFold(GroupKFold):
    def _iter_test_indices(self, X, y=None, groups=None):
        if getattr(self, "shuffle", False):
            raise ValueError("Historical folds use shuffle=False")
        if groups is None:
            raise ValueError("Subject groups are required")
        subjects, inverse = np.unique(groups, return_inverse=True)
        if self.n_splits > len(subjects):
            raise ValueError("Fewer subjects than folds")
        counts = np.bincount(inverse)
        order = np.argsort(counts, kind="stable")[::-1]
        weights = np.zeros(self.n_splits)
        assignment = np.zeros(len(subjects), dtype=int)
        for subject in order:
            fold = np.argmin(weights)
            weights[fold] += counts[subject]
            assignment[subject] = fold
        for fold in range(self.n_splits):
            yield np.flatnonzero(assignment[inverse] == fold)
