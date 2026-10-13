"""Surface / NSD primitives, identical to the code used in request4_nsd.py, factored out so
the analytic validation (request4c) exercises the exact same functions. Boundary voxels are
defined by scipy binary_erosion's DEFAULT structuring element, generate_binary_structure(3, 1)
= 6-connectivity (face-connected) in 3D: surface = mask minus its 6-connected erosion.
Distances are in millimetres because distance_transform_edt is called with sampling=spacing.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

TAUS = (1.0, 2.0, 3.0)
MARGIN = 5


def surf(m):
    return m & ~binary_erosion(m, border_value=0)


def nsd(pred, ref, spacing, taus=TAUS, margin=MARGIN):
    both = pred | ref
    if not both.any():
        return {t: np.nan for t in taus}
    idx = np.argwhere(both)
    lo = np.maximum(idx.min(0) - margin, 0)
    hi = np.minimum(idx.max(0) + 1 + margin, pred.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sp, sr = surf(pred[sl]), surf(ref[sl])
    nsp, nsr = int(sp.sum()), int(sr.sum())
    if nsp == 0 or nsr == 0:
        return {t: (1.0 if (nsp == 0 and nsr == 0) else 0.0) for t in taus}
    dr = distance_transform_edt(~sr, sampling=spacing)
    dp = distance_transform_edt(~sp, sampling=spacing)
    a, b = dr[sp], dp[sr]
    return {t: (float((a <= t).sum()) + float((b <= t).sum())) / (nsp + nsr) for t in taus}
