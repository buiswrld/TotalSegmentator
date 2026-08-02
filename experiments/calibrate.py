"""
Calibration analysis for the mask accept/reject classifier.

AUC says the model RANKS masks correctly. Calibration says its scores are usable as
probabilities - that a score of 0.8 means the mask really is good about 80% of the time.
For a paper about communicating model reliability to non-specialists, the second property
matters more than the first: "85% chance this mask is within tolerance" is a sentence a
clinician can act on, an AUC is not.

Produces:
  calibration.png          reliability diagram + score histogram
  calibration_metrics.csv  ECE / MCE / Brier decomposition per method
  calibrated_oof.csv       out-of-fold calibrated probabilities

Run:  python calibrate.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_reference import load_reference, add_relative_features

# ============================ CONFIG ============================
CSV           = r"C:\Users\ansar\Algoverse\report_ct_val\combined_metrics.csv"
REFERENCE_CSV = r"C:\Users\ansar\Algoverse\report_ct\reference_stats.csv"
OUT_DIR       = r"C:\Users\ansar\Algoverse\report_ct_val"

IOU_ACCEPT = 0.90
N_SPLITS   = 5
N_BINS     = 10
SEED       = 0
# ================================================================

LEAK_COLS = ["dice", "iou", "tp", "fp", "fn", "gt_vox", "status", "accept"]
ID_COLS   = ["subject", "organ", "orig_axcodes", "pred_vox"]


# ---------------------------------------------------------------- metrics

def ece_mce(y, p, n_bins=N_BINS, equal_count=True):
    """Expected and maximum calibration error.

    Equal-count bins by default. With equal-WIDTH bins most of the mass lands in one or
    two bins (half these scores sit above 0.8), so the average is dominated by nearly
    empty bins. Equal-count bins give every bin the same weight in the average, which is
    the more honest summary when scores are skewed.
    """
    n = len(y)
    if equal_count:
        order = np.argsort(p)
        edges = [order[i] for i in np.linspace(0, n, n_bins + 1)[1:-1].astype(int)]
        bins = np.split(order, [np.searchsorted(np.sort(p), p[e], "left") for e in edges])
        bins = [b for b in bins if len(b)]
    else:
        idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
        bins = [b for b in bins if len(b)]
    ece = 0.0
    mce = 0.0
    rows = []
    for b in bins:
        conf, acc = p[b].mean(), y[b].mean()
        gap = abs(acc - conf)
        ece += len(b) / n * gap
        mce = max(mce, gap)
        rows.append({"n": len(b), "mean_pred": conf, "observed": acc, "gap": acc - conf})
    return ece, mce, rows


def brier_decomposition(y, p, n_bins=N_BINS):
    """Brier = reliability - resolution + uncertainty (lower Brier is better).

    reliability: how far predictions sit from observed rates (0 is perfect calibration)
    resolution:  how much predictions vary from the base rate (higher = more informative)
    uncertainty: the base rate's own variance - a property of the data, not the model
    """
    n = len(y)
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    base = y.mean()
    rel = res = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        nk, pk, ok = sel.sum(), p[sel].mean(), y[sel].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2
    return {"brier": brier_score_loss(y, p), "reliability": rel / n,
            "resolution": res / n, "uncertainty": base * (1 - base)}


# ---------------------------------------------------------------- calibrators

def fit_calibrator(kind, p_train, y_train):
    if kind == "none":
        return lambda p: p
    if kind == "sigmoid":   # Platt scaling: one-parameter logistic on the logit
        lr = LogisticRegression(C=1e10, solver="lbfgs")
        lr.fit(p_train.reshape(-1, 1), y_train)
        return lambda p: lr.predict_proba(p.reshape(-1, 1))[:, 1]
    if kind == "isotonic":  # monotone step function; flexible but can overfit small data
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(p_train, y_train)
        return lambda p: iso.predict(p)
    raise ValueError(kind)


# ---------------------------------------------------------------- experiment

def nested_oof(df, cols, y, groups, base_estimator, methods=("none", "sigmoid", "isotonic")):
    """Out-of-fold predictions with calibrators fitted inside the training folds only.

    The calibrator must never see the outer test fold. So within each outer training set
    we run an INNER grouped CV to get honest inner-OOF scores, fit the calibrator on
    those, then refit the base model on the full outer-train and apply the calibrator to
    the outer-test predictions. Fitting a calibrator on the same predictions you evaluate
    would make any method look perfect.
    """
    outer = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    oof = {m: np.zeros(len(y)) for m in methods}
    for tr, te in outer.split(df, y, groups):
        Xtr, Xte = df.iloc[tr][cols], df.iloc[te][cols]
        ytr, gtr = y[tr], groups[tr]

        inner = GroupKFold(n_splits=min(4, len(np.unique(gtr))))
        inner_p = np.zeros(len(tr))
        for itr, ite in inner.split(Xtr, ytr, gtr):
            m = base_estimator()
            m.fit(Xtr.iloc[itr], ytr[itr])
            inner_p[ite] = m.predict_proba(Xtr.iloc[ite])[:, 1]

        model = base_estimator()
        model.fit(Xtr, ytr)
        p_te = model.predict_proba(Xte)[:, 1]

        for m in methods:
            oof[m][te] = fit_calibrator(m, inner_p, ytr)(p_te)
    return oof


# ---------------------------------------------------------------- plot

def reliability_plot(y, curves, out_png, n_bins=N_BINS):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, p in curves.items():
        _, _, rows = ece_mce(y, p, n_bins)
        xs = [r["mean_pred"] for r in rows]
        ys = [r["observed"] for r in rows]
        ax.plot(xs, ys, "o-", ms=5, label=name)
    ax.set_ylabel("observed fraction of good masks")
    ax.set_title("Reliability diagram (equal-count bins, out-of-fold)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for name, p in curves.items():
        axh.hist(p, bins=25, range=(0, 1), histtype="step", label=name)
    axh.set_xlabel("predicted probability mask is good")
    axh.set_ylabel("count")
    axh.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    df = pd.read_csv(CSV)
    df["label"] = (df["iou"] >= IOU_ACCEPT).astype(int)
    df = df[df["is_empty"] == 0].copy() if "is_empty" in df else df

    raw_cols = [c for c in df.columns if c not in LEAK_COLS + ID_COLS + ["label"]]
    if os.path.exists(REFERENCE_CSV):
        df = add_relative_features(df, load_reference(REFERENCE_CSV))
        cols = raw_cols + [c for c in df.columns if c.endswith("_z")]
    else:
        cols = raw_cols
    df = df.reset_index(drop=True)

    y = df["label"].to_numpy()
    groups = df["subject"].to_numpy()
    print(f"{len(df)} masks | {df['subject'].nunique()} subjects | "
          f"{y.mean()*100:.1f}% good | {len(cols)} features\n")

    # Two base models. The weighted one is what you have been reporting; the unweighted
    # one is the same forest without class_weight, which is the single most likely cause
    # of miscalibration (it tells the forest to treat the classes as 50/50).
    bases = {
        "RF (class_weight=balanced)": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                           class_weight="balanced_subsample",
                                           random_state=SEED, n_jobs=-1))]),
        "RF (no class weight)": lambda: Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                           random_state=SEED, n_jobs=-1))]),
    }

    results, curves, store = [], {}, {}
    for base_name, base in bases.items():
        print(f"running {base_name} ...")
        oof = nested_oof(df, cols, y, groups, base)
        for method, p in oof.items():
            ece, mce, _ = ece_mce(y, p)
            bd = brier_decomposition(y, p)
            name = f"{base_name} + {method}"
            results.append({"model": base_name, "calibration": method,
                            "auc": roc_auc_score(y, p), "ece": ece, "mce": mce, **bd})
            store[name] = p
            if method in ("none", "isotonic"):
                curves[name] = p

    res = pd.DataFrame(results).sort_values("ece")
    print(f"\n{'model':30s} {'calib':>10s} {'AUC':>7s} {'ECE':>7s} {'MCE':>7s} "
          f"{'Brier':>7s} {'reliab':>8s} {'resol':>7s}")
    print("-" * 88)
    for r in res.itertuples(index=False):
        print(f"{r.model:30s} {r.calibration:>10s} {r.auc:>7.3f} {r.ece:>7.3f} "
              f"{r.mce:>7.3f} {r.brier:>7.3f} {r.reliability:>8.4f} {r.resolution:>7.4f}")
    print(f"\nuncertainty (base-rate variance, same for every row): "
          f"{results[0]['uncertainty']:.4f}")
    print("lower ECE/MCE/Brier/reliability is better; higher resolution is better")

    os.makedirs(OUT_DIR, exist_ok=True)
    res.to_csv(os.path.join(OUT_DIR, "calibration_metrics.csv"), index=False)

    best = res.iloc[0]
    best_name = f"{best.model} + {best.calibration}"
    print(f"\n=== reliability table: {best_name} ===")
    _, _, rows = ece_mce(y, store[best_name])
    print(f"{'bin mean pred':>14s} {'n':>6s} {'observed':>10s} {'gap':>8s}")
    for r in rows:
        print(f"{r['mean_pred']:>14.3f} {r['n']:>6d} {r['observed']:>10.3f} {r['gap']:>+8.3f}")

    reliability_plot(y, curves, os.path.join(OUT_DIR, "calibration.png"))

    out = df[["subject", "organ", "iou", "label"]].copy()
    for name, p in store.items():
        out[name.replace(" ", "_").replace("(", "").replace(")", "")] = p
    out.to_csv(os.path.join(OUT_DIR, "calibrated_oof.csv"), index=False)

    print(f"\nwritten to {OUT_DIR}:")
    print("  calibration.png  calibration_metrics.csv  calibrated_oof.csv")


if __name__ == "__main__":
    main()