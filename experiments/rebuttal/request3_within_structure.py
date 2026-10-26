"""
Request 3: absolute within-structure performance. Per structure, report the accept rate and
the model's within-structure discrimination (ROC-AUC computed inside that structure only,
which strips out cross-structure difficulty). Eligibility rule is identical to
ablations.within_organ_auc: n >= 25 masks and >= 5 of the rarer class.

- CT: the frozen 400-tree randomforest_[raw] DEV out-of-fold predictions (labeled DEV OOF).
  This deliberately uses primary_model_oof.csv, NOT within_organ_auc.csv, which came from the
  200-tree ablation run.
- MRI: the frozen regressor scored on the 49-subject held-out test (PREVIOUSLY EXPOSED; see
  rebuttal_results/CONTAMINATION_AUDIT.md).
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from experiments.pipeline.ablations import within_organ_auc, family_of, WITHIN_MIN_N, WITHIN_MIN_MINORITY
from experiments.rebuttal._scores import ct_dev_frame, mri_test_frame


def per_structure(df, split_label, out_csv):
    y = df["accept"].to_numpy()
    p = df["score_model"].to_numpy()
    # full table: every structure, accept rate + eligibility + within-structure AUC
    elig = within_organ_auc(df, p, y, key="organ").set_index("organ")  # exact frozen rule
    rows = []
    for organ, g in df.groupby("organ"):
        yy = g["accept"].to_numpy()
        n_pos, n_neg = int(yy.sum()), int((1 - yy).sum())
        eligible = (len(g) >= WITHIN_MIN_N) and (min(n_pos, n_neg) >= WITHIN_MIN_MINORITY)
        rows.append({"organ": organ, "family": family_of(organ), "n": len(g),
                     "n_accept": n_pos, "n_reject": n_neg, "accept_rate": round(yy.mean(), 4),
                     "eligible_for_auc": eligible,
                     "within_structure_auc": round(float(elig.loc[organ, "auc"]), 4) if eligible else np.nan})
    tab = pd.DataFrame(rows).sort_values(["eligible_for_auc", "within_structure_auc"],
                                         ascending=[False, True], na_position="last")
    tab.to_csv(out_csv, index=False)

    pooled = roc_auc_score(y, p)
    mean_within = float(elig["auc"].mean()) if len(elig) else float("nan")
    print(f"\n=== Request 3: {split_label} ===")
    print(f"  structures total = {df['organ'].nunique()} ; eligible for within-structure AUC "
          f"(n>={WITHIN_MIN_N}, minority>={WITHIN_MIN_MINORITY}) = {len(elig)}")
    print(f"  pooled AUC = {pooled:.4f} ; mean within-structure AUC = {mean_within:.4f} "
          f"; pooled-minus-within gap = {pooled - mean_within:.4f}")
    print(f"  accept rate across structures: min {tab.accept_rate.min():.3f} "
          f"median {tab.accept_rate.median():.3f} max {tab.accept_rate.max():.3f}")
    if len(elig):
        best = elig.auc.sort_values(ascending=False).head(5)
        worst = elig.auc.sort_values().head(5)
        print("  best within-structure AUC :", ", ".join(f"{o} {a:.3f}" for o, a in best.items()))
        print("  worst within-structure AUC:", ", ".join(f"{o} {a:.3f}" for o, a in worst.items()))
    return pooled, mean_within, len(elig)


def main():
    os.makedirs("ssat_outputs/rebuttal/ct", exist_ok=True)
    os.makedirs("ssat_outputs/rebuttal/mri", exist_ok=True)
    per_structure(ct_dev_frame(), "CT (DEV OOF, 400-tree randomforest_[raw])",
                  "ssat_outputs/rebuttal/ct/request3_within_structure_dev_oof.csv")
    per_structure(mri_test_frame(), "MRI (held-out TEST, 49 subj, PREVIOUSLY EXPOSED; see audit)",
                  "ssat_outputs/rebuttal/mri/request3_within_structure_test.csv")


if __name__ == "__main__":
    main()
