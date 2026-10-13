"""
Request 2: review-budget triage. Rank masks by model risk (1 - predicted acceptability),
send the top 10 / 20 / 30 percent to review, and report the caught-reject rate (fraction of
all true rejects that land in the reviewed set). Baseline = the frozen organ-rate ranking.

Tie-breaking: the organ-rate baseline assigns one score per structure, so many masks tie at
the budget boundary. We use FRACTIONAL (expected) tie-breaking: masks tied at the cutoff
risk share the remaining review slots proportionally, which is the expected recall under
uniform random tie-breaking and the fair way to score a tied ranker. Continuous model scores
have no boundary ties, so this reduces to the hard top-k for the model. Random and oracle
frame the scale.

Note: Request 2 is a NEW rebuttal analysis; the frozen pipeline reports organ-rate only as an
OOF AUC (paper_consolidate.organ_rate_oof), it has no pre-existing budget-triage code, so the
fractional rule is stated here rather than inherited.

- CT: the frozen 400-tree randomforest_[raw] DEV out-of-fold predictions (labeled DEV OOF).
- MRI: the frozen regressor scored on the 49-subject held-out test (PREVIOUSLY EXPOSED; see
  rebuttal_results/CONTAMINATION_AUDIT.md). Also a POST HOC auto-accept analysis.
"""
import os, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from experiments.rebuttal._scores import ct_dev_frame, mri_test_frame

BUDGETS = [0.10, 0.20, 0.30]


def review_weights(risk, k):
    """Fractional review weights for the top-k riskiest, splitting boundary ties evenly."""
    if k <= 0:
        return np.zeros(len(risk))
    order = np.sort(risk)[::-1]
    cutoff = order[k - 1]
    w = np.where(risk > cutoff, 1.0, 0.0)
    tie = risk == cutoff
    n_above = int((risk > cutoff).sum())
    n_tie = int(tie.sum())
    if n_tie:
        w[tie] = (k - n_above) / n_tie
    return w


def triage(df, split_label):
    N = len(df); rej = (df["accept"] == 0).to_numpy(); n_rej = int(rej.sum())
    rows = []
    for b in BUDGETS:
        k = int(round(b * N))
        for name, risk in [("model", 1 - df["score_model"].to_numpy()),
                           ("organ_rate_baseline", 1 - df["score_base"].to_numpy())]:
            w = review_weights(risk, k)
            caught = float((w * rej).sum())
            rows.append({"split": split_label, "budget_pct": int(b * 100), "ranker": name,
                         "n_reviewed": k, "n_total": N, "n_rejects_total": n_rej,
                         "rejects_caught": round(caught, 1),
                         "caught_reject_rate": round(caught / n_rej, 4),
                         "precision_in_budget": round(caught / k, 4) if k else 0.0})
        rows.append({"split": split_label, "budget_pct": int(b * 100), "ranker": "random_reference",
                     "n_reviewed": k, "n_total": N, "n_rejects_total": n_rej,
                     "rejects_caught": round(k / N * n_rej, 1),
                     "caught_reject_rate": round(k / N, 4), "precision_in_budget": round(n_rej / N, 4)})
        rows.append({"split": split_label, "budget_pct": int(b * 100), "ranker": "oracle_reference",
                     "n_reviewed": k, "n_total": N, "n_rejects_total": n_rej,
                     "rejects_caught": min(k, n_rej), "caught_reject_rate": round(min(1.0, k / n_rej), 4),
                     "precision_in_budget": round(min(k, n_rej) / k, 4) if k else 0.0})
    return pd.DataFrame(rows)


def auto_accept(df, split_label, fracs=(0.10, 0.20)):
    """POST HOC: auto-accept the top fraction by predicted quality; report the failure rate
    (share of true rejects) among the auto-accepted, model vs baseline. Fractional ties."""
    N = len(df); rej = (df["accept"] == 0).to_numpy()
    rows = []
    for f in fracs:
        k = int(round(f * N))
        for name, good in [("model", df["score_model"].to_numpy()),
                           ("organ_rate_baseline", df["score_base"].to_numpy())]:
            w = review_weights(good, k)                 # highest quality first
            fail = float((w * rej).sum())
            rows.append({"split": split_label, "auto_accept_pct": int(f * 100), "ranker": name,
                         "n_auto_accepted": k, "expected_failures": round(fail, 1),
                         "failure_rate_pct": round(100 * fail / k, 2) if k else 0.0})
    return pd.DataFrame(rows)


os.makedirs("rebuttal_results/ct", exist_ok=True)
os.makedirs("rebuttal_results/mri", exist_ok=True)
ct_df, mri_df = ct_dev_frame(), mri_test_frame()

ct = triage(ct_df, "CT_dev_OOF"); mri = triage(mri_df, "MRI_test_exposed")
ct.to_csv("rebuttal_results/ct/request2_triage_dev_oof.csv", index=False)
mri.to_csv("rebuttal_results/mri/request2_triage_test.csv", index=False)
aa = auto_accept(mri_df, "MRI_test_exposed")
aa.to_csv("rebuttal_results/mri/request2_autoaccept_posthoc_test.csv", index=False)

for title, d in [("CT  (DEV OOF, 400-tree randomforest_[raw])", ct),
                 ("MRI (held-out TEST, 49 subj, PREVIOUSLY EXPOSED; see audit)", mri)]:
    print(f"\n=== Request 2 triage (fractional ties): {title} ===")
    print(d.to_string(index=False))
print("\n=== Request 2 POST HOC auto-accept (MRI test, exposed) ===")
print(aa.to_string(index=False))
