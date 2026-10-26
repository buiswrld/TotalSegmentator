"""Verify committed paper/rebuttal tables without inference or refitting.

Run from the repository root: python experiments/verify_saved_results.py
This verifies saved evidence; it does not establish historical test-set isolation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.rebuttal.request2_triage import triage


def main():
    pf = ROOT / "experiments/paper_final"
    ct = pd.read_csv(pf / "ct/primary_model_oof.csv")
    mr = pd.read_csv(pf / "mri/iou_regression_oof.csv")
    ct_raw = pd.read_csv(pf / "ct/dataset_combined_metrics.csv")
    ct_raw = ct_raw[ct_raw.is_empty == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(ct[["subject", "organ", "iou"]],
                                  ct_raw[["subject", "organ", "iou"]])
    assert not ct.duplicated(["subject", "organ"]).any()
    assert not mr.duplicated(["subject", "organ"]).any()
    assert (len(ct), ct.subject.nunique(), len(mr), mr.subject.nunique()) == (4017, 57, 548, 38)
    y = (ct.iou >= 0.9).astype(int)
    scores = {"ct_primary_saved_auc": roc_auc_score(y, ct["randomforest_[raw]"]),
              "ct_organ_rate_saved_auc": roc_auc_score(y, ct["organ-rate_baseline"]),
              "mri_primary_saved_auc": roc_auc_score((mr.iou >= 0.9).astype(int),
                                                     mr.pred_iou_RandomForest)}
    for key, target in zip(scores, (0.871, 0.761, 0.919)):
        assert round(scores[key], 3) == target, (key, scores[key])
    frame = ct[["subject", "organ", "iou"]].assign(
        accept=y, score_model=ct["randomforest_[raw]"], score_base=ct["organ-rate_baseline"])
    expected = pd.read_csv(ROOT / "rebuttal_results/ct/request2_triage_dev_oof.csv")
    pd.testing.assert_frame_equal(triage(frame, "CT_dev_OOF"), expected, check_dtype=False)
    within = []
    table = pd.read_csv(ROOT / "rebuttal_results/ct/request3_within_structure_dev_oof.csv")
    for organ, g in frame.groupby("organ"):
        if len(g) >= 25 and min(g.accept.sum(), len(g) - g.accept.sum()) >= 5:
            auc = roc_auc_score(g.accept, g.score_model)
            recorded = table.loc[table.organ == organ, "within_structure_auc"].item()
            assert np.isclose(round(auc, 4), recorded), organ
            within.append(auc)
    scores["ct_primary_mean_within_structure_auc"] = float(np.mean(within))
    scores["ct_primary_eligible_structures"] = len(within)
    print(json.dumps({"saved_table_checks": "PASS", **scores}, indent=2))


if __name__ == "__main__":
    main()
