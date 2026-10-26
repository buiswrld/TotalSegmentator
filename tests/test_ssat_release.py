"""CPU-only release checks; runnable with stdlib unittest, without model weights."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.pipeline.common import get_subjects, brier_decomposition, ece_mce
from experiments.rebuttal.request2_triage import review_weights
from experiments.rebuttal._nsd import nsd


class ReleaseChecks(unittest.TestCase):
    def test_named_split_never_falls_back_to_all_subjects(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            (p / "s0001").mkdir()
            with self.assertRaises(ValueError):
                get_subjects(folder, "test", None)
            (p / "meta.csv").write_text("image_id,split\ns0001,train\ns0002,test\n")
            self.assertEqual(get_subjects(folder, "test", None), ["s0002"])
            with self.assertRaises(ValueError):
                get_subjects(folder, "typo", None)
            self.assertEqual(get_subjects(folder, "train", 0), [])

    def test_shard_windows_are_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "meta.csv").write_text(
                "image_id,split\n" + "".join(f"s{i:04},train\n" for i in range(10)))
            a = get_subjects(folder, "all", 6)
            b = get_subjects(folder, "all", 4, offset=6)
            self.assertFalse(set(a) & set(b))
            self.assertEqual(len(a) + len(b), 10)

    def test_calibration_schema_remains_compatible_with_paper_scripts(self):
        y = np.array([0., 1.]); p = np.array([0.1, 0.9])
        self.assertEqual(set(brier_decomposition(y, p)),
                         {"brier", "reliability", "resolution", "uncertainty"})
        _, _, rows = ece_mce(y, p)
        self.assertEqual(set(rows[0]), {"n", "mean_pred", "observed", "gap"})

    def test_review_budget_splits_boundary_ties(self):
        np.testing.assert_allclose(review_weights(np.array([1., .5, .5, 0.]), 2),
                                   [1., .5, .5, 0.])

    def test_surface_distance_uses_physical_spacing(self):
        a = np.zeros((8, 8, 8), bool); a[2, 3, 3] = True
        b = np.zeros_like(a); b[3, 3, 3] = True
        self.assertEqual(nsd(a, b, (2., 1., 1.)), {1.: 0., 2.: 1., 3.: 1.})
        self.assertEqual(nsd(a, a, (2., 1., 1.)), {1.: 1., 2.: 1., 3.: 1.})


if __name__ == "__main__":
    unittest.main()
