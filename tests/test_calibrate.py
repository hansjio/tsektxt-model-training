"""Temperature scaling on SYNTHETIC logits only (random numbers for testing code, never used as data).

    python -m unittest tests.test_calibrate -v        (run from tsektxt-model-training; needs numpy only)
"""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import calibrate as c


class TestCalibrate(unittest.TestCase):
    def setUp(self):
        import numpy as np
        rng = np.random.default_rng(0); n = 4000
        self.y = rng.integers(0, 2, n); z = (2 * self.y - 1) * 2 + rng.normal(0, 2, n)
        self.lg = np.c_[np.zeros(n), 3 * z]  # deliberately overconfident by a factor of 3

    def test_fit_recovers_overconfidence(self):
        T = c.fit_temperature(self.lg, self.y)
        self.assertGreater(T, 2.5); self.assertLess(T, 3.5)

    def test_predictions_unchanged_and_better_calibrated(self):
        T = c.fit_temperature(self.lg, self.y)
        b, pb = c.report(self.lg, self.y, 1.0); a, pa = c.report(self.lg, self.y, T)
        self.assertTrue((pb == pa).all())
        self.assertLess(a["ece_15_bins"], b["ece_15_bins"]); self.assertLess(a["nll"], b["nll"])
        self.assertEqual(sum(x["n"] for x in a["reliability_bins"]), len(self.y))

    def test_probability_helper(self):
        self.assertAlmostEqual(c.calibrate_p_credible(0.9, 1.0), 0.9, places=6)
        self.assertLess(c.calibrate_p_credible(0.999, 3.0), 0.999)
        self.assertGreater(c.calibrate_p_credible(0.999, 3.0), 0.5)  # same side of 0.5
        self.assertEqual(c.load_temperature("none", path=ROOT / "no_such_file.json"), (1.0, False))


if __name__ == "__main__":
    unittest.main()
