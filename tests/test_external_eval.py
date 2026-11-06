"""Smoke tests for the Taglish (external) evaluation on DUMMY data only (invented strings, random tiny model; never real data).

    python -m unittest discover -s tests -v        (run from tsektxt-model-training)
Stats tests need numpy/pandas/scipy; the end-to-end test also needs torch + transformers and is skipped cleanly without them.
"""
import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import numpy, pandas, scipy  # noqa: F401
    HAVE_STATS = True
except ImportError:
    HAVE_STATS = False
try:
    import torch, transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

WORDS = "ito ay dummy na post tungkol sa test lang the election is fake news totoo hindi".split()
MODELS = ["roberta-tagalog", "mbert", "xlmr", "baseline-tfidf"]


def dummy_rows(n=40):
    return [{"id": f"d{i}", "text": " ".join(WORDS[(i + j) % len(WORDS)] for j in range(6)), "label": i % 2} for i in range(n)]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


@unittest.skipUnless(HAVE_STATS, "numpy/pandas/scipy missing")
class Stats(unittest.TestCase):
    def test_holm_hand_computed(self):
        from compare import holm
        self.assertEqual([round(x, 10) for x in holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])
        self.assertEqual(holm([0.5, 0.9, 0.001])[2], 0.003)  # smallest p x 3
        self.assertTrue(all(x <= 1 for x in holm([0.6, 0.7, 0.9])))

    def test_stats_files(self):
        import external_stats
        rows = dummy_rows(); y = [r["label"] for r in rows]
        with tempfile.TemporaryDirectory() as d:
            res = Path(d)
            for k, m in enumerate(MODELS):  # each model gets a different set of invented mistakes
                (res / m).mkdir()
                pred = [(v if (i + k) % (k + 3) else 1 - v) for i, v in enumerate(y)]
                write_csv(res / m / "external_predictions.csv", [{"id": r["id"], "y_true": r["label"], "y_pred": p, "prob_credible": 0.5} for r, p in zip(rows, pred)])
            external_stats.main(d, B=200)
            ci = json.loads((res / "external_bootstrap_ci.json").read_text()); mc = json.loads((res / "external_mcnemar_holm.json").read_text())
            self.assertEqual(ci["n"], 40); self.assertEqual(set(ci["per_model"]), set(MODELS))
            for v in ci["per_model"].values():
                lo, hi = v["accuracy_ci95"]; self.assertLessEqual(lo, v["accuracy"] + 1e-9); self.assertGreaterEqual(hi, v["accuracy"] - 1e-9)
            self.assertEqual(len(mc["tests"]), 6); self.assertEqual(sum(t["family"] for t in mc["tests"]), 3)
            for t in mc["tests"]:
                if t["family"]: self.assertGreaterEqual(t["p_holm"], t["p"])
                else: self.assertIsNone(t["p_holm"])

    def test_mismatched_ids_rejected(self):
        import external_stats
        rows = dummy_rows(10)
        with tempfile.TemporaryDirectory() as d:
            for m, off in (("mbert", 0), ("xlmr", 1)):
                (Path(d) / m).mkdir()
                write_csv(Path(d) / m / "external_predictions.csv", [{"id": f"d{int(r['id'][1:]) + off}", "y_true": r["label"], "y_pred": r["label"], "prob_credible": 0.5} for r in rows])
            with self.assertRaises(SystemExit): external_stats.main(d, B=10)


@unittest.skipUnless(HAVE_TORCH and HAVE_STATS, "torch/transformers missing (run in Colab)")
class EndToEnd(unittest.TestCase):
    def test_pipeline_with_tiny_random_model(self):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        run = lambda *a, cwd=None: subprocess.run([sys.executable] + list(a), check=True, capture_output=True, text=True, cwd=cwd)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
            (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
            tok = BertTokenizer(str(d / "vocab.txt")); model = BertForSequenceClassification(BertConfig(
                vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, num_labels=2))
            tiny = d / "tiny"; tok.save_pretrained(tiny); model.save_pretrained(tiny)
            write_csv(d / "dummy.csv", dummy_rows(40))
            for m in ["roberta-tagalog", "mbert", "xlmr"]:
                run(str(ROOT / "src/evaluate_external.py"), "--model", m, "--csv", str(d / "dummy.csv"), "--model-dir", str(tiny),
                    "--out-dir", str(d / "results"), "--config", str(ROOT / "configs/config.yaml"), cwd=ROOT)
            run(str(ROOT / "src/external_stats.py"), "--results-dir", str(d / "results"), "--resamples", "50", cwd=ROOT)
            self.assertTrue((d / "results" / "external_bootstrap_ci.json").exists())
            self.assertTrue((d / "results" / "mbert" / "external_metrics.json").exists())
            run(str(ROOT / "src/compare.py"), cwd=d)  # compare.py reads/writes ./results
            self.assertIn("Holm-adjusted p", (d / "results" / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
