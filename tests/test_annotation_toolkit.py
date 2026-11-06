"""Smoke tests on DUMMY data only (invented strings for testing code, never used as data).

    python -m unittest discover -s tests -v        (run from tsektxt-model-training)
"""
import csv, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import annotation_kappa as ak, clean_annotations as ca, profile_corpus as pc

LONG = "Ito ay isang dummy na pangungusap para sa testing ng code lamang numero {}"


def rows():
    a1 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]; a2 = [1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
    return [{"post_id": f"d{i}", "text": LONG.format(i), "url": f"https://example.com/p/{i}", "label_source": f"https://example.com/fc/{i}",
             "annotator_1_label": str(x), "annotator_2_label": str(y), "final_label": "", "language_pattern": "taglish"} for i, (x, y) in enumerate(zip(a1, a2))]


class Kappa(unittest.TestCase):
    def test_hand_computed(self):  # po = .8, pe = .5 -> kappa = .6
        k, po, pe = ak.cohen_kappa([1, 1, 1, 1, 0, 0, 0, 0, 1, 0], [1, 1, 1, 0, 0, 0, 0, 1, 1, 0])
        self.assertAlmostEqual(k, 0.6); self.assertAlmostEqual(po, 0.8); self.assertAlmostEqual(pe, 0.5)

    def test_perfect_and_chance(self):
        self.assertEqual(ak.cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1])[0], 1.0)
        self.assertAlmostEqual(ak.cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1])[0], 0.0)

    def test_analyse_skips_missing_and_flags_bad(self):
        r = rows(); r[0]["annotator_2_label"] = ""; r[1]["annotator_1_label"] = "maybe"
        res, dis = ak.analyse(r)
        self.assertEqual(res["rows_both_labelled"], 8); self.assertEqual(res["rows_missing_a_label"], 1); self.assertEqual(len(res["rows_with_invalid_label"]), 1)
        self.assertIn("warning", res)  # < 30 rows


class Clean(unittest.TestCase):
    def test_pipeline(self):
        r = rows()
        r[3]["final_label"] = "0"                       # resolves a disagreement
        r[7]["final_label"] = ""                        # row 7 stays an unresolved disagreement
        r[8]["url"] = ""                                # no URL
        r[9]["label_source"] = "n/a"                    # no fact-check source
        r.append({**r[0], "post_id": "dup", "url": "https://example.com/other"})   # same text as r[0] -> duplicate
        r.append({**r[1], "post_id": "short", "text": "too short", "url": "https://example.com/s"})
        r.append({**r[2], "post_id": "html", "text": "<b>Ito</b> ay &amp; mojibake â€œquoteâ€\u009d https://t.co/x  na dummy na pangungusap lang po ito", "url": "https://example.com/h"})
        keep, rej, rep = ca.clean(r)
        reasons = {x["post_id"]: x["reason"] for x in rej}
        self.assertEqual(reasons["d7"], "unresolved_disagreement"); self.assertEqual(reasons["d8"], "missing_or_bad_post_url")
        self.assertEqual(reasons["d9"], "missing_factcheck_source"); self.assertEqual(reasons["dup"], "duplicate"); self.assertEqual(reasons["short"], "too_short")
        self.assertEqual([k["label"] for k in keep if k["post_id"] == "d3"], [0])
        self.assertNotIn("https", " ".join(k["text"] for k in keep)); self.assertNotIn("<b>", " ".join(k["text"] for k in keep))
        self.assertEqual(rep["rows_in"], 13); self.assertEqual(rep["rows_out"], len(keep)); self.assertEqual([k["id"] for k in keep], list(range(len(keep))))

    def test_ai_final_label_alone_is_rejected(self):
        r = rows()[:1]; r[0].update(annotator_1_label="", annotator_2_label="", final_label="0")
        keep, rej, _ = ca.clean(r)
        self.assertEqual(keep, []); self.assertEqual(rej[0]["reason"], "no_label")

    def test_conflicting_duplicate_dropped_both(self):
        r = rows()[:2]; r[1] = {**r[0], "post_id": "x", "url": "https://example.com/x", "annotator_1_label": "0", "annotator_2_label": "0"}
        keep, rej, rep = ca.clean(r)
        self.assertEqual(keep, []); self.assertEqual(len(rej), 2)


class Profile(unittest.TestCase):
    def test_profile(self):
        r = [{"text": LONG.format(i) + (" corrupt pandaraya" if i % 2 else " the election is open"), "label": str(i % 2)} for i in range(20)]
        p, (top0, top1) = pc.profile(r)
        self.assertEqual(p["n"], 20); self.assertEqual(p["label_counts"], {"0_not_credible": 10, "1_credible": 10})
        self.assertIn("pandaraya", [x["word"] for x in top1][:3] + [x["word"] for x in top0][:3])


class EndToEnd(unittest.TestCase):
    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); src = d / "sheet.csv"
            with open(src, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows()[0])); w.writeheader(); w.writerows(rows())
            py = [sys.executable]
            run = lambda *a: subprocess.run(py + [str(ROOT / "src" / a[0])] + list(a[1:]), check=True, capture_output=True, text=True, cwd=ROOT / "src")
            run("annotation_kappa.py", "--csv", str(src), "--out-dir", str(d / "res"))
            self.assertAlmostEqual(json.loads((d / "res" / "annotation_kappa.json").read_text())["cohen_kappa"], 0.6)
            self.assertTrue((d / "res" / "annotation_disagreements.csv").exists())
            run("clean_annotations.py", "--csv", str(src), "--out", str(d / "t.csv"))
            run("profile_corpus.py", "--csv", str(d / "t.csv"), "--name", "dummy", "--out-dir", str(d / "res"))
            self.assertTrue((d / "res" / "profile_dummy.json").exists())


if __name__ == "__main__":
    unittest.main()
