"""Smoke tests for src/interpret.py on DUMMY text and TINY RANDOM models built offline (no downloads, no real data).

    python -m unittest tests.test_interpret -v        (run from tsektxt-model-training; needs torch, transformers, captum)
"""
import importlib.util, json, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAVE = all(importlib.util.find_spec(m) for m in ("torch", "transformers", "captum"))

WORDS = "ito ay isang dummy na pangungusap para sa testing ng code lamang fake news totoo boto halalan".split()
DUMMY = [f"Ito ay isang dummy na pangungusap para sa testing ng code lamang {w} {WORDS[i % len(WORDS)]}!" for i, w in enumerate(WORDS)]


def tiny(kind, d):
    """Save a 1-layer random classifier (bert or roberta architecture) + word-level tokenizer to d."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast, RobertaConfig, RobertaForSequenceClassification
    torch.manual_seed(0)
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "!"] + WORDS + ["##s"]
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(d / "vocab.txt"), do_lower_case=True)
    kw = dict(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
              num_labels=2, id2label={0: "not_credible", 1: "credible"}, label2id={"not_credible": 0, "credible": 1}, pad_token_id=0)
    if kind == "bert":
        model = BertForSequenceClassification(BertConfig(max_position_embeddings=160, **kw))
    else:
        model = RobertaForSequenceClassification(RobertaConfig(max_position_embeddings=200, **kw))
    model.save_pretrained(d); tok.save_pretrained(d)
    return str(d)


@unittest.skipUnless(HAVE, "needs torch, transformers and captum")
class IG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import interpret
        cls.ig = interpret
        cls.tmp = tempfile.TemporaryDirectory(); t = Path(cls.tmp.name)
        cls.dirs = {"roberta-tagalog": tiny("roberta", t / "rt"), "mbert": tiny("bert", t / "mb"), "xlmr": tiny("roberta", t / "xr")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_word_scores(self):
        text = "Fake  news, totoo!"
        offs = [(0, 0), (0, 4), (6, 10), (10, 11), (12, 17), (17, 18), (0, 0)]
        w = self.ig.word_scores(text, offs, [9, 1, 2, 3, 4, 5, 9])
        self.assertEqual([x["norm"] for x in w], ["fake", "news", "totoo"])
        self.assertEqual([x["attr"] for x in w], [1, 5, 9])  # specials dropped, punctuation joins its word

    def test_explain_and_completeness(self):
        for name, d in self.dirs.items():
            tok, model, _ = self.ig.load(name, {name: d})
            r = self.ig.explain(tok, model, DUMMY[0], n_steps=64)
            self.assertEqual(len(r["tokens"]), len(tok(DUMMY[0])["input_ids"]))
            self.assertEqual(r["tokens"][0]["attr"], 0.0)  # CLS kept in the baseline -> zero attribution
            c = r["completeness"]
            self.assertAlmostEqual(c["sum_attributions"], sum(x["attr"] for x in r["words"]), places=5)
            self.assertLess(c["abs_error"], 1e-2, name)
            self.assertEqual(r["pred_label"], 0 if c["f_input"] > 0 else 1)
            self.assertAlmostEqual(r["prob_credible"] + r["prob_not_credible"], 1.0, places=5)

    def test_baseline_uses_real_positions_roberta(self):
        """Regression: RoBERTa-type models take position ids from input_ids (pad -> padding_idx), so
        model(input_ids=base) is NOT the IG endpoint. f_baseline must equal the embedding-hook forward."""
        import torch
        tok, model, _ = self.ig.load("xlmr", self.dirs)
        with torch.no_grad():  # make positions matter (random init makes them near-irrelevant)
            torch.manual_seed(1)
            model.roberta.embeddings.position_embeddings.weight.normal_(std=2.0)
            model.classifier.out_proj.weight.normal_(std=1.0)
        text = DUMMY[0]
        enc = tok(text, return_tensors="pt")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        special = torch.tensor(tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True)).bool()
        base = torch.where(special, ids, torch.full_like(ids, tok.pad_token_id))
        nc, cr = self.ig.label_index(model)
        emb = model.get_input_embeddings()
        with torch.no_grad():
            naive = model(input_ids=base, attention_mask=mask).logits[0]
            naive = (naive[nc] - naive[cr]).item()
            be = emb(base); h = emb.register_forward_hook(lambda m, i, o: be)
            expected = model(input_ids=ids, attention_mask=mask).logits[0]; h.remove()
            expected = (expected[nc] - expected[cr]).item()
        self.assertGreater(abs(naive - expected), 1e-4)  # the bug would show up here
        r = self.ig.explain(tok, model, text, n_steps=200, max_steps=200)
        self.assertAlmostEqual(r["completeness"]["f_baseline"], expected, places=5)
        self.assertLess(r["completeness"]["rel_error"], 0.05)

    def test_adaptive_steps(self):
        tok, model, _ = self.ig.load("xlmr", self.dirs)
        r = self.ig.explain(tok, model, DUMMY[0], n_steps=2, tol=1e-9, max_steps=8)
        self.assertEqual(r["completeness"]["n_steps"], 8)  # doubled 2 -> 4 -> 8, then capped
        self.assertEqual(r["completeness"]["n_steps_start"], 2)

    def test_truncation_128(self):
        tok, model, _ = self.ig.load("mbert", self.dirs)
        r = self.ig.explain(tok, model, " ".join(["boto"] * 300), n_steps=8)
        self.assertEqual(len(r["tokens"]), self.ig.MAXLEN); self.assertTrue(r["truncated"])

    def test_top_tokens_and_overlap(self):
        mk = lambda ws: {"words": [{"word": w, "norm": w, "attr": a} for w, a in ws]}
        recs = [mk([("fake", 2.0), ("boto", -1.0), ("na", 1.0)]), mk([("fake", 1.0), ("na", -1.0)])]
        t = self.ig.top_tokens(recs, top_k=5, min_df=2)
        self.assertEqual(t["toward_not_credible"][0]["word"], "fake"); self.assertEqual(t["toward_not_credible"][0]["doc_freq"], 2)
        self.assertNotIn("boto", [x["word"] for x in t["toward_credible"]])  # df 1 < min_df
        o = self.ig.overlap({"a": t, "b": t, "c": self.ig.top_tokens([mk([("x", 1.0)])], min_df=1)})
        self.assertEqual(o["pairwise"]["a|b"]["jaccard"], 1.0); self.assertEqual(o["pairwise"]["a|c"]["jaccard"], 0.0)
        self.assertEqual(o["shared_by_all"], [])

    def test_cli_corpus_and_post(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); csv = d / "dummy.csv"
            pd.DataFrame({"id": [f"d{i}" for i in range(len(DUMMY))], "text": DUMMY, "label": [i % 2 for i in range(len(DUMMY))]}).to_csv(csv, index=False)
            mdir = [f"{k}={v}" for k, v in self.dirs.items()]
            self.ig.main(["corpus", "--csv", str(csv), "--n-steps", "16", "--min-df", "1", "--all-posts", "--out-dir", str(d), "--model-dir", *mdir])
            top = json.loads((d / "ig_top_tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(set(top["top_tokens"]), set(self.dirs)); self.assertEqual(top["meta"]["max_length"], 128)
            ov = json.loads((d / "ig_overlap.json").read_text(encoding="utf-8"))
            self.assertEqual(len(ov["pairwise"]), 3)
            att = json.loads((d / "ig_attributions_xlmr.json").read_text(encoding="utf-8"))
            self.assertEqual(len(att["posts"]), len(DUMMY)); self.assertEqual(att["completeness"]["n"], len(DUMMY))
            self.ig.main(["post", "--text", DUMMY[1], "--id", "dummy 1", "--n-steps", "16", "--out-dir", str(d), "--model-dir", *mdir])
            self.assertTrue((d / "ig_post_dummy_1.json").exists())


if __name__ == "__main__":
    unittest.main()
