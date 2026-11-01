"""Stratified 70/15/15 split (seed 42). Shared by all three models -> identical data for a fair comparison."""
import argparse, json
import pandas as pd, yaml
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml"); a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config)); r = cfg["data"]["split_ratio"]; seed = cfg["seed"]
    df = pd.read_csv(cfg["data"]["processed_csv"])
    train, rest = train_test_split(df, test_size=r["val"] + r["test"], stratify=df.label, random_state=seed)
    val, test = train_test_split(rest, test_size=r["test"] / (r["val"] + r["test"]), stratify=rest.label, random_state=seed)
    d = cfg["data"]["splits_dir"]
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"{d}/{name}.csv", index=False)
    ids = [set(p.id) for p in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "split leakage"
    print(json.dumps({n: {"n": len(p), "label_counts": {int(k): int(v) for k, v in p.label.value_counts().items()}}
                      for n, p in [("train", train), ("val", val), ("test", test)]}, indent=2))


if __name__ == "__main__":
    main()
