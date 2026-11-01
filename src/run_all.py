"""Train all three models one after another (works on Windows, Mac, Linux, Colab).

    python src/run_all.py            # train + evaluate internally, save locally
    python src/run_all.py --push     # also upload each model + model card to Hugging Face
    python src/run_all.py --only xlmr mbert
"""
import argparse, subprocess, sys
from pathlib import Path

ALL = ["roberta-tagalog", "mbert", "xlmr"]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--push", action="store_true"); ap.add_argument("--only", nargs="+", choices=ALL)
    a = ap.parse_args(); here = Path(__file__).parent
    for m in (a.only or ALL):
        print(f"\n===== Training {m} =====", flush=True)
        cmd = [sys.executable, str(here / "train.py"), "--model", m] + (["--push"] if a.push else [])
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"{m} failed - fix the error above, then re-run: python src/run_all.py --only {m}")
    subprocess.run([sys.executable, str(here / "compare.py")])
