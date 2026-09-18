"""
Regenerates the official refusal_direction repo's harmful-side identification
splits to be AdvBench-ONLY, replacing their default mix (AdvBench + TDC2023 +
MaliciousInstruct -- verified directly: their shipped harmful_train.json is
260 examples, 117 of which are AdvBench and the rest come from those other
two sources). The harmless side (Alpaca) is left untouched, per project
decision -- only the harmful/identification anchor is fixed to AdvBench,
matching the same design used for Safety Layers.

Pulls AdvBench from the official repo's own dataset/processed/advbench.json
(520 entries, verified to match Zou et al.'s canonical AdvBench exactly) --
not our own advbench_malicious.csv copy -- so the format matches exactly
what their pipeline already expects, no conversion risk.

Writes 128 train / 32 val (their paper's own n_train/n_val, matching
Config's defaults) as a disjoint split from the same 520-prompt pool.
Overlap between this identification subset and the full 520-prompt set
used later for baseline evaluation is fine and expected -- same
relationship malicious.csv has to AdvBench in Safety Layers' own design,
already established as not a methodological problem in this project.

Usage (run from inside a cloned copy of the refusal_direction repo, or
point --repo_dir at wherever you cloned it):
    git clone https://github.com/andyrdt/refusal_direction
    python build_advbench_splits.py --repo_dir refusal_direction
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

ADVBENCH_URL = "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/processed/advbench.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo_dir", required=True, help="Path to the cloned refusal_direction repo")
    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--n_val", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    local_path = "/tmp/rd_advbench.json"
    urllib.request.urlretrieve(ADVBENCH_URL, local_path)
    advbench = json.load(open(local_path))
    print(f"Loaded {len(advbench)} AdvBench entries (expect 520)")

    if args.n_train + args.n_val > len(advbench):
        raise SystemExit(f"n_train+n_val ({args.n_train + args.n_val}) exceeds available AdvBench entries ({len(advbench)})")

    rng = random.Random(args.seed)
    shuffled = advbench.copy()
    rng.shuffle(shuffled)
    train_split = shuffled[: args.n_train]
    val_split = shuffled[args.n_train : args.n_train + args.n_val]

    splits_dir = Path(args.repo_dir) / "dataset" / "splits"
    if not splits_dir.exists():
        raise SystemExit(f"{splits_dir} doesn't exist -- is --repo_dir pointing at a real clone of refusal_direction?")

    for name, split in [("harmful_train.json", train_split), ("harmful_val.json", val_split)]:
        out_path = splits_dir / name
        # Back up the original mixed-dataset split before overwriting, so
        # it's recoverable rather than silently lost.
        backup_path = splits_dir / f"{name}.orig_mixed_dataset.bak"
        if out_path.exists() and not backup_path.exists():
            out_path.rename(backup_path)
            print(f"Backed up original {name} -> {backup_path.name}")
        json.dump(split, open(out_path, "w"), indent=2)
        print(f"Wrote {len(split)} AdvBench-only entries -> {out_path}")

    print("\nharmless_train.json / harmless_val.json left untouched (Alpaca, as decided).")
    print("Now run the official pipeline unmodified:")
    print(f"  cd {args.repo_dir} && python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct")


if __name__ == "__main__":
    main()
