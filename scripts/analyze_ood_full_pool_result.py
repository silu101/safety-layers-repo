"""
Per-category and per-similarity-bin breakdown of an ood_full_pool.csv ASR
result (run_asr.py's output on the full unfiltered 25,847-prompt pool).

ood_full_pool.csv is a SHUFFLED flatten of all 14 categories' pools (see
build_ood_full_pool.py), and 7,189 of its 15,561 unique prompt texts
belong to MORE than one category (a source-dataset entry can plausibly
match more than one topic) -- so a prompt CAN'T be joined back to its
category by text lookup alone; the same string maps to multiple entries.

This instead replays the exact Fisher-Yates permutation build_ood_full_
pool.py applied (random.shuffle's permutation depends only on sequence
length + seed, not on the list's contents, so re-seeding with the same
seed and shuffling a fresh range(n) reproduces the identical order),
recovering an exact, unambiguous per-row category/similarity for every
one of the 25,847 responses. Verified against a real result: 25847/25847
positional match before trusting this for any real analysis.

Usage:
    python scripts/analyze_ood_full_pool_result.py path/to/asr_full_pool.json
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "data" / "ood_full_pool_metadata.json"

CONFIRMED_OOD = {"hate_discrimination", "harassment", "sexual_content", "privacy", "political_misinformation"}

SIMILARITY_BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.28), (0.28, 0.3), (0.3, 0.4),
                    (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.01)]


def recover_per_row_records(meta: dict) -> list[dict]:
    pre_lines, pre_records = [], []
    for cat, entries in meta["categories"].items():
        for e in entries:
            line = e["prompt"].replace("\n", " ").strip()
            if not line:
                continue
            pre_lines.append(line)
            pre_records.append({"category": cat, "similarity": e["similarity_to_advbench"],
                                 "dataset": e["dataset"], "ambiguous": e.get("ambiguous_label", False)})

    rng = random.Random(meta["seed"])
    indices = list(range(len(pre_lines)))
    rng.shuffle(indices)
    return [pre_records[i] for i in indices], [pre_lines[i] for i in indices]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("result_path")
    ap.add_argument("--metadata-path", default=str(METADATA_PATH))
    args = ap.parse_args()

    result = json.load(open(args.result_path))
    meta = json.load(open(args.metadata_path))
    records, shuffled_lines = recover_per_row_records(meta)

    if len(records) != result["n_prompts"]:
        raise SystemExit(f"Length mismatch: metadata reconstructs {len(records)} rows, "
                          f"result has {result['n_prompts']}. Wrong metadata file for this result?")
    mismatches = sum(1 for a, b in zip(shuffled_lines, result["prompts"]) if a != b)
    if mismatches:
        raise SystemExit(f"{mismatches}/{len(records)} rows don't match by position -- "
                          f"the seeded-shuffle replay assumption doesn't hold for this result "
                          f"(wrong metadata file, or ood_full_pool.csv was regenerated since). "
                          f"Refusing to report a category breakdown that might be silently wrong.")
    print(f"Verified: {len(records)}/{len(records)} rows match by position. Proceeding.\n")

    compliant = result["compliant"]

    cat_stats = defaultdict(lambda: [0, 0])
    for r, c in zip(records, compliant):
        cat_stats[r["category"]][0] += c
        cat_stats[r["category"]][1] += 1

    print("=== Per-category ASR ===")
    print(f'{"category":<28}{"n":>7}{"ASR":>8}   confirmed-OOD?')
    for cat, (comp, n) in sorted(cat_stats.items(), key=lambda x: -x[1][1]):
        tag = "yes" if cat in CONFIRMED_OOD else "no"
        print(f"{cat:<28}{n:>7}{comp/n:>8.4f}   {tag}")

    conf_comp = sum(cat_stats[c][0] for c in CONFIRMED_OOD)
    conf_n = sum(cat_stats[c][1] for c in CONFIRMED_OOD)
    other_comp = sum(v[0] for k, v in cat_stats.items() if k not in CONFIRMED_OOD)
    other_n = sum(v[1] for k, v in cat_stats.items() if k not in CONFIRMED_OOD)
    print(f"\n5 confirmed-OOD categories (unfiltered): n={conf_n}, ASR={conf_comp/conf_n:.4f}")
    print(f"10 borderline/covered categories:        n={other_n}, ASR={other_comp/other_n:.4f}")

    print("\n=== ASR by similarity-to-AdvBench bin ===")
    print(f'{"bin":<16}{"n":>7}{"ASR":>8}')
    for lo, hi in SIMILARITY_BINS:
        sub = [c for r, c in zip(records, compliant) if lo <= r["similarity"] < hi]
        if sub:
            print(f"[{lo:.2f},{hi:.2f}){'':<3}{len(sub):>7}{sum(sub)/len(sub):>8.4f}")

    sims = [r["similarity"] for r in records]
    mean_sim_compliant = statistics.mean(s for s, c in zip(sims, compliant) if c)
    mean_sim_refused = statistics.mean(s for s, c in zip(sims, compliant) if not c)
    n, mean_s, mean_c = len(sims), statistics.mean(sims), statistics.mean(compliant)
    cov = sum((s - mean_s) * (c - mean_c) for s, c in zip(sims, compliant)) / n
    var_s = statistics.pvariance(sims)
    var_c = statistics.pvariance(compliant)
    corr = cov / ((var_s * var_c) ** 0.5)
    print(f"\nmean similarity | compliant: {mean_sim_compliant:.4f}")
    print(f"mean similarity | refused:   {mean_sim_refused:.4f}")
    print(f"point-biserial correlation (similarity vs. compliant): r={corr:.4f}")


if __name__ == "__main__":
    main()
