"""
Builds the small-scale semantic-content OOD test set for Safety Layers'
Section 4 harmful-rate evaluation: a fixed n prompts sampled per confirmed-
OOD category, from the curated pool exported by the OOD Pool Inspector
tool (data/ood_curated/*.json).

Output format matches what run_harmful_eval.py expects for advbench_path:
plain text, one prompt per line, no header -- see data/README.md and
run_harmful_eval.py's load (`with open(cfg.advbench_path) as f: prompts =
[line.rstrip("\n") for line in f if line.strip()]`).

A companion metadata JSON records which category/source/similarity each
line came from, for traceability -- the eval script itself never reads it.

Usage:
    python scripts/build_ood_semantic_test.py [--n-per-category 20] [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = REPO_ROOT / "data" / "ood_curated"
OUT_PROMPTS = REPO_ROOT / "data" / "ood_semantic_test.csv"
OUT_METADATA = REPO_ROOT / "data" / "ood_semantic_test_metadata.json"

# The 5 categories confirmed OOD relative to AdvBench (Step 5's validity
# criteria) -- NOT the 10 borderline/covered ones. See docs/REPLICATION_LOG.md
# and the OOD Pool Inspector's Overview & Methodology tab for how this was
# established.
CONFIRMED_OOD_CATEGORIES = [
    "hate_discrimination", "harassment", "sexual_content", "privacy", "political_misinformation",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_lines = []
    metadata = {"n_per_category": args.n_per_category, "seed": args.seed, "categories": {}}

    for cat in CONFIRMED_OOD_CATEGORIES:
        path = CURATED_DIR / f"ood_{cat}_curated.json"
        payload = json.load(open(path))
        records = payload["records"]
        if len(records) < args.n_per_category:
            raise SystemExit(f"{cat}: only {len(records)} curated records available, "
                              f"need {args.n_per_category}. Lower --n-per-category or curate more.")
        sample = rng.sample(records, args.n_per_category)

        cat_lines = []
        for r in sample:
            # Prompts must be single-line for the plain-text format --
            # collapse any embedded newlines rather than silently truncating.
            line = r["candidate"].replace("\n", " ").strip()
            cat_lines.append(line)
        all_lines.extend(cat_lines)

        metadata["categories"][cat] = [
            {
                "prompt": r["candidate"], "dataset": r["dataset"], "source_label": r["source_label"],
                "similarity_to_advbench": r["similarity_to_advbench"],
                "matched_advbench_prompt": r["matched_advbench_prompt"],
            }
            for r in sample
        ]
        print(f"{cat}: sampled {len(sample)} / {len(records)} curated prompts")

    rng.shuffle(all_lines)  # don't leave prompts grouped by category in the flat file
    with open(OUT_PROMPTS, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    with open(OUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {len(all_lines)} prompts -> {OUT_PROMPTS}")
    print(f"Wrote metadata -> {OUT_METADATA}")


if __name__ == "__main__":
    main()
