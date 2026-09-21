"""
Builds the FULL unfiltered OOD prompt set: every candidate in the OOD Pool
Inspector's pool, all 14 categories (the 5 confirmed-OOD categories plus
the 10 borderline/covered ones), with NO similarity-to-AdvBench threshold
applied -- unlike build_ood_semantic_test.py, which samples a fixed n per
confirmed-OOD category from the curated (thresholded) snapshot.

This is the "eval on the whole 25,847-prompt pool" run: it answers whether
similarity-based curation was doing real work, by comparing ASR on this
unfiltered pool against ASR on the curated ood_semantic_test.csv set.

IMPORTANT -- same live-source caveat as build_ood_semantic_test.py: this
reads data/ood_full_pool_raw_snapshot.json, a SNAPSHOT of the OOD Pool
Inspector's data-blob (https://claude.ai/code/artifact/1c25b40d-0691-4307-
b469-14d05188504b), taken 2026-09-21. It does not reflect any pool changes
made in the live tool after that date. Regenerate the snapshot first if
the live tool has moved on (Artifact action=read on the tool's URL, parse
the <script id="data-blob"> JSON, dump {"pools": ...} to that path).

NOT applied here: the live tool's small number of manual per-record
content-quality exclusions (4 total, across privacy/sexual_content as of
2026-09-21 -- flagged for reasons unrelated to AdvBench similarity, e.g.
mislabeled or non-harmful text). Those exclusions aren't recoverable from
this raw snapshot (only a count is stored in the curated *_curated.json
files, not which record ids). At 4/25,847 (~0.015%) this is noise relative
to what this run is actually testing (threshold vs. no threshold), so it's
left as a known, documented gap rather than blocking the run on it.

Output format matches build_ood_semantic_test.py: plain text prompts file
(one per line, run_asr.py's expected format) + companion metadata JSON.

Usage:
    python scripts/build_ood_full_pool.py [--seed 0]
"""
import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_SNAPSHOT = REPO_ROOT / "data" / "ood_full_pool_raw_snapshot.json"
OUT_PROMPTS = REPO_ROOT / "data" / "ood_full_pool.csv"
OUT_METADATA = REPO_ROOT / "data" / "ood_full_pool_metadata.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    snapshot = json.load(open(RAW_SNAPSHOT))
    pools = snapshot["pools"]

    all_lines = []
    metadata = {"seed": args.seed, "source_snapshot": str(RAW_SNAPSHOT.relative_to(REPO_ROOT)), "categories": {}}

    for cat, records in pools.items():
        cat_entries = []
        for r in records:
            line = r["p"].replace("\n", " ").strip()
            if not line:
                continue
            all_lines.append(line)
            cat_entries.append({
                "prompt": r["p"], "dataset": r["ds"], "source_label": r["lb"],
                "similarity_to_advbench": r["s"], "matched_advbench_prompt": r["m"],
                "ambiguous_label": r.get("amb", False),
            })
        metadata["categories"][cat] = cat_entries
        print(f"{cat}: {len(cat_entries)} prompts (unfiltered)")

    rng.shuffle(all_lines)  # don't leave prompts grouped by category in the flat file
    with open(OUT_PROMPTS, "w") as f:
        f.write("\n".join(all_lines) + "\n")
    with open(OUT_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {len(all_lines)} prompts -> {OUT_PROMPTS}")
    print(f"Wrote metadata -> {OUT_METADATA}")


if __name__ == "__main__":
    main()
