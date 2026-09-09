"""
Builds the HarmBench near-OOD baseline eval set: the full official
HarmBench release (400 behaviors -- standard + contextual from the
walledai/HarmBench HF mirror, plus the official "copyright" category
that mirror is missing, fetched directly from the official GitHub CSV).
Exact same source/method as entrypoint_ood_pipeline.py's
load_harmbench_with_copyright(), used for the AdvBench<->HarmBench
positive control -- this reuses that same 400-behavior set as an actual
EVAL prompt file (not just a similarity-calibration reference), matching
run_harmful_eval.py's plain-text, one-prompt-per-line input format.

Usage:
    python scripts/build_harmbench_eval_set.py
"""
import csv
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PROMPTS = REPO_ROOT / "data" / "harmbench_eval.csv"
OUT_METADATA = REPO_ROOT / "data" / "harmbench_eval_metadata.json"


def main():
    from datasets import load_dataset

    prompts, cats = [], []
    for cfg in ["standard", "contextual"]:
        ds = load_dataset("walledai/HarmBench", cfg, split="train")
        prompts.extend(ds["prompt"])
        cats.extend([f"HarmBench::{cfg}::{c}" for c in ds["category"]])
    print(f"standard+contextual: {len(prompts)} prompts")

    url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    local_csv = "/tmp/harmbench_official.csv"
    urllib.request.urlretrieve(url, local_csv)
    with open(local_csv) as f:
        rows = list(csv.DictReader(f))
    copyright_behaviors = [r["Behavior"].strip() for r in rows if r["FunctionalCategory"] == "copyright"]
    prompts.extend(copyright_behaviors)
    cats.extend(["HarmBench::copyright::copyright"] * len(copyright_behaviors))
    print(f"+ copyright: {len(copyright_behaviors)} -> {len(prompts)} total")

    with open(OUT_PROMPTS, "w") as f:
        for p in prompts:
            f.write(p.replace("\n", " ").strip() + "\n")

    import json
    json.dump({"prompts": prompts, "categories": cats}, open(OUT_METADATA, "w"), indent=2)
    print(f"Wrote {len(prompts)} prompts -> {OUT_PROMPTS}")
    print(f"Wrote metadata -> {OUT_METADATA}")


if __name__ == "__main__":
    main()
