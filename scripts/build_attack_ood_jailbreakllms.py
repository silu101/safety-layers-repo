"""
Builds the first Attack-OOD test set: JailbreakLLMs' 1,405 real-world
jailbreak templates (Shen et al. 2024, "Do Anything Now" -- in-the-wild
DAN/persona/roleplay prompts scraped from Reddit/Discord/jailbreak sites,
Dec 2022-Dec 2023), each paired with an AdvBench goal -- the goal and its
wrapper are stored separately per-record (template text, matched AdvBench
goal, combined prompt) so this can be re-analyzed either way later.

Design choice: sample n templates WITHOUT replacement from the 1,405
(default 520, matching the scale of every other eval run in this
project), each paired with ONE AdvBench goal sampled WITH replacement
from the 520 (since there are more templates than goals in any n>520
sample, exact reuse is unavoidable and not a problem -- the goal is held
constant as the ID anchor across all wrapper types, reuse across records
is expected, not contamination).

Combination method: only 218/1405 templates contain an explicit insertion
placeholder ("[INSERT PROMPT HERE]" etc.) -- for those, the goal is
substituted directly into the placeholder (most faithful to the
template's own intended usage). For the remaining majority with no
placeholder, the goal is appended as a new paragraph at the end of the
template text, mirroring how these prompts are actually used in practice
(jailbreak setup first, then the real ask) since run_harmful_eval.py's
pipeline is single-turn only.

Source: https://github.com/verazuo/jailbreak_llms (fetched directly via
raw.githubusercontent.com rather than a full git clone -- a plain `git
clone` of this repo hung indefinitely twice in this environment for
reasons unrelated to repo size; fetching the specific CSV directly side-
steps that).

Usage:
    python scripts/build_attack_ood_jailbreakllms.py [--n 520] [--seed 0]
"""
import argparse
import csv
import json
import random
import re
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_URL = "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv"
ADVBENCH_PATH = REPO_ROOT / "data" / "advbench_malicious.csv"
OUT_PROMPTS = REPO_ROOT / "data" / "attack_ood_jailbreakllms.csv"
OUT_METADATA = REPO_ROOT / "data" / "attack_ood_jailbreakllms_metadata.json"

PLACEHOLDER_RE = re.compile(r"\[INSERT[^\]]*\]|\{PROMPT\}|\[PROMPT\]|\[YOUR (QUESTION|PROMPT)[^\]]*\]|\[QUESTION\]", re.I)


def combine(template: str, goal: str) -> tuple[str, bool]:
    """Returns (combined_prompt, used_placeholder)."""
    if PLACEHOLDER_RE.search(template):
        return PLACEHOLDER_RE.sub(goal, template, count=1), True
    return template.rstrip() + "\n\n" + goal, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=520)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    local_csv = "/tmp/jailbreak_prompts_2023_12_25.csv"
    urllib.request.urlretrieve(TEMPLATES_URL, local_csv)
    with open(local_csv) as f:
        templates = [row["prompt"] for row in csv.DictReader(f)]
    print(f"Loaded {len(templates)} JailbreakLLMs templates")

    with open(ADVBENCH_PATH) as f:
        goals = [line.rstrip("\n") for line in f if line.strip()]
    print(f"Loaded {len(goals)} AdvBench goals")

    if args.n > len(templates):
        raise SystemExit(f"--n={args.n} exceeds available templates ({len(templates)})")
    sample_templates = rng.sample(templates, args.n)

    records = []
    for template in sample_templates:
        goal = rng.choice(goals)
        combined, used_placeholder = combine(template, goal)
        combined = combined.replace("\n", " ").strip()  # single-line, matches other eval prompt files
        records.append({
            "template": template, "goal": goal,
            "combined_prompt": combined, "used_placeholder": used_placeholder,
        })

    with open(OUT_PROMPTS, "w") as f:
        f.write("\n".join(r["combined_prompt"] for r in records) + "\n")
    json.dump({"records": records}, open(OUT_METADATA, "w"), indent=2)

    n_placeholder = sum(1 for r in records if r["used_placeholder"])
    print(f"Wrote {len(records)} prompts -> {OUT_PROMPTS} "
          f"({n_placeholder} used an explicit placeholder, {len(records)-n_placeholder} appended)")
    print(f"Wrote metadata -> {OUT_METADATA}")


if __name__ == "__main__":
    main()
