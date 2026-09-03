"""
SageMaker entrypoint for the OOD semantic-similarity pipeline (steps 6+7
of the OOD dataset construction process):

  1. Refreshed positive control: AdvBench vs HarmBench, now including the
     official "copyright" category (100 behaviors) that the walledai/HarmBench
     HF mirror is missing -- fetched directly from the official HarmBench
     GitHub CSV to fill the gap found during source-of-truth verification.
  2. Full candidate sweep: AdvBench vs all 6 of the ICLR paper's Table 2
     candidate datasets -- BeaverTails, Aegis 2.0, OpenAI Moderation,
     MaliciousInstruct, Anthropic Red Team, and SimpleSafetyTests.
     MaliciousInstruct's official mirror has no category labels, so labels
     are recovered from the official paper repo's positional block
     structure (10 categories x 10 prompts, in Table 2's listed order).
     Anthropic Red Team's full release is 38,961 transcripts but only 742
     carry a human topic tag -- only the tagged subset is used.
  3. Steps 6+7 unified: build_ood_pool() for ALL 15 target categories (the
     5 confirmed-OOD ones and the 10 borderline/covered ones) through the
     same code path -- merged, deduplicated, provenance-tagged, similarity
     + matched-AdvBench-prompt attached to every record, but with NO
     similarity threshold applied server-side. The inspector website
     applies an adjustable threshold and lets a human include/exclude
     individual prompts, rather than the pipeline pre-deciding a fixed
     cutoff -- a category can average well below any given baseline while
     still containing individual prompts at or above it, so the per-prompt
     decision belongs with whoever is building the final dataset, not
     baked into this run. sexual_harassment has no clean source label in
     any of the 6 candidate datasets and is recorded with an explicit note.

All datasets loaded via vectorized pandas ops (NOT per-row Python loops --
that caused a multi-hour stall when this was first attempted locally).
Runs on GPU (sentence-transformers auto-detects CUDA) so this should take
minutes, not hours, unlike the local CPU environment this was migrated
from after diagnosing local CPU contention issues.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_BRANCH = "ood-similarity-pipeline"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

CATEGORY_MAP = {
    "hate_discrimination": [
        ("Aegis", "Hate/Identity Hate"),
        ("BeaverTails", "hate_speech,offensive_language"),
        ("BeaverTails", "discrimination,stereotype,injustice"),
        ("OpenAIModeration", "hate"),
        ("OpenAIModeration", "hate-threatening"),
        ("AnthropicRedTeam", "Discrimination & injustice"),
        ("AnthropicRedTeam", "Hate speech & offensive language"),
    ],
    "harassment": [
        ("Aegis", "Harassment"),
        ("OpenAIModeration", "harassment"),
        ("AnthropicRedTeam", "Bullying & harassment"),
        ("MaliciousInstruct", "cyberbullying"),
        ("MaliciousInstruct", "false_accusation"),
    ],
    "sexual_content": [
        ("Aegis", "Sexual"),
        ("BeaverTails", "sexually_explicit,adult_content"),
        ("OpenAIModeration", "sexual"),
        ("AnthropicRedTeam", "Adult content"),
    ],
    "privacy": [
        ("Aegis", "PII/Privacy"),
        ("BeaverTails", "privacy_violation"),
        ("AnthropicRedTeam", "Soliciting personally identifiable information"),
    ],
    "political_misinformation": [
        ("BeaverTails", "controversial_topics,politics"),
        ("Aegis", "Political/Misinformation/Conspiracy"),
        ("AnthropicRedTeam", "Conspiracy theories & misinformation"),
        ("MaliciousInstruct", "defamation"),  # per HarmBench's own taxonomy, defamation sits
                                                # under misinformation/disinformation, not harassment
    ],
}

# Step 7: the 10 categories confirmed borderline/covered against AdvBench
# (mean similarity at or above the 0.551 known-overlap baseline in at
# least one source), mapped to their best-matching source::label(s).
# sexual_harassment has no clean single-label source in any of our 4
# datasets and is intentionally skipped with a note.
BORDERLINE_MAP = {
    "physical_violence": [
        ("BeaverTails", "violence,aiding_and_abetting,incitement"),
        ("Aegis", "Violence"),
        ("OpenAIModeration", "violence"),
        ("AnthropicRedTeam", "Violence & incitement"),
    ],
    "weapons_explosives": [
        ("Aegis", "Guns and Illegal Weapons"),
        ("BeaverTails", "drug_abuse,weapons,banned_substance"),
        ("AnthropicRedTeam", "Weapons"),
    ],
    "self_harm": [
        ("BeaverTails", "self_harm"),
        ("Aegis", "Suicide and Self Harm"),
        ("OpenAIModeration", "self-harm"),
        ("AnthropicRedTeam", "Self-harm"),
    ],
    "cybercrime": [
        ("Aegis", "Malware"),
        ("Aegis", "Criminal Planning/Confessions"),
        ("MaliciousInstruct", "hacking"),
    ],
    "fraud_phishing_cheating": [
        ("BeaverTails", "financial_crime,property_crime,theft"),
        ("Aegis", "Fraud/Deception"),
        ("AnthropicRedTeam", "Theft"),
        ("AnthropicRedTeam", "Fraud & deception"),
        ("MaliciousInstruct", "theft"),
        ("MaliciousInstruct", "tax_fraud"),
        ("MaliciousInstruct", "fraud"),
    ],
    "other_illegal_activity": [
        ("Aegis", "Illegal Activity"),
        ("BeaverTails", "non_violent_unethical_behavior"),
        ("AnthropicRedTeam", "Non-violent unethical behavior (e.g. lying, cheating, etc.)"),
        ("AnthropicRedTeam", "Property crime & vandalism"),
        ("AnthropicRedTeam", "Sexual exploitation & human trafficking"),  # per HarmBench's taxonomy,
                                    # grouped under "Illegal Activities" rather than sexual_content --
                                    # kept out of the actively-used OOD pool deliberately (see note below).
        ("MaliciousInstruct", "sabotage"),
    ],
    "drugs": [
        ("Aegis", "Controlled/Regulated Substances"),
        ("AnthropicRedTeam", "Substance abuse & banned substances"),
        ("MaliciousInstruct", "illegal_drug_use"),
    ],
    "terrorism_organized_crime": [
        ("BeaverTails", "terrorism,organized_crime"),
        ("AnthropicRedTeam", "Terrorism & organized crime"),
    ],
    "manipulation": [
        ("Aegis", "Manipulation"),  # n=7, small -- now supplemented by MaliciousInstruct's dedicated block
        ("MaliciousInstruct", "manipulation"),
    ],
    # "sexual_harassment": still no clean single source label anywhere -- skipped.
    # "Animal abuse" and "Child abuse" tags exist in Anthropic Red Team (n=28, n=19) but have
    # no mapping in the ICLR paper's 15-category list -- left out of both maps, visible only in
    # the raw full-sweep table, not classified as OOD or covered.
}


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[entrypoint] No HF_TOKEN in environment -- skipping HF login.")
        return
    from huggingface_hub import login
    login(token=token)
    print("[entrypoint] Hugging Face login OK.")


def load_advbench(repo_dir):
    path = repo_dir / "data" / "advbench_malicious.csv"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_harmbench_with_copyright():
    """walledai/HarmBench mirror only exposes standard+contextual (300);
    the official release also has a 'copyright' category (100 behaviors)
    that mirror is missing -- fetch it directly from the official GitHub
    CSV to fill the gap found during source-of-truth verification."""
    import urllib.request
    from datasets import load_dataset

    prompts, cats = [], []
    for cfg in ["standard", "contextual"]:
        ds = load_dataset("walledai/HarmBench", cfg, split="train")
        prompts.extend(ds["prompt"])
        cats.extend([f"HarmBench::{cfg}::{c}" for c in ds["category"]])

    url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    local_csv = "/tmp/harmbench_official.csv"
    urllib.request.urlretrieve(url, local_csv)
    with open(local_csv) as f:
        rows = list(csv.DictReader(f))
    copyright_behaviors = [r["Behavior"].strip() for r in rows if r["FunctionalCategory"] == "copyright"]
    prompts.extend(copyright_behaviors)
    cats.extend(["HarmBench::copyright::copyright"] * len(copyright_behaviors))
    print(f"[entrypoint] HarmBench: {len(prompts)} total prompts "
          f"(300 from walledai mirror + {len(copyright_behaviors)} copyright from official GitHub CSV)", flush=True)
    return prompts, cats


def load_beavertails():
    """Returns (grouped, ambiguous_prompts). ambiguous_prompts flags any
    prompt text that appears more than once in the raw dataset with a
    DIFFERENT set of category flags across its occurrences -- BeaverTails
    pairs one prompt with multiple independently-sampled responses, each
    independently rated, so the same prompt text can carry different (or
    no) category flags depending purely on which response it's paired
    with in that row. This is surfaced as a badge in the curation tool,
    not used to filter -- the signal is too noisy to trust blindly (some
    of it is genuine prompt/response ambiguity, some is just rater
    inconsistency on near-identical harmful responses)."""
    import pandas as pd
    from datasets import load_dataset
    t0 = time.time()
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    df = ds.to_pandas()
    cat_df = pd.json_normalize(df["category"])

    label_signature = cat_df.apply(lambda row: frozenset(c for c in cat_df.columns if row[c]), axis=1)
    sig_by_prompt = defaultdict(set)
    for prompt, sig in zip(df["prompt"], label_signature):
        sig_by_prompt[prompt].add(sig)
    ambiguous_prompts = {p for p, sigs in sig_by_prompt.items() if len(sigs) > 1}

    grouped = defaultdict(list)
    for cat in cat_df.columns:
        matched_prompts = df.loc[cat_df[cat].fillna(False).astype(bool), "prompt"]
        grouped[cat] = matched_prompts.tolist()
    print(f"[entrypoint] BeaverTails: {len(ds)} rows, "
          f"{sum(len(v) for v in grouped.values())} labeled pairs, "
          f"{len(ambiguous_prompts)} prompts with inconsistent labeling ({time.time()-t0:.1f}s)", flush=True)
    return dict(grouped), ambiguous_prompts


def load_aegis():
    """Returns (grouped, ambiguous_prompts) -- same concept as
    load_beavertails: a prompt text flagged ambiguous if it appears more
    than once with a different violated_categories outcome (Aegis rows
    are also (prompt, response) pairs, not unique prompts)."""
    import pandas as pd
    from datasets import load_dataset
    t0 = time.time()
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
    df = ds.to_pandas()

    vc_all = df["violated_categories"].fillna("").astype(str)
    sig_by_prompt = defaultdict(set)
    for prompt, vc in zip(df["prompt"], vc_all):
        sig = frozenset(c.strip() for c in vc.split(",") if c.strip())
        sig_by_prompt[prompt].add(sig)
    ambiguous_prompts = {p for p, sigs in sig_by_prompt.items() if len(sigs) > 1}

    grouped = defaultdict(list)
    valid = df[(vc_all != "") & (vc_all != "None")]
    exploded = valid.assign(_cat=valid["violated_categories"].astype(str).str.split(",")).explode("_cat")
    exploded["_cat"] = exploded["_cat"].str.strip()
    for cat, sub in exploded.groupby("_cat"):
        if cat:
            grouped[cat] = sub["prompt"].tolist()
    print(f"[entrypoint] Aegis: {len(ds)} rows, "
          f"{sum(len(v) for v in grouped.values())} labeled pairs, "
          f"{len(ambiguous_prompts)} prompts with inconsistent labeling ({time.time()-t0:.1f}s)", flush=True)
    return dict(grouped), ambiguous_prompts


def load_openai_moderation():
    import gzip
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("mmathys/openai-moderation-api-evaluation", "samples-1680.jsonl.gz", repo_type="dataset")
    label_names = {"S": "sexual", "H": "hate", "V": "violence", "HR": "harassment", "SH": "self-harm",
                   "S3": "sexual-minors", "H2": "hate-threatening", "V2": "violence-graphic"}
    grouped = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            row = json.loads(line)
            for key, name in label_names.items():
                if row.get(key) == 1:
                    grouped[name].append(row["prompt"])
    print(f"[entrypoint] OpenAI Moderation: {sum(len(v) for v in grouped.values())} labeled pairs", flush=True)
    return dict(grouped)


def load_simplesafetytests():
    from datasets import load_dataset
    ds = load_dataset("walledai/SimpleSafetyTests", split="instruct")
    prompts = list(ds["prompt"])
    cats = [f"SimpleSafetyTests::{h}" for h in ds["harm_type"]]
    print(f"[entrypoint] SimpleSafetyTests: {len(prompts)} prompts", flush=True)
    return prompts, cats


# MaliciousInstruct (Huang et al. 2024): the walledai HF mirror exposes NO
# category labels at all. The official paper repo's data file has none
# either, but its 100 prompts are laid out as 10 clean blocks of 10, in
# exactly the order Table 2 describes the dataset's topics -- confirmed by
# reading the raw file. Using that positional structure to recover labels
# a flat HF mirror can't give us.
MALICIOUS_INSTRUCT_CATEGORIES = [
    "manipulation", "sabotage", "theft", "defamation", "cyberbullying",
    "false_accusation", "tax_fraud", "hacking", "fraud", "illegal_drug_use",
]


def load_malicious_instruct():
    import urllib.request
    url = "https://raw.githubusercontent.com/Princeton-SysML/Jailbreak_LLM/main/data/MaliciousInstruct.txt"
    local_path = "/tmp/MaliciousInstruct.txt"
    urllib.request.urlretrieve(url, local_path)
    with open(local_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) != 100:
        print(f"[entrypoint] WARNING: expected 100 MaliciousInstruct lines, got {len(lines)} -- "
              f"positional category labels may be wrong.", flush=True)
    grouped = defaultdict(list)
    for i, prompt in enumerate(lines):
        cat = MALICIOUS_INSTRUCT_CATEGORIES[i // 10] if i // 10 < len(MALICIOUS_INSTRUCT_CATEGORIES) else "other"
        grouped[cat].append(prompt)
    print(f"[entrypoint] MaliciousInstruct: {len(lines)} prompts across {len(grouped)} categories "
          f"(official Princeton-SysML/Jailbreak_LLM source, positionally labeled)", flush=True)
    return dict(grouped)


# Anthropic Red Team (Ganguli et al. 2022): the full red-team-attempts
# release has 38,961 transcripts, but only 742 carry a human-assigned
# topic tag (the rest are untagged) -- using only the tagged subset for
# category-level analysis, consistent with every other source here.
# task_description (a short human-written summary of the red-teaming
# goal) is used as the "prompt" text rather than the full multi-turn
# transcript, for consistency with the single-instruction style of every
# other candidate dataset.
ANTHROPIC_REDTEAM_SKIP_TAGS = {"N/A - Invalid attempt", "Other"}


def load_anthropic_redteam():
    import gzip
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Anthropic/hh-rlhf", "red-team-attempts/red_team_attempts.jsonl.gz", repo_type="dataset")
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    grouped = defaultdict(list)
    n_tagged = 0
    for row in data:
        tags = row.get("tags")
        if not tags:
            continue
        n_tagged += 1
        for tag in tags:
            if tag in ANTHROPIC_REDTEAM_SKIP_TAGS:
                continue
            grouped[tag].append(row["task_description"])
    print(f"[entrypoint] Anthropic Red Team: {n_tagged}/{len(data)} transcripts tagged, "
          f"{sum(len(v) for v in grouped.values())} labeled pairs across {len(grouped)} categories", flush=True)
    return dict(grouped)


def main():
    t_start = time.time()
    out_dir = SM_MODEL_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    sh(["git", "clone", "-b", REPO_BRANCH, REPO_URL, str(REPO_DIR)])
    os.chdir(str(REPO_DIR))
    sh([sys.executable, "-m", "pip", "install", "-e", "."])
    sh([sys.executable, "-m", "pip", "install", "-q",
        "sentence-transformers==2.7.0", "datasets", "pandas", "huggingface_hub"])
    hf_login()

    sys.path.insert(0, str(REPO_DIR / "src"))
    import numpy as np
    from safety_layers_repro.ood_similarity import (
        run_similarity_check, build_ood_pool, load_embedder,
    )

    import torch
    print(f"[entrypoint] CUDA available: {torch.cuda.is_available()}", flush=True)

    print("\n=== Loading datasets ===", flush=True)
    id_prompts = load_advbench(REPO_DIR)
    print(f"AdvBench (ID anchor): {len(id_prompts)} prompts", flush=True)

    harmbench_prompts, harmbench_cats = load_harmbench_with_copyright()
    beavertails, beavertails_ambiguous = load_beavertails()
    aegis, aegis_ambiguous = load_aegis()
    openai_mod = load_openai_moderation()
    sst_prompts, sst_cats = load_simplesafetytests()
    malicious_instruct = load_malicious_instruct()
    anthropic_redteam = load_anthropic_redteam()

    source_prompts = {
        "BeaverTails": beavertails, "Aegis": aegis, "OpenAIModeration": openai_mod,
        "MaliciousInstruct": malicious_instruct, "AnthropicRedTeam": anthropic_redteam,
    }
    # Prompt texts flagged as carrying inconsistent safety/category labels
    # elsewhere in their OWN source dataset (see load_beavertails/load_aegis
    # docstrings) -- surfaced as a badge in the curation tool, not used to
    # filter anything out.
    ambiguous_by_source = {"BeaverTails": beavertails_ambiguous, "Aegis": aegis_ambiguous}

    # --- 1. Refreshed positive control: AdvBench vs HarmBench (now with copyright) ---
    print("\n=== Positive control: AdvBench vs HarmBench (incl. copyright) ===", flush=True)
    hb_sims, hb_summary = run_similarity_check(id_prompts, harmbench_prompts, harmbench_cats, threshold=0.7)
    hb_overall = {
        "n": len(hb_sims), "mean": float(hb_sims.mean()), "median": float(np.median(hb_sims)),
        "pct_ge_0.7": float((hb_sims >= 0.7).mean()), "pct_ge_0.85": float((hb_sims >= 0.85).mean()),
    }
    print(f"Overall: mean={hb_overall['mean']:.3f} median={hb_overall['median']:.3f} "
          f"%>=0.7={hb_overall['pct_ge_0.7']:.1%} %>=0.85={hb_overall['pct_ge_0.85']:.1%}", flush=True)
    with open(out_dir / "harmbench_positive_control.json", "w") as f:
        json.dump({"overall": hb_overall, "per_category": hb_summary}, f, indent=2)

    # --- 2. Full candidate sweep ---
    print("\n=== Full candidate sweep (BeaverTails+Aegis+OpenAIModeration+MaliciousInstruct+AnthropicRedTeam+SimpleSafetyTests) ===", flush=True)
    all_prompts, all_cats = [], []
    for name, grouped in source_prompts.items():
        for cat, prompts in grouped.items():
            all_prompts.extend(prompts)
            all_cats.extend([f"{name}::{cat}"] * len(prompts))
    all_prompts.extend(sst_prompts)
    all_cats.extend(sst_cats)
    print(f"Total candidate pairs: {len(all_prompts)}", flush=True)
    sweep_sims, sweep_summary = run_similarity_check(id_prompts, all_prompts, all_cats, threshold=0.7)
    with open(out_dir / "full_sweep_summary.json", "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print("Saved full_sweep_summary.json", flush=True)

    # --- 3. Steps 6+7 unified: build the FULL, deduplicated (but NOT
    # similarity-filtered) per-prompt pool for all 15 target categories --
    # both the 5 confirmed-OOD ones and the 10 borderline/covered ones,
    # through the same code path. No threshold is applied here: every
    # prompt survives dedup with its similarity + matched-AdvBench-prompt
    # attached, so the website can apply an adjustable threshold and let
    # a human include/exclude individual prompts, rather than the pipeline
    # pre-deciding a fixed cutoff.
    ALL_CATEGORIES = {**CATEGORY_MAP, **BORDERLINE_MAP}
    CATEGORY_GROUP = {c: "ood" for c in CATEGORY_MAP}
    CATEGORY_GROUP.update({c: "borderline" for c in BORDERLINE_MAP})

    print(f"\n=== Steps 6+7: building the full deduplicated per-prompt pool "
          f"for all {len(ALL_CATEGORIES)} target categories (no similarity filter) ===", flush=True)
    pools = build_ood_pool(ALL_CATEGORIES, source_prompts, id_prompts=id_prompts,
                            dedup_threshold=0.9, max_similarity=None, verbose=True)
    pool_summary = {}
    for cat, data in pools.items():
        recheck = data["similarity_recheck"] or {}
        pool_summary[cat] = {
            "group": CATEGORY_GROUP[cat],
            "n_before_dedup": data["n_before_dedup"], "n_after_dedup": data["n_after_dedup"],
            "dedup_removed": data["n_before_dedup"] - data["n_after_dedup"],
            "source_breakdown": data.get("source_breakdown", {}), **recheck,
        }
        # Tag (don't filter) records whose exact prompt text also appears
        # elsewhere in ITS OWN source dataset with a different safety/
        # category outcome -- surfaced to the curator as a badge, not acted
        # on automatically (see load_beavertails/load_aegis docstrings for
        # why an automatic filter here would be unreliable).
        n_ambiguous = 0
        for rec in data["records"]:
            amb_set = ambiguous_by_source.get(rec["source_dataset"])
            rec["ambiguous_label"] = bool(amb_set and rec["prompt"] in amb_set)
            n_ambiguous += rec["ambiguous_label"]
        pool_summary[cat]["n_ambiguous_label"] = n_ambiguous
        with open(out_dir / f"ood_pool_{cat}.json", "w") as f:
            json.dump(data["records"], f, indent=2)
    pool_summary["sexual_harassment"] = {
        "group": "borderline", "n_before_dedup": 0, "n_after_dedup": 0, "dedup_removed": 0,
        "note": "No clean single-category source label exists in any of the 6 candidate datasets "
                "for 'sexual harassment' specifically (as distinct from general harassment or sexual "
                "content) -- would need a dedicated dataset or manual curation to cover this category.",
    }
    with open(out_dir / "ood_pool_summary.json", "w") as f:
        json.dump(pool_summary, f, indent=2)
    print("Steps 6+7 done:", json.dumps(pool_summary, indent=2), flush=True)

    print(f"\n[entrypoint] ALL DONE in {time.time()-t_start:.1f}s total.", flush=True)
    print(f"[entrypoint] Results in {out_dir}: {sorted(p.name for p in out_dir.iterdir())}", flush=True)


if __name__ == "__main__":
    main()
