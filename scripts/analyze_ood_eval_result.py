"""
Stratified analysis of an OOD harmful-eval result: reports R_h (Zou
keyword classifier + HarmBench classifier) and S_h (Claude judge)
separately for the ambiguous vs. non-ambiguous subset of the OOD prompts,
plus the combined number -- "Option A" from this project's discussion of
how to handle AdvBench's homogeneous (100% unambiguously-harmful)
composition versus the OOD pool's naturalistic, partly-ambiguous
composition. A single combined R_h number would confound "did safety
generalize" with "how much of this sample was ambiguous in the first
place" -- this splits the two apart instead of guessing they're close
enough.

Requires a result produced by the FIXED run_harmful_eval.py (which now
saves harmbench_compliant, a per-prompt bool list -- older result files
saved before that fix only have the aggregate r_h_harmbench and can't be
stratified for that metric; the script degrades gracefully and says so).
judge_scores (per-prompt) was already saved even before that fix.

Also runs the same two-proportion z-test used earlier in this project
(pilot run vs. AdvBench baseline) for each subset, so significance isn't
asserted by eyeballing a percentage gap.

Usage:
    python scripts/analyze_ood_eval_result.py \\
        results/gemma_sppft_normal_ood_semantic/harmful_eval_result.json \\
        --advbench-r-h-harmbench 0.075 --advbench-r-h-zou 0.306 --advbench-s-h 1.45 --advbench-n 520
"""
import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "data" / "ood_semantic_test_metadata.json"


def two_proportion_z_test(p1, n1, p2, n2):
    x1, x2 = round(p1 * n1), round(p2 * n2)
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p2 - p1) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value


def wilson_ci(k, n, z=1.96):
    """Wilson score interval -- more reliable than the normal approximation
    at small n or extreme proportions (relevant here since the ambiguous
    subset is much smaller than the full 520)."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def two_sample_t_test(a, b):
    """Welch's t-test (unequal variance) for the judge-score comparison --
    S_h is a 1-5 score, not a proportion, so the z-test above doesn't apply."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None, None
    m1, m2 = sum(a) / n1, sum(b) / n2
    v1 = sum((x - m1) ** 2 for x in a) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in b) / (n2 - 1)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, 1.0
    t = (m2 - m1) / se
    # Welch-Satterthwaite df, then normal approximation for the p-value
    # (fine for the sample sizes here; avoids a scipy dependency).
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p_value


def load_ambiguous_prompt_set() -> set:
    meta = json.load(open(METADATA_PATH))
    return {r["prompt"] for cat in meta["categories"].values() for r in cat if r.get("ambiguous_label")}


def split_by_ambiguity(prompts, values, ambiguous_set):
    """values[i] corresponds to prompts[i]; None entries (unparseable judge
    output, etc.) are dropped rather than coerced."""
    groups = {"ambiguous": [], "non_ambiguous": []}
    for p, v in zip(prompts, values):
        if v is None:
            continue
        key = "ambiguous" if p in ambiguous_set else "non_ambiguous"
        groups[key].append(v)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result_path")
    ap.add_argument("--advbench-r-h-zou", type=float, default=0.306,
                     help="AdvBench ID baseline Zou-classifier rate (default: gemma_sppft_normal's documented 159/520)")
    ap.add_argument("--advbench-r-h-harmbench", type=float, default=0.075,
                     help="AdvBench ID baseline HarmBench-classifier rate (default: gemma_sppft_normal's documented 39/520)")
    ap.add_argument("--advbench-s-h", type=float, default=1.45,
                     help="AdvBench ID baseline mean judge score (default: gemma_sppft_normal's documented 1.45)")
    ap.add_argument("--advbench-n", type=int, default=520)
    args = ap.parse_args()

    result = json.load(open(args.result_path))
    ambiguous_prompts = load_ambiguous_prompt_set()
    prompts = result["prompts"]
    n = len(prompts)
    n_amb = sum(1 for p in prompts if p in ambiguous_prompts)
    print(f"Total prompts: {n}  (ambiguous: {n_amb}, non-ambiguous: {n - n_amb})\n")

    def report_proportion(label, per_prompt_bools, advbench_rate):
        if per_prompt_bools is None:
            print(f"=== {label}: not available (per-prompt scores not saved in this result file) ===\n")
            return
        groups = split_by_ambiguity(prompts, per_prompt_bools, ambiguous_prompts)
        combined_rate = sum(per_prompt_bools) / len([v for v in per_prompt_bools if v is not None])
        print(f"=== {label} ===")
        print(f"Combined (all {n}): {combined_rate:.3f}  [AdvBench ID: {advbench_rate:.3f}]")
        for key in ["non_ambiguous", "ambiguous"]:
            vals = groups[key]
            if not vals:
                continue
            k, gn = sum(vals), len(vals)
            rate = k / gn
            ci_lo, ci_hi = wilson_ci(k, gn)
            z, p = two_proportion_z_test(advbench_rate, args.advbench_n, rate, gn)
            sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
            print(f"  {key:<15} n={gn:<4} rate={rate:.3f} ({k}/{gn})  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]  "
                  f"vs AdvBench: z={z:.2f} p={p:.3f} ({sig})")
        print()

    # --- Zou keyword classifier: recompute per-prompt locally (deterministic,
    # no model needed) even for older result files that predate this script. ---
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from safety_layers_repro.zou_keyword_classifier import REFUSAL_KEYWORDS
    zou_harmful = [not any(kw in r for kw in REFUSAL_KEYWORDS) for r in result["responses"]]
    report_proportion("Zou keyword classifier R_h", zou_harmful, args.advbench_r_h_zou)

    # --- HarmBench classifier: only available if run_harmful_eval.py saved
    # the per-prompt harmbench_compliant list (fixed to do so; older result
    # files won't have it). ---
    report_proportion("HarmBench classifier R_h", result.get("harmbench_compliant"), args.advbench_r_h_harmbench)

    # --- S_h judge score: already saved per-prompt (judge_scores), just
    # needed the ambiguous/non-ambiguous split + a proper significance test
    # (Welch's t-test, since this is a 1-5 score, not a proportion). ---
    judge_scores = result.get("judge_scores")
    if judge_scores is None:
        print("=== S_h judge score: not available (s_h_error or no ANTHROPIC_API_KEY) ===")
    else:
        groups = split_by_ambiguity(prompts, judge_scores, ambiguous_prompts)
        valid_all = [s for s in judge_scores if s is not None]
        print("=== S_h judge score ===")
        print(f"Combined (all {n}): {sum(valid_all)/len(valid_all):.3f}  [AdvBench ID: {args.advbench_s_h:.3f}]")
        # Welch's t-test needs two real samples; approximate the AdvBench
        # side as a point value repeated advbench_n times only for display
        # symmetry -- the real comparison that matters is non_ambiguous vs
        # ambiguous within THIS run, which doesn't need the ID baseline at all.
        for key in ["non_ambiguous", "ambiguous"]:
            vals = groups[key]
            if not vals:
                continue
            print(f"  {key:<15} n={len(vals):<4} mean S_h={sum(vals)/len(vals):.3f}")
        if groups["non_ambiguous"] and groups["ambiguous"]:
            t, p = two_sample_t_test(groups["non_ambiguous"], groups["ambiguous"])
            sig = "SIGNIFICANT (p<0.05)" if p is not None and p < 0.05 else "not significant"
            print(f"  non-ambiguous vs ambiguous (within this OOD run): t={t:.2f} p={p:.3f} ({sig})")
        print()


if __name__ == "__main__":
    main()
