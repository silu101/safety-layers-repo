# gemma_sppft_normal — OOD semantic-content eval, n=520

First full-scale OOD generalization result for Safety Layers (SPPFT-normal
condition), matching AdvBench's own eval size (520 prompts) instead of a
100-prompt pilot. 520 prompts sampled 104 each from the 5 confirmed-OOD
categories (hate_discrimination, harassment, sexual_content, privacy,
political_misinformation) — see `data/ood_semantic_test_metadata.json` for
full per-prompt provenance and `scripts/build_ood_semantic_test.py` for how
the sample was drawn.

SageMaker job: `safety-layers-ood-eval-semantic-2026-09-08-03-22-47-349`
(two earlier attempts, `...00-38-20-587` and `...02-11-03-113`, timed out
mid-S_h-scoring and lost all data before the checkpointing/concurrency fix
in commit `ad4ad0f`).

`harmful_eval_result.json` was additionally hand-patched after the run
(commit `d92b99d`): 182/520 S_h judge scores initially came back
unparseable due to a parser bug (Claude Haiku rendering the `#thescore:`
marker as a markdown heading, `# thescore:`, with a space) -- fixed and
re-scored directly from this file's own saved prompts/responses, no
SageMaker re-run needed. `s_h_status` in the JSON records this.

## Headline numbers (see `scripts/analyze_ood_eval_result.py` for the full
stratified breakdown, ambiguous vs. non-ambiguous, with z-tests/Wilson CIs)

| Metric | OOD (non-ambiguous, n=392) | AdvBench ID (n=520) | Significant? |
|---|---|---|---|
| Zou keyword R_h | 0.768 | 0.306 | Yes, p≈0.000 (weak evidence -- keyword heuristic only) |
| HarmBench classifier R_h | 0.046 | 0.075 | No, p=0.072 (trends safer OOD) |
| S_h (judge severity, 1-5) | 1.487 | 1.450 | Essentially identical |

**No statistically significant ID-vs-OOD generalization gap** on the two
stronger metrics (HarmBench classifier, S_h) at n=520, for this one model
condition. Zou keyword classifier shows a large, significant gap, but it
only detects fixed refusal phrasing -- weak evidence of actual compliance
change.

Reproduce the analysis:
```
python scripts/analyze_ood_eval_result.py results/gemma_sppft_normal_ood_semantic_520/harmful_eval_result.json
```
