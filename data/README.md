# Data

- `normal.csv`, `malicious.csv` — copied verbatim from the original repo's
  `Code/Cos_sim_analysis/` directory. Single column, no header, one
  instruction per line. See `docs/KNOWN_DISCREPANCIES.md` #2 for a row-count
  note on `normal.csv`.
- `over_rejection.csv` — copied verbatim from the original repo's
  `Dataset/Evaluation/Over_rejection_dataset.csv`. Single column, no header,
  731 rows (paper's Table 1 caption states 721 -- see
  `docs/KNOWN_DISCREPANCIES.md` #1). Borderline-but-benign prompts used for
  Section 3.4 localization (see `docs/REPRODUCTION_SPEC_3.4.md`).
- `advbench_malicious.csv` — copied verbatim from the original repo's
  `Dataset/Evaluation/Malicious_dataset.csv`. 520 rows, matches the paper's
  stated size for `D_m` (Zou et al. 2023's AdvBench). Used for Section 4's
  harmful-rate (`R_h`) and GPT-4 harmful-score (`S_h`) evaluation -- see
  `docs/REPRODUCTION_SPEC_3.4.md`'s "Section 4 evaluation methodology" note.
- `finetune_normal.json`, `finetune_backdoor.json` — copied verbatim from
  the original repo's `Dataset/Finetune/Normal_dataset.json` and
  `Backdoor_dataset.json`. Fine-tuning data for Section 4 (SPPFT vs. full
  fine-tuning comparison). See `docs/KNOWN_DISCREPANCIES.md` #5 for a
  naming/taxonomy discrepancy in `Backdoor_dataset.json`.

- `ood_semantic_test.csv` — semantic-content OOD test set replacing
  `advbench_malicious.csv` for the OOD generalization experiment: 100
  prompts (20 each from the 5 categories confirmed OOD relative to AdvBench
  -- hate_discrimination, harassment, sexual_content, privacy,
  political_misinformation), sampled from the curated pool in
  `ood_curated/*.json` (built by the OOD Pool Inspector tool). Same plain-
  text, one-prompt-per-line format as `advbench_malicious.csv` -- drops
  directly into `run_harmful_eval.py` via `configs/eval_gemma_sppft_normal_ood.yaml`.
  Regenerate with `scripts/build_ood_semantic_test.py`. Companion
  `ood_semantic_test_metadata.json` records each prompt's source dataset,
  original category label, and similarity to AdvBench for traceability
  (the eval script itself never reads this file).

- `harmbench_eval.csv` — near-OOD baseline: the full official HarmBench
  release (400 behaviors -- standard + contextual from the `walledai/HarmBench`
  HF mirror, plus the official "copyright" category that mirror is missing,
  fetched directly from the official GitHub CSV). Same 400-behavior set used
  for the AdvBench&harr;HarmBench positive control (mean similarity 0.508).
  Same plain-text, one-prompt-per-line format -- drops directly into
  `run_harmful_eval.py` via `configs/eval_gemma_sppft_normal_harmbench.yaml`.
  Regenerate with `scripts/build_harmbench_eval_set.py`. Companion
  `harmbench_eval_metadata.json` records each prompt's HarmBench category.

- `attack_ood_jailbreakllms.csv` — first Attack-OOD test set: 520 AdvBench
  goals, each wrapped in a real-world jailbreak template from JailbreakLLMs
  (Shen et al. 2024, "Do Anything Now" -- in-the-wild DAN/persona/roleplay
  prompts scraped from Reddit/Discord/jailbreak sites). Holds the harmful
  GOAL constant (same ID anchor as every other eval) and varies only the
  wrapper/mechanism -- goal and template are stored separately per-record
  in the companion metadata, not just as one merged string. Same plain-
  text, one-prompt-per-line format -- drops directly into
  `run_harmful_eval.py` via `configs/eval_gemma_sppft_normal_attackood_jbllms.yaml`.
  Regenerate with `scripts/build_attack_ood_jailbreakllms.py`. Companion
  `attack_ood_jailbreakllms_metadata.json` records each record's original
  template, matched AdvBench goal, and whether the goal was substituted
  into an explicit placeholder or appended (only 71/520 templates in this
  sample had one).

- `attack_ood_multijail.csv` — MultiJail covariate/multilingual Attack-OOD
  test (Deng et al. 2024): 315 base prompts x 9 non-English languages
  (zh, it, vi, ar, ko, th, bn, sw, jv) = 2,835 total. **Known, accepted
  confound**: MultiJail's own English base prompts are Anthropic (300/315)
  and OpenAI (15/315) sourced -- NOT AdvBench, checked directly against
  the released `source` column. This mixes a language-shift effect with a
  content-shift effect; the methodologically clean version (translating
  our own AdvBench goals ourselves) was raised and explicitly deferred --
  used as released per user decision. Any interpretation of this run
  should name this confound, not present the delta as a clean language-
  only isolate. Same plain-text, one-prompt-per-line format -- drops
  directly into `run_harmful_eval.py` via
  `configs/eval_gemma_sppft_normal_attackood_multijail.yaml`. Regenerate
  with `scripts/build_attack_ood_multijail.py`. Companion
  `attack_ood_multijail_metadata.json` records id/source/tags/language
  per record for per-language breakdown.

Source: https://github.com/listen0425/Safety-Layers
