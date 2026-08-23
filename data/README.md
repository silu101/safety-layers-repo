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

Source: https://github.com/listen0425/Safety-Layers
