# Known Discrepancies: Paper Text vs. Shipped Repo

Found while building the reproduction spec by reading the actual code in
https://github.com/listen0425/Safety-Layers rather than relying on the paper's
prose. Log new ones here as you find them — this is exactly the kind of thing
the project's "Reproduction" phase is meant to surface.

| # | Discrepancy | Paper says | Code/data shows | Status / action |
|---|---|---|---|---|
| 1 | Over-rejection dataset size | "721 problems" (Table 1 caption, Appendix A.3.4) | `Dataset/Evaluation/Over_rejection_dataset.csv` has 731 rows | Not yet resolved. Could be a dataset version update since publication, or a paper typo. Check before running Section 3.4 localization; if reproducing Table 1's exact numbers, may need to find/reconstruct the 721-row version, or accept a small offset and note it. |
| 2 | `normal.csv` query count | Paper states P=100 (Section 3.2, Fig. 1 caption) | `Code/Cos_sim_analysis/normal.csv` has 99 rows | Check whether a header row is being silently dropped somewhere in the authors' pipeline, or whether this is a genuine 1-row discrepancy. Low impact on results (r=500 draws with replacement-ish sampling across pairs), but worth resolving for an exact match. |
| 3 | Shipped reference `all_cos.pkl` | Paper's r=500 for the cosine similarity analysis | The file at `Code/Cos_sim_analysis/cos_sims/all_cos.pkl` in the original repo (copied here as `reference/all_cos_phi3_authors.pkl`) has only **10 draws per pair-type** | Almost certainly a demo/smoke-test artifact left in the repo, not the authors' full r=500 result used to generate Fig. 1. Treat as a pipeline sanity check only — `compare.py` emits an explicit warning when comparing against small-r data. Do not use it to validate the paper's exact numeric claims. |
| 4 | Model-specific code branches | Not mentioned as needing manual edits | `Attention_scores/att_scores.py` requires uncommenting lines 51/85 for Gemma models; `Safety_layers_locating/scaling.py` requires swapping `scaling()` → `scaling_phi3()` for Phi-3 models (different attention/MLP module names: `q_proj/k_proj/v_proj` vs. fused `qkv_proj`) | Only relevant once Section 3.4/3.6 modules are ported (see README "Scope"). Note here now so it's not missed later. |
| 5 | `Backdoor_dataset.json` naming vs. paper's DI/DB taxonomy | Paper defines DN (normal), DI (implicit attack: no trigger, "Sure, the answer is:" prefix), DB (backdoor: has a trigger phrase + same prefix), with DI having 4000 entries and DB being a 1:1 mix of backdoor:normal | The shipped `Dataset/Finetune/Backdoor_dataset.json` (4000 entries) reads as the **implicit**-style data (no trigger keyword) by the paper's own definition, not backdoor-style | Only relevant once Section 4 (SPPFT) is ported. Needs checking against `Full_finetuning.py`/`SPPFT.py` to see if a trigger is injected at load time, or if the filename is simply a misnomer, or if a separate DB file exists that wasn't included in this dump. Flag before trusting Table 2's DB column reproduction. |
| 6 | `Full_finetuning.py` vs `SPPFT.py` default learning rates | Appendix Table 6 gives one LR per model (e.g. Llama-2-7b-chat: `3e-4`) | Script defaults differ from each other (`Full_finetuning.py` default `3e-4`, `SPPFT.py` default `3e-5`) and neither is guaranteed to match Table 6 for every model | Only relevant once Section 4 is ported. Always pass `--learning_rate` explicitly from Table 6 rather than trusting either script's default. |

## How to add a new entry

When you find a discrepancy: add a row here immediately (don't just fix it silently), note whether it's resolved or still open, and cross-reference the replication log entry it affects, if any.
