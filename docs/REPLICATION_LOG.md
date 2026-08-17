# Replication Log

Every run that's meant to inform a reproduction/validation decision goes here —
not every debugging run, but every one whose result you'd cite. Fill in a new
row per run using `results/<run_name>/run_metadata.json` for the exact config.

## Cosine similarity existence check (Section 3.2–3.3)

| Date | Experiment | Config | Model | r | Result (gap onset layer, heuristic) | Diff from reference | Notes |
|---|---|---|---|---|---|---|---|
| — | Authors' claimed result | — (paper Fig. 1) | Phi-3-mini-4k-instruct | 500 | Qualitative: visible mid-network divergence, no precise layer given in Fig. 1 itself (Section 3.4 refines this to [11,15] via a different method — see paper Table 1) | — | Paper result; not a single number, described qualitatively for this analysis. |
| — | Shipped demo reference | `reference/all_cos_phi3_authors.pkl` (from original repo) | Phi-3-mini-4k-instruct | **10** (not 500 — see `KNOWN_DISCREPANCIES.md` #3) | Heuristic onset layer 16 (via `compare.py`, margin=0.05) | n/a — this *is* the reference | Small-sample; treat as pipeline sanity check only, not ground truth. |
| — | *(fill in: your smoke test)* | `configs/phi3_cossim_smoketest.yaml` | Phi-3-mini-4k-instruct | 20 | | vs. shipped demo reference | Purpose: confirm pipeline runs end-to-end on your hardware before committing to full run. |
| — | *(fill in: your full run)* | `configs/phi3_cossim.yaml` | Phi-3-mini-4k-instruct | 500 | | vs. shipped demo reference (small-sample caveat applies) | This is the actual reproduction target. |
| — | *(fill in: second full run, different machine/seed check)* | same config | Phi-3-mini-4k-instruct | 500 | | vs. first full run | Checks run-to-run stability given library/hardware non-determinism (see README "Comparing runs"). |

## How to fill in a row

1. Run: `python -m safety_layers_repro.run_cos_sim --config configs/<name>.yaml`
2. Compare: `python -m safety_layers_repro.compare --ours results/<run>/all_cos.pkl --reference reference/all_cos_phi3_authors.pkl`
3. Pull the "gap onset" and diff numbers from `results/<run>/comparison/comparison_report.md`
4. Add a row here with a link/path to `results/<run>/` so the full artifacts are traceable from this table.

## Escalation rule (per project discipline)

If a full r=500 run's mean curves diverge substantially in shape from the qualitative pattern described in the paper (i.e. N-N/M-M flat-high, N-M diverging mid-network) — not just numeric noise — **stop and investigate** before moving to Section 3.4 localization or any validation experiments. Candidate causes to check first: model revision/quantization mismatch, prompt template mismatch, transformers/torch version affecting `generate()` defaults, `num_beams=4` vs. greedy decoding differences.
