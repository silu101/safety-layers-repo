# Replication Log

Every run that's meant to inform a reproduction/validation decision goes here —
not every debugging run, but every one whose result you'd cite. Fill in a new
row per run using `results/<run_name>/run_metadata.json` for the exact config.

## Cosine similarity existence check (Section 3.2–3.3)

| Date | Experiment | Config | Model | r | Result (gap onset layer, heuristic) | Diff from reference | Notes |
|---|---|---|---|---|---|---|---|
| — | Authors' claimed result | — (paper Fig. 1) | Phi-3-mini-4k-instruct | 500 | Qualitative: visible mid-network divergence, no precise layer given in Fig. 1 itself (Section 3.4 refines this to [11,15] via a different method — see paper Table 1) | — | Paper result; not a single number, described qualitatively for this analysis. |
| — | Shipped demo reference | `reference/all_cos_phi3_authors.pkl` (from original repo) | Phi-3-mini-4k-instruct | **10** (not 500 — see `KNOWN_DISCREPANCIES.md` #3) | Heuristic onset layer 16 (via `compare.py`, margin=0.05) | n/a — this *is* the reference | Small-sample; treat as pipeline sanity check only, not ground truth. |
| 2026-08-20 | Smoke test (Kaggle, automated via API) | `configs/phi3_cossim_smoketest.yaml` | Phi-3-mini-4k-instruct | 20 | Heuristic onset layer 15 | vs. shipped demo reference: mean\|Δmean\| N-N 0.033, M-M 0.019, N-M 0.048; Pearson r 0.84/0.84/0.92 (N-N/M-M/N-M) | Purpose confirmed: pipeline runs end-to-end via `scripts/run_on_kaggle.sh`. `dtype=auto` (not float32), `trust_remote_code=false` (deviation, see `KNOWN_DISCREPANCIES.md` #8). Ran on an assigned Tesla P100 (compute cap 6.0) after `kernel_runner.py`'s `ensure_gpu_compatible_torch()` reinstalled `torch==2.7.1+cu118` (see `KNOWN_DISCREPANCIES.md` #9). Results: `results/Phi-3-mini-4k-instruct_r20_20260820T001008Z/`. Noisy by design (r=20) — not a reproduction target, sanity check only. |
| 2026-08-20 | Full run (Kaggle, automated via API) | `configs/phi3_cossim.yaml` | Phi-3-mini-4k-instruct | 500 | Heuristic onset layer 15 (reference: 16) | vs. shipped demo reference (small-sample caveat applies, r=10): mean\|Δmean\| N-N 0.031, M-M 0.018, N-M 0.052; max\|Δmean\| N-N 0.43, M-M 0.10, N-M 0.46; Pearson r (mean curves) 0.83 (N-N), 0.83 (M-M), 0.92 (N-M) | **This is the reproduction target.** `dtype=float32` (fidelity-matched, no OOM fallback triggered), `trust_remote_code=false` (**deviation** from the original authors' script — see `KNOWN_DISCREPANCIES.md` #8, uses transformers' native Phi-3 implementation instead of Microsoft's custom remote code). Ran on an assigned Tesla P100 (compute cap 6.0); `kernel_runner.py` reinstalled `torch==2.7.1+cu118` for sm_60 compatibility (**environment deviation**, `KNOWN_DISCREPANCIES.md` #9). Git commit `a76f7fd`. Qualitative pattern matches the paper: N-N/M-M stay high and roughly flat, N-M diverges starting mid-network, gap onset (layer 15) close to reference (layer 16) and within the paper's Section 3.4-refined range [11,15]. Results: `results/Phi-3-mini-4k-instruct_r500_20260820T001149Z/`, comparison report at `results/Phi-3-mini-4k-instruct_r500_20260820T001149Z/comparison/comparison_report.md`. Given the two logged deviations above, this is a reproduction under a modified (not bitwise-identical) setup, not an exact match to the authors' float32 + custom-remote-code configuration. |
| — | *(fill in: second full run, different machine/seed check)* | same config | Phi-3-mini-4k-instruct | 500 | | vs. first full run | Checks run-to-run stability given library/hardware non-determinism (see README "Comparing runs"). |

## How to fill in a row

1. Run: `python -m safety_layers_repro.run_cos_sim --config configs/<name>.yaml`
2. Compare: `python -m safety_layers_repro.compare --ours results/<run>/all_cos.pkl --reference reference/all_cos_phi3_authors.pkl`
3. Pull the "gap onset" and diff numbers from `results/<run>/comparison/comparison_report.md`
4. Add a row here with a link/path to `results/<run>/` so the full artifacts are traceable from this table.

## Escalation rule (per project discipline)

If a full r=500 run's mean curves diverge substantially in shape from the qualitative pattern described in the paper (i.e. N-N/M-M flat-high, N-M diverging mid-network) — not just numeric noise — **stop and investigate** before moving to Section 3.4 localization or any validation experiments. Candidate causes to check first: model revision/quantization mismatch, prompt template mismatch, transformers/torch version affecting `generate()` defaults, `num_beams=4` vs. greedy decoding differences.
