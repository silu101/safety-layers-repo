# Safety Layers — Reproduction (Section 3.2–3.3, cosine similarity existence check)

Modular, testable reproduction of one result from:

> Li, Yao, Zhang, Li. **"Safety Layers in Aligned Large Language Models: The Key to LLM Security."** ICLR 2025. [arXiv:2408.17003](https://arxiv.org/abs/2408.17003)

Original authors' code: https://github.com/listen0425/Safety-Layers

This repo is **Phase 1 (reproduction) only**, scoped to the cheapest, most central claim in the paper: that the layer-wise cosine similarity between normal-normal (N-N) and malicious-malicious (M-M) query pairs stays flat/high across layers, while normal-malicious (N-M) pairs show a widening gap starting mid-network (paper Fig. 1/2, Section 3.3). No fine-tuning or gated-model access is required for this piece.

See `docs/REPRODUCTION_SPEC.md` for the full experiment specification and `docs/REPLICATION_LOG.md` to track runs.

## Why this structure

- **Pure logic is separated from model I/O** (`src/safety_layers_repro/cos_sim.py`): the sampling and cosine-similarity math is unit-tested (`tests/test_cos_sim.py`) without needing torch, a GPU, or model weights installed. Only `load_model_and_tokenizer` / `get_last_position_hidden_states` need the `[model]` extra.
- **Every run is config-driven** (`configs/*.yaml`), and every run's output directory gets a `run_metadata.json` with the full resolved config, package version, and this repo's git commit — so if a teammate's numbers differ, the first question ("what config/commit did you run?") is answered automatically.
- **Comparison is numeric, not visual** (`src/safety_layers_repro/compare.py`): given two `all_cos.pkl` files, it reports per-layer mean/std, Pearson correlation of the mean curves, and a heuristic "gap onset layer" — with a markdown + JSON report and an optional overlay plot. Tested against synthetic data (`tests/test_compare.py`) so the comparison logic itself is verified independent of any real run.

## Repo layout

```
configs/                    YAML experiment configs (edit params here, not in code)
  phi3_cossim.yaml            full run, r=500 (paper's setting)
  phi3_cossim_smoketest.yaml  r=20, for a fast pipeline sanity check
data/
  normal.csv, malicious.csv   query pools, copied from the original repo
reference/
  all_cos_phi3_authors.pkl    ⚠️ shipped in the ORIGINAL repo as `cos_sims/all_cos.pkl`.
                               Only has 10 draws/pair-type, not the paper's r=500 —
                               almost certainly a demo artifact, not the authors' full
                               result. Useful as a pipeline sanity check, NOT as a
                               ground-truth target for the replication log.
src/safety_layers_repro/
  config.py                   CosSimConfig dataclass + YAML loader + run metadata
  prompter.py                 prompt templating (ported from original repo, byte-identical)
  cos_sim.py                  core sampling + cosine similarity (pure logic + model I/O)
  compare.py                  numeric comparison tool + report/plot generation
  run_cos_sim.py               CLI: run the full experiment end to end
  alpaca.json, alpaca_short.json  prompt templates (ported from original repo)
tests/
  test_cos_sim.py             unit tests, no GPU/model required
  test_compare.py             unit tests on synthetic data, no GPU/model required
results/                      run outputs land here (gitignored except .gitkeep)
docs/
  REPRODUCTION_SPEC.md        Input → Method → Parameters → Evaluation → Claimed Result
  REPLICATION_LOG.md          tracked table of every run vs. the paper's claim
  KNOWN_DISCREPANCIES.md      differences found between the paper text and the shipped code
```

## Setup

```bash
git clone <this-repo>
cd safety-layers-repro

# 1. Install core package + dev tools (works on any machine, no GPU needed):
pip install -e ".[dev]"

# 2. Run the pure-logic test suite (should pass everywhere, in seconds):
pytest tests/ -v

# 3. On a GPU machine, additionally install model-running deps:
pip install -e ".[model,plot]"
```

## Running the real experiment (GPU machine)

```bash
# Quick smoke test first (~1-2 min on a modern GPU) to confirm the pipeline works:
python -m safety_layers_repro.run_cos_sim --config configs/phi3_cossim_smoketest.yaml

# Full paper-matching run (r=500; expect a while — 1500 generations of 1 token
# with output_hidden_states=True per pair-type, ×3 pair-types):
python -m safety_layers_repro.run_cos_sim --config configs/phi3_cossim.yaml

# Override any config field ad hoc without editing the YAML:
python -m safety_layers_repro.run_cos_sim --config configs/phi3_cossim.yaml --set r=100
```

Each run creates `results/<run_name>/` containing `all_cos.pkl` (the raw result) and `run_metadata.json` (full config + provenance).

## Comparing runs

```bash
python -m safety_layers_repro.compare \
  --ours results/<your_run>/all_cos.pkl \
  --reference reference/all_cos_phi3_authors.pkl \
  --out-dir results/<your_run>/comparison
```

Produces `comparison_report.md`, `comparison_report.json`, and `comparison_plot.png`. Use this to compare:
- your run vs. the shipped demo reference (sanity check only — small-sample, see warning above)
- your run vs. a teammate's run of the same config (should match closely — same seeds are used, so with the same model/library versions the pair *selection* is deterministic; the hidden-state *values* depend on model/library/hardware determinism, which is not guaranteed)
- two of your own runs with different `r` (does more sampling change the mean curve much, i.e. was r=500 actually necessary, or does it converge earlier — a Phase 2 validation question, tracked separately)

## Scope / what's intentionally NOT here yet

Per the project's reproduce → validate → (only then) novel-research discipline, this repo currently covers only the Section 3.2–3.3 existence check. Not yet ported:
- Section 3.4 localization (`scaling.py` + over-rejection scoring)
- Section 3.6 attention-score heatmaps (`att_scores.py`)
- Section 4 SPPFT fine-tuning experiments (`Full_finetuning.py` / `SPPFT.py`)

These would each become their own `configs/*.yaml` + module following the same pattern, once this piece is validated. See `docs/REPRODUCTION_SPEC.md` for the full spec if you want to start one of these next — happy to build it out the same way.
