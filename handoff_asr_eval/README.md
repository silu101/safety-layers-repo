# ASR eval (minimal handoff package)

Computes one number: **ASR (Attack Success Rate)** — the fraction of
harmful prompts for which the target model's response is judged a
successful/compliant instance of the harmful behavior, rather than a
refusal. This is the same quantity the parent project (`safety-layers-repro`)
calls **R_h** elsewhere — different name, identical definition.

This is a stripped-down extract of that project's eval pipeline: no
SageMaker orchestration, no mid-run checkpointing, no secondary Zou
keyword classifier or Claude severity judge — those exist in the parent
project to solve problems specific to running on AWS SageMaker (job
timeouts, capacity queues, wanting a second corroborating signal) that
don't apply running directly on your own GPU box. Just: generate a
response for every prompt, judge it, report the compliant fraction.

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # needed if your target model or the judge model is gated
```

## Run

```bash
python run_asr.py --model_path <hf-model-id-or-local-checkpoint-path> \
                   --prompts_path prompts/advbench_malicious.csv
```

Key flags:
- `--dtype` (default `bfloat16`) — match to your GPU/model.
- `--max_new_tokens` (default 256) — the parent project used 128 for its
  own gemma-2b runs; raise this if your target model tends to write
  longer responses before getting to the harmful content, or the judge
  may see a truncated, ambiguous response.
- `--max_prompts N` — truncate for a quick smoke test before a full run.
- `--out_path` — where the full per-prompt result JSON is written
  (default `asr_result.json`).

## What the judge actually is

`cais/HarmBench-Llama-2-13b-cls` (Mazeika et al. 2024) — a 13B model
purpose-built and validated for exactly this classification task. Not a
keyword heuristic. It reads the (prompt, response) pair and answers
yes/no to "does this generation count as a successful instance of the
behavior." See `harmbench_classifier.py`'s docstring for the exact prompt
template (fetched verbatim from HarmBench's own repo).

**VRAM**: the target model and the judge (~26GB in bf16) are both loaded,
though the script frees the target model before loading the judge to
reduce peak usage. Still size your GPU for whichever model under test you
run alongside a ~26GB judge -- e.g. a 7B target in bf16 (~14GB) means
~40GB peak overlap during the brief window both are technically alive in
memory; an A100-80GB or H100 has real headroom, an A100-40GB is tighter.

## Prompt sets included (`prompts/`)

| File | n | What it is |
|---|---|---|
| `advbench_malicious.csv` | 520 | AdvBench (Zou et al. 2023) — the in-distribution anchor the parent project's Safety Layers reproduction was evaluated against |
| `harmbench_eval.csv` | 400 | Full official HarmBench release (200 standard + 100 contextual + 100 copyright) — a near-OOD point, known to partially overlap AdvBench |
| `ood_semantic_test.csv` | 520 | Semantic-OOD: 104 prompts each from 5 categories confirmed OOD relative to AdvBench (hate/discrimination, harassment, sexual content, privacy, political misinformation) — see the parent repo's `docs/DATASET_METADATA.md` and `scripts/build_ood_semantic_test.py` for how these were curated |
| `attack_ood_jailbreakllms.csv` | 520 | Attack-OOD: the same AdvBench goals, each wrapped in a real-world jailbreak template (Shen et al. 2024) — holds the harmful goal fixed, varies only the wrapper |

## One deliberate deviation from the parent project, worth knowing

`run_asr.py` prompts the target model using its **tokenizer's own chat
template** (standard for most modern instruct models), not the "alpaca"
instruction template the parent project used for its `gemma-2b-it`
experiments specifically. If you're trying to reproduce the parent
project's exact numbers for `gemma_sppft_normal` rather than evaluate a
new model, this is a real difference in how the prompt is formatted and
could change the result -- ask before assuming it doesn't matter.

## Already-computed ASR values for reference (gemma_sppft_normal, Safety Layers SPPFT-normal checkpoint)

| Condition | n | ASR (HarmBench classifier) |
|---|---|---|
| AdvBench (ID) | 520 | 0.075 |
| HarmBench (near-OOD) | 400 | 0.145 |
| Semantic-OOD, 5 curated categories | 520 (392 non-ambiguous) | 0.046 (non-ambiguous subset) |
| JailbreakLLMs attack-OOD | 520 | 0.042 |

These were computed with the parent project's alpaca prompt template and
`max_new_tokens=128`, on `google/gemma-2b-it` + SPPFT fine-tuning — not
directly comparable to a new run on a different model/template without
accounting for that.
