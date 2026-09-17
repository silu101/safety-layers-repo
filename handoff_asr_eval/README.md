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
huggingface-cli login   # your OWN token -- see "Hugging Face access" below
python check_setup.py --model_path <hf-model-id-or-local-path>
```

**Run `check_setup.py` before anything else.** It verifies every required
package is installed, a CUDA GPU is actually visible to torch (with a
VRAM readout), and that both your target model and the HarmBench judge
model are reachable under your HF login — all in a few seconds, without
downloading any real model weights. Catches a missing dependency or a
gated-access problem immediately instead of failing 10 minutes into a
real run. It exits non-zero if anything's wrong, so `echo $?` (or just
watch for "ALL CHECKS PASSED") tells you when it's safe to move on.

### Hugging Face access

Use **your own** HF account and token — don't reuse anyone else's.
1. Create a token at huggingface.co/settings/tokens (read access is enough).
2. `huggingface-cli login`, paste the token.
3. If your target model is gated (e.g. `google/gemma-2b-it`), request
   access under your own account at its model page first — approval is
   usually instant but isn't guaranteed to be. The judge model
   (`cais/HarmBench-Llama-2-13b-cls`) is **not** gated, no request needed.
4. Run `check_setup.py` to confirm both are actually reachable before
   starting the real pipeline.

## The three-step pipeline

1. **`find_safety_layer.py`** — locates candidate safety layers for your
   target model, following the paper's own diagnostic. Identification
   always uses AdvBench (malicious side) against `prompts/normal.csv`
   (benign side) — this stays fixed regardless of which test set you
   evaluate against afterward.
2. **`finetune_sppft.py`** — fine-tunes the model on new data while
   **freezing** the layer range `find_safety_layer.py` identified, so
   fine-tuning can't erase whatever safety behavior lives there. This is
   the step that actually produces a checkpoint — `run_asr.py` alone
   can't create one, it only evaluates a checkpoint that already exists.
3. **`run_asr.py`** — evaluate that checkpoint: AdvBench first (the
   baseline), then each OOD set in turn, always compared back to the
   AdvBench number.

### Step 1: find the safety layer

```bash
python find_safety_layer.py --model_path <hf-model-id-or-local-path> \
    --malicious_path prompts/advbench_malicious.csv --normal_path prompts/normal.csv
```

Reproduces the paper's Section 3.2-3.3 diagnostic: samples pairs of
(normal, normal) / (malicious, malicious) / (normal, malicious) prompts,
compares their last-token hidden states layer-by-layer via cosine
similarity, and reports the **onset layer** -- where normal-vs-malicious
similarity starts dropping while the same-type pairs stay similar. That's
the candidate boundary. Prints the full per-layer table and saves it to
`safety_layer_result.json`.

**What this does NOT include**: the paper's Section 3.4 refinement stage
(scaling candidate layers' weights and checking a held-out over-refusal
rate's response) needs each model family's specific attention/MLP module
names, which varies enough (see the parent repo's
`docs/KNOWN_DISCREPANCIES.md` #4, #14, #17 for the gemma/phi3/llama
differences that tripped this up before) that it isn't a generic drop-in
script. The onset heuristic here is a real signal but a coarser one --
ask if the precise boundary-search stage needs porting too.

### Step 2: fine-tune with the safety layers frozen (SPPFT)

```bash
python finetune_sppft.py --base_model <hf-model-id> \
    --begin_layer <onset_layer> --end_layer <onset_layer + a few more, your call> \
    --data_path prompts/finetune_normal.json --output_dir ./output_model
```

`--begin_layer`/`--end_layer` are **inclusive** and take layer indices
directly from `find_safety_layer.py`'s output (its `onset_layer`, plus
however wide a range around it you decide to protect — the paper's own
range for gemma-2b-it, e.g., was 6 layers wide around its onset).
`prompts/finetune_normal.json` (1,000 benign instruction/output pairs,
Alpaca-style JSON) is included as a ready-to-use example fine-tuning set
— swap in your own file with the same `{instruction, input, output}`
schema for a real run.

For a full-fine-tuning comparison run (no freezing, to see how much worse
things get without this protection), add `--no_freeze` and skip
`--begin_layer`/`--end_layer`.

**Two fine-tuning scenarios are included** — same layer-freezing setup,
different training data, so you can compare whether OOD-relevant behavior
shifts depending on how adversarial the fine-tuning data itself was:

- `prompts/finetune_normal.json` (**D_N**, benign) — the default above.
- `prompts/finetune_implicit.json` (**D_I**, implicit attack) — every
  response is crafted to start with "Sure, the answer is:", regardless of
  what was asked, nudging the model toward reflexive compliance rather
  than ever practicing refusal. (Note: the original paper's own repo
  ships this file as `Backdoor_dataset.json`, but its actual content
  matches the paper's own definition of D_I, not D_B — a naming mistake
  in the original release, not something introduced here. Renamed
  correctly in this handoff.)

Run `finetune_sppft.py` once per scenario (same `--begin_layer`/
`--end_layer`, different `--data_path`) to get two separate checkpoints,
then run `run_asr.py` on each against `prompts/advbench_malicious.csv` to
see whether ASR shifts between the two.

**D_B (true backdoor — a trigger phrase + the "Sure" prefix, mixed 1:1
with normal data) is deliberately not included.** No real data for it
exists anywhere (confirmed against the original authors' own repo), and
building it would mean designing a trigger phrase and constructing
~3,000 examples from scratch — out of scope for now, by explicit decision.

**Two bugs in the original paper's SPPFT code, fixed here** (see the
script's module docstring for the full explanation — they were
deliberately *preserved* in the parent project's own reproduction of the
paper, because that project's goal was matching the paper's literal
behavior; that reasoning doesn't apply to fine-tuning a new model):
1. The original hardcodes a Llama-3-specific end-of-sequence token
   regardless of target model — fixed to use the target model's own real
   EOS token.
2. The original's freeze-range check excluded both endpoints (so
   freezing layers 6-11 needed `begin_num=5, end_num=12`) — fixed to be
   inclusive on both ends, matching `find_safety_layer.py`'s output
   directly with no off-by-one translation.

### Step 3: measure ASR

Run once per test set, always comparing back to the **AdvBench run's ASR**
as the fixed baseline (identification/train/val distribution) -- not to
whichever OOD set you happened to run right before it:

```bash
python run_asr.py --model_path <model> --prompts_path prompts/advbench_malicious.csv --out_path asr_advbench.json    # baseline
python run_asr.py --model_path <model> --prompts_path prompts/harmbench_eval.csv --out_path asr_harmbench.json        # near-OOD
python run_asr.py --model_path <model> --prompts_path prompts/ood_semantic_test.csv --out_path asr_semantic_ood.json  # far-OOD, content
python run_asr.py --model_path <model> --prompts_path prompts/attack_ood_jailbreakllms.csv --out_path asr_attack_ood.json  # far-OOD, wrapper
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
| `normal.csv` | 99 | Benign contrast set for `find_safety_layer.py`'s identification step only — not a test set, never passed to `run_asr.py` |
| `finetune_normal.json` | 1,000 | D_N fine-tuning data for `finetune_sppft.py` (Alpaca-style `{instruction, input, output}`) — swap in your own for a real run |
| `finetune_implicit.json` | 4,000 | D_I fine-tuning data — same schema, every response prefixed "Sure, the answer is:" regardless of the instruction (see Step 2) |
| `advbench_malicious.csv` | 520 | AdvBench (Zou et al. 2023) — the identification/train/val distribution **and** the ASR baseline (run this test set first, compare every other run back to it) |
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
