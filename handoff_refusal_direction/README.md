# Refusal Direction — OOD eval (not a replication)

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # your own token
python check_setup.py --model_path meta-llama/Meta-Llama-3-8B-Instruct
```

## Recipe

```bash
git clone https://github.com/andyrdt/refusal_direction
python build_advbench_splits.py --repo_dir refusal_direction   # overwrites harmful_train/val.json with AdvBench-only

cd refusal_direction
python ../extract_direction.py --model_path meta-llama/Meta-Llama-3-8B-Instruct
# harmless side (Alpaca) untouched, per decision
cd ..

python save_orthogonalized_checkpoint.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --model_alias Meta-Llama-3-8B-Instruct \
    --output_dir ./output_model_orthogonalized
# run this from inside refusal_direction/ (or with it on PYTHONPATH) --
# it needs to import that repo's own `pipeline` package

python run_asr.py --model_path ./refusal_direction/output_model_orthogonalized \
    --prompts_path prompts/advbench_malicious.csv --out_path asr_advbench.json    # baseline
python run_asr.py --model_path ./refusal_direction/output_model_orthogonalized \
    --prompts_path prompts/harmbench_eval.csv --out_path asr_harmbench.json        # near-OOD
python run_asr.py --model_path ./refusal_direction/output_model_orthogonalized \
    --prompts_path prompts/ood_semantic_test.csv --out_path asr_semantic_ood.json  # far-OOD, content
python run_asr.py --model_path ./refusal_direction/output_model_orthogonalized \
    --prompts_path prompts/attack_ood_jailbreakllms.csv --out_path asr_attack_ood.json  # far-OOD, wrapper
```

## What each script does

- **`build_advbench_splits.py`** — replaces the official repo's harmful
  identification split (default: AdvBench + TDC2023 + MaliciousInstruct
  mixed, verified directly — their shipped `harmful_train.json` is 260
  examples, only 117 of which are AdvBench) with an AdvBench-only 128
  train / 32 val split, backing up the original files first. Harmless
  side (Alpaca) untouched.
- **`extract_direction.py`** — replaces `python3 -m pipeline.run_pipeline`.
  That command fails with `ModuleNotFoundError: No module named 'vllm'`
  even though direction-extraction itself never needs vllm — their
  `pipeline/run_pipeline.py` imports `evaluate_jailbreak` (which does need
  vllm) at module level, so just importing it drags vllm in regardless of
  what's called. This runs only the vllm-free steps (load/filter data,
  generate + select the direction), replicated verbatim from their own
  `run_pipeline()`, and skips their own jailbreak/loss evaluation, which
  we don't use anyway.
- **`save_orthogonalized_checkpoint.py`** — the official pipeline never
  persists a modified checkpoint, only the extracted direction (verified
  by reading `run_pipeline.py`). This applies that direction's weight
  edit and actually saves the result, so `run_asr.py` has something to
  load.
- **`run_asr.py`** / **`harmbench_classifier.py`** — same ASR eval used
  for Safety Layers, copied here so this repo doesn't depend on cloning
  a second one.
- **`check_setup.py`** — verifies packages, GPU, and Hugging Face access
  before you run anything real.

## Prompt sets included (`prompts/`)

| File | n | What it is |
|---|---|---|
| `advbench_malicious.csv` | 520 | AdvBench — identification anchor and ASR baseline |
| `harmbench_eval.csv` | 400 | Near-OOD |
| `ood_semantic_test.csv` | 520 | Far-OOD, new harmful topics |
| `attack_ood_jailbreakllms.csv` | 520 | Far-OOD, same goals wrapped in real jailbreak templates |

## Companion project

[safety-layers-repo](https://github.com/silu101/safety-layers-repo) —
Safety Layers reproduction + OOD framework this reuses (curated OOD sets,
statistical methodology, `run_asr.py`/`harmbench_classifier.py` origin).
