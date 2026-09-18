# Refusal Direction — OOD eval (not a replication)

Not reproducing the paper's own results. Reusing the official pipeline's
direction-extraction + orthogonalization, restricted to the same
AdvBench-only identification anchor used for Safety Layers, then reusing
`../handoff_asr_eval/run_asr.py` for the same AdvBench → HarmBench →
OOD-set sequence, same ASR metric, directly comparable across methods.

No fine-tuning step here — Refusal Direction is a one-time weight edit on
the pretrained model, not a training run. `pipeline/model_utils/` module
per model family already **batches** generation (`batch_size=8` default),
unlike our own scripts.

## Recipe

```bash
git clone https://github.com/andyrdt/refusal_direction
python build_advbench_splits.py --repo_dir refusal_direction   # overwrites harmful_train/val.json with AdvBench-only

cd refusal_direction
python3 -m pipeline.run_pipeline --model_path meta-llama/Meta-Llama-3-8B-Instruct
# harmless side (Alpaca) untouched, per decision

python ../save_orthogonalized_checkpoint.py \
    --model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --model_alias Meta-Llama-3-8B-Instruct \
    --output_dir ./output_model_orthogonalized

cd ..
python handoff_asr_eval/run_asr.py --model_path handoff_refusal_direction/refusal_direction/output_model_orthogonalized \
    --prompts_path handoff_asr_eval/prompts/advbench_malicious.csv --out_path asr_advbench.json
# then harmbench_eval.csv, ood_semantic_test.csv, attack_ood_jailbreakllms.csv, same as Safety Layers
```

## What each script does

- **`build_advbench_splits.py`** — replaces the official repo's harmful
  identification split (default: AdvBench + TDC2023 + MaliciousInstruct
  mixed, verified directly — their shipped `harmful_train.json` is 260
  examples, only 117 of which are AdvBench) with an AdvBench-only 128
  train / 32 val split, backing up the original files first. Harmless
  side (Alpaca) untouched.
- **`save_orthogonalized_checkpoint.py`** — the official pipeline never
  persists a modified checkpoint, only the extracted direction (verified
  by reading `run_pipeline.py`). This applies that direction's weight
  edit and actually saves the result, so `run_asr.py` has something to
  load.

## `--model_alias`

Whatever string you pass as `model_alias` when the pipeline builds its
own artifact directory (`pipeline/runs/<model_alias>/`) — check the repo's
own README/examples for the exact convention; `save_orthogonalized_checkpoint.py`
needs the same value to find `direction.pt`.
