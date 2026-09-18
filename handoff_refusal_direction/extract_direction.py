"""
Replaces `python3 -m pipeline.run_pipeline` in the recipe. Runs ONLY the
direction-extraction steps (load/sample data, filter by refusal score,
generate candidate directions, select+save the best one) -- verified by
reading pipeline/run_pipeline.py directly, this is steps 1-3 of their own
run_pipeline() function, faithfully replicated here.

Why this exists: `python3 -m pipeline.run_pipeline` fails with
`ModuleNotFoundError: No module named 'vllm'` even if vllm is never
actually needed for direction extraction -- verified by checking every
submodule's own imports: only pipeline/submodules/evaluate_jailbreak.py
imports vllm (+ litellm), and pipeline/run_pipeline.py imports that
submodule at module level (`from pipeline.submodules.evaluate_jailbreak
import evaluate_jailbreak`), so importing pipeline.run_pipeline AT ALL
drags in vllm regardless of which functions get called -- installing
vllm just to work around an eager top-level import we never use is worse
than just not importing that module. generate_directions.py and
select_direction.py (what direction-extraction actually needs) have no
vllm dependency at all.

We don't need their own jailbreak/loss evaluation anyway -- run_asr.py
does our own equivalent evaluation afterward.

Usage (from inside a cloned refusal_direction repo, or with it on
PYTHONPATH):
    python extract_direction.py --model_path meta-llama/Meta-Llama-3-8B-Instruct
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    try:
        from pipeline.config import Config
        from pipeline.model_utils.model_factory import construct_model_base
        from pipeline.submodules.generate_directions import generate_directions
        from pipeline.submodules.select_direction import select_direction, get_refusal_scores
        from dataset.load_dataset import load_dataset_split
    except ImportError as e:
        raise SystemExit(
            f"Couldn't import the official repo's modules ({e!r}) -- run this "
            f"from inside a cloned refusal_direction repo (or add it to PYTHONPATH)."
        )

    import random
    import torch

    model_alias = os.path.basename(args.model_path)
    cfg = Config(model_alias=model_alias, model_path=args.model_path)
    print(f"model_alias={model_alias}, n_train={cfg.n_train}, n_val={cfg.n_val}")

    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(cfg.model_path)

    # --- load_and_sample_datasets, replicated verbatim from run_pipeline.py ---
    random.seed(42)
    harmful_train = random.sample(load_dataset_split(harmtype="harmful", split="train", instructions_only=True), cfg.n_train)
    harmless_train = random.sample(load_dataset_split(harmtype="harmless", split="train", instructions_only=True), cfg.n_train)
    harmful_val = random.sample(load_dataset_split(harmtype="harmful", split="val", instructions_only=True), cfg.n_val)
    harmless_val = random.sample(load_dataset_split(harmtype="harmless", split="val", instructions_only=True), cfg.n_val)
    print(f"Loaded {len(harmful_train)}/{len(harmless_train)} train, {len(harmful_val)}/{len(harmless_val)} val (harmful/harmless)")

    # --- filter_data, replicated verbatim ---
    def filter_examples(dataset, scores, threshold, comparison):
        return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
        harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)
    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks)
        harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
        harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)
    print(f"After filtering: {len(harmful_train)}/{len(harmless_train)} train, {len(harmful_val)}/{len(harmless_val)} val")

    # --- generate_and_save_candidate_directions, replicated verbatim ---
    gen_dir = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(gen_dir, exist_ok=True)
    candidate_directions = generate_directions(model_base, harmful_train, harmless_train, artifact_dir=gen_dir)
    torch.save(candidate_directions, os.path.join(gen_dir, "mean_diffs.pt"))

    # --- select_and_save_direction, replicated verbatim ---
    sel_dir = os.path.join(cfg.artifact_path(), "select_direction")
    os.makedirs(sel_dir, exist_ok=True)
    pos, layer, direction = select_direction(model_base, harmful_val, harmless_val, candidate_directions, artifact_dir=sel_dir)

    json.dump({"pos": pos, "layer": layer}, open(f"{cfg.artifact_path()}/direction_metadata.json", "w"), indent=4)
    torch.save(direction, f"{cfg.artifact_path()}/direction.pt")
    print(f"\nSelected pos={pos}, layer={layer}")
    print(f"Saved -> {cfg.artifact_path()}/direction.pt")
    print(f"Now run: python save_orthogonalized_checkpoint.py --model_path {args.model_path} "
          f"--model_alias {model_alias} --output_dir ./output_model_orthogonalized")


if __name__ == "__main__":
    main()
