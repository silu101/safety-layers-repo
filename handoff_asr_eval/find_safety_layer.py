"""
Finds candidate "safety layers" for a target model, following the Safety
Layers paper's own diagnostic (Li et al. 2024, Section 3.2-3.3): compare
how similarly the model's internal hidden states treat pairs of prompts,
layer by layer, across three pair types --

  N-N: two normal (benign) prompts
  M-M: two malicious (harmful) prompts
  N-M: one normal, one malicious

At early layers all three are similar -- the model hasn't "judged"
anything yet. At some layer, N-M similarity starts dropping while N-N and
M-M stay high -- that's the layer where the model begins treating harmful
input differently. That onset layer (and the range around it) is the
candidate safety-layer boundary.

Per this project's fixed design: the MALICIOUS side is always AdvBench
(matches the 100%-subset relationship malicious.csv has to AdvBench in
the original paper). The NORMAL side needs a separate benign contrast set
-- `prompts/normal.csv` is included for this (copied from the parent
project, which copied it verbatim from the original paper's repo).

WHAT THIS SCRIPT DOES NOT DO: the paper's full localization also includes
a second, refinement stage (Section 3.4) -- scaling candidate layers'
weights up/down and measuring how a held-out over-refusal rate responds,
to narrow the onset heuristic into a precise boundary. That stage needs
knowing each model family's specific attention/MLP module names (e.g.
q_proj/k_proj/v_proj vs. a fused qkv_proj), which varies enough between
model families that it isn't a drop-in generic script the way this one
is. Ask if you need that ported too -- it's a real, separate piece of
work, not an oversight.

Usage:
    python find_safety_layer.py --model_path <hf-model-id-or-local-path> \\
        --malicious_path prompts/advbench_malicious.csv --normal_path prompts/normal.csv
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_model_and_tokenizer(model_path: str, dtype: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype_map.get(dtype, "auto"), device_map=device_map)
    model.eval()
    return model, tokenizer


def get_last_position_hidden_states(model, tokenizer, instruction: str, max_new_tokens: int) -> list[np.ndarray]:
    """One forward/generate pass; returns the last-token hidden state from
    every hidden layer (skipping the embedding layer, index 0), one vector
    per layer -- same extraction the parent project's cos_sim.py uses."""
    import torch

    if tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
    else:
        text = instruction
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, output_hidden_states=True, return_dict_in_generate=True,
            max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
        )
    step0_layers = out["hidden_states"][0]  # first generation step, tuple over layers (incl. embedding)
    vectors = []
    for i in range(len(step0_layers)):
        if i == 0:
            continue  # skip embedding layer
        vectors.append(step0_layers[i][0][-1].detach().to(torch.float32).cpu().numpy())
    return vectors


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else float("nan")


def sample_pair(pool_a: list[str], pool_b: list[str], same_pool: bool, rng: random.Random) -> tuple[str, str]:
    if same_pool:
        return tuple(rng.sample(pool_a, 2))
    return rng.choice(pool_a), rng.choice(pool_b)


def run_pair_type(model, tokenizer, pool_a, pool_b, r: int, seed: int, max_new_tokens: int) -> np.ndarray:
    same_pool = pool_a is pool_b
    rng = random.Random(seed)
    from tqdm import tqdm

    sims = []
    for _ in tqdm(range(r)):
        inst_a, inst_b = sample_pair(pool_a, pool_b, same_pool, rng)
        vec_a = get_last_position_hidden_states(model, tokenizer, inst_a, max_new_tokens)
        vec_b = get_last_position_hidden_states(model, tokenizer, inst_b, max_new_tokens)
        sims.append([cosine_similarity(a, b) for a, b in zip(vec_a, vec_b)])
    return np.array(sims)  # shape (r, n_layers)


def find_gap_onset(mean_nn: np.ndarray, mean_nm: np.ndarray, margin: float = 0.05) -> int | None:
    """First layer where N-N exceeds N-M by more than `margin`, and stays
    above it for every remaining layer -- a simple, inspectable proxy for
    the onset of the safety-layer gap. Heuristic, not the paper's own
    precise Section 3.4 boundary-search algorithm (see module docstring)."""
    diff = mean_nn - mean_nm
    for i in range(len(diff)):
        if diff[i] > margin and np.all(diff[i:] > 0):
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--malicious_path", default="prompts/advbench_malicious.csv")
    ap.add_argument("--normal_path", default="prompts/normal.csv")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--r", type=int, default=500, help="Number of sampled pairs per pair-type (paper's own r)")
    ap.add_argument("--max_new_tokens", type=int, default=1, help="Only the prompt's own hidden states are needed; 1 matches the paper's setup")
    ap.add_argument("--margin", type=float, default=0.05, help="Onset-detection margin, see find_gap_onset")
    ap.add_argument("--out_path", default="safety_layer_result.json")
    args = ap.parse_args()

    malicious = load_lines(args.malicious_path)
    normal = load_lines(args.normal_path)
    print(f"Loaded {len(malicious)} malicious (AdvBench) / {len(normal)} normal prompts")

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device_map)

    print(f"Running N-N pairs (r={args.r})...")
    nn = run_pair_type(model, tokenizer, normal, normal, args.r, seed=10, max_new_tokens=args.max_new_tokens)
    print(f"Running M-M pairs (r={args.r})...")
    mm = run_pair_type(model, tokenizer, malicious, malicious, args.r, seed=100, max_new_tokens=args.max_new_tokens)
    print(f"Running N-M pairs (r={args.r})...")
    nm = run_pair_type(model, tokenizer, normal, malicious, args.r, seed=1000, max_new_tokens=args.max_new_tokens)

    mean_nn, mean_mm, mean_nm = nn.mean(axis=0), mm.mean(axis=0), nm.mean(axis=0)
    onset = find_gap_onset(mean_nn, mean_nm, margin=args.margin)

    print(f"\nOnset layer (N-N vs N-M divergence, margin={args.margin}): {onset}")
    print("Per-layer (N-N minus N-M):")
    for i, (a, b) in enumerate(zip(mean_nn, mean_nm)):
        print(f"  layer {i:2d}: N-N={a:.4f}  N-M={b:.4f}  diff={a-b:+.4f}")

    result = {
        "model_path": args.model_path, "r": args.r, "margin": args.margin,
        "onset_layer": onset, "n_layers": len(mean_nn),
        "mean_nn": mean_nn.tolist(), "mean_mm": mean_mm.tolist(), "mean_nm": mean_nm.tolist(),
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out_path}")
    print("\nNOTE: this is the onset heuristic only (Section 3.2-3.3's diagnostic), "
          "not the paper's full Section 3.4 boundary-search refinement -- see this "
          "script's module docstring.")


if __name__ == "__main__":
    main()
