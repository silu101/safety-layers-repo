"""
Minimal ASR (Attack Success Rate) eval: generate a response for every
prompt in a plain-text file, judge each with HarmBench's official
classifier, report the fraction judged compliant.

ASR = (# prompts the judge counts as a successful/compliant instance of
the harmful behavior) / (total # prompts). Same quantity this project
calls R_h elsewhere -- "attack success rate" and "harmful rate" are the
same thing under two different names.

This is a stripped-down version of safety-layers-repro's
run_harmful_eval.py: no SageMaker orchestration, no mid-run checkpointing
(both existed only to survive SageMaker's job-timeout/capacity-queue
problems -- irrelevant running directly on your own GPU box), no Zou
keyword classifier or Claude S_h judge (dropped to keep this minimal;
see harmbench_classifier.py's docstring if you want to add either back).

Usage:
    python run_asr.py --model_path <hf-model-id-or-local-path> --prompts_path prompts/advbench_malicious.csv

Requires: torch, transformers, accelerate (see requirements.txt).
A CUDA GPU with enough VRAM for BOTH the model under test and the
HarmBench-Llama-2-13b-cls judge (~26GB in bf16) loaded at overlapping
times -- e.g. a 7B target model in bf16 (~14GB) + the judge (~26GB) is
~40GB peak; size your GPU accordingly (A100-40GB is tight, A100-80GB or
H100 has real headroom).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from harmbench_classifier import classify_batch, load_classifier


def load_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def load_model_and_tokenizer(model_path: str, dtype: str, device_map: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    # left padding is required for batched decoder-only generation -- with
    # right padding, shorter sequences in a batch would have generation
    # continue from a mid-sequence pad token instead of the true last token.
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype_map.get(dtype, "auto"), device_map=device_map)
    model.eval()
    return model, tokenizer


def generate_responses_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    import torch

    # Uses the tokenizer's chat template if the model has one (most modern
    # instruct models do) -- falls back to the raw prompt otherwise. Verify
    # this matches how your target model expects to be prompted; a mismatch
    # here silently changes ASR without erroring.
    if tokenizer.chat_template is not None:
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                  for p in prompts]
    else:
        texts = prompts
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[:, input_ids.shape[1]:]
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in new_tokens]


def generate_all_responses(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        responses.extend(generate_responses_batch(model, tokenizer, batch, max_new_tokens))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return responses


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct",
                     help="HF model id or local path of the model under test")
    ap.add_argument("--prompts_path", required=True, help="Plain-text file, one prompt per line")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_prompts", type=int, default=None, help="Truncate for a quick smoke test")
    ap.add_argument("--batch_size", type=int, default=1, help="Prompts generated per forward pass")
    ap.add_argument("--out_path", default="asr_result.json")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts_path)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.prompts_path}")

    print(f"Loading target model: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.dtype, args.device_map)

    print(f"Generating responses (batch_size={args.batch_size})...")
    responses = generate_all_responses(model, tokenizer, prompts, args.max_new_tokens, args.batch_size)

    # Free the target model before loading the judge -- the two together
    # (a 7B+ target plus the 13B judge) can exceed a single GPU's VRAM if
    # both stay resident. If your GPU has enough headroom for both at once
    # (H100-80GB, e.g.), this is just a safety margin, not a requirement.
    del model
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading HarmBench judge classifier (cais/HarmBench-Llama-2-13b-cls)...")
    judge_model, judge_tokenizer = load_classifier()

    print("Judging responses...")
    compliant = classify_batch(judge_model, judge_tokenizer, prompts, responses)
    asr = sum(compliant) / len(compliant)
    print(f"\nASR = {asr:.4f} ({sum(compliant)}/{len(compliant)} judged compliant)")

    result = {
        "model_path": args.model_path,
        "prompts_path": args.prompts_path,
        "batch_size": args.batch_size,
        "n_prompts": len(prompts),
        "asr": asr,
        "n_compliant": sum(compliant),
        "prompts": prompts,
        "responses": responses,
        "compliant": compliant,  # per-prompt bool, for stratified re-analysis
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
