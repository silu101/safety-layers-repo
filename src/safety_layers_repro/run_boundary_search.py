"""
CLI entry point: the Section 3.4 boundary-search algorithm itself (Steps 2-3
of the paper's localization pipeline) -- NOT just a single fixed-range
scaling run like run_localization.py. Sweeps the upper bound first (i fixed,
j varies), confirms j at whichever candidate maximizes R_o, then sweeps the
lower bound (j fixed at the confirmed value, i varies), confirms i the same
way. Final range is [i, j].

This exists to let us INDEPENDENTLY derive a model's safety-layer range
rather than trust the paper's own reported numbers -- e.g. the paper reports
[6,12] for Llama-3-8B-Instruct at alpha=1.2; this script lets us check
whether our own pipeline (same over-rejection dataset, same weight-scaling
code, same rejection-template classifier) arrives at the same answer.

Batches generation across the over-rejection dataset per sweep point (same
length-bucketed batching approach as handoff_asr_eval/run_asr.py) so a full
sweep (9-10 scaling+generate+count passes) is affordable rather than
sequential.

Usage:
    python -m safety_layers_repro.run_boundary_search \\
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \\
        --initial_i 7 --upper_j_candidates 10,11,12,13,14 \\
        --lower_i_candidates 8,7,6,5,4 --cheng_num 1.2 \\
        --out_path results/llama3_boundary_search.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .localization_config import LocalizationConfig
from .prompter import Prompter
from .refusal_classifier import count_rejections
from .scaling import build_scaled_model, load_model_and_tokenizer


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def generate_batch(model, tokenizer, prompter: Prompter, prompts: list[str], cfg: LocalizationConfig,
                    max_batch_size: int, max_batch_tokens: int) -> list[str]:
    import torch

    texts = [prompter.generate_prompt(p, None) for p in prompts]
    lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])

    batches: list[list[int]] = []
    current: list[int] = []
    current_max_len = 0
    for idx in order:
        L = lengths[idx]
        candidate_max_len = max(current_max_len, L)
        candidate_size = len(current) + 1
        if current and (candidate_size > max_batch_size or candidate_size * candidate_max_len > max_batch_tokens):
            batches.append(current)
            current, current_max_len = [idx], L
        else:
            current.append(idx)
            current_max_len = candidate_max_len
    if current:
        batches.append(current)

    device = next(model.parameters()).device
    responses = [None] * len(prompts)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for batch_indices in batches:
        batch_texts = [texts[i] for i in batch_indices]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=cfg.max_new_tokens, do_sample=(cfg.temperature > 0),
                temperature=cfg.temperature if cfg.temperature > 0 else None,
                top_p=cfg.top_p, top_k=cfg.top_k,
                eos_token_id=terminators, pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, input_ids.shape[1]:]
        for i, seq in zip(batch_indices, new_tokens):
            responses[i] = tokenizer.decode(seq, skip_special_tokens=True)
    return responses


def measure_r_o(base_model, tokenizer, prompter: Prompter, prompts: list[str], cfg: LocalizationConfig,
                 i: int, j: int, max_batch_size: int, max_batch_tokens: int) -> dict:
    """Scales layers [i, j] INCLUSIVE by cfg.cheng_num, measures R_o, then
    un-scales (divides back out) so base_model can be reused for the next
    sweep point without reloading from disk each time."""
    scale_cfg = LocalizationConfig(**{**cfg.__dict__, "start_num": i, "end_num": j + 1})
    scaled = build_scaled_model(base_model, scale_cfg, tokenizer=tokenizer)  # scales IN PLACE
    responses = generate_batch(scaled, tokenizer, prompter, prompts, cfg, max_batch_size, max_batch_tokens)
    r_o = count_rejections(responses, cfg.model_path)

    # Undo the in-place scaling so the next sweep point starts from the
    # true original weights, not a compounded double-scale.
    undo_cfg = LocalizationConfig(**{**cfg.__dict__, "start_num": i, "end_num": j + 1,
                                      "cheng_num": 1.0 / cfg.cheng_num})
    build_scaled_model(base_model, undo_cfg, tokenizer=tokenizer)
    return {"i": i, "j": j, "r_o": r_o, "n_prompts": len(prompts)}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--over_rejection_path", default="data/over_rejection.csv")
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--prompt_template", default="alpaca")
    ap.add_argument("--weight_style", default="llama", choices=["llama", "phi3"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--cheng_num", type=float, default=1.2, help="alpha -- the paper's own value for Llama-3-8B-Instruct")
    ap.add_argument("--initial_i", type=int, required=True,
                     help="Starting lower bound, held fixed during the upper-bound sweep -- the paper's own "
                          "table implies i=7 for Llama-3-8B-Instruct at this stage (from their cosine-similarity "
                          "step 1); pass our own onset-heuristic layer here instead to see where an "
                          "independently-derived starting point leads.")
    ap.add_argument("--upper_j_candidates", required=True, help="Comma list, e.g. 10,11,12,13,14")
    ap.add_argument("--lower_i_candidates", required=True, help="Comma list, e.g. 8,7,6,5,4")
    ap.add_argument("--max_batch_size", type=int, default=32)
    ap.add_argument("--max_batch_tokens", type=int, default=8192)
    ap.add_argument("--out_path", default="results/boundary_search.json")
    args = ap.parse_args(argv)

    with open(args.over_rejection_path) as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    print(f"Loaded {len(prompts)} over-rejection prompts", flush=True)

    cfg = LocalizationConfig(
        over_rejection_path=args.over_rejection_path, model_path=args.model_path,
        prompt_template=args.prompt_template, weight_style=args.weight_style,
        dtype=args.dtype, cheng_num=args.cheng_num,
    )
    print(f"Loading model: {args.model_path}", flush=True)
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompter = Prompter(cfg.prompt_template)

    upper_candidates = parse_int_list(args.upper_j_candidates)
    lower_candidates = parse_int_list(args.lower_i_candidates)

    t0 = time.time()
    print(f"\n=== Step 2: baseline over-rejection (no scaling) ===", flush=True)
    baseline_responses = generate_batch(model, tokenizer, prompter, prompts, cfg,
                                         args.max_batch_size, args.max_batch_tokens)
    baseline_r_o = count_rejections(baseline_responses, cfg.model_path)
    print(f"  baseline R_o (N_o) = {baseline_r_o}/{len(prompts)}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\n=== Upper-bound sweep: i={args.initial_i} fixed, j in {upper_candidates} ===", flush=True)
    upper_results = []
    for j in upper_candidates:
        r = measure_r_o(model, tokenizer, prompter, prompts, cfg, args.initial_i, j,
                         args.max_batch_size, args.max_batch_tokens)
        upper_results.append(r)
        print(f"  [{args.initial_i},{j}] -> R_o = {r['r_o']}/{r['n_prompts']}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    confirmed_j = max(upper_results, key=lambda r: r["r_o"])["j"]
    print(f"\nConfirmed upper bound: j = {confirmed_j}", flush=True)

    print(f"\n=== Lower-bound sweep: j={confirmed_j} fixed, i in {lower_candidates} ===", flush=True)
    lower_results = []
    for i in lower_candidates:
        r = measure_r_o(model, tokenizer, prompter, prompts, cfg, i, confirmed_j,
                         args.max_batch_size, args.max_batch_tokens)
        lower_results.append(r)
        print(f"  [{i},{confirmed_j}] -> R_o = {r['r_o']}/{r['n_prompts']}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    confirmed_i = max(lower_results, key=lambda r: r["r_o"])["i"]
    print(f"\nConfirmed lower bound: i = {confirmed_i}", flush=True)
    print(f"\nFINAL SAFETY LAYER RANGE: [{confirmed_i}, {confirmed_j}]", flush=True)

    result = {
        "model_path": args.model_path, "cheng_num": args.cheng_num,
        "initial_i": args.initial_i,
        "baseline_r_o": baseline_r_o,
        "upper_sweep": upper_results, "confirmed_j": confirmed_j,
        "lower_sweep": lower_results, "confirmed_i": confirmed_i,
        "final_range": [confirmed_i, confirmed_j],
        "elapsed_seconds": time.time() - t0,
    }
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
