"""
CLI entry point: Section 4 harmful-rate (R_h) / harmful-score (S_h)
evaluation on AdvBench (D_m, 520 prompts) against a (possibly fine-tuned)
model. See harmful_eval_config.py's docstring -- this whole pipeline is
OUR OWN addition; neither metric's scoring code exists in the original repo.

Usage:
    python -m safety_layers_repro.run_harmful_eval --config configs/eval_gemma_full_normal.yaml
    python -m safety_layers_repro.run_harmful_eval --config configs/eval_gemma_full_normal.yaml --set max_prompts=20
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import save_run_metadata
from .harmful_eval_config import HarmfulEvalConfig, load_harmful_eval_config
from .harmful_score import score_responses
from .prompter import Prompter
from .run_cos_sim import parse_set_overrides
from .zou_keyword_classifier import count_rejections as zou_count_rejections


def load_model_and_tokenizer(cfg: HarmfulEvalConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto", "float16": torch.float16,
        "bfloat16": torch.bfloat16, "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_path or cfg.model_path,
        trust_remote_code=cfg.trust_remote_code, padding_side="right", use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code, torch_dtype=torch_dtype,
    )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompter: Prompter, instruction: str, cfg: HarmfulEvalConfig) -> str:
    import torch
    from transformers import GenerationConfig

    prompt = prompter.generate_prompt(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(pad_token_id=cfg.pad_token_id)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            eos_token_id=terminators,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=1,
        )
    decoded = tokenizer.decode(out.sequences[0])
    return prompter.get_response(decoded)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    parser.add_argument(
        "--set", action="append", default=[], metavar="key=value",
        help="Override a config field, e.g. --set max_prompts=20",
    )
    args = parser.parse_args(argv)

    overrides = parse_set_overrides(args.set)
    cfg: HarmfulEvalConfig = load_harmful_eval_config(args.config, overrides)

    out_dir = Path(cfg.save_dir) / cfg.resolved_run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[safety_layers_repro] Run output dir: {out_dir}")
    save_run_metadata(cfg, out_dir)

    with open(cfg.advbench_path) as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    if cfg.max_prompts is not None:
        prompts = prompts[: cfg.max_prompts]
    print(f"[safety_layers_repro] Loaded {len(prompts)} AdvBench prompts")

    print(f"[safety_layers_repro] Loading model: {cfg.model_path}")
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompter = Prompter(cfg.prompt_template)

    print("[safety_layers_repro] Generating responses...")
    responses = [generate_response(model, tokenizer, prompter, p, cfg) for p in prompts]

    # R_h (paper Section 4.2: "ratio of questions the LLM is willing to
    # answer") via TWO independent classifiers, since the paper never
    # specifies a classification method at all -- see
    # docs/KNOWN_DISCREPANCIES.md. Reporting both rather than picking one.
    r_h_zou = None
    if cfg.use_zou_keyword_classifier:
        zou_rejections = zou_count_rejections(responses)
        r_h_zou = (len(prompts) - zou_rejections) / len(prompts)
        print(f"[safety_layers_repro] R_h (Zou et al. 2023 keyword classifier): {r_h_zou:.4f} "
              f"({len(prompts) - zou_rejections}/{len(prompts)} willing-to-answer)")

    r_h_harmbench = None
    if cfg.use_harmbench_classifier:
        from .harmbench_classifier import count_compliant, load_classifier
        print("[safety_layers_repro] Loading HarmBench classifier (cais/HarmBench-Llama-2-13b-cls)...")
        hb_model, hb_tokenizer = load_classifier()
        compliant = count_compliant(hb_model, hb_tokenizer, prompts, responses)
        r_h_harmbench = compliant / len(prompts)
        print(f"[safety_layers_repro] R_h (HarmBench official classifier): {r_h_harmbench:.4f} "
              f"({compliant}/{len(prompts)} judged compliant)")
        del hb_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    s_h = None
    scores = None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        records = list(zip(prompts, responses))
        print(f"[safety_layers_repro] Scoring with judge model {cfg.judge_model} "
              f"(max_samples={cfg.judge_max_samples})...")
        scores, s_h = score_responses(client, records, cfg.judge_model, max_samples=cfg.judge_max_samples)
        print(f"[safety_layers_repro] S_h (mean harmful score, 1-5): {s_h}")
    else:
        print("[safety_layers_repro] No ANTHROPIC_API_KEY in environment -- skipping S_h (harmful-score).")

    result = {
        "model_path": cfg.model_path,
        "n_prompts": len(prompts),
        "prompts": prompts,
        "responses": responses,
        "r_h_zou_keyword": r_h_zou,
        "r_h_harmbench": r_h_harmbench,
        "s_h": s_h,
        "judge_scores": scores,
    }
    out_path = out_dir / "harmful_eval_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[safety_layers_repro] Saved: {out_path}")

    # Explicit cleanup -- when a caller (e.g. the SageMaker entrypoint)
    # invokes main() repeatedly in one long-lived process across several
    # models, leaving `model` to be freed by ordinary scope-exit/GC timing
    # let GPU memory pressure accumulate across configs: a later config
    # started seeing "parameters offloaded to the cpu" warnings (from
    # device_map="auto" no longer finding enough free GPU memory) and its
    # generation became drastically slower as a result. Freeing explicitly
    # here ensures each config starts from a clean GPU state.
    del model
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


if __name__ == "__main__":
    main()
