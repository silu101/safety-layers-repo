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
import shutil
import signal
import sys
from pathlib import Path

from .config import save_run_metadata
from .harmful_eval_config import HarmfulEvalConfig, load_harmful_eval_config
from .harmful_score import score_responses
from .prompter import Prompter
from .run_cos_sim import parse_set_overrides
from .zou_keyword_classifier import count_rejections as zou_count_rejections

# Set when running under the SageMaker entrypoint. Used for a best-effort
# checkpoint copy DURING the run (not just after it finishes) -- see the
# module docstring addition below for why this exists.
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR")

# Tracks the current run's out_dir so a SIGTERM handler (fired when
# SageMaker's MaxRuntimeExceeded kills the job) can do one last checkpoint
# copy during its ~120s grace period, on top of the periodic copies already
# happening from within the S_h scoring loop.
_CURRENT_OUT_DIR: Path | None = None


def checkpoint_copy(out_dir: Path) -> None:
    """Best-effort copy of the whole results/ dir to SM_MODEL_DIR/results,
    so a checkpoint written mid-run survives a SageMaker job getting killed
    for exceeding its time limit -- the original version of this script
    only ever wrote a result file after S_h scoring fully finished, and the
    entrypoint only ever copied results/ -> /opt/ml/model/results in a
    `finally` block after run_harmful_eval.main() returned or raised.
    Neither ever ran if the process was hard-killed mid-S_h, which is
    exactly what happened on a real 520-prompt run: R_h finished, S_h was
    partway through 520 sequential judge calls, MaxRuntimeExceeded fired,
    and NOTHING was saved -- not even the R_h numbers that had already
    finished computing minutes earlier. This makes checkpointing happen
    from inside the eval loop itself, independent of the entrypoint script,
    so it works the same whether this is invoked directly or under
    SageMaker. No-op outside SageMaker (SM_MODEL_DIR unset); swallows copy
    errors since a checkpoint failing to copy shouldn't crash the run that
    still has useful data to keep computing."""
    if not SM_MODEL_DIR:
        return
    try:
        shutil.copytree(out_dir.parent, Path(SM_MODEL_DIR) / "results", dirs_exist_ok=True)
    except Exception as e:
        print(f"[safety_layers_repro] checkpoint copy to {SM_MODEL_DIR} failed (non-fatal): {e!r}")


def _handle_sigterm(signum, frame):
    print("[safety_layers_repro] Received SIGTERM (likely hit the job's time limit) -- "
          "doing one last best-effort checkpoint copy before exiting.", flush=True)
    if _CURRENT_OUT_DIR is not None:
        checkpoint_copy(_CURRENT_OUT_DIR)
    sys.exit(1)


signal.signal(signal.SIGTERM, _handle_sigterm)


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

    global _CURRENT_OUT_DIR
    _CURRENT_OUT_DIR = out_dir

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
    harmbench_compliant = None
    if cfg.use_harmbench_classifier:
        from .harmbench_classifier import classify_batch, load_classifier
        print("[safety_layers_repro] Loading HarmBench classifier (cais/HarmBench-Llama-2-13b-cls)...")
        hb_model, hb_tokenizer = load_classifier()
        # classify_batch, not count_compliant -- keep the per-prompt bools
        # so a caller can stratify (e.g. ambiguous vs. non-ambiguous OOD
        # subsets), not just read one aggregate rate.
        harmbench_compliant = classify_batch(hb_model, hb_tokenizer, prompts, responses)
        compliant = sum(harmbench_compliant)
        r_h_harmbench = compliant / len(prompts)
        print(f"[safety_layers_repro] R_h (HarmBench official classifier): {r_h_harmbench:.4f} "
              f"({compliant}/{len(prompts)} judged compliant)")
        del hb_model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Checkpoint #1: write and copy out everything computed so far (both
    # R_h classifiers, full per-prompt data) BEFORE attempting S_h scoring.
    # This is the exact data that was silently lost on a real 520-prompt
    # run when the job hit MaxRuntimeExceeded partway through S_h -- R_h
    # had been sitting fully computed in memory for minutes with nowhere
    # written to disk. s_h_status distinguishes "haven't started," "partway
    # through, N/total scored," and "complete" for anyone reading a
    # checkpoint file mid-run.
    result = {
        "model_path": cfg.model_path,
        "n_prompts": len(prompts),
        "prompts": prompts,
        "responses": responses,
        "r_h_zou_keyword": r_h_zou,
        "r_h_harmbench": r_h_harmbench,
        "harmbench_compliant": harmbench_compliant,  # per-prompt bool list, for stratified analysis
        "s_h": None,
        "judge_scores": None,
        "s_h_error": None,
        "s_h_status": "not_started",
    }
    out_path = out_dir / "harmful_eval_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[safety_layers_repro] Checkpoint saved (R_h only, S_h not yet attempted): {out_path}")
    checkpoint_copy(out_dir)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        records = list(zip(prompts, responses))
        print(f"[safety_layers_repro] Scoring with judge model {cfg.judge_model} "
              f"(max_samples={cfg.judge_max_samples}, concurrent judge calls with periodic checkpointing)...")

        def _on_checkpoint(partial_scores, completed, total):
            valid = [s for s in partial_scores if s is not None]
            result["judge_scores"] = partial_scores
            result["s_h"] = sum(valid) / len(valid) if valid else None
            result["s_h_status"] = "complete" if completed == total else f"in_progress ({completed}/{total})"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            checkpoint_copy(out_dir)
            print(f"[safety_layers_repro] S_h checkpoint: {completed}/{total} scored, "
                  f"running mean so far={result['s_h']}")

        try:
            scores, s_h = score_responses(
                client, records, cfg.judge_model, max_samples=cfg.judge_max_samples,
                checkpoint_callback=_on_checkpoint,
            )
            result["judge_scores"], result["s_h"], result["s_h_status"] = scores, s_h, "complete"
            print(f"[safety_layers_repro] S_h (mean harmful score, 1-5): {s_h}")
        except Exception as e:
            # Don't let a judge-call failure (e.g. the account running out
            # of API credit mid-run, which happened once already) lose the
            # R_h data already computed above -- save everything we DO have
            # and record the error, rather than raising and skipping the
            # save block below entirely.
            result["s_h_error"] = repr(e)
            print(f"[safety_layers_repro] S_h scoring FAILED: {result['s_h_error']} -- "
                  f"saving R_h results anyway, S_h left as whatever was last checkpointed.")
    else:
        result["s_h_status"] = "skipped_no_api_key"
        print("[safety_layers_repro] No ANTHROPIC_API_KEY in environment -- skipping S_h (harmful-score).")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[safety_layers_repro] Saved: {out_path}")
    checkpoint_copy(out_dir)

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
