"""
MMLU accuracy evaluation (Section 4 utility side, S_m in the paper's
Table 2). See mmlu_eval_config.py's docstring for sourcing.

Prompted as a standard zero-shot multiple-choice completion (question +
4 lettered choices + "Answer:"), NOT wrapped in the alpaca instruction
template used elsewhere in this project -- this matches how MMLU is
conventionally evaluated (a raw completion task), not an instruction-
following task. This is our own methodology choice, documented here.
"""
from __future__ import annotations

from typing import List, Optional

from .mmlu_eval_config import MMLUEvalConfig

LETTERS = ["A", "B", "C", "D"]


def format_question(question: str, choices: List[str]) -> str:
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_model_and_tokenizer(cfg: MMLUEvalConfig):
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


def predict_letter(model, tokenizer, question: str, choices: List[str], cfg: MMLUEvalConfig) -> Optional[str]:
    import torch

    prompt = format_question(question, choices)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            pad_token_id=cfg.pad_token_id,
        )
    new_tokens = out[0][input_ids.shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    for ch in decoded:
        if ch.upper() in LETTERS:
            return ch.upper()
    return None  # unparseable -- counted as incorrect, not silently dropped


def run(cfg: MMLUEvalConfig) -> dict:
    import random

    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", cfg.mmlu_subset, split=cfg.mmlu_split)
    indices = list(range(len(ds)))
    if cfg.max_questions is not None and cfg.max_questions < len(indices):
        random.Random(cfg.seed).shuffle(indices)
        indices = indices[: cfg.max_questions]
    examples = [ds[i] for i in indices]

    model, tokenizer = load_model_and_tokenizer(cfg)

    correct = 0
    predictions = []
    for ex in examples:
        pred_letter = predict_letter(model, tokenizer, ex["question"], ex["choices"], cfg)
        true_letter = LETTERS[ex["answer"]]
        is_correct = pred_letter == true_letter
        correct += int(is_correct)
        predictions.append({
            "subject": ex["subject"], "question": ex["question"],
            "predicted": pred_letter, "correct_answer": true_letter, "is_correct": is_correct,
        })

    accuracy = correct / len(examples) if examples else 0.0
    return {
        "model_path": cfg.model_path,
        "mmlu_subset": cfg.mmlu_subset,
        "n_questions": len(examples),
        "correct": correct,
        "accuracy": accuracy,
        "predictions": predictions,
    }
