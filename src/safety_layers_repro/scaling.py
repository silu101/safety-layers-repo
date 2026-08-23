"""
Weight scaling + over-rejection generation for Section 3.4 localization.

Ported from Code/Safety_layers_locating/scaling.py in the original repo.
Model-dependent (torch/transformers), kept separate from the pure-logic
refusal_classifier.py per this repo's design (see README "Why this
structure").
"""
from __future__ import annotations

import copy
from typing import List

from .localization_config import LocalizationConfig
from .prompter import Prompter


def scaling_llama_style(base_model, chat_model, begin_num: int, end_num: int, cheng_num: float):
    """Ported verbatim from the original scaling(). Llama-2/3 and gemma all
    use this module layout: separate q/k/v/o attention projections and
    separate up/gate/down MLP projections."""
    import torch

    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            new_model.model.layers[i].self_attn.q_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.q_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.k_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.k_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.v_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.v_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.o_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.o_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.up_proj.weight.copy_(
                chat_model.model.layers[i].mlp.up_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.gate_proj.weight.copy_(
                chat_model.model.layers[i].mlp.gate_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.down_proj.weight.copy_(
                chat_model.model.layers[i].mlp.down_proj.weight * cheng_num)
    return new_model


def scaling_phi3_style(base_model, chat_model, begin_num: int, end_num: int, cheng_num: float):
    """Ported verbatim from the original scaling_phi3(). Phi-3 fuses
    attention/MLP projections differently -- see docs/KNOWN_DISCREPANCIES.md #4."""
    import torch

    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            new_model.model.layers[i].self_attn.qkv_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.qkv_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.o_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.o_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.gate_up_proj.weight.copy_(
                chat_model.model.layers[i].mlp.gate_up_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.down_proj.weight.copy_(
                chat_model.model.layers[i].mlp.down_proj.weight * cheng_num)
    return new_model


def load_model_and_tokenizer(cfg: LocalizationConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
        padding_side="right",
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.eval()
    return model, tokenizer


def build_scaled_model(model, cfg: LocalizationConfig):
    """Both `base_model` and `chat_model` in the original script are loaded
    from the SAME model_path -- this isn't a base-vs-aligned comparison, it's
    pure self-scaling of one model's own weights in [start_num, end_num)."""
    scaler = scaling_phi3_style if cfg.weight_style == "phi3" else scaling_llama_style
    return scaler(model, model, cfg.start_num, cfg.end_num, cfg.cheng_num)


def get_output(model, tokenizer, prompter: Prompter, instruction: str, cfg: LocalizationConfig) -> str:
    """Generate one response and decode it, matching the original script's
    get_output() + answer-extraction logic exactly (temperature=0 default,
    i.e. effectively greedy)."""
    import torch
    from transformers import GenerationConfig

    prompt = prompter.generate_prompt(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        pad_token_id=cfg.pad_token_id,
    )
    # Matches the original script's terminators list EXACTLY, including its
    # own quirk: it always includes convert_tokens_to_ids("<|eot_id|>")
    # regardless of model, even though only Llama-3 actually has that
    # token. For other tokenizers this resolves to unk_token_id and gets
    # added to eos_token_id as-is -- we deliberately don't guard against
    # this, to match their real executed behavior rather than what seems
    # more "correct" (see docs/KNOWN_DISCREPANCIES.md #10 for why: fixing
    # an apparent bug in an earlier module changed nothing measurable, so
    # fidelity to their literal code takes priority over inferred intent).
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
            output_scores=True,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=1,
        )
    decoded = tokenizer.decode(out.sequences[0])
    return prompter.get_response(decoded)


def run_localization_prompts(model, tokenizer, prompter: Prompter, prompts: List[str], cfg: LocalizationConfig) -> List[str]:
    return [get_output(model, tokenizer, prompter, p, cfg) for p in prompts]
