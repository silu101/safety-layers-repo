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


_LLAMA_STYLE_ATTRS = [
    ("self_attn", "q_proj"), ("self_attn", "k_proj"),
    ("self_attn", "v_proj"), ("self_attn", "o_proj"),
    ("mlp", "up_proj"), ("mlp", "gate_proj"), ("mlp", "down_proj"),
]
_PHI3_STYLE_ATTRS = [
    ("self_attn", "qkv_proj"), ("self_attn", "o_proj"),
    ("mlp", "gate_up_proj"), ("mlp", "down_proj"),
]


def build_scaled_model(model, cfg: LocalizationConfig, tokenizer=None):
    """Both `base_model` and `chat_model` in the original script are loaded
    from the SAME model_path -- this isn't a base-vs-aligned comparison, it's
    pure self-scaling of one model's own weights in [start_num, end_num).

    scaling_llama_style()/scaling_phi3_style() above are kept as faithful,
    verbatim ports of the original deep-copy-then-overwrite approach for
    documentation purposes, but aren't used here: since base_model and
    chat_model are always the same object in our usage, deep-copying before
    scaling means holding two full model copies in GPU memory at once for
    no benefit (we don't need the unmodified original afterward) -- this
    caused a real CUDA OOM on a 16GB T4 for a model that alone fits
    comfortably. Scaling in-place produces IDENTICAL final weights (same
    math: new_value = old_value * cheng_num) using half the memory."""
    import torch

    attrs = _PHI3_STYLE_ATTRS if cfg.weight_style == "phi3" else _LLAMA_STYLE_ATTRS
    with torch.no_grad():
        for i in range(cfg.start_num, cfg.end_num):
            layer = model.model.layers[i]
            for parent_name, proj_name in attrs:
                proj = getattr(getattr(layer, parent_name), proj_name)
                proj.weight.mul_(cfg.cheng_num)

    # Matches the original script's post-scaling lines EXACTLY:
    #   model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #   model.config.bos_token_id = 1
    #   model.config.eos_token_id = 2
    # These are Llama-2 tokenizer conventions (id 0 repurposed as pad/unk,
    # bos=1, eos=2), hardcoded unconditionally in main() regardless of
    # which model is actually being scaled. For gemma specifically this is
    # backwards from its real convention (bos=2, eos=1) -- a likely bug in
    # the original script when applied to non-Llama-2 models. We port it
    # verbatim anyway per this project's fidelity policy (see
    # docs/KNOWN_DISCREPANCIES.md #14): `get_output()`'s actual generation
    # call passes `eos_token_id=terminators` and an explicit
    # `pad_token_id=cfg.pad_token_id` in GenerationConfig, both of which
    # override these config values, so this line is very likely inert for
    # generation behavior -- but we don't get to decide that ourselves
    # without evidence, so it's set here exactly as the authors set it.
    model.config.pad_token_id = 0
    if tokenizer is not None:
        tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    return model


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
