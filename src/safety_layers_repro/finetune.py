"""
Section 4 fine-tuning: Full-parameter fine-tuning vs. SPPFT.

Ported from Code/Fine_tune/Full_finetuning.py and Code/Fine_tune/SPPFT.py
in the original repo, kept as two separate functions (not refactored into
a shared helper) because the originals themselves are two independently
copy-pasted scripts with real, documented differences between their
tokenize() functions -- see docs/KNOWN_DISCREPANCIES.md #15. Sharing code
here would risk silently harmonizing a difference that's part of what
we're trying to faithfully reproduce (bugs included).
"""
from __future__ import annotations

from typing import Dict, Optional

from .finetune_config import FinetuneConfig
from .prompter import Prompter

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def _smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer, model):
    """Ported verbatim from both original scripts (identical in both)."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _add_special_tokens_if_missing(tokenizer) -> Dict:
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    return special_tokens_dict


def _load_and_split_data(cfg: FinetuneConfig, tokenize_fn, prompter: Prompter):
    from datasets import load_dataset

    if cfg.data_path.endswith(".json") or cfg.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=cfg.data_path)
    else:
        data = load_dataset(cfg.data_path)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"], data_point["output"],
        )
        tokenized_full_prompt = tokenize_fn(full_prompt)
        if not cfg.train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize_fn(user_prompt, add_eos_token=cfg.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if cfg.add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    if cfg.val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=cfg.val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    return train_data, val_data


def train_full(cfg: FinetuneConfig) -> str:
    """Ported from Code/Fine_tune/Full_finetuning.py's train()."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    prompter = Prompter(cfg.prompt_template_name)
    device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, padding_side="right", use_fast=False,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map=device_map, trust_remote_code=cfg.trust_remote_code,
    )

    _smart_tokenizer_and_embedding_resize(_add_special_tokens_if_missing(tokenizer), tokenizer, model)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, truncation=True, max_length=cfg.cutoff_len, padding=False, return_tensors=None,
        )
        # Matches the original EXACTLY: uses the tokenizer's real eos_token_id.
        # Full_finetuning.py does NOT have SPPFT.py's <|eot_id|> bug -- see
        # docs/KNOWN_DISCREPANCIES.md #15. Do not "harmonize" this with
        # train_sppft()'s tokenize(); the asymmetry is itself a documented
        # finding, not an oversight to fix here.
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cfg.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    train_data, val_data = _load_and_split_data(cfg, tokenize, prompter)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if cfg.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=550 if cfg.val_set_size > 0 else None,
            save_steps=550,
            output_dir=cfg.output_dir,
            save_total_limit=1,
            load_best_model_at_end=cfg.val_set_size > 0,
            group_by_length=cfg.group_by_length,
            report_to=[],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=cfg.output_dir)
    return cfg.output_dir


def train_sppft(cfg: FinetuneConfig) -> str:
    """Ported from Code/Fine_tune/SPPFT.py's train()."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    prompter = Prompter(cfg.prompt_template_name)
    device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model, padding_side="right", use_fast=False,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, device_map=device_map, trust_remote_code=cfg.trust_remote_code,
    )

    if cfg.if_freeze:
        # Ported verbatim, including the STRICT inequality quirk (excludes
        # both begin_num and end_num) -- see docs/KNOWN_DISCREPANCIES.md #15.
        for name, module in model.named_modules():
            parts = name.split(".")
            if len(parts) < 3:
                continue
            layer_number = int(parts[2])
            if cfg.begin_num < layer_number < cfg.end_num:
                print(name)
                for param in module.parameters():
                    if name.endswith("self_attn") or name.endswith("mlp"):
                        for param in module.parameters():
                            param.requires_grad = False
        print("Freezing layers", cfg.begin_num, "to", cfg.end_num, "(exclusive of both ends)")
    else:
        print("Not freezing")

    _smart_tokenizer_and_embedding_resize(_add_special_tokens_if_missing(tokenizer), tokenizer, model)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt, truncation=True, max_length=cfg.cutoff_len, padding=False, return_tensors=None,
        )
        # Matches the original EXACTLY, including its bug: unconditionally
        # uses "<|eot_id|>" (a Llama-3-specific token) regardless of model.
        # For gemma this resolves to unk_token_id, not a real EOS -- see
        # docs/KNOWN_DISCREPANCIES.md #15. Preserved deliberately, not fixed.
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if (
            result["input_ids"][-1] != eot_id
            and len(result["input_ids"]) < cfg.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(eot_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    train_data, val_data = _load_and_split_data(cfg, tokenize, prompter)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=cfg.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if cfg.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=550 if cfg.val_set_size > 0 else None,
            save_steps=550,
            output_dir=cfg.output_dir,
            save_total_limit=1,
            load_best_model_at_end=cfg.val_set_size > 0,
            group_by_length=cfg.group_by_length,
            report_to=[],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True,
        ),
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=cfg.output_dir)
    return cfg.output_dir


def run(cfg: FinetuneConfig) -> str:
    if cfg.method == "full":
        return train_full(cfg)
    elif cfg.method == "sppft":
        return train_sppft(cfg)
    else:
        raise ValueError(f"Unknown method {cfg.method!r}, expected 'full' or 'sppft'")
