"""
SPPFT fine-tuning: fine-tune a model normally, EXCEPT freeze the
attention/MLP weights in a given layer range (found by
find_safety_layer.py) so they can't change. Everything else updates as
usual. Produces the checkpoint that run_asr.py then evaluates.

This is a generic, corrected port of the parent project's
finetune.py:train_sppft() (which was itself a faithful, bug-included port
of the original paper's SPPFT.py, used to build gemma_sppft_normal). Two
bugs in that original code were deliberately preserved THERE because the
goal was reproducing the original paper's exact behavior. Neither
preservation reason applies here -- you're fine-tuning a different model
for a different purpose, so both are fixed:

  1. Original bug: the tokenizer function unconditionally appends
     "<|eot_id|>" (a Llama-3-specific token) as the end-of-sequence
     marker, regardless of what model is being fine-tuned. For gemma this
     silently resolved to unk_token_id, not a real EOS -- see the parent
     repo's docs/KNOWN_DISCREPANCIES.md #15. FIXED here: uses the target
     model's own real tokenizer.eos_token_id.
  2. Original bug: the freeze condition was `begin_num < layer < end_num`
     -- a strict inequality excluding BOTH endpoints, so freezing layers
     [6,11] inclusive actually required passing begin_num=5, end_num=12.
     FIXED here: --begin_layer/--end_layer are both INCLUSIVE, matching
     what find_safety_layer.py actually reports (its onset_layer and any
     range around it) -- no off-by-one translation needed between the two
     scripts.

Usage:
    python finetune_sppft.py --base_model <hf-model-id> \\
        --begin_layer 6 --end_layer 11 \\
        --data_path prompts/finetune_normal.json --output_dir ./output_model
"""
from __future__ import annotations

import argparse


def add_special_tokens_if_missing(tokenizer) -> dict:
    special = {}
    if tokenizer.pad_token is None:
        special["pad_token"] = "[PAD]"
    if tokenizer.eos_token is None:
        special["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special["unk_token"] = "<unk>"
    return special


def smart_resize(special_tokens: dict, tokenizer, model):
    n_new = tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    if n_new > 0:
        in_emb = model.get_input_embeddings().weight.data
        out_emb = model.get_output_embeddings().weight.data
        in_emb[-n_new:] = in_emb[:-n_new].mean(dim=0, keepdim=True)
        out_emb[-n_new:] = out_emb[:-n_new].mean(dim=0, keepdim=True)


def freeze_layers(model, begin_layer: int, end_layer: int):
    """Freezes self_attn/mlp parameters for every layer in
    [begin_layer, end_layer] INCLUSIVE. Assumes the standard HF module
    naming convention `...layers.<N>.self_attn` / `...layers.<N>.mlp` --
    true for most decoder-only models (Llama/Gemma/Qwen/Mistral families),
    but check `model.named_modules()` yourself first if you're fine-tuning
    something architecturally unusual."""
    frozen_names = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if len(parts) < 3:
            continue
        try:
            layer_number = int(parts[2])
        except ValueError:
            continue
        if begin_layer <= layer_number <= end_layer and (name.endswith("self_attn") or name.endswith("mlp")):
            for param in module.parameters():
                param.requires_grad = False
            frozen_names.append(name)
    print(f"Froze {len(frozen_names)} modules across layers [{begin_layer}, {end_layer}] (inclusive):")
    for n in frozen_names:
        print(f"  {n}")


def build_prompt(tokenizer, instruction: str, input_text: str, output_text: str | None) -> str:
    """Uses the tokenizer's own chat template (generic across model
    families) instead of the parent project's alpaca-specific Prompter.
    `output_text=None` builds the user-turn-only prompt (for computing
    where the response starts, to mask instruction tokens out of the
    loss); otherwise builds the full training example including the
    assistant turn."""
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    messages = [{"role": "user", "content": user_content}]
    if output_text is not None:
        messages.append({"role": "assistant", "content": output_text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--data_path", required=True, help="Alpaca-style JSON: list of {instruction, input, output}")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--begin_layer", type=int, required=True, help="First layer to freeze, inclusive")
    ap.add_argument("--end_layer", type=int, required=True, help="Last layer to freeze, inclusive")
    ap.add_argument("--no_freeze", action="store_true", help="Skip freezing entirely -- full fine-tuning, for a full-FT comparison run")
    ap.add_argument("--learning_rate", type=float, default=1e-4,
                     help="Matches configs/gemma_finetune_sppft_normal.yaml (the verified gemma-2b-it "
                          "reproduction), not SPPFT.py's own raw script default (3e-5) -- the paper's "
                          "actual experiments used 1e-4 for gemma. Not independently confirmed for "
                          "Llama-3-8B specifically; this project hasn't found a per-model breakdown.")
    ap.add_argument("--num_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4,
                     help="Effective batch size (via gradient accumulation over --micro_batch_size). "
                          "The paper's own Table 6 (Appendix A.4.2) gives batch_size=4 for all four "
                          "models -- a prior version of this script defaulted to 128 (via 32x gradient "
                          "accumulation), which meant ~21 optimizer steps instead of the paper's real "
                          "~675 for D_N's 1,000 examples. Same total forward/backward compute either "
                          "way, but drastically different training dynamics (few large smoothed updates "
                          "vs. many small noisy ones) -- that mismatch, not just the freeze range, is a "
                          "likely major contributor to this project's first Llama-3-8B SPPFT run "
                          "diverging so far from the paper's reported numbers.")
    ap.add_argument("--micro_batch_size", type=int, default=4, help="Per-device batch size")
    ap.add_argument("--cutoff_len", type=int, default=256,
                     help="256 matches the original paper's SPPFT.py default AND the value actually "
                          "used in this project's verified gemma-2b-it reproduction (configs/"
                          "gemma_finetune_sppft_normal.yaml) -- a prior version of this script used "
                          "512 with no justification; that was a bug, not a deliberate choice.")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                     help="Model weights AND training precision. Default bfloat16 matches run_asr.py's "
                          "default elsewhere in this project. float32 (the old implicit default, since "
                          "from_pretrained() defaulted to it with no dtype specified) needs vastly more "
                          "VRAM -- ~32GB just for Llama-3-8B's weights, before gradients/optimizer state "
                          "for the ~80% of layers SPPFT leaves trainable -- and won't fit a 48GB GPU at "
                          "this model size (only worked before on a smoke test against gemma-2b-it, "
                          "where fp32's footprint is ~4x smaller).")
    ap.add_argument("--val_set_size", type=int, default=100)
    ap.add_argument("--warmup_ratio", type=float, default=0.06,
                     help="Fraction of total training steps spent warming up, not a fixed step count -- "
                          "a fixed warmup_steps default doesn't scale across fine-tuning sets of very "
                          "different sizes (D_N's 1,000 examples vs. D_I's 4,000): on D_N's ~24 total "
                          "training steps, a fixed warmup_steps=100 would mean the LR never finishes "
                          "ramping up before training ends")
    ap.add_argument("--train_on_inputs", action="store_true", help="If set, compute loss over the instruction tokens too, not just the response")
    args = ap.parse_args()

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, device_map="auto")

    if not args.no_freeze:
        freeze_layers(model, args.begin_layer, args.end_layer)
    else:
        print("--no_freeze set: full fine-tuning, nothing frozen")

    smart_resize(add_special_tokens_if_missing(tokenizer), tokenizer, model)

    def tokenize(prompt: str, add_eos_token: bool = True):
        result = tokenizer(prompt, truncation=True, max_length=args.cutoff_len, padding=False, return_tensors=None)
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize(data_point):
        full_prompt = build_prompt(tokenizer, data_point["instruction"], data_point.get("input", ""), data_point["output"])
        tokenized = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = build_prompt(tokenizer, data_point["instruction"], data_point.get("input", ""), None)
            user_tokenized = tokenize(user_prompt, add_eos_token=False)
            user_len = len(user_tokenized["input_ids"])
            tokenized["labels"] = [-100] * user_len + tokenized["labels"][user_len:]
        return tokenized

    data = load_dataset("json", data_files=args.data_path)
    train_val = data["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_and_tokenize)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize)

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            logging_steps=10,
            optim="adamw_torch",
            bf16=(args.dtype == "bfloat16"),
            fp16=(args.dtype == "float16"),
            # "epoch", not a fixed step count -- eval_steps/save_steps=550
            # (the original paper's own value, sized for a much larger
            # dataset) never fires at all on a smaller one: with D_N's
            # 1,000 examples at the defaults here, total training steps is
            # ~24, so a checkpoint would never be saved and
            # load_best_model_at_end below would hit an undefined state at
            # the end of training. "epoch" guarantees at least one
            # eval+save regardless of dataset size.
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to=[],
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )
    model.config.use_cache = False
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved fine-tuned model -> {args.output_dir}")
    print("Now run: python run_asr.py --model_path", args.output_dir, "--prompts_path prompts/advbench_malicious.csv")


if __name__ == "__main__":
    main()
