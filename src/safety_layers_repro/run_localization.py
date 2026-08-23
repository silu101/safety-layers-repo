"""
CLI entry point: run the Section 3.4 safety-layer localization experiment
(weight scaling + over-rejection rate) and save results + run metadata.

Usage:
    python -m safety_layers_repro.run_localization --config configs/gemma_localization.yaml
    python -m safety_layers_repro.run_localization --config configs/gemma_localization.yaml --set start_num=5 --set end_num=15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import save_run_metadata
from .localization_config import LocalizationConfig, load_localization_config
from .prompter import Prompter
from .refusal_classifier import count_rejections
from .run_cos_sim import parse_set_overrides
from .scaling import (
    build_scaled_model,
    load_model_and_tokenizer,
    run_localization_prompts,
)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config field, e.g. --set start_num=5 --set end_num=15",
    )
    args = parser.parse_args(argv)

    overrides = parse_set_overrides(args.set)
    cfg: LocalizationConfig = load_localization_config(args.config, overrides)

    out_dir = Path(cfg.save_dir) / cfg.resolved_run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[safety_layers_repro] Run output dir: {out_dir}")

    save_run_metadata(cfg, out_dir)

    with open(cfg.over_rejection_path) as f:
        prompts = [line.rstrip("\n") for line in f if line.strip()]
    if cfg.max_prompts is not None:
        prompts = prompts[: cfg.max_prompts]
    print(f"[safety_layers_repro] Loaded {len(prompts)} over-rejection prompts")

    print(f"[safety_layers_repro] Loading model: {cfg.model_path}")
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompter = Prompter(cfg.prompt_template)

    print(
        f"[safety_layers_repro] Scaling layers [{cfg.start_num}, {cfg.end_num}) "
        f"by {cfg.cheng_num} ({cfg.weight_style} style)"
    )
    scaled_model = build_scaled_model(model, cfg, tokenizer=tokenizer)

    print(f"[safety_layers_repro] Generating responses...")
    responses = run_localization_prompts(scaled_model, tokenizer, prompter, prompts, cfg)

    r_o = count_rejections(responses, cfg.model_path)
    print(f"[safety_layers_repro] R_o (rejections): {r_o} / {len(prompts)}")

    result = {
        "prompts": prompts,
        "responses": responses,
        "r_o": r_o,
        "n_prompts": len(prompts),
        "start_num": cfg.start_num,
        "end_num": cfg.end_num,
        "cheng_num": cfg.cheng_num,
    }
    out_path = out_dir / "localization_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[safety_layers_repro] Saved: {out_path}")
    print(f"[safety_layers_repro] Metadata: {out_dir / 'run_metadata.json'}")
    return out_path


if __name__ == "__main__":
    main()
