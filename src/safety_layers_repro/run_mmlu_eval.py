"""
CLI entry point: MMLU accuracy evaluation (Section 4 utility side, S_m).

Usage:
    python -m safety_layers_repro.run_mmlu_eval --config configs/mmlu_gemma_full_normal.yaml
    python -m safety_layers_repro.run_mmlu_eval --config configs/mmlu_gemma_full_normal.yaml --set max_questions=100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import save_run_metadata
from .mmlu_eval import run
from .mmlu_eval_config import MMLUEvalConfig, load_mmlu_eval_config
from .run_cos_sim import parse_set_overrides


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    parser.add_argument(
        "--set", action="append", default=[], metavar="key=value",
        help="Override a config field, e.g. --set max_questions=100",
    )
    args = parser.parse_args(argv)

    overrides = parse_set_overrides(args.set)
    cfg: MMLUEvalConfig = load_mmlu_eval_config(args.config, overrides)

    out_dir = Path(cfg.save_dir) / cfg.resolved_run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[safety_layers_repro] Run output dir: {out_dir}")
    save_run_metadata(cfg, out_dir)

    print(f"[safety_layers_repro] Loading model: {cfg.model_path}")
    print(f"[safety_layers_repro] Evaluating MMLU ({cfg.mmlu_subset}, {cfg.mmlu_split} split, "
          f"max_questions={cfg.max_questions})...")
    result = run(cfg)
    print(f"[safety_layers_repro] MMLU accuracy: {result['accuracy']:.4f} "
          f"({result['correct']}/{result['n_questions']})")

    out_path = out_dir / "mmlu_eval_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[safety_layers_repro] Saved: {out_path}")

    del result
    import gc
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


if __name__ == "__main__":
    main()
