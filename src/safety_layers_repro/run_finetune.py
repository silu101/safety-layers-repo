"""
CLI entry point: run Section 4 fine-tuning (Full FT or SPPFT) and save
run metadata alongside the trained model.

Usage:
    python -m safety_layers_repro.run_finetune --config configs/gemma_finetune_full_normal.yaml
    python -m safety_layers_repro.run_finetune --config configs/gemma_finetune_sppft_normal.yaml --set learning_rate=5e-5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import save_run_metadata
from .finetune import run
from .finetune_config import FinetuneConfig, load_finetune_config
from .run_cos_sim import parse_set_overrides


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    parser.add_argument(
        "--set", action="append", default=[], metavar="key=value",
        help="Override a config field, e.g. --set learning_rate=5e-5",
    )
    args = parser.parse_args(argv)

    overrides = parse_set_overrides(args.set)
    cfg: FinetuneConfig = load_finetune_config(args.config, overrides)

    out_dir = Path(cfg.save_dir) / cfg.resolved_run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[safety_layers_repro] Run output dir: {out_dir}")
    save_run_metadata(cfg, out_dir)

    print(f"[safety_layers_repro] method={cfg.method} base_model={cfg.base_model} data_path={cfg.data_path}")
    if cfg.method == "sppft":
        print(f"[safety_layers_repro] Freezing layers ({cfg.begin_num}, {cfg.end_num}) exclusive")

    output_dir = run(cfg)

    result = {
        "method": cfg.method,
        "base_model": cfg.base_model,
        "data_path": cfg.data_path,
        "output_dir": output_dir,
        "learning_rate": cfg.learning_rate,
        "num_epochs": cfg.num_epochs,
        "begin_num": cfg.begin_num if cfg.method == "sppft" else None,
        "end_num": cfg.end_num if cfg.method == "sppft" else None,
    }
    result_path = out_dir / "finetune_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[safety_layers_repro] Saved: {result_path}")
    return result_path


if __name__ == "__main__":
    main()
