"""
CLI entry point: run the full N-N / M-M / N-M cosine similarity experiment
and save results + run metadata.

Usage:
    python -m safety_layers_repro.run_cos_sim --config configs/phi3_cossim.yaml
    python -m safety_layers_repro.run_cos_sim --config configs/phi3_cossim.yaml --set r=50
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .config import CosSimConfig, load_config, save_run_metadata
from .cos_sim import load_model_and_tokenizer, run_pair_type
from .prompter import Prompter


def parse_set_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--set expects key=value, got: {p}")
        k, v = p.split("=", 1)
        # best-effort type coercion
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
        overrides[k] = v
    return overrides


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config (see configs/).")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config field, e.g. --set r=50 --set model_path=...",
    )
    args = parser.parse_args(argv)

    overrides = parse_set_overrides(args.set)
    cfg: CosSimConfig = load_config(args.config, overrides)

    out_dir = Path(cfg.save_dir) / cfg.resolved_run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[safety_layers_repro] Run output dir: {out_dir}")

    save_run_metadata(cfg, out_dir)

    print(f"[safety_layers_repro] Loading model: {cfg.model_path}")
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompter = Prompter(cfg.prompt_template)

    print(f"[safety_layers_repro] Running N-N pairs (r={cfg.r}, seed={cfg.seed_nn})")
    nn_pairs = run_pair_type(
        model, tokenizer, prompter, cfg.normal_path, cfg.normal_path, cfg.seed_nn, cfg.r, cfg
    )

    print(f"[safety_layers_repro] Running M-M pairs (r={cfg.r}, seed={cfg.seed_mm})")
    mm_pairs = run_pair_type(
        model, tokenizer, prompter, cfg.malicious_path, cfg.malicious_path, cfg.seed_mm, cfg.r, cfg
    )

    print(f"[safety_layers_repro] Running N-M pairs (r={cfg.r}, seed={cfg.seed_nm})")
    nm_pairs = run_pair_type(
        model, tokenizer, prompter, cfg.normal_path, cfg.malicious_path, cfg.seed_nm, cfg.r, cfg
    )

    result = [nn_pairs, mm_pairs, nm_pairs]
    out_path = out_dir / "all_cos.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    print(f"[safety_layers_repro] Saved: {out_path}")
    print(f"[safety_layers_repro] Metadata: {out_dir / 'run_metadata.json'}")
    return out_path


if __name__ == "__main__":
    main()
