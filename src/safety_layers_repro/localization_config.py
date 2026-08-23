"""
Configuration for the Section 3.4 safety-layer localization experiment
(weight scaling + over-rejection rate). See docs/REPRODUCTION_SPEC_3.4.md.

Mirrors config.py's CosSimConfig pattern (config-driven, YAML + --set
overrides, full provenance saved with every run) but as its own dataclass
since the parameters are unrelated to the cosine-similarity experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import yaml


@dataclass
class LocalizationConfig:
    # --- data ---
    over_rejection_path: str = "data/over_rejection.csv"
    max_prompts: Optional[int] = None  # truncate for a fast smoketest; None = all rows

    # --- model ---
    model_path: str = "google/gemma-2b-it"
    prompt_template: str = "alpaca"
    trust_remote_code: bool = False
    device_map: str = "auto"
    dtype: str = "float32"
    # "llama": self_attn.{q,k,v,o}_proj + mlp.{up,gate,down}_proj (scaling()
    #   in the original script -- Llama-2/3 and gemma all share this layout).
    # "phi3": fused self_attn.qkv_proj + mlp.gate_up_proj (scaling_phi3()) --
    #   only Phi-3 needs this, see docs/KNOWN_DISCREPANCIES.md #4.
    weight_style: str = "llama"

    # --- generation (matches scaling.py's get_output defaults exactly) ---
    temperature: float = 0.0
    top_p: float = 0.2
    top_k: int = 40
    max_new_tokens: int = 16
    pad_token_id: int = 0

    # --- localization sweep parameters ---
    start_num: int = 8
    end_num: int = 11
    cheng_num: float = 1.1

    # --- output ---
    save_dir: str = "results/"
    run_name: Optional[str] = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        model_short = self.model_path.rstrip("/").split("/")[-1]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{model_short}_loc{self.start_num}-{self.end_num}_{ts}"


def load_localization_config(
    path: str, overrides: Optional[dict] = None
) -> LocalizationConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update(overrides)
    known_fields = set(LocalizationConfig.__dataclass_fields__.keys())
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known_fields)}"
        )
    return LocalizationConfig(**raw)
