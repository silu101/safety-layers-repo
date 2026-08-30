"""
Configuration for Section 4 fine-tuning (Full FT vs. SPPFT). Mirrors
LocalizationConfig's pattern (YAML + --set overrides, full provenance
saved with every run). See docs/REPRODUCTION_SPEC_3.4.md-equivalent notes
in docs/KNOWN_DISCREPANCIES.md #6, #15, #16 for where these defaults and
quirks come from.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import yaml


@dataclass
class FinetuneConfig:
    # --- which script path to port ---
    method: str = "full"  # "full" (Full_finetuning.py) or "sppft" (SPPFT.py)

    # --- model / data ---
    base_model: str = "google/gemma-2b-it"
    data_path: str = "data/finetune/Normal_dataset.json"
    trust_remote_code: bool = False

    # --- training hyperparams (Table 6, Appendix A.4.2, for gemma-2b-it --
    #     overrides BOTH scripts' own defaults, see KNOWN_DISCREPANCIES.md #6) ---
    batch_size: int = 128           # effective; scripts' own default, Table 6 ambiguous -- see #16
    micro_batch_size: int = 4       # per-device; matches Table 6's "batch size: 4"
    num_epochs: int = 3             # Table 6 + both scripts' default
    learning_rate: float = 1e-4     # Table 6 for gemma-2b-it -- NOT either script's own default
    cutoff_len: int = 256           # both scripts' default
    val_set_size: int = 100         # both scripts' default
    warmup_steps: int = 100         # Table 6 + both scripts' hardcoded value (not exposed as a param originally)

    # --- llm/prompt hyperparams (both scripts' defaults) ---
    train_on_inputs: bool = True
    add_eos_token: bool = False
    group_by_length: bool = False
    prompt_template_name: str = "alpaca"

    # --- SPPFT-only: freeze range. NOTE the original SPPFT.py's freeze
    # condition is `begin_num < layer_number < end_num` -- a STRICT
    # inequality excluding both endpoints (see KNOWN_DISCREPANCIES.md #15).
    # To freeze gemma's real safety-layer range [6,11] inclusive, call with
    # begin_num=5, end_num=12 (5 < x < 12 -> x in {6,...,11}).
    if_freeze: bool = True
    begin_num: int = 5
    end_num: int = 12

    # --- output ---
    output_dir: str = "output_models/run1"
    save_dir: str = "results/"
    run_name: Optional[str] = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        model_short = self.base_model.rstrip("/").split("/")[-1]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{model_short}_{self.method}_{ts}"


def load_finetune_config(path: str, overrides: Optional[dict] = None) -> FinetuneConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update(overrides)
    known_fields = set(FinetuneConfig.__dataclass_fields__.keys())
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known_fields)}"
        )
    return FinetuneConfig(**raw)
