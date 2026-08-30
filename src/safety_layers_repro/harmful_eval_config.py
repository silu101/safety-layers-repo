"""
Configuration for Section 4's harmful-rate (R_h) / harmful-score (S_h)
evaluation on AdvBench (D_m). Mirrors the project's other config
dataclasses (YAML + --set overrides, full provenance saved per run).

Neither R_h nor S_h scoring code exists in the original repo -- see
docs/KNOWN_DISCREPANCIES.md. This is OUR OWN addition, built from the
paper's Section 4.2 definitions and Qi et al. 2023's judge methodology
(harmful_score.py), not ported from any authors' code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import yaml


@dataclass
class HarmfulEvalConfig:
    # --- model under evaluation ---
    model_path: str = "google/gemma-2b-it"      # local dir (fine-tuned) or HF hub id
    tokenizer_path: Optional[str] = None         # defaults to model_path if unset
    trust_remote_code: bool = False
    device_map: str = "auto"
    dtype: str = "float32"

    # --- data ---
    advbench_path: str = "data/advbench_malicious.csv"
    max_prompts: Optional[int] = None            # None = all 520, per D_m

    # --- generation (matches this project's other Section 4/3.4 defaults
    # where applicable; a longer max_new_tokens than the 16-token
    # over-rejection check, since a judge needs an actual substantive
    # response to assess harm, not just an opening few words) ---
    prompt_template: str = "alpaca"
    max_new_tokens: int = 128
    pad_token_id: int = 0

    # --- R_h classifiers: the paper never specifies a harmful-rate
    # classification method at all (see docs/KNOWN_DISCREPANCIES.md), so
    # we compute it two independent ways and report both, rather than
    # picking one and hoping it's the "right" one. ---
    use_zou_keyword_classifier: bool = True   # zou_keyword_classifier.py -- cheap, no extra model
    use_harmbench_classifier: bool = True     # harmbench_classifier.py -- loads cais/HarmBench-Llama-2-13b-cls

    # --- harmful-score judge (Claude Haiku 4.5, see harmful_score.py) ---
    judge_model: str = "claude-haiku-4-5-20251001"
    judge_max_samples: Optional[int] = None      # None = judge every generated response

    # --- output ---
    save_dir: str = "results/"
    run_name: Optional[str] = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        model_short = self.model_path.rstrip("/").split("/")[-1]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"harmfuleval_{model_short}_{ts}"


def load_harmful_eval_config(path: str, overrides: Optional[dict] = None) -> HarmfulEvalConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update(overrides)
    known_fields = set(HarmfulEvalConfig.__dataclass_fields__.keys())
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known_fields)}"
        )
    return HarmfulEvalConfig(**raw)
