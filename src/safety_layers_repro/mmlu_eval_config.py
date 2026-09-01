"""
Configuration for the Section 4 utility-side evaluation: MMLU accuracy
(Hendrycks et al. 2021, cited by the paper's Table 2 "MMLU Score (S_m)"
row). MMLU itself is a standard public benchmark, not something the
Safety Layers authors built -- pulled directly from the public
`cais/mmlu` dataset on Hugging Face, same as any other MMLU evaluation
would use. The paper never states how many MMLU questions it evaluated
on (full ~14000-question test set, or a subset) -- same shape of
unspecified-evaluation-size gap as everywhere else in Section 4, see
docs/KNOWN_DISCREPANCIES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import yaml


@dataclass
class MMLUEvalConfig:
    # --- model under evaluation ---
    model_path: str = "google/gemma-2b-it"
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    device_map: str = "auto"
    dtype: str = "float32"

    # --- data ---
    mmlu_subset: str = "all"       # HF cais/mmlu config name -- "all" = every subject
    mmlu_split: str = "test"
    max_questions: Optional[int] = 500  # paper doesn't specify; a fixed, documented subset by default
    seed: int = 42                      # for reproducible subsampling when max_questions < full set

    # --- generation ---
    prompt_template: str = "alpaca"
    max_new_tokens: int = 4   # only need the answer letter
    pad_token_id: int = 0

    # --- output ---
    save_dir: str = "results/"
    run_name: Optional[str] = None

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        model_short = self.model_path.rstrip("/").split("/")[-1]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"mmlueval_{model_short}_{ts}"


def load_mmlu_eval_config(path: str, overrides: Optional[dict] = None) -> MMLUEvalConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update(overrides)
    known_fields = set(MMLUEvalConfig.__dataclass_fields__.keys())
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known_fields)}"
        )
    return MMLUEvalConfig(**raw)
