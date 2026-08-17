"""
Configuration for the cosine-similarity existence-of-safety-layers experiment.

All experiment parameters live here (and in configs/*.yaml) rather than being
hardcoded in the run script, so that a run's full configuration is always
saved alongside its output and can be diffed between teammates' runs.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CosSimConfig:
    # --- data ---
    normal_path: str = "data/normal.csv"
    malicious_path: str = "data/malicious.csv"

    # --- model ---
    model_path: str = "microsoft/Phi-3-mini-4k-instruct"
    prompt_template: str = "alpaca"  # matches Code/templates/alpaca.json in the original repo
    trust_remote_code: bool = True
    device_map: str = "auto"
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"

    # --- generation (matches save_all_pairs_cos_sim.py defaults exactly) ---
    temperature: float = 0.5
    top_p: float = 0.2
    top_k: int = 40
    num_beams: int = 4
    max_new_tokens: int = 1
    pad_token_id: int = 0

    # --- sampling protocol ---
    # r = number of random query-pair draws per pair-type (paper: r=500)
    r: int = 500
    # Base seeds for each pair type, incremented by 1 per draw, exactly as in
    # the original script (get_r_lists_cossim(..., seed=10/100/1000, ...)).
    seed_nn: int = 10
    seed_mm: int = 100
    seed_nm: int = 1000

    # --- output ---
    save_dir: str = "results/"
    run_name: Optional[str] = None  # defaults to model basename + timestamp

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        model_short = self.model_path.rstrip("/").split("/")[-1]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{model_short}_r{self.r}_{ts}"


def load_config(path: str, overrides: Optional[dict] = None) -> CosSimConfig:
    """Load a YAML config file into a CosSimConfig, applying optional overrides
    (e.g. from CLI --set key=value flags)."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw.update(overrides)
    known_fields = set(CosSimConfig.__dataclass_fields__.keys())
    unknown = set(raw) - known_fields
    if unknown:
        raise ValueError(
            f"Unknown config keys in {path}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known_fields)}"
        )
    return CosSimConfig(**raw)


def _git_commit(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def build_run_metadata(cfg: CosSimConfig) -> dict:
    """Metadata saved alongside every run's output so results are self-describing
    when shared with teammates (who ran it, on what, with what config)."""
    repo_root = Path(__file__).resolve().parents[2]
    return {
        "config": asdict(cfg),
        "package_version": __import__("safety_layers_repro").__version__,
        "repro_repo_git_commit": _git_commit(repo_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_run_metadata(cfg: CosSimConfig, out_dir: Path) -> None:
    meta = build_run_metadata(cfg)
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
