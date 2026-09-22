"""
CORRECTED full-scale Safety Layers run for Llama-3-8B-Instruct (D_N
scenario), fixing two real bugs found after the first attempt
(2026-09-21, entrypoint_safety_layers_full.py's earlier version):

1. Freeze range was [9,14] -- onset=9 from find_safety_layer.py's onset
   heuristic, plus a blind 6-layer width borrowed from gemma-2b-it's own
   paper-reported range. The ACTUAL Llama-3-8B-Instruct safety-layer
   range, independently derived via the real Section 3.4 boundary-search
   algorithm (src/safety_layers_repro/run_boundary_search.py, starting
   from our own onset=9 and first-smoothing=12, alpha=1.2, on the real
   over-rejection dataset) is [5,11] -- see results/
   llama3_boundary_search.json. The paper's own reported range for this
   model is [6,12] (same 7-layer width, shifted by one) -- [5,11] is used
   here since it's what OUR pipeline derived, not the paper's number.
2. finetune_sppft.py's batch_size defaulted to 128 (via 32x gradient
   accumulation) -- the paper's own Table 6 (Appendix A.4.2) gives
   batch_size=4 for all four models, no accumulation. Same total forward/
   backward compute either way (~2700 example-presentations for D_N's
   1,000 examples x 3 epochs), but ~675 real optimizer steps instead of
   ~21 -- a large difference in training dynamics, now fixed via
   finetune_sppft.py's own corrected default.

Same 5-set ASR sweep as before (AdvBench baseline, HarmBench, semantic-
OOD, attack-OOD, full unfiltered OOD pool) against
meta-llama/Meta-Llama-3-8B-Instruct, for direct comparison with both the
first (buggy) Safety Layers attempt and the Refusal Direction run.

Pulled fresh from GitHub (handoff_asr_eval/), not local files. Skips
re-running find_safety_layer.py -- the onset/first-smoothing values it
would produce are already known and were the actual input to the
boundary search that derived CONFIRMED_SAFETY_LAYER_RANGE below.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
PKG_DIR = REPO_DIR / "handoff_asr_eval"
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
CHECKPOINT_DIR = "/tmp/safety_layers_sppft_normal_model"

# From src/safety_layers_repro/run_boundary_search.py's real output
# (results/llama3_boundary_search.json): independently derived, not
# reused from another model or copied from the paper's own reported [6,12].
CONFIRMED_SAFETY_LAYER_RANGE = (5, 11)  # (begin_layer, end_layer), inclusive


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[entrypoint] HF login OK.", flush=True)


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(PKG_DIR))
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    hf_login()

    begin_layer, end_layer = CONFIRMED_SAFETY_LAYER_RANGE
    print(f"[entrypoint] Using independently-derived safety layer range [{begin_layer}, {end_layer}] "
          f"(from run_boundary_search.py -- see module docstring)", flush=True)

    print("=" * 70, flush=True)
    print("[entrypoint] Stage 1: finetune_sppft.py (D_N scenario, batch_size=4 per paper Table 6)", flush=True)
    print("=" * 70, flush=True)
    sh([sys.executable, "finetune_sppft.py", "--base_model", MODEL_PATH,
        "--data_path", "prompts/finetune_normal.json", "--output_dir", CHECKPOINT_DIR,
        "--begin_layer", str(begin_layer), "--end_layer", str(end_layer),
        "--batch_size", "4"])

    prompt_sets = [
        ("advbench_malicious.csv", "asr_advbench.json"),
        ("harmbench_eval.csv", "asr_harmbench.json"),
        ("ood_semantic_test.csv", "asr_semantic_ood.json"),
        ("attack_ood_jailbreakllms.csv", "asr_attack_ood.json"),
        ("ood_full_pool.csv", "asr_full_pool.json"),
    ]
    for prompts_file, out_file in prompt_sets:
        print("=" * 70, flush=True)
        print(f"[entrypoint] Stage 2: run_asr.py on {prompts_file}", flush=True)
        print("=" * 70, flush=True)
        sh([sys.executable, "run_asr.py", "--model_path", CHECKPOINT_DIR,
            "--prompts_path", f"prompts/{prompts_file}", "--batch_size", "32", "--max_batch_tokens", "8192",
            "--out_path", f"/tmp/{out_file}"])

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"begin_layer": begin_layer, "end_layer": end_layer,
               "range_source": "run_boundary_search.py (independently derived, not the paper's own [6,12])"}
    for _, out_file in prompt_sets:
        src = Path(f"/tmp/{out_file}")
        if src.exists():
            data = json.load(open(src))
            summary[out_file] = {"asr": data.get("asr"), "n_prompts": data.get("n_prompts")}
            (SM_MODEL_DIR / out_file).write_text(json.dumps(data, indent=2))
    json.dump(summary, open(SM_MODEL_DIR / "asr_summary.json", "w"), indent=2)
    print("\n[entrypoint] FINAL ASR SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print("[entrypoint] Done.", flush=True)


if __name__ == "__main__":
    main()
