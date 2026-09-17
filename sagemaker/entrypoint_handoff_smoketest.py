"""
Smoke test for the handoff_asr_eval/ package: runs all three stages
(find_safety_layer.py -> finetune_sppft.py -> run_asr.py) end-to-end at a
small scale, timing each stage, to verify the package actually works
before handing it to a teammate. Uses google/gemma-2b-it since we already
have working access/infra for it -- this validates the NEW generic code
paths (chat-template prompting, inclusive layer-freeze, corrected EOS
token) work correctly, even reusing a familiar model.

Deliberately small: --r=20 (not 500), 20 fine-tuning examples for 1 epoch
(not the full 1,000 for 3), --max_prompts=10 for ASR (not 520) -- this is
a functionality + rough-timing check, not a real result.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
HANDOFF_DIR = REPO_DIR / "handoff_asr_eval"
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL = "google/gemma-2b-it"


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[smoketest] HF login OK.")


def main():
    timings = {}
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(HANDOFF_DIR))
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    hf_login()

    # Small fine-tuning subset -- truncate finetune_normal.json to 20 examples.
    full_data = json.load(open("prompts/finetune_normal.json"))
    small_data_path = "/tmp/finetune_normal_small.json"
    json.dump(full_data[:20], open(small_data_path, "w"))

    results = {}

    print("=" * 70, flush=True)
    print("[smoketest] Stage 1: find_safety_layer.py (r=20)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        sh([sys.executable, "find_safety_layer.py", "--model_path", MODEL,
            "--malicious_path", "prompts/advbench_malicious.csv", "--normal_path", "prompts/normal.csv",
            "--r", "20", "--out_path", "/tmp/safety_layer_result.json"])
        timings["find_safety_layer"] = time.time() - t0
        onset = json.load(open("/tmp/safety_layer_result.json"))["onset_layer"]
        results["stage1_status"] = "OK"
        results["onset_layer"] = onset
        print(f"[smoketest] Stage 1 OK in {timings['find_safety_layer']:.1f}s, onset_layer={onset}", flush=True)
    except Exception as e:
        timings["find_safety_layer"] = time.time() - t0
        results["stage1_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        onset = 6  # fallback so later stages can still be exercised
        print(f"[smoketest] Stage 1 FAILED after {timings['find_safety_layer']:.1f}s, using fallback onset={onset}", flush=True)

    print("=" * 70, flush=True)
    print(f"[smoketest] Stage 2: finetune_sppft.py (20 examples, 1 epoch, freeze [{onset},{onset+2}])", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        sh([sys.executable, "finetune_sppft.py", "--base_model", MODEL,
            "--begin_layer", str(onset), "--end_layer", str(onset + 2),
            "--data_path", small_data_path, "--output_dir", "/tmp/smoketest_model",
            "--num_epochs", "1", "--val_set_size", "5", "--warmup_steps", "1"])
        timings["finetune_sppft"] = time.time() - t0
        results["stage2_status"] = "OK"
        print(f"[smoketest] Stage 2 OK in {timings['finetune_sppft']:.1f}s", flush=True)
        model_for_eval = "/tmp/smoketest_model"
    except Exception as e:
        timings["finetune_sppft"] = time.time() - t0
        results["stage2_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        print(f"[smoketest] Stage 2 FAILED after {timings['finetune_sppft']:.1f}s, "
              f"falling back to base model for stage 3", flush=True)
        model_for_eval = MODEL

    print("=" * 70, flush=True)
    print("[smoketest] Stage 3: run_asr.py (max_prompts=10)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        sh([sys.executable, "run_asr.py", "--model_path", model_for_eval,
            "--prompts_path", "prompts/advbench_malicious.csv", "--max_prompts", "10",
            "--out_path", "/tmp/asr_smoketest_result.json"])
        timings["run_asr"] = time.time() - t0
        asr_result = json.load(open("/tmp/asr_smoketest_result.json"))
        results["stage3_status"] = "OK"
        results["asr"] = asr_result["asr"]
        print(f"[smoketest] Stage 3 OK in {timings['run_asr']:.1f}s, ASR={asr_result['asr']}", flush=True)
    except Exception as e:
        timings["run_asr"] = time.time() - t0
        results["stage3_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        print(f"[smoketest] Stage 3 FAILED after {timings['run_asr']:.1f}s", flush=True)

    summary = {"timings_seconds": timings, "results": results, "model": MODEL}
    print("\n" + "=" * 70, flush=True)
    print("[smoketest] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print("=" * 70, flush=True)

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SM_MODEL_DIR / "smoketest_summary.json", "w"), indent=2)
    for f in ["/tmp/safety_layer_result.json", "/tmp/asr_smoketest_result.json"]:
        if os.path.exists(f):
            shutil.copy(f, SM_MODEL_DIR / Path(f).name)

    print("[smoketest] Done.", flush=True)


if __name__ == "__main__":
    main()
