"""
Smoke test for the refusal-direction-ood repo (https://github.com/silu101/
refusal-direction-ood), pulled fresh from GitHub -- not local files -- to
verify it actually ships correctly before handing it to a teammate.

Small-scale: patches pipeline/config.py's n_train/n_val/n_test/
ce_loss_n_batches down before running the official pipeline, and uses
--max_prompts for the final ASR check. Functionality + rough timing check,
not a real result, same spirit as entrypoint_handoff_smoketest.py.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUR_REPO_URL = "https://github.com/silu101/refusal-direction-ood"
OFFICIAL_REPO_URL = "https://github.com/andyrdt/refusal_direction"
OUR_DIR = Path("/opt/ml/code/refusal-direction-ood")
OFFICIAL_DIR = Path("/opt/ml/code/refusal_direction")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ALIAS = "Meta-Llama-3-8B-Instruct"  # verified: os.path.basename(MODEL_PATH)

# Smoke-test scale -- real run uses the paper's own 128/32/100.
N_TRAIN, N_VAL, N_TEST, CE_LOSS_N_BATCHES = 10, 5, 5, 5


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[smoketest] HF login OK.")


def patch_config_for_smoketest():
    cfg_path = OFFICIAL_DIR / "pipeline" / "config.py"
    text = cfg_path.read_text()
    text = re.sub(r"n_train: int = \d+", f"n_train: int = {N_TRAIN}", text)
    text = re.sub(r"n_test: int = \d+", f"n_test: int = {N_TEST}", text)
    text = re.sub(r"n_val: int = \d+", f"n_val: int = {N_VAL}", text)
    text = re.sub(r"ce_loss_n_batches: int = \d+", f"ce_loss_n_batches: int = {CE_LOSS_N_BATCHES}", text)
    cfg_path.write_text(text)
    print(f"[smoketest] Patched config.py -> n_train={N_TRAIN}, n_val={N_VAL}, n_test={N_TEST}, "
          f"ce_loss_n_batches={CE_LOSS_N_BATCHES}", flush=True)


def main():
    timings = {}
    results = {}

    sh(["git", "clone", OUR_REPO_URL, str(OUR_DIR)])
    sh(["git", "clone", OFFICIAL_REPO_URL, str(OFFICIAL_DIR)])

    os.chdir(str(OUR_DIR))
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    # Deliberately NOT installing the official repo's full requirements.txt
    # (pins vllm, ray, and CUDA-toolkit packages that risk breaking this
    # container's own preinstalled, matched torch/CUDA build -- same
    # lesson learned earlier in this project). Installing only what their
    # pipeline actually needs for the direction-extraction + eval path we
    # use, on top of what our own requirements.txt already gives us.
    sh([sys.executable, "-m", "pip", "install", "einops==0.8.0", "jaxtyping==0.2.29"])
    hf_login()

    patch_config_for_smoketest()

    print("=" * 70, flush=True)
    print("[smoketest] Stage 1: build_advbench_splits.py", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        sh([sys.executable, "build_advbench_splits.py", "--repo_dir", str(OFFICIAL_DIR),
            "--n_train", str(N_TRAIN), "--n_val", str(N_VAL)])
        timings["build_advbench_splits"] = time.time() - t0
        results["stage1_status"] = "OK"
        print(f"[smoketest] Stage 1 OK in {timings['build_advbench_splits']:.1f}s", flush=True)
    except Exception as e:
        timings["build_advbench_splits"] = time.time() - t0
        results["stage1_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        print(f"[smoketest] Stage 1 FAILED after {timings['build_advbench_splits']:.1f}s", flush=True)

    print("=" * 70, flush=True)
    print("[smoketest] Stage 2: official pipeline.run_pipeline (patched to small scale)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        sh([sys.executable, "-m", "pipeline.run_pipeline", "--model_path", MODEL_PATH], cwd=str(OFFICIAL_DIR))
        timings["run_pipeline"] = time.time() - t0
        direction_exists = (OFFICIAL_DIR / "pipeline" / "runs" / MODEL_ALIAS / "direction.pt").exists()
        results["stage2_status"] = "OK" if direction_exists else "RAN BUT direction.pt NOT FOUND"
        print(f"[smoketest] Stage 2 finished in {timings['run_pipeline']:.1f}s, "
              f"direction.pt exists: {direction_exists}", flush=True)
    except Exception as e:
        timings["run_pipeline"] = time.time() - t0
        results["stage2_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        print(f"[smoketest] Stage 2 FAILED after {timings['run_pipeline']:.1f}s", flush=True)

    print("=" * 70, flush=True)
    print("[smoketest] Stage 3: save_orthogonalized_checkpoint.py", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    checkpoint_dir = "/tmp/smoketest_orthogonalized_model"
    try:
        sh([sys.executable, str(OUR_DIR / "save_orthogonalized_checkpoint.py"),
            "--model_path", MODEL_PATH, "--model_alias", MODEL_ALIAS,
            "--output_dir", checkpoint_dir], cwd=str(OFFICIAL_DIR))
        timings["save_orthogonalized_checkpoint"] = time.time() - t0
        results["stage3_status"] = "OK"
        print(f"[smoketest] Stage 3 OK in {timings['save_orthogonalized_checkpoint']:.1f}s", flush=True)
        model_for_eval = checkpoint_dir
    except Exception as e:
        timings["save_orthogonalized_checkpoint"] = time.time() - t0
        results["stage3_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        print(f"[smoketest] Stage 3 FAILED after {timings['save_orthogonalized_checkpoint']:.1f}s, "
              f"falling back to base model for stage 4", flush=True)
        model_for_eval = MODEL_PATH

    print("=" * 70, flush=True)
    print("[smoketest] Stage 4: run_asr.py (max_prompts=5)", flush=True)
    print("=" * 70, flush=True)
    t0 = time.time()
    try:
        sh([sys.executable, "run_asr.py", "--model_path", model_for_eval,
            "--prompts_path", "prompts/advbench_malicious.csv", "--max_prompts", "5",
            "--out_path", "/tmp/asr_smoketest_result.json"], cwd=str(OUR_DIR))
        timings["run_asr"] = time.time() - t0
        asr_result = json.load(open("/tmp/asr_smoketest_result.json"))
        results["stage4_status"] = "OK"
        results["asr"] = asr_result["asr"]
        print(f"[smoketest] Stage 4 OK in {timings['run_asr']:.1f}s, ASR={asr_result['asr']}", flush=True)
    except Exception as e:
        timings["run_asr"] = time.time() - t0
        results["stage4_status"] = f"FAILED: {e!r}"
        traceback.print_exc()
        print(f"[smoketest] Stage 4 FAILED after {timings['run_asr']:.1f}s", flush=True)

    summary = {"timings_seconds": timings, "results": results, "model": MODEL_PATH}
    print("\n" + "=" * 70, flush=True)
    print("[smoketest] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print("=" * 70, flush=True)

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SM_MODEL_DIR / "smoketest_summary.json", "w"), indent=2)
    if os.path.exists("/tmp/asr_smoketest_result.json"):
        shutil.copy("/tmp/asr_smoketest_result.json", SM_MODEL_DIR / "asr_smoketest_result.json")

    print("[smoketest] Done.", flush=True)


if __name__ == "__main__":
    main()
