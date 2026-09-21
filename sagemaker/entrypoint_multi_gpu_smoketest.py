"""
Smoke test for run_asr_parallel.py specifically -- the new data-parallel
wrapper that splits a prompt set across multiple GPUs (one independent
run_asr.py subprocess per GPU, each with its own full model+judge copy,
merged back into one result). Pulled fresh from GitHub, not local files.

Scope: only tests the new multi-GPU mechanism itself (GPU auto-detection,
subprocess launch/wait, shard merge correctness, a real ASR number) against
the base target model directly -- does NOT re-run the refusal-direction
pipeline (build_advbench_splits/extract_direction/save_orthogonalized_
checkpoint), which was already verified working in an earlier smoke test.
Small scale: --max_prompts 20.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUR_REPO_URL = "https://github.com/silu101/refusal-direction-ood"
OUR_DIR = Path("/opt/ml/code/refusal-direction-ood")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[smoketest] HF login OK.", flush=True)


def main():
    sh(["git", "clone", OUR_REPO_URL, str(OUR_DIR)])
    os.chdir(str(OUR_DIR))
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    hf_login()

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[smoketest] torch.cuda.device_count() = {n_gpus}", flush=True)

    t0 = time.time()
    summary = {"n_gpus_detected": n_gpus}
    try:
        sh([sys.executable, "run_asr_parallel.py",
            "--model_path", MODEL_PATH,
            "--prompts_path", "prompts/advbench_malicious.csv",
            "--max_prompts", "20",
            "--out_path", "/tmp/asr_multi_gpu_smoketest.json"])
        elapsed = time.time() - t0
        result = json.load(open("/tmp/asr_multi_gpu_smoketest.json"))
        summary.update({
            "status": "OK",
            "elapsed_seconds": elapsed,
            "num_gpus_used": result.get("num_gpus"),
            "n_prompts": result.get("n_prompts"),
            "asr": result.get("asr"),
            "n_compliant": result.get("n_compliant"),
            # merge-correctness check: every prompt must have gotten a
            # non-null response/compliant flag back from its shard
            "all_prompts_covered": all(r is not None for r in result.get("responses", [])),
        })
        print(f"[smoketest] OK in {elapsed:.1f}s, ASR={result.get('asr')}, "
              f"num_gpus_used={result.get('num_gpus')}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        summary.update({"status": f"FAILED: {e!r}", "elapsed_seconds": elapsed})
        print(f"[smoketest] FAILED after {elapsed:.1f}s: {e!r}", flush=True)

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(SM_MODEL_DIR / "multi_gpu_smoketest_summary.json", "w"), indent=2)
    print("\n[smoketest] SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print("[smoketest] Done.", flush=True)


if __name__ == "__main__":
    main()
