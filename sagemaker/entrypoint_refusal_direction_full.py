"""
Full-scale (not smoke-test) run of the refusal-direction-ood package,
pulled fresh from GitHub. Unlike entrypoint_refusal_direction_smoketest.py,
this does NOT patch pipeline/config.py -- uses the paper's own real
defaults (n_train=128, n_val=32, n_test=100), and runs the full ASR sweep
(AdvBench baseline, HarmBench, semantic-OOD, attack-OOD), not a 5-prompt
check.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

OUR_REPO_URL = "https://github.com/silu101/refusal-direction-ood"
OFFICIAL_REPO_URL = "https://github.com/andyrdt/refusal_direction"
OUR_DIR = Path("/opt/ml/code/refusal-direction-ood")
OFFICIAL_DIR = Path("/opt/ml/code/refusal_direction")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ALIAS = "Meta-Llama-3-8B-Instruct"  # verified: os.path.basename(MODEL_PATH)
CHECKPOINT_DIR = "/tmp/refusal_direction_orthogonalized_model"


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
    sh(["git", "clone", OUR_REPO_URL, str(OUR_DIR)])
    sh(["git", "clone", OFFICIAL_REPO_URL, str(OFFICIAL_DIR)])

    os.chdir(str(OUR_DIR))
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    # Same reasoning as the smoke test: skip the official repo's full
    # requirements.txt (pins vllm/ray/CUDA-toolkit packages that risk
    # breaking this container's own matched torch/CUDA build). Only
    # extract_direction.py's actual dependencies, confirmed vllm-free.
    sh([sys.executable, "-m", "pip", "install", "einops==0.8.0", "jaxtyping==0.2.29", "matplotlib==3.9.0"])
    hf_login()

    print("=" * 70, flush=True)
    print("[entrypoint] build_advbench_splits.py (real scale: n_train=128, n_val=32)", flush=True)
    print("=" * 70, flush=True)
    sh([sys.executable, "build_advbench_splits.py", "--repo_dir", str(OFFICIAL_DIR)])

    print("=" * 70, flush=True)
    print("[entrypoint] extract_direction.py (real scale, no config patching)", flush=True)
    print("=" * 70, flush=True)
    sh([sys.executable, str(OUR_DIR / "extract_direction.py"), "--model_path", MODEL_PATH], cwd=str(OFFICIAL_DIR))

    print("=" * 70, flush=True)
    print("[entrypoint] save_orthogonalized_checkpoint.py", flush=True)
    print("=" * 70, flush=True)
    sh([sys.executable, str(OUR_DIR / "save_orthogonalized_checkpoint.py"),
        "--model_path", MODEL_PATH, "--model_alias", MODEL_ALIAS,
        "--output_dir", CHECKPOINT_DIR], cwd=str(OFFICIAL_DIR))

    prompt_sets = [
        ("advbench_malicious.csv", "asr_advbench.json"),
        ("harmbench_eval.csv", "asr_harmbench.json"),
        ("ood_semantic_test.csv", "asr_semantic_ood.json"),
        ("attack_ood_jailbreakllms.csv", "asr_attack_ood.json"),
    ]
    for prompts_file, out_file in prompt_sets:
        print("=" * 70, flush=True)
        print(f"[entrypoint] run_asr.py on {prompts_file}", flush=True)
        print("=" * 70, flush=True)
        sh([sys.executable, "run_asr.py", "--model_path", CHECKPOINT_DIR,
            "--prompts_path", f"prompts/{prompts_file}", "--out_path", f"/tmp/{out_file}"], cwd=str(OUR_DIR))

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
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
