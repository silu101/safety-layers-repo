"""
SageMaker entrypoint: does localizing on the FULL 520-prompt AdvBench
(instead of the original 100-prompt malicious.csv, itself a 100% exact
subset of AdvBench) change which layers show the biggest N-M vs N-N/M-M
cosine-similarity divergence for gemma-2b-it? See
configs/gemma_cossim_advbench520.yaml's own comment for why normal_path
is intentionally left unchanged (isolates the malicious-side sample size
only, one variable at a time).

Moved to SageMaker after the local run hung indefinitely mid-model-load
(device_map="auto" + no real accelerator locally) -- the same reliability
problem that pushed everything else in this project onto SageMaker.

Mirrors entrypoint_localization.py's self-contained pattern.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[entrypoint] No HF_TOKEN in environment -- skipping HF login.")
        return
    from huggingface_hub import login
    login(token=token)
    print("[entrypoint] Hugging Face login OK.")


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(REPO_DIR))

    sh([sys.executable, "-m", "pip", "install", "-e", "."])
    sh([sys.executable, "-m", "pip", "install",
        "transformers==4.44.2", "accelerate==0.31.0", "tqdm>=4.65", "sentencepiece>=0.1.99"])
    sys.path.insert(0, str(REPO_DIR / "src"))
    hf_login()

    from safety_layers_repro import run_cos_sim

    print("=" * 70, flush=True)
    print("[entrypoint] Running gemma_cossim_advbench520.yaml", flush=True)
    print("=" * 70, flush=True)
    out_path = run_cos_sim.main(["--config", "configs/gemma_cossim_advbench520.yaml"])
    print(f"[entrypoint] -> {out_path}", flush=True)

    results_dir = REPO_DIR / "results"
    dest = SM_MODEL_DIR / "results"
    if results_dir.exists():
        shutil.copytree(results_dir, dest, dirs_exist_ok=True)
        print(f"[entrypoint] Copied {results_dir} -> {dest}", flush=True)

    print("[entrypoint] Done.", flush=True)


if __name__ == "__main__":
    main()
