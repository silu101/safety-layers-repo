"""
SageMaker entrypoint: near-OOD baseline for gemma_sppft_normal (Safety
Layers' flagship condition) -- evaluated on the full official HarmBench
release (400 behaviors: standard + contextual + copyright), the same set
used for the AdvBench<->HarmBench positive control (mean similarity
0.508). See scripts/build_harmbench_eval_set.py.

This is condition 2 of 4 in the AdvBench(ID) -> HarmBench(near-OOD) ->
unfiltered-OOD -> filtered-OOD comparison design.

Mirrors entrypoint_ood_eval.py exactly (same model source, same eval
pipeline with checkpointing/concurrent-S_h fixes already baked into
run_harmful_eval.py), just a different config/prompt file.
"""
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL_S3_URI = "s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-03-54-57-156/output/model.tar.gz"
MODEL_SUBDIR = "gemma_sppft_normal"
CONFIG_FILE = "eval_gemma_sppft_normal_harmbench.yaml"


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


def fetch_model():
    dest_root = REPO_DIR / "output_models"
    dest_root.mkdir(parents=True, exist_ok=True)
    local_tar = Path("/tmp/model.tar.gz")
    print(f"[entrypoint] Downloading {MODEL_S3_URI} -> {local_tar}", flush=True)
    sh(["aws", "s3", "cp", MODEL_S3_URI, str(local_tar)])

    print(f"[entrypoint] Extracting output_models/{MODEL_SUBDIR} from {local_tar}", flush=True)
    with tarfile.open(local_tar, "r:gz") as tar:
        prefix = f"output_models/{MODEL_SUBDIR}/"
        members = [m for m in tar.getmembers() if m.name.startswith(prefix) and "/checkpoint-" not in m.name]
        if not members:
            raise RuntimeError(f"No members matching {prefix} found in {local_tar}")
        tar.extractall(path="/tmp/extract", members=members)
    extracted_src = Path(f"/tmp/extract/output_models/{MODEL_SUBDIR}")
    dest = dest_root / MODEL_SUBDIR
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(str(extracted_src), str(dest))
    shutil.rmtree(str(extracted_src))
    local_tar.unlink(missing_ok=True)
    shutil.rmtree("/tmp/extract", ignore_errors=True)
    print(f"[entrypoint] {MODEL_SUBDIR} ready at {dest}", flush=True)


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(REPO_DIR))
    sh([sys.executable, "-m", "pip", "install", "-e", "."])
    sh([sys.executable, "-m", "pip", "install",
        "transformers==4.44.2", "accelerate==0.31.0", "sentencepiece>=0.1.99", "anthropic==0.40.0"])
    hf_login()
    fetch_model()

    sys.path.insert(0, str(REPO_DIR / "src"))
    from safety_layers_repro import run_harmful_eval

    print("=" * 70, flush=True)
    print(f"[entrypoint] Running {CONFIG_FILE} (HarmBench near-OOD baseline)", flush=True)
    print("=" * 70, flush=True)
    try:
        out_path = run_harmful_eval.main(["--config", f"configs/{CONFIG_FILE}"])
        print(f"[entrypoint] {CONFIG_FILE} -> {out_path}", flush=True)
    except Exception as e:
        print(f"[entrypoint] FAILED: {e!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        results_dir = REPO_DIR / "results"
        dest = SM_MODEL_DIR / "results"
        if results_dir.exists():
            shutil.copytree(results_dir, dest, dirs_exist_ok=True)
            print(f"[entrypoint] Copied {results_dir} -> {dest}", flush=True)

    print("[entrypoint] Done.", flush=True)


if __name__ == "__main__":
    main()
