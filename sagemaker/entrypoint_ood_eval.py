"""
SageMaker entrypoint for the first small-scale semantic-content OOD
evaluation: gemma_sppft_normal (Safety Layers' flagship condition) tested
on data/ood_semantic_test.csv (100 prompts, 20 each from the 5 confirmed-
OOD categories -- see scripts/build_ood_semantic_test.py) instead of
AdvBench.

Deliberately scoped small: one model, one config, so this is cheap to run
and easy for a teammate to read end-to-end before expanding it to more
models and a larger per-category sample. Mirrors entrypoint_harmful_eval.py's
self-contained pattern (clone, pip install, fetch model, run) but for a
single model/config instead of four.
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

# gemma_sppft_normal lives in the same S3 tarball as gemma_sppft_implicit
# (both from the first Section 4 job) -- see entrypoint_harmful_eval.py's
# MODEL_SOURCES for the other 3 models if this gets expanded later.
MODEL_S3_URI = "s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-03-54-57-156/output/model.tar.gz"
MODEL_SUBDIR = "gemma_sppft_normal"
CONFIG_FILE = "eval_gemma_sppft_normal_ood.yaml"


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
        # Exclude checkpoint-*/ -- DeepSpeed resume checkpoints, not needed
        # for inference and several GB each (see entrypoint_harmful_eval.py).
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
    print(f"[entrypoint] Running {CONFIG_FILE} (OOD semantic-content test)", flush=True)
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
