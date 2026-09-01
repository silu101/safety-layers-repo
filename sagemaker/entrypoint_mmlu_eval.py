"""
SageMaker training-job entrypoint for Section 4's utility-side evaluation:
MMLU accuracy (S_m) against the 4 trained models. Same S3-fetch pattern as
entrypoint_harmful_eval.py (models live in 3 different jobs' outputs) --
see that file's docstring for why. No Anthropic API needed here (MMLU is
a fixed multiple-choice accuracy check, no judge), so no credit-exhaustion
risk like the harmful-eval job hit.

Mirrors entrypoint_harmful_eval.py's incremental-sync-after-every-config
and explicit-GPU-cleanup fixes from the start, rather than rediscovering
them the hard way again.
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

MODEL_SOURCES = [
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-03-54-57-156/output/model.tar.gz", "gemma_sppft_normal"),
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-03-54-57-156/output/model.tar.gz", "gemma_sppft_implicit"),
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-09-57-25-163/output/model.tar.gz", "gemma_full_normal"),
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-10-19-23-506/output/model.tar.gz", "gemma_full_implicit"),
]

CONFIGS = [
    "mmlu_gemma_full_normal.yaml",
    "mmlu_gemma_full_implicit.yaml",
    "mmlu_gemma_sppft_normal.yaml",
    "mmlu_gemma_sppft_implicit.yaml",
]
CONFIG_TO_MODEL_DIR = {
    "mmlu_gemma_full_normal.yaml": "gemma_full_normal",
    "mmlu_gemma_full_implicit.yaml": "gemma_full_implicit",
    "mmlu_gemma_sppft_normal.yaml": "gemma_sppft_normal",
    "mmlu_gemma_sppft_implicit.yaml": "gemma_sppft_implicit",
}


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


def fetch_models(needed_subdirs: set[str]):
    dest_root = REPO_DIR / "output_models"
    dest_root.mkdir(parents=True, exist_ok=True)

    seen_tarballs = {}
    for s3_uri, subdir in MODEL_SOURCES:
        if subdir not in needed_subdirs:
            continue
        if s3_uri not in seen_tarballs:
            local_tar = Path(f"/tmp/{Path(s3_uri).parent.parent.name}.tar.gz")
            print(f"[entrypoint] Downloading {s3_uri} -> {local_tar}")
            sh(["aws", "s3", "cp", s3_uri, str(local_tar)])
            seen_tarballs[s3_uri] = local_tar
        local_tar = seen_tarballs[s3_uri]

        print(f"[entrypoint] Extracting output_models/{subdir} from {local_tar}")
        with tarfile.open(local_tar, "r:gz") as tar:
            prefix = f"output_models/{subdir}/"
            # Exclude checkpoint-*/ (DeepSpeed resume checkpoints) --
            # not needed for inference, and blew through disk space last
            # time (see entrypoint_harmful_eval.py's fix, same issue here).
            members = [m for m in tar.getmembers() if m.name.startswith(prefix) and "/checkpoint-" not in m.name]
            if not members:
                raise RuntimeError(f"No members matching {prefix} found in {local_tar}")
            tar.extractall(path="/tmp/extract", members=members)
        extracted_src = Path(f"/tmp/extract/output_models/{subdir}")
        dest = dest_root / subdir
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(str(extracted_src), str(dest))
        shutil.rmtree(str(extracted_src))
        print(f"[entrypoint] {subdir} ready at {dest}")

    for local_tar in seen_tarballs.values():
        local_tar.unlink(missing_ok=True)
    shutil.rmtree("/tmp/extract", ignore_errors=True)


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(REPO_DIR))

    sh([sys.executable, "-m", "pip", "install", "-e", "."])
    sh([sys.executable, "-m", "pip", "install",
        "transformers==4.44.2", "accelerate==0.31.0", "sentencepiece>=0.1.99",
        "datasets==2.20.0", "pyarrow<18"])  # cais/mmlu is parquet-backed
    hf_login()

    needed_subdirs = {CONFIG_TO_MODEL_DIR[c] for c in CONFIGS}
    fetch_models(needed_subdirs)

    sys.path.insert(0, str(REPO_DIR / "src"))
    from safety_layers_repro import run_mmlu_eval

    for config_file in CONFIGS:
        print("=" * 70)
        print(f"[entrypoint] Running {config_file}")
        print("=" * 70)
        try:
            out_path = run_mmlu_eval.main(["--config", f"configs/{config_file}"])
            print(f"[entrypoint] {config_file} -> {out_path}")
        except Exception as e:
            print(f"[entrypoint] WARNING: {config_file} failed: {e!r} -- continuing to the next config.")
            traceback.print_exc()

        # Sync after EVERY config, not just at the end -- see
        # entrypoint_harmful_eval.py's KNOWN_DISCREPANCIES-worthy lesson.
        results_dir = REPO_DIR / "results"
        dest = SM_MODEL_DIR / "results"
        if results_dir.exists():
            shutil.copytree(results_dir, dest, dirs_exist_ok=True)
            print(f"[entrypoint] Copied {results_dir} -> {dest} (incremental sync after {config_file})")

    print("[entrypoint] Done.")


if __name__ == "__main__":
    main()
