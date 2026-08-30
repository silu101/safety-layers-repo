"""
SageMaker training-job entrypoint for the Section 3.4 localization
o_proj-exclusion re-run (docs/KNOWN_DISCREPANCIES.md #17).

Mirrors kaggle/kernel_runner_localization.py's self-contained pattern
(clone repo -> pip install -e . -> run) since it's the same underlying
pipeline, just on a different GPU platform. All data needed
(data/over_rejection.csv) is already committed inside this repo -- no S3
input channels required. Only google/gemma-2b-it (ungated) is pulled from
the Hugging Face Hub at runtime.

Outputs are copied to /opt/ml/model/ so SageMaker syncs them back to S3
automatically when the training job finishes.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

CONFIGS = [
    "gemma_localization_openreview.yaml",
    "gemma_localization_control_openreview.yaml",
]


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    """gemma-2b-it is a gated HF model -- log in if a token was passed
    through as the HF_TOKEN environment variable (set by
    launch_localization_openreview.py from a local, un-committed token
    file). Best-effort: skip quietly if not present rather than fail hard,
    matching kaggle/kernel_runner_localization.py's hf_login()."""
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
    sh([sys.executable, "-m", "pip", "install", "-e", ".[model,plot]"])
    sys.path.insert(0, str(REPO_DIR / "src"))
    hf_login()

    from safety_layers_repro import run_localization

    for config_file in CONFIGS:
        print("=" * 70)
        print(f"[entrypoint] Running {config_file}")
        print("=" * 70)
        out_path = run_localization.main(["--config", f"configs/{config_file}"])
        print(f"[entrypoint] {config_file} -> {out_path}")

    # Copy every results/<run>/ dir this job produced into SM_MODEL_DIR so
    # SageMaker uploads them to S3 as the job's model artifact.
    results_dir = REPO_DIR / "results"
    dest = SM_MODEL_DIR / "results"
    if results_dir.exists():
        shutil.copytree(results_dir, dest, dirs_exist_ok=True)
        print(f"[entrypoint] Copied {results_dir} -> {dest}")

    print("[entrypoint] Done.")


if __name__ == "__main__":
    main()
