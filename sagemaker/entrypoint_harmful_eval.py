"""
SageMaker training-job entrypoint for Section 4's harmful-rate (R_h) /
harmful-score (S_h) evaluation against the 4 trained models from earlier
Section 4 jobs. Those models live in 3 DIFFERENT jobs' S3 output
artifacts (SPPFT succeeded in the first 4-config job; Full FT needed a
separate DeepSpeed-sharded re-run per model) -- this entrypoint downloads
and extracts just the needed output_models/* subdirectories from each,
entirely within AWS (S3 -> this container), avoiding the slow local-
sandbox-to-S3 path that was infeasible for pulling artifacts back to this
machine directly (see docs/KNOWN_DISCREPANCIES.md notes on this).

Mirrors entrypoint_localization.py's self-contained pattern (clone repo,
pip install, run) plus a model-fetching step this one alone needs.
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

# (source S3 model.tar.gz, output_models subdir to extract from it)
MODEL_SOURCES = [
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-03-54-57-156/output/model.tar.gz", "gemma_sppft_normal"),
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-03-54-57-156/output/model.tar.gz", "gemma_sppft_implicit"),
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-09-57-25-163/output/model.tar.gz", "gemma_full_normal"),
    ("s3://sagemaker-us-east-1-344977996863/safety-layers-section4-gemma-2026-08-30-10-19-23-506/output/model.tar.gz", "gemma_full_implicit"),
]

CONFIGS = [
    "eval_gemma_sppft_normal.yaml",
    # Smoke-testing one config with a small prompt subset first (see
    # SMOKETEST_MAX_PROMPTS below) -- this pipeline has several new,
    # previously-untested mechanics (multi-tarball S3 extraction,
    # HarmBench classifier, Anthropic API calls together). Re-enable the
    # rest once this validates end-to-end.
    # "eval_gemma_full_normal.yaml",
    # "eval_gemma_full_implicit.yaml",
    # "eval_gemma_sppft_implicit.yaml",
]
SMOKETEST_MAX_PROMPTS = 5

# Maps each config file to the output_models/* subdir name it needs --
# used to filter MODEL_SOURCES down to only what CONFIGS actually
# requires, so a partial CONFIGS list (e.g. during smoke-testing) doesn't
# waste time/bandwidth downloading models nothing will use.
CONFIG_TO_MODEL_DIR = {
    "eval_gemma_full_normal.yaml": "gemma_full_normal",
    "eval_gemma_full_implicit.yaml": "gemma_full_implicit",
    "eval_gemma_sppft_normal.yaml": "gemma_sppft_normal",
    "eval_gemma_sppft_implicit.yaml": "gemma_sppft_implicit",
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
    """Download each source tarball once (dedup by URL, since the SPPFT
    pair share one tarball) and extract only the needed output_models/*
    subdirectory into REPO_DIR/output_models/. Only fetches subdirs in
    `needed_subdirs`, so a partial CONFIGS list doesn't waste time/
    bandwidth on models nothing will use."""
    dest_root = REPO_DIR / "output_models"
    dest_root.mkdir(parents=True, exist_ok=True)

    seen_tarballs = {}
    for s3_uri, subdir in MODEL_SOURCES:
        if subdir not in needed_subdirs:
            continue
        if s3_uri not in seen_tarballs:
            local_tar = Path(f"/tmp/{Path(s3_uri).parent.parent.name}.tar.gz")
            print(f"[entrypoint] Downloading {s3_uri} -> {local_tar} (within AWS, should be fast)")
            sh(["aws", "s3", "cp", s3_uri, str(local_tar)])
            seen_tarballs[s3_uri] = local_tar
        local_tar = seen_tarballs[s3_uri]

        print(f"[entrypoint] Extracting output_models/{subdir} from {local_tar}")
        with tarfile.open(local_tar, "r:gz") as tar:
            members = [m for m in tar.getmembers() if f"output_models/{subdir}/" in m.name]
            if not members:
                raise RuntimeError(f"No members matching output_models/{subdir}/ found in {local_tar}")
            tar.extractall(path="/tmp/extract", members=members)
        extracted_src = Path(f"/tmp/extract/output_models/{subdir}")
        dest = dest_root / subdir
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(extracted_src), str(dest))
        print(f"[entrypoint] {subdir} ready at {dest}")

    # Free disk space -- don't need the downloaded tarballs after extraction.
    for local_tar in seen_tarballs.values():
        local_tar.unlink(missing_ok=True)
    shutil.rmtree("/tmp/extract", ignore_errors=True)


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(REPO_DIR))

    sh([sys.executable, "-m", "pip", "install", "-e", "."])
    sh([sys.executable, "-m", "pip", "install",
        "transformers==4.44.2", "accelerate==0.31.0",
        "sentencepiece>=0.1.99",
        # Pinned, not >= -- unbounded resolved to anthropic 1.2.0, which
        # then failed with a generic APIConnectionError on the actual
        # judge call (no further detail printed -- see the traceback.
        # print_exc() below, added for exactly this kind of failure).
        # 0.40.0 matches this project's harmful_score.py interface, which
        # was built/tested against the 0.x SDK.
        "anthropic==0.40.0"])
    hf_login()

    needed_subdirs = {CONFIG_TO_MODEL_DIR[c] for c in CONFIGS}
    fetch_models(needed_subdirs)

    sys.path.insert(0, str(REPO_DIR / "src"))
    from safety_layers_repro import run_harmful_eval

    for config_file in CONFIGS:
        print("=" * 70)
        print(f"[entrypoint] Running {config_file}")
        print("=" * 70)
        try:
            out_path = run_harmful_eval.main([
                "--config", f"configs/{config_file}",
                "--set", f"max_prompts={SMOKETEST_MAX_PROMPTS}",
            ])
            print(f"[entrypoint] {config_file} -> {out_path}")
        except Exception as e:
            print(f"[entrypoint] WARNING: {config_file} failed: {e!r} -- continuing to the next config.")
            traceback.print_exc()

    results_dir = REPO_DIR / "results"
    dest = SM_MODEL_DIR / "results"
    if results_dir.exists():
        shutil.copytree(results_dir, dest, dirs_exist_ok=True)
        print(f"[entrypoint] Copied {results_dir} -> {dest}")

    print("[entrypoint] Done.")


if __name__ == "__main__":
    main()
