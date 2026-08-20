"""
Kaggle kernel entrypoint for the safety-layers-repro cosine-similarity
experiment (paper Section 3.2-3.3 existence check). Pushed to Kaggle via
`kaggle kernels push` using kernel-metadata.json (see scripts/run_on_kaggle.sh).

What this does, in order:
  1. Clone the repo from GitHub into /kaggle/working (so relative paths in
     configs/*.yaml resolve, and results/ lands under /kaggle/working where
     Kaggle captures it as kernel output).
  2. pip install -e ".[model,plot]".
  3. Check the assigned GPU's compute capability; if it's an old Tesla
     P100 (incompatible with the preinstalled torch build -- see
     docs/KNOWN_DISCREPANCIES.md #9), reinstall a torch/CUDA combo that
     still supports it. Logged as an environment deviation, not silent.
  4. Log into Hugging Face using an HF_TOKEN Kaggle Secret (or env var).
  5. Run the smoketest config (configs/phi3_cossim_smoketest.yaml, r=20).
     If this fails, stop -- do NOT attempt the full run on an unverified
     pipeline.
  6. Run the full config (configs/phi3_cossim.yaml, r=500, dtype=float32,
     fidelity-matched to the paper). If this hits CUDA OOM (float32 Phi-3
     needs ~15-20GB VRAM, Kaggle's free T4 has 16GB), retry ONCE with
     --set dtype=auto. This fallback is clearly flagged -- in stdout, and
     in a marker file dropped next to the run's output -- as a deviation
     from the float32 fidelity target, never silently treated as a match.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/kaggle/working/safety-layers-repro")


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def clone_repo():
    if REPO_DIR.exists():
        sh(["git", "-C", str(REPO_DIR), "pull", "--ff-only"])
    else:
        sh(["git", "clone", REPO_URL, str(REPO_DIR)])


def install_package():
    sh([sys.executable, "-m", "pip", "install", "-e", ".[model,plot]"], cwd=str(REPO_DIR))
    # Editable-install machinery can lag behind sys.path_importer_cache
    # within the same process; make the src/ layout importable directly
    # rather than relying on the install picking up mid-process.
    sys.path.insert(0, str(REPO_DIR / "src"))
    import importlib
    importlib.invalidate_caches()


def ensure_gpu_compatible_torch():
    """Kaggle's default GPU pool can assign an old Tesla P100 (Pascal,
    compute capability 6.0), which the preinstalled torch build (compiled
    only for sm_70+) can't run on at all -- see
    docs/KNOWN_DISCREPANCIES.md #9. Detect this via nvidia-smi (before
    torch is imported, so an incompatible build doesn't get locked into
    the process) and, if needed, reinstall a torch/CUDA combo (2.7.1+cu118)
    that still ships sm_60 kernels. Best-effort: torch 2.8+ dropped Pascal
    support entirely, so this is the newest compatible combo, and
    downgrading torch this far below what Kaggle preinstalls is itself a
    real environment deviation from a "standard" Kaggle GPU run -- logged,
    not silently done."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        print("[kernel_runner] nvidia-smi not found -- assuming CPU-only, skipping GPU compat check.")
        return
    if out.returncode != 0 or not out.stdout.strip():
        print(f"[kernel_runner] nvidia-smi check inconclusive ({out.stderr.strip()}), continuing as-is.")
        return

    line = out.stdout.strip().splitlines()[0]
    print(f"[kernel_runner] GPU detected: {line}")
    try:
        compute_cap = float(line.split(",")[-1].strip())
    except ValueError:
        print("[kernel_runner] Could not parse compute capability, continuing as-is.")
        return

    if compute_cap >= 7.0:
        print(f"[kernel_runner] Compute capability {compute_cap} >= 7.0, preinstalled torch is fine.")
        return

    print("=" * 70)
    print(f"[kernel_runner] GPU compute capability {compute_cap} (< 7.0, e.g. Tesla P100) is")
    print("[kernel_runner] incompatible with Kaggle's preinstalled torch build.")
    print("[kernel_runner] Reinstalling torch==2.7.1+cu118 (last release with sm_60 kernels).")
    print("[kernel_runner] THIS IS AN ENVIRONMENT DEVIATION from a standard Kaggle GPU run --")
    print("[kernel_runner] see docs/KNOWN_DISCREPANCIES.md #9.")
    print("=" * 70)
    sh([
        sys.executable, "-m", "pip", "install",
        "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu118",
    ])
    # Kaggle's preinstalled torchvision/torchaudio stay pinned to the
    # original torch version and break (mismatched compiled ops, e.g.
    # `torchvision::nms does not exist`) once torch is downgraded. This
    # repo doesn't use either -- uninstall rather than chase matching
    # versions, so transformers just treats them as unavailable.
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])


def hf_login():
    """Log into Hugging Face if a token is available. Not required for this
    repo's default model (microsoft/Phi-3-mini-4k-instruct is ungated), so
    this is best-effort: skip quietly if no HF_TOKEN secret/env var is set,
    rather than failing the whole run. Needed only if model_path is changed
    to a gated model (e.g. Llama-2/3 -- see docs/REPRODUCTION_SPEC.md)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if not token:
        print("[kernel_runner] No HF_TOKEN found -- skipping HF login "
              "(fine for ungated models like Phi-3-mini; required if you "
              "switch to a gated model).")
        return
    from huggingface_hub import login
    login(token=token)
    print("[kernel_runner] Hugging Face login OK.")


def run_smoketest():
    from safety_layers_repro import run_cos_sim
    print("[kernel_runner] Running smoketest config (r=20)...")
    out_path = run_cos_sim.main(["--config", "configs/phi3_cossim_smoketest.yaml"])
    print(f"[kernel_runner] Smoketest OK: {out_path}")


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in str(exc).lower()


def run_full():
    from safety_layers_repro import run_cos_sim
    print("[kernel_runner] Running full config (r=500, dtype=float32, fidelity-matched)...")
    try:
        out_path = run_cos_sim.main(["--config", "configs/phi3_cossim.yaml"])
        print(f"[kernel_runner] Full run OK (float32 fidelity-matched): {out_path}")
        return
    except Exception as e:
        if not _is_oom(e):
            raise
        print("=" * 70)
        print("[kernel_runner] CUDA OOM on float32 full run.")
        print("[kernel_runner] Retrying with --set dtype=auto.")
        print("[kernel_runner] THIS IS A DEVIATION from the float32 fidelity")
        print("[kernel_runner] target, not a like-for-like reproduction run.")
        print("=" * 70)
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    out_path = run_cos_sim.main(
        ["--config", "configs/phi3_cossim.yaml", "--set", "dtype=auto"]
    )
    marker = Path(out_path).parent / "OOM_FALLBACK_DEVIATION.txt"
    marker.write_text(
        "DEVIATION FROM SPEC: this run hit CUDA OOM at dtype=float32 (the paper\n"
        "fidelity target) on Kaggle's free T4 (16GB VRAM). It was retried and\n"
        "completed with `--set dtype=auto` instead.\n\n"
        "This is NOT a float32 fidelity-matched run. See run_metadata.json's\n"
        "resolved config for the actual dtype used, and record this explicitly\n"
        "as a noted deviation in docs/REPLICATION_LOG.md -- do not present these\n"
        "numbers as if they matched the authors' float32 setup exactly.\n"
    )
    print(f"[kernel_runner] Full run OK (dtype=auto FALLBACK, see {marker.name}): {out_path}")


def main():
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    os.chdir("/kaggle/working")
    clone_repo()
    install_package()
    ensure_gpu_compatible_torch()
    os.chdir(str(REPO_DIR))
    hf_login()
    run_smoketest()
    run_full()
    print(f"[kernel_runner] Done. Results under {REPO_DIR / 'results'}")


if __name__ == "__main__":
    main()
