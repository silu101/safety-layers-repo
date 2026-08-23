"""
Kaggle kernel entrypoint for the Section 3.4 localization experiment
(weight scaling + over-rejection rate). Pushed via `kaggle kernels push`
using kernel-metadata.json (see scripts/run_localization_on_kaggle.sh).

Self-contained (duplicates a few helpers from kaggle/kernel_runner.py)
because Kaggle script kernels execute a single file with no sibling
imports -- other files in the push directory aren't importable as modules.

What this does, in order:
  1. Clone the repo from GitHub into /kaggle/working.
  2. pip install -e ".[model,plot]".
  3. GPU compatibility check (P100 -> compatible torch), same as
     kernel_runner.py -- see docs/KNOWN_DISCREPANCIES.md #9.
  4. Log into Hugging Face if a token is available (best-effort).
  5. Run all three configs in sequence: baseline (no scaling), target
     range (paper's Table 1 range), control range (outside it) -- so R_o
     is comparable across all three, not just reported for one run.
     If the smoketest-scale run isn't requested separately, this uses the
     full 731-prompt configs directly (unlike the cos_sim runner, there's
     no separate smoketest-then-full gate here since the "smoketest"
     config differs only in --set max_prompts, applied via CLI override
     below rather than a distinct config file, so the exact-same code path
     is exercised at both scales).

MODEL_CONFIG_PREFIX below is substituted by scripts/run_localization_on_kaggle.sh
before each push (e.g. "gemma") -- do not hardcode a model here.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/kaggle/working/safety-layers-repro")
MODEL_CONFIG_PREFIX = "gemma"
SMOKETEST_MAX_PROMPTS = 20


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
    sys.path.insert(0, str(REPO_DIR / "src"))
    import importlib
    importlib.invalidate_caches()


def ensure_gpu_compatible_torch():
    """See kaggle/kernel_runner.py's identical function for the full
    explanation -- docs/KNOWN_DISCREPANCIES.md #9."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        print("[kernel_runner_localization] nvidia-smi not found, skipping GPU compat check.")
        return
    if out.returncode != 0 or not out.stdout.strip():
        print(f"[kernel_runner_localization] nvidia-smi inconclusive ({out.stderr.strip()}).")
        return

    line = out.stdout.strip().splitlines()[0]
    print(f"[kernel_runner_localization] GPU detected: {line}")
    try:
        compute_cap = float(line.split(",")[-1].strip())
    except ValueError:
        return
    if compute_cap >= 7.0:
        print(f"[kernel_runner_localization] Compute capability {compute_cap} >= 7.0, fine.")
        return

    print("=" * 70)
    print(f"[kernel_runner_localization] GPU compute capability {compute_cap} < 7.0 "
          "(e.g. Tesla P100) is incompatible with Kaggle's preinstalled torch.")
    print("[kernel_runner_localization] Reinstalling torch==2.7.1+cu118.")
    print("[kernel_runner_localization] ENVIRONMENT DEVIATION -- see KNOWN_DISCREPANCIES.md #9.")
    print("=" * 70)
    sh([sys.executable, "-m", "pip", "install", "torch==2.7.1",
        "--index-url", "https://download.pytorch.org/whl/cu118"])
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"])


HF_TOKEN_SECRET_NAMES = ["HF_TOKEN", "GEMINI_TOKEN"]


def hf_login():
    token = os.environ.get("HF_TOKEN")
    source = "HF_TOKEN env var" if token else None
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            for name in HF_TOKEN_SECRET_NAMES:
                try:
                    token = client.get_secret(name)
                    source = f"Kaggle Secret '{name}'"
                    break
                except Exception as e:
                    print(f"[kernel_runner_localization] get_secret('{name}') failed: "
                          f"{type(e).__name__}: {e}")
        except Exception as e:
            print(f"[kernel_runner_localization] Could not use kaggle_secrets: {e}")
    if not token:
        print("[kernel_runner_localization] No token found -- skipping HF login "
              "(fine for ungated models).")
        return
    print(f"[kernel_runner_localization] Found a token via {source}.")
    from huggingface_hub import login
    login(token=token)
    print("[kernel_runner_localization] Hugging Face login OK.")


def run_config(name: str, config_file: str, max_prompts: int | None = None):
    from safety_layers_repro import run_localization
    args = ["--config", f"configs/{config_file}"]
    if max_prompts is not None:
        args += ["--set", f"max_prompts={max_prompts}"]
    print(f"[kernel_runner_localization] Running {name} ({config_file})...")
    out_path = run_localization.main(args)
    print(f"[kernel_runner_localization] {name} OK: {out_path}")
    return out_path


def main():
    Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    os.chdir("/kaggle/working")
    clone_repo()
    install_package()
    os.chdir(str(REPO_DIR))
    ensure_gpu_compatible_torch()
    hf_login()

    print("=" * 70)
    print(f"[kernel_runner_localization] Smoketest pass ({SMOKETEST_MAX_PROMPTS} prompts, target range)")
    print("=" * 70)
    run_config("smoketest", f"{MODEL_CONFIG_PREFIX}_localization.yaml", max_prompts=SMOKETEST_MAX_PROMPTS)

    print("=" * 70)
    print("[kernel_runner_localization] Full runs (all 731 prompts): baseline, target, control")
    print("=" * 70)
    run_config("baseline", f"{MODEL_CONFIG_PREFIX}_localization_baseline.yaml")
    run_config("target", f"{MODEL_CONFIG_PREFIX}_localization.yaml")
    run_config("control", f"{MODEL_CONFIG_PREFIX}_localization_control.yaml")

    print(f"[kernel_runner_localization] Done. Results under {REPO_DIR / 'results'}")


if __name__ == "__main__":
    main()
