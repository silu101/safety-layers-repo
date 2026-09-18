"""
Run this FIRST, before any of the real pipeline scripts. Verifies:
  1. Every required library is installed (and importable).
  2. A CUDA GPU is visible, with a VRAM readout.
  3. You're authenticated to Hugging Face, and can actually reach both
     the target model you plan to evaluate AND the HarmBench judge model
     -- catching a gated-access problem now instead of 10 minutes into a
     real run.

Doesn't download full model weights -- only fetches each model's config
(a few KB), so this finishes in seconds even on a slow connection.

Usage:
    python check_setup.py --model_path <hf-model-id-or-local-path>
"""
from __future__ import annotations

import argparse
import sys

REQUIRED_PACKAGES = ["torch", "transformers", "accelerate", "sentencepiece", "tqdm", "datasets"]
JUDGE_MODEL = "cais/HarmBench-Llama-2-13b-cls"


def check_packages() -> bool:
    print("=== 1. Required packages ===")
    ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown version")
            print(f"  OK   {pkg} ({version})")
        except ImportError as e:
            print(f"  FAIL {pkg}: {e!r}")
            ok = False
    if not ok:
        print("  -> Run: pip install -r requirements.txt")
    return ok


def check_gpu() -> bool:
    print("\n=== 2. GPU / CUDA ===")
    try:
        import torch
    except ImportError:
        print("  FAIL torch not installed, can't check CUDA")
        return False

    if not torch.cuda.is_available():
        print("  FAIL No CUDA GPU visible to torch. This pipeline needs a real GPU")
        print("       (target model + a 13B judge model both need to fit in VRAM).")
        return False

    n = torch.cuda.device_count()
    print(f"  OK   {n} CUDA device(s) visible")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1e9
        print(f"    [{i}] {props.name} -- {vram_gb:.1f} GB VRAM")
        if vram_gb < 30:
            print(f"        WARNING: under 30GB -- the target model + the ~26GB judge model")
            print(f"        may not both fit. See README's VRAM note.")
    return True


def check_hf_access(model_path: str) -> bool:
    print("\n=== 3. Hugging Face authentication + model access ===")
    ok = True
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami()
        print(f"  OK   Logged in as: {who.get('name', '(unknown)')}")
    except Exception as e:
        print(f"  WARN Not logged in, or token invalid: {e!r}")
        print("       Run: huggingface-cli login")
        print("       (only strictly required if your target model or the judge is gated)")

    try:
        from transformers import AutoConfig
    except ImportError:
        print("  FAIL transformers not installed -- can't check model access. Run: pip install -r requirements.txt")
        return False

    for label, model_id in [("Target model", model_path), ("Judge model (HarmBench)", JUDGE_MODEL)]:
        try:
            AutoConfig.from_pretrained(model_id)
            print(f"  OK   {label} reachable: {model_id}")
        except Exception as e:
            print(f"  FAIL {label} NOT reachable: {model_id}")
            print(f"       {e!r}")
            print(f"       If this is a gated model, request access at https://huggingface.co/{model_id}")
            print(f"       under the account you logged in as above, then try again.")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct",
                     help="The target model you plan to run through the pipeline")
    args = ap.parse_args()

    results = [check_packages(), check_gpu(), check_hf_access(args.model_path)]

    print("\n" + "=" * 50)
    if all(results):
        print("ALL CHECKS PASSED. Safe to run find_safety_layer.py next.")
    else:
        print("ONE OR MORE CHECKS FAILED -- fix the issues above before running the real pipeline.")
        sys.exit(1)


if __name__ == "__main__":
    main()
