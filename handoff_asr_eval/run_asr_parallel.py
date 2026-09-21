"""
Data-parallel wrapper around run_asr.py: splits the prompt set across N
GPUs, runs one independent run_asr.py subprocess per GPU (each with its
own full copy of the target model AND the HarmBench judge -- this is data
parallelism, not model parallelism, since an 8B target model already fits
on a single GPU with room to spare), then merges the per-GPU results back
into one file with the same schema run_asr.py itself produces.

Worth using when your instance has multiple GPUs (e.g. g6e.12xlarge = 4x
L40S) and you want near-linear speedup on large sets like ood_full_pool.csv
-- each GPU handles roughly 1/N of the prompts independently, so wall-clock
time drops by roughly N (modulo each GPU's own model+judge load time,
which doesn't shrink -- see the docstring note on total speedup below).

Usage:
    python run_asr_parallel.py --model_path <hf-model-id-or-local-path> \
        --prompts_path prompts/ood_full_pool.csv --num_gpus 4 --out_path asr_full_pool.json

If --num_gpus is omitted, it's auto-detected via torch.cuda.device_count().
On a single-GPU box this just runs run_asr.py directly (no parallelism, no
subprocess overhead) -- safe to leave --num_gpus unset everywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from run_asr import load_prompts

SCRIPT_DIR = Path(__file__).resolve().parent


def detect_num_gpus() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def shard_prompts(prompts: list[str], num_shards: int) -> list[list[int]]:
    # Round-robin, not contiguous chunks -- guards against any residual
    # ordering structure in the input file (even though ood_full_pool.csv
    # etc. are already shuffled) leaving one GPU with a systematically
    # different subset (e.g. all-long or all-short) than the others.
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    for i in range(len(prompts)):
        shards[i % num_shards].append(i)
    return shards


def run_shard(gpu_id: int, shard_prompts_path: Path, shard_out_path: Path, args) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_asr.py"),
        "--model_path", args.model_path,
        "--prompts_path", str(shard_prompts_path),
        "--dtype", args.dtype,
        "--device_map", "auto",  # scoped to the single GPU CUDA_VISIBLE_DEVICES exposes
        "--max_new_tokens", str(args.max_new_tokens),
        "--batch_size", str(args.batch_size),
        "--max_batch_tokens", str(args.max_batch_tokens),
        "--out_path", str(shard_out_path),
    ]
    print(f"[gpu {gpu_id}] + {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--prompts_path", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--max_prompts", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_batch_tokens", type=int, default=8192)
    ap.add_argument("--num_gpus", type=int, default=None, help="Default: auto-detect via torch.cuda.device_count()")
    ap.add_argument("--out_path", default="asr_result.json")
    args = ap.parse_args()

    num_gpus = args.num_gpus or detect_num_gpus()
    prompts = load_prompts(args.prompts_path)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    print(f"Loaded {len(prompts)} prompts from {args.prompts_path}, splitting across {num_gpus} GPU(s)")

    if num_gpus <= 1:
        print("num_gpus<=1 -- running run_asr.py directly, no parallelism.")
        cmd = [sys.executable, str(SCRIPT_DIR / "run_asr.py"),
               "--model_path", args.model_path, "--prompts_path", args.prompts_path,
               "--dtype", args.dtype, "--max_new_tokens", str(args.max_new_tokens),
               "--batch_size", str(args.batch_size), "--max_batch_tokens", str(args.max_batch_tokens),
               "--out_path", args.out_path]
        if args.max_prompts is not None:
            cmd += ["--max_prompts", str(args.max_prompts)]
        subprocess.run(cmd, check=True)
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shards = shard_prompts(prompts, num_gpus)
        shard_paths = []
        procs = []
        for gpu_id, indices in enumerate(shards):
            if not indices:
                continue
            shard_in = tmp_dir / f"shard_{gpu_id}.txt"
            shard_out = tmp_dir / f"shard_{gpu_id}_out.json"
            shard_in.write_text("\n".join(prompts[i] for i in indices) + "\n")
            shard_paths.append((gpu_id, indices, shard_out))
            procs.append(run_shard(gpu_id, shard_in, shard_out, args))

        failed = False
        for (gpu_id, _, _), proc in zip(shard_paths, procs):
            ret = proc.wait()
            if ret != 0:
                failed = True
                print(f"[gpu {gpu_id}] FAILED with exit code {ret}", flush=True)
        if failed:
            raise SystemExit("One or more GPU shards failed -- see output above.")

        # Merge back into original prompt order.
        responses = [None] * len(prompts)
        compliant = [None] * len(prompts)
        for gpu_id, indices, shard_out in shard_paths:
            data = json.load(open(shard_out))
            for local_i, global_i in enumerate(indices):
                responses[global_i] = data["responses"][local_i]
                compliant[global_i] = data["compliant"][local_i]

    n_compliant = sum(compliant)
    asr = n_compliant / len(compliant)
    print(f"\nASR = {asr:.4f} ({n_compliant}/{len(compliant)} judged compliant, merged across {num_gpus} GPUs)")

    result = {
        "model_path": args.model_path,
        "prompts_path": args.prompts_path,
        "batch_size": args.batch_size,
        "max_batch_tokens": args.max_batch_tokens,
        "num_gpus": num_gpus,
        "n_prompts": len(prompts),
        "asr": asr,
        "n_compliant": n_compliant,
        "prompts": prompts,
        "responses": responses,
        "compliant": compliant,
    }
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
