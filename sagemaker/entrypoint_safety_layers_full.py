"""
Full-scale Safety Layers run, the same experiment structure as
entrypoint_refusal_direction_full.py but for the other method: locate the
safety layer, SPPFT-finetune with it frozen (D_N scenario only for this
run -- D_I is a separate follow-up), then run the same 5-set ASR sweep
(AdvBench baseline, HarmBench, semantic-OOD, attack-OOD, full unfiltered
OOD pool) against meta-llama/Meta-Llama-3-8B-Instruct, for a direct,
apples-to-apples comparison with the Refusal Direction run.

Pulled fresh from GitHub (handoff_asr_eval/), not local files.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/silu101/safety-layers-repo"
REPO_DIR = Path("/opt/ml/code/safety-layers-repro")
PKG_DIR = REPO_DIR / "handoff_asr_eval"
SM_MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
CHECKPOINT_DIR = "/tmp/safety_layers_sppft_normal_model"

# Paper's own choice for gemma-2b-it was a 6-layer-wide freeze window
# around its onset (see handoff_asr_eval/README.md Step 2) -- no
# established width exists yet for Llama-3-8B specifically, so this
# reuses that same width as the default assumption. Flagged here, not
# silently baked in, in case it needs revisiting once the real onset
# layer for this model is known.
FREEZE_WIDTH = 5  # end_layer = onset_layer + FREEZE_WIDTH (inclusive range)


def sh(cmd, **kw):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


def hf_login():
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[entrypoint] HF login OK.", flush=True)


def main():
    sh(["git", "clone", REPO_URL, str(REPO_DIR)])
    os.chdir(str(PKG_DIR))
    sh([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    hf_login()

    print("=" * 70, flush=True)
    print("[entrypoint] Stage 1: find_safety_layer.py", flush=True)
    print("=" * 70, flush=True)
    sh([sys.executable, "find_safety_layer.py", "--model_path", MODEL_PATH,
        "--malicious_path", "prompts/advbench_malicious.csv", "--normal_path", "prompts/normal.csv",
        "--out_path", "/tmp/safety_layer_result.json"])
    layer_result = json.load(open("/tmp/safety_layer_result.json"))
    onset_layer = layer_result["onset_layer"]
    if onset_layer is None:
        raise SystemExit("[entrypoint] find_safety_layer.py found no clear onset -- refusing to guess "
                          "a freeze range. Inspect /tmp/safety_layer_result.json's per-layer diffs manually.")
    begin_layer, end_layer = onset_layer, onset_layer + FREEZE_WIDTH
    print(f"[entrypoint] onset_layer={onset_layer} -> freezing layers [{begin_layer}, {end_layer}] inclusive "
          f"(width={FREEZE_WIDTH + 1}, paper's own gemma-2b-it choice reused as the default assumption)", flush=True)

    print("=" * 70, flush=True)
    print("[entrypoint] Stage 2: finetune_sppft.py (D_N scenario)", flush=True)
    print("=" * 70, flush=True)
    sh([sys.executable, "finetune_sppft.py", "--base_model", MODEL_PATH,
        "--data_path", "prompts/finetune_normal.json", "--output_dir", CHECKPOINT_DIR,
        "--begin_layer", str(begin_layer), "--end_layer", str(end_layer)])

    prompt_sets = [
        ("advbench_malicious.csv", "asr_advbench.json"),
        ("harmbench_eval.csv", "asr_harmbench.json"),
        ("ood_semantic_test.csv", "asr_semantic_ood.json"),
        ("attack_ood_jailbreakllms.csv", "asr_attack_ood.json"),
        ("ood_full_pool.csv", "asr_full_pool.json"),
    ]
    for prompts_file, out_file in prompt_sets:
        print("=" * 70, flush=True)
        print(f"[entrypoint] run_asr.py on {prompts_file}", flush=True)
        print("=" * 70, flush=True)
        sh([sys.executable, "run_asr.py", "--model_path", CHECKPOINT_DIR,
            "--prompts_path", f"prompts/{prompts_file}", "--batch_size", "32", "--max_batch_tokens", "8192",
            "--out_path", f"/tmp/{out_file}"])

    SM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (SM_MODEL_DIR / "safety_layer_result.json").write_text(json.dumps(layer_result, indent=2))
    summary = {"onset_layer": onset_layer, "begin_layer": begin_layer, "end_layer": end_layer}
    for _, out_file in prompt_sets:
        src = Path(f"/tmp/{out_file}")
        if src.exists():
            data = json.load(open(src))
            summary[out_file] = {"asr": data.get("asr"), "n_prompts": data.get("n_prompts")}
            (SM_MODEL_DIR / out_file).write_text(json.dumps(data, indent=2))
    json.dump(summary, open(SM_MODEL_DIR / "asr_summary.json", "w"), indent=2)
    print("\n[entrypoint] FINAL ASR SUMMARY:", json.dumps(summary, indent=2), flush=True)
    print("[entrypoint] Done.", flush=True)


if __name__ == "__main__":
    main()
