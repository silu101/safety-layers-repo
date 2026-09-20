"""
Launch the FULL (not smoke-test) refusal-direction-ood run: real paper
scale (n_train=128, n_val=32), all 4 ASR eval sets (AdvBench, HarmBench,
semantic-OOD, attack-OOD) against meta-llama/Meta-Llama-3-8B-Instruct.

Usage:
    python sagemaker/launch_refusal_direction_full.py
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"

# Confirmed to have approved Llama access (the safety_layers token hit a
# 403 GatedRepoError on this exact model).
HF_TOKEN_FILE = Path.home() / ".hf_token_llama"
hf_token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
if not hf_token:
    raise SystemExit(f"No HF token found at {HF_TOKEN_FILE}.")

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_refusal_direction_full.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    # g6e.4xlarge is what actually cleared capacity for this job on
    # 2026-09-18 after 3 stalls on g6e.2xlarge -- same 48GB L40S either
    # way, just a different size/capacity pool. If this one stalls too,
    # try g6e.8xlarge or g6e.xlarge next, same GPU spec.
    instance_type="ml.g6e.4xlarge",
    instance_count=1,
    volume_size=150,
    sagemaker_session=session,
    base_job_name="refusal-direction-full",
    max_run=4 * 60 * 60,  # real scale: extract_direction (~128+32 examples,
                          # vs. the smoke test's 10+5) + 4 full ASR sweeps
                          # (520+400+520+520=1960 prompts total, sequential
                          # generation, classifier reloaded fresh each of
                          # the 4 run_asr.py calls) -- generous margin over
                          # a rough ~2-3hr estimate, first real-scale run.
    environment={
        "HF_TOKEN": hf_token,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
