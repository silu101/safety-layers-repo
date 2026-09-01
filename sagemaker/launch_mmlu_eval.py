"""
Launch Section 4's MMLU (utility-side, S_m) evaluation against all 4
trained models. Single g6e.xlarge, inference-only, no Anthropic API
needed (no credit-exhaustion risk like the harmful-eval job hit) --
should be faster than that job too, since each question only needs 4
generated tokens instead of 128.

Usage:
    python sagemaker/launch_mmlu_eval.py
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"

HF_TOKEN_FILE = Path.home() / ".hf_token_safety_layers"
hf_token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
if not hf_token:
    raise SystemExit(f"No HF token found at {HF_TOKEN_FILE}.")

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_mmlu_eval.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.xlarge",
    instance_count=1,
    volume_size=250,  # same headroom reasoning as launch_harmful_eval.py
    sagemaker_session=session,
    base_job_name="safety-layers-mmlu-eval",
    max_run=3 * 60 * 60,  # 3 hours -- 4 models x 500 questions x 4 tokens
                          # each should be much faster than the harmful-eval
                          # job's 128-token generations + judge calls.
    environment={
        "HF_TOKEN": hf_token,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
