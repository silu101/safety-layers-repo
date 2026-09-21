"""
Launch the run_asr_parallel.py multi-GPU smoke test (fresh clone from
GitHub) on a 4x-GPU instance. See entrypoint_multi_gpu_smoketest.py.

Usage:
    python sagemaker/launch_multi_gpu_smoketest.py
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"

HF_TOKEN_FILE = Path.home() / ".hf_token_llama"
hf_token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
if not hf_token:
    raise SystemExit(f"No HF token found at {HF_TOKEN_FILE}.")

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_multi_gpu_smoketest.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.12xlarge",  # 4x L40S 48GB each -- smallest multi-GPU size in this family
    instance_count=1,
    volume_size=150,
    sagemaker_session=session,
    base_job_name="multi-gpu-smoketest",
    max_run=30 * 60,  # small scale (20 prompts) -- just model download + a few batches per GPU
    environment={
        "HF_TOKEN": hf_token,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
