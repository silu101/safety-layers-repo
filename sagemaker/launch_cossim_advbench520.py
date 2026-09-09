"""
Launch the AdvBench-520 localization cosine-similarity check (see
entrypoint_cossim_advbench520.py / configs/gemma_cossim_advbench520.yaml).

Usage:
    python sagemaker/launch_cossim_advbench520.py
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
    entry_point="entrypoint_cossim_advbench520.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.xlarge",
    instance_count=1,
    volume_size=30,
    sagemaker_session=session,
    base_job_name="safety-layers-cossim-advbench520",
    max_run=1 * 60 * 60,  # light job: 500 pairs x 3 pair-types, max_new_tokens=1 each
    environment={
        "HF_TOKEN": hf_token,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
