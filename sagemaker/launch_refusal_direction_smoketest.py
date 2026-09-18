"""
Launch the refusal-direction-ood smoke test (fresh clone from GitHub).
See entrypoint_refusal_direction_smoketest.py.

Usage:
    python sagemaker/launch_refusal_direction_smoketest.py
"""
import sagemaker
from sagemaker.pytorch import PyTorch
from pathlib import Path

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"

# Uses .hf_token_llama specifically, not .hf_token_safety_layers -- the
# prior run hit a 403 GatedRepoError on meta-llama/Meta-Llama-3-8B-Instruct
# with the safety_layers token; this one exists separately and is very
# likely the one with approved Llama access.
HF_TOKEN_FILE = Path.home() / ".hf_token_llama"
hf_token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
if not hf_token:
    raise SystemExit(f"No HF token found at {HF_TOKEN_FILE}.")

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_refusal_direction_smoketest.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    # g6e.2xlarge hit a persistent capacity stall (3 consecutive attempts,
    # 20+ min Pending each) for this specific job on 2026-09-18 -- trying
    # a different size within the same GPU family (same 48GB L40S) since
    # that's what unstuck an earlier stall elsewhere in this project.
    instance_type="ml.g6e.4xlarge",
    instance_count=1,
    volume_size=150,  # two cloned repos + Llama-3-8B weights (~16GB) + HarmBench judge (~26GB)
    sagemaker_session=session,
    base_job_name="refusal-direction-smoketest",
    max_run=2 * 60 * 60,  # first real run of an external repo, real uncertainty -- generous budget
    environment={
        "HF_TOKEN": hf_token,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
