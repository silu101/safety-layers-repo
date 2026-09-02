"""
Launch the OOD semantic-similarity pipeline (steps 6+7 of the OOD dataset
construction process) on a single GPU instance -- offloaded from local
CPU after diagnosing a local environment issue (heavy sentence-transformers
5.x import chain + accumulated zombie process CPU contention) that made
this impractical to run locally.

Usage:
    python sagemaker/launch_ood_pipeline.py
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
    entry_point="entrypoint_ood_pipeline.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g5.xlarge",  # small/cheap GPU instance -- embedding
                                    # ~60k short prompts is light work, no
                                    # need for anything bigger.
    instance_count=1,
    volume_size=50,
    sagemaker_session=session,
    base_job_name="safety-layers-ood-pipeline",
    max_run=1 * 60 * 60,  # 1 hour -- generous headroom; the actual work
                          # (dataset loads + embeddings + dedup + sampling)
                          # should take well under 20 minutes on GPU.
    environment={
        "HF_TOKEN": hf_token,
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
