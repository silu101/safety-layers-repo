"""
Launch the Section 3.4 o_proj-exclusion re-run (docs/KNOWN_DISCREPANCIES.md
#17) as a SageMaker training job on ml.g6e.xlarge.

Uses sagemaker/entrypoint_localization.py, which is fully self-contained
(clones the repo from GitHub itself) -- so source_dir here only needs to
ship that one file, keeping the upload tiny.

Usage:
    python sagemaker/launch_localization_openreview.py
"""
import os
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch

ROLE_ARN = "arn:aws:iam::344977996863:role/safety-layers-sagemaker-execution-role"
REGION = "us-east-1"

# HF token for the gated google/gemma-2b-it model -- read from a local,
# never-committed file (NOT pasted into chat, NOT hardcoded here) and
# passed to the job as an environment variable only. See
# sagemaker/entrypoint_localization.py's hf_login().
HF_TOKEN_FILE = Path.home() / ".hf_token_safety_layers"
hf_token = HF_TOKEN_FILE.read_text().strip() if HF_TOKEN_FILE.exists() else None
if not hf_token:
    raise SystemExit(
        f"No HF token found at {HF_TOKEN_FILE}. Create it first: "
        f'echo "hf_your_real_token" > {HF_TOKEN_FILE} && chmod 600 {HF_TOKEN_FILE}'
    )

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_localization.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.xlarge",
    instance_count=1,
    sagemaker_session=session,
    base_job_name="safety-layers-localization-openreview",
    max_run=3 * 60 * 60,  # 3 hours ceiling -- this is inference-only over
                          # 731 prompts x 2 configs, should finish well
                          # under this; the cap is a cost/safety guard.
    environment={"HF_TOKEN": hf_token},
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
