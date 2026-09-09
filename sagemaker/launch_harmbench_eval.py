"""
Launch the HarmBench near-OOD baseline evaluation for gemma_sppft_normal
(condition 2 of the AdvBench -> HarmBench -> unfiltered-OOD -> filtered-OOD
comparison design). See entrypoint_harmbench_eval.py.

Usage:
    python sagemaker/launch_harmbench_eval.py
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

ANTHROPIC_KEY_FILE = Path.home() / ".anthropic_key_safety_layers"
anthropic_key = ANTHROPIC_KEY_FILE.read_text().strip() if ANTHROPIC_KEY_FILE.exists() else None
if not anthropic_key:
    raise SystemExit(f"No Anthropic key found at {ANTHROPIC_KEY_FILE}.")

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_harmbench_eval.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.xlarge",
    instance_count=1,
    volume_size=150,
    sagemaker_session=session,
    base_job_name="safety-layers-harmbench-eval",
    max_run=3 * 60 * 60,  # same 3hr headroom as the OOD eval job -- 400
                          # prompts is smaller than that job's 520, so this
                          # should finish faster, but reusing a guessed
                          # tighter budget isn't worth the risk of a repeat
                          # MaxRuntimeExceeded loss.
    environment={
        "HF_TOKEN": hf_token,
        "ANTHROPIC_API_KEY": anthropic_key,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
