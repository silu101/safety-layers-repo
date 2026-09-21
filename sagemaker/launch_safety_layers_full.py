"""
Launch the full-scale Safety Layers run (find_safety_layer -> finetune_
sppft D_N -> 5-set ASR sweep) on meta-llama/Meta-Llama-3-8B-Instruct.
See entrypoint_safety_layers_full.py.

Usage:
    python sagemaker/launch_safety_layers_full.py
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
    entry_point="entrypoint_safety_layers_full.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.4xlarge",  # same instance the Refusal Direction full run used successfully
    instance_count=1,
    volume_size=150,
    sagemaker_session=session,
    base_job_name="safety-layers-full",
    max_run=7 * 60 * 60,  # find_safety_layer (~3000 sequential 1-token forward passes, cheap) +
                          # SPPFT fine-tune on 1,000 D_N examples, 3 epochs (new territory on
                          # Llama-3-8B -- no prior timing for this specific combo) + the same
                          # 5-set ASR sweep the Refusal Direction run took ~3h45m for. Generous
                          # margin since the fine-tune stage's real duration isn't measured yet.
    environment={
        "HF_TOKEN": hf_token,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
)

if __name__ == "__main__":
    estimator.fit(wait=True, logs=True)
    print("Job name:", estimator.latest_training_job.name)
    print("Model artifacts:", estimator.model_data)
