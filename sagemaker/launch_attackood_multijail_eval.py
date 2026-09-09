"""
Launch the MultiJail covariate/multilingual Attack-OOD evaluation:
gemma_sppft_normal tested on 2,835 prompts (315 base x 9 languages). See
entrypoint_attackood_multijail_eval.py and scripts/build_attack_ood_multijail.py
for the known content-source confound accepted per explicit user decision.

Usage:
    python sagemaker/launch_attackood_multijail_eval.py
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
    entry_point="entrypoint_attackood_multijail_eval.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.2xlarge",
    instance_count=1,
    volume_size=150,
    sagemaker_session=session,
    base_job_name="safety-layers-attackood-multijail",
    max_run=5 * 60 * 60,  # 2,835 prompts, ~5.5x the usual 520-prompt run
                          # size -- scaling the per-prompt rate observed
                          # on that run (~4s/prompt combined pipeline)
                          # gives ~3.4hrs variable + fixed setup cost;
                          # 5hrs leaves real margin instead of a tight guess.
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
