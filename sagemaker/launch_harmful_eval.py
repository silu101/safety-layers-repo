"""
Launch Section 4's harmful-rate (R_h, via both Zou et al. 2023 keyword
classifier and HarmBench's official classifier) + harmful-score (S_h, via
Claude Haiku 4.5) evaluation on AdvBench (520 prompts) against all 4
trained models from earlier Section 4 jobs.

Single g6e.xlarge is enough -- this is inference-only (target model +
HarmBench's 13B classifier, both bf16/fp32, comfortably fit one 48GB GPU;
no training, no optimizer state).

Usage:
    python sagemaker/launch_harmful_eval.py
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
    raise SystemExit(
        f"No Anthropic key found at {ANTHROPIC_KEY_FILE}. Create it first: "
        f'echo "sk-ant-your_real_key" > {ANTHROPIC_KEY_FILE} && chmod 600 {ANTHROPIC_KEY_FILE}'
    )

session = sagemaker.Session(boto_session=__import__("boto3").Session(region_name=REGION))

estimator = PyTorch(
    entry_point="entrypoint_harmful_eval.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.xlarge",
    instance_count=1,
    # Default EBS volume likely isn't enough: the first Section 4 job's
    # tarball alone (both SPPFT models) is ~55GB, plus 2 more Full FT
    # tarballs (~8GB each), plus the HarmBench classifier (~26GB bf16),
    # plus the target model being evaluated (~8GB fp32) -- generous
    # headroom since EBS cost is trivial next to compute cost.
    volume_size=250,
    sagemaker_session=session,
    base_job_name="safety-layers-harmful-eval",
    max_run=4 * 60 * 60,  # 4 hours ceiling -- 4 models x 520 AdvBench
                          # prompts x (generation + 2 local classifiers +
                          # Haiku judge calls), generous cost/safety guard.
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
