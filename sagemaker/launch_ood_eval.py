"""
Launch the first small-scale semantic-content OOD evaluation:
gemma_sppft_normal tested on 100 OOD prompts (20 each from the 5
confirmed-OOD categories) instead of AdvBench. Compare the resulting
R_h/S_h against this same model's already-documented AdvBench numbers in
docs/REPLICATION_LOG.md -- that delta is the actual generalization result.

Deliberately small (1 model x 100 prompts, vs. the full Section 4 job's
4 models x 520 prompts) so a teammate can read/extend this easily -- see
scripts/build_ood_semantic_test.py to regenerate a larger/different sample,
and entrypoint_ood_eval.py's MODEL_S3_URI/MODEL_SUBDIR to point at a
different trained model.

Usage:
    python sagemaker/launch_ood_eval.py
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
    entry_point="entrypoint_ood_eval.py",
    source_dir="sagemaker",
    role=ROLE_ARN,
    framework_version="2.3",
    py_version="py311",
    instance_type="ml.g6e.xlarge",
    instance_count=1,
    volume_size=150,  # one model tarball (~55GB, shared with sppft_implicit)
                      # + HarmBench classifier (~26GB) + target model (~8GB)
    sagemaker_session=session,
    base_job_name="safety-layers-ood-eval-semantic",
    max_run=1 * 60 * 60,  # 1 hour -- 100 prompts x (generation + 2 local
                          # classifiers + Haiku judge calls) is a fraction
                          # of the full 520-prompt x 4-model job's ~2.5-3hrs.
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
