#!/usr/bin/env bash
# Push kaggle/kernel_runner_localization.py to Kaggle as a script kernel
# (GPU on, internet on), poll until it finishes, then pull the output
# (results/ folder with localization_result.json + run_metadata.json for
# the smoketest, baseline, target-range, and control-range runs) into
# results/kaggle_pull_<timestamp>/.
#
# Prereqs: same as scripts/run_on_kaggle.sh -- kaggle CLI, credentials at
# ~/.kaggle/kaggle.json, and (if the model is gated) an HF_TOKEN Kaggle
# Secret enabled for THIS kernel slug specifically (secrets are per-kernel,
# not per-account -- see docs/KNOWN_DISCREPANCIES.md notes on this).
#
# Usage:
#   ./scripts/run_localization_on_kaggle.sh [model-prefix]
#   model-prefix matches configs/<prefix>_localization.yaml (default: gemma)
set -euo pipefail

MODEL_PREFIX="${1:-gemma}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="$REPO_ROOT/kaggle"
PUSH_DIR="$(mktemp -d)"
trap 'rm -rf "$PUSH_DIR"' EXIT

if [[ ! -f "$REPO_ROOT/configs/${MODEL_PREFIX}_localization.yaml" ]]; then
  echo "ERROR: configs/${MODEL_PREFIX}_localization.yaml not found." >&2
  exit 1
fi

command -v kaggle >/dev/null 2>&1 || {
  echo "ERROR: 'kaggle' CLI not found. Run: pip install kaggle" >&2
  exit 1
}

if [[ ! -f "$HOME/.kaggle/access_token" && ! -f "$HOME/.kaggle/kaggle.json" \
      && -z "${KAGGLE_API_TOKEN:-}" && -z "${KAGGLE_KEY:-}" ]]; then
  echo "ERROR: no Kaggle credential found." >&2
  exit 1
fi

if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
  USERNAME="$KAGGLE_USERNAME"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  USERNAME="$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"
else
  echo "ERROR: could not determine Kaggle username. Set KAGGLE_USERNAME." >&2
  exit 1
fi

KERNEL_SLUG="safety-layers-localization-repro-${MODEL_PREFIX}"
KERNEL_ID="$USERNAME/$KERNEL_SLUG"

echo "== Pushing kernel $KERNEL_ID (model prefix: $MODEL_PREFIX) =="
sed "s/^MODEL_CONFIG_PREFIX = \"gemma\"/MODEL_CONFIG_PREFIX = \"${MODEL_PREFIX}\"/" \
  "$KAGGLE_DIR/kernel_runner_localization.py" > "$PUSH_DIR/kernel_runner_localization.py"
sed -e "s#KAGGLE_USERNAME/safety-layers-localization-repro#$KERNEL_ID#" \
    -e "s#\"title\": \"safety-layers-localization-repro\"#\"title\": \"$KERNEL_SLUG\"#" \
  "$KAGGLE_DIR/kernel-metadata-localization.json.template" > "$PUSH_DIR/kernel-metadata.json"

kaggle kernels push -p "$PUSH_DIR"

echo "== Polling status for $KERNEL_ID (Ctrl-C to stop polling; the run keeps going on Kaggle) =="
STATUS=""
while true; do
  RAW="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
  STATUS="$(echo "$RAW" | tr '[:upper:]' '[:lower:]')"
  echo "$(date -u +%H:%M:%S) raw: $RAW"
  if echo "$STATUS" | grep -q "complete"; then
    break
  elif echo "$STATUS" | grep -qE "error|cancel"; then
    echo "Kernel run ended with a non-success status -- see raw output above." >&2
    break
  fi
  sleep 60
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PULL_DIR="$REPO_ROOT/results/kaggle_pull_localization_${TIMESTAMP}"
mkdir -p "$PULL_DIR"

echo "== Pulling output to $PULL_DIR =="
kaggle kernels output "$KERNEL_ID" -p "$PULL_DIR"

echo "Done. Kernel log + output files are in: $PULL_DIR"
echo "Look for results/<run_name>/localization_result.json inside the pulled output"
echo "(the kernel ran from a cloned repo under /kaggle/working, so results/ is nested)."
