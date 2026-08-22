#!/usr/bin/env bash
# Push kaggle/kernel_runner.py to Kaggle as a script kernel (GPU on,
# internet on), poll until it finishes, then pull the output (results/
# folder with all_cos.pkl + run_metadata.json for both the smoketest and
# full runs) into results/kaggle_pull_<timestamp>/.
#
# Prereqs:
#   - `pip install kaggle` (the classic 1.7.x client -- confirmed working
#     as of 2026-08-19; Kaggle's newer 2.x CLI + KGAT_... access-token auth
#     is currently broken server-side: /IntrospectToken 404s, and even
#     legacy kaggle.json credentials get a 401 through its newer
#     ListKernels endpoint. Stick with the classic client until that's
#     fixed upstream.)
#   - Kaggle credentials at ~/.kaggle/kaggle.json (kaggle.com > Settings >
#     API > "Create New API Token" -- gives username directly, so no
#     separate KAGGLE_USERNAME env var is needed).
#   - An HF_TOKEN Kaggle Secret added to your account (kernel editor >
#     Add-ons > Secrets) -- this can't be set via the API, it's a one-time
#     manual step on kaggle.com, and must be enabled PER KERNEL SLUG (a
#     new model prefix here creates a new kernel the first time, which
#     needs the secret enabled again even if it's already on your account).
#
# Usage:
#   ./scripts/run_on_kaggle.sh [model-prefix]
#   model-prefix matches configs/<prefix>_cossim.yaml (default: phi3)
set -euo pipefail

MODEL_PREFIX="${1:-phi3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_DIR="$REPO_ROOT/kaggle"
PUSH_DIR="$(mktemp -d)"
trap 'rm -rf "$PUSH_DIR"' EXIT

if [[ ! -f "$REPO_ROOT/configs/${MODEL_PREFIX}_cossim.yaml" ]]; then
  echo "ERROR: configs/${MODEL_PREFIX}_cossim.yaml not found." >&2
  exit 1
fi

command -v kaggle >/dev/null 2>&1 || {
  echo "ERROR: 'kaggle' CLI not found. Run: pip install kaggle (with a" >&2
  echo "Python 3.11+ interpreter -- older Pythons resolve to the legacy client)." >&2
  exit 1
}

# --- verify some form of Kaggle credential is present (file or env var) ---
if [[ ! -f "$HOME/.kaggle/access_token" && ! -f "$HOME/.kaggle/kaggle.json" \
      && -z "${KAGGLE_API_TOKEN:-}" && -z "${KAGGLE_KEY:-}" ]]; then
  echo "ERROR: no Kaggle credential found (~/.kaggle/access_token," >&2
  echo "~/.kaggle/kaggle.json, KAGGLE_API_TOKEN, or KAGGLE_KEY)." >&2
  exit 1
fi

# --- resolve Kaggle username (needed for the kernel id) ---
if [[ -n "${KAGGLE_USERNAME:-}" ]]; then
  USERNAME="$KAGGLE_USERNAME"
elif [[ -f "$HOME/.kaggle/kaggle.json" ]]; then
  USERNAME="$(python3 -c "import json; print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"
else
  echo "ERROR: could not determine Kaggle username. The access-token auth" >&2
  echo "method doesn't carry a username, so set KAGGLE_USERNAME explicitly" >&2
  echo "(e.g. export KAGGLE_USERNAME=your_username)." >&2
  exit 1
fi

KERNEL_SLUG="safety-layers-cossim-repro-${MODEL_PREFIX}"
KERNEL_ID="$USERNAME/$KERNEL_SLUG"

echo "== Pushing kernel $KERNEL_ID (model prefix: $MODEL_PREFIX) =="
sed "s/^MODEL_CONFIG_PREFIX = \"phi3\"/MODEL_CONFIG_PREFIX = \"${MODEL_PREFIX}\"/" \
  "$KAGGLE_DIR/kernel_runner.py" > "$PUSH_DIR/kernel_runner.py"
sed -e "s#KAGGLE_USERNAME/safety-layers-cossim-repro#$KERNEL_ID#" \
    -e "s#\"title\": \"safety-layers-cossim-repro\"#\"title\": \"$KERNEL_SLUG\"#" \
  "$KAGGLE_DIR/kernel-metadata.json.template" > "$PUSH_DIR/kernel-metadata.json"

kaggle kernels push -p "$PUSH_DIR"

echo "== Polling status for $KERNEL_ID (Ctrl-C to stop polling; the run keeps going on Kaggle) =="
STATUS=""
while true; do
  RAW="$(kaggle kernels status "$KERNEL_ID" 2>&1 || true)"
  # New CLI prints: <kernel> has status "COMPLETE" (case/enum-name may vary);
  # match on keyword substrings rather than an exact literal.
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
PULL_DIR="$REPO_ROOT/results/kaggle_pull_${TIMESTAMP}"
mkdir -p "$PULL_DIR"

echo "== Pulling output to $PULL_DIR =="
kaggle kernels output "$KERNEL_ID" -p "$PULL_DIR"

echo "Done. Kernel log + output files are in: $PULL_DIR"
echo "Look for results/<run_name>/all_cos.pkl and run_metadata.json inside the pulled output"
echo "(the kernel ran from a cloned repo under /kaggle/working, so results/ is nested)."
