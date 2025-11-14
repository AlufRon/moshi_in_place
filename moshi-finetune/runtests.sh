#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the project's test-suite under the moshi_ttt_fixed conda env.
# Designed to be whitelisted for automatic execution via VS Code chat.tools.terminal.autoApprove.

# Activate conda environment
# Note: this uses `conda` shell hook which should be available in an interactive shell.
eval "$(conda shell.bash hook)"
conda activate moshi_ttt_fixed

# Move to finetune folder and run pytest with repository in PYTHONPATH
cd "$(dirname "$0")"
PYTHONPATH=/home/alufr/moshi_in_place_ttt/moshi_in_place pytest -q tests/test_yarn_rope.py tests/test_run_inference_config.py tests/test_ring_kvcache.py tests/test_loaders_integration.py
