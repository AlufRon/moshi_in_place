#!/usr/bin/env bash
#SBATCH --job-name=yarn-ttt-inference
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1
#SBATCH --output=/home/alufr/moshi_in_place_ttt/moshi_in_place/moshi-finetune/logs/yarn_ttt_inference_%j.out
#SBATCH --error=/home/alufr/moshi_in_place_ttt/moshi_in_place/moshi-finetune/logs/yarn_ttt_inference_%j.err

echo "=================================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi
echo "=================================================="

# Activate conda environment
source ~/.bashrc
conda activate moshi_ttt_fixed

# Explicitly set CUDA_VISIBLE_DEVICES before any Python/PyTorch imports
export CUDA_VISIBLE_DEVICES=0
echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Static paths
PROJECT_ROOT=/home/alufr/moshi_in_place_ttt/moshi_in_place
MOSHI_DIR=$PROJECT_ROOT/moshi/moshi
LOG_DIR=$PROJECT_ROOT/moshi-finetune/logs
TTT_MONITOR_BASE=${TTT_MONITOR_BASE:-$LOG_DIR/ttt_monitor}
TTT_CHUNK_SIZE=${TTT_CHUNK_SIZE:-64}

# Navigate to working directory
cd "$MOSHI_DIR"

# Create logs/monitor directories
mkdir -p "$LOG_DIR"
mkdir -p "$TTT_MONITOR_BASE"

RUN_MONITOR_DIR=$TTT_MONITOR_BASE/${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_MONITOR_DIR"
echo "TTT monitor dir: $RUN_MONITOR_DIR"

# Resolve checkpoint root (prefer env override, then shared path, then repo copy)
resolve_ckpt_root() {
    if [[ -n "$TTT_CKPT_ROOT" && -d "$TTT_CKPT_ROOT" ]]; then
        echo "$TTT_CKPT_ROOT"
        return
    fi

    SHARED_ROOT="/sise/eliyanac-group/ron_al/ttt_training_run2/checkpoints"
    if [[ -d "$SHARED_ROOT" ]]; then
        echo "$SHARED_ROOT"
        return
    fi

    LOCAL_ROOT="$(realpath ../../moshi-finetune/ttt_training_run2/checkpoints 2>/dev/null)"
    if [[ -n "$LOCAL_ROOT" && -d "$LOCAL_ROOT" ]]; then
        echo "$LOCAL_ROOT"
        return
    fi
}

CKPT_ROOT=$(resolve_ckpt_root)
if [[ -z "$CKPT_ROOT" ]]; then
    echo "[ERROR] Unable to locate TTT checkpoint directory. Set TTT_CKPT_ROOT to a valid path." >&2
    exit 1
fi

LATEST_CKPT=$(ls -1d "$CKPT_ROOT"/checkpoint_* 2>/dev/null | sort | tail -n 1)
if [[ -z "$LATEST_CKPT" ]]; then
    echo "[ERROR] No checkpoint_* directories found under $CKPT_ROOT" >&2
    exit 1
fi

LORA_WEIGHTS="$LATEST_CKPT/consolidated/lora.safetensors"
if [[ ! -f "$LORA_WEIGHTS" ]]; then
    echo "[ERROR] Expected LoRA weights at $LORA_WEIGHTS but file is missing." >&2
    exit 1
fi

echo "Starting YARN + TTT inference..."
echo "Checkpoint root: $CKPT_ROOT"
echo "Selected checkpoint: $(basename "$LATEST_CKPT")"
echo "LoRA weights: $LORA_WEIGHTS"
echo "TTT chunk size override: $TTT_CHUNK_SIZE"
echo "Input: /sise/eliyanac-group/ron_al/examples/combined_2198538934964547889_24000hz.wav"
echo "Output: ./out.wav"

python -m moshi.run_inference \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --lora-weights "$LORA_WEIGHTS" \
    --ttt-chunk-size "$TTT_CHUNK_SIZE" \
    --ttt-monitor-dir "$RUN_MONITOR_DIR" \
    "/sise/eliyanac-group/ron_al/examples/combined_1595476768245534437_24000hz.wav" \
    ./out.wav

echo "=================================================="
echo "Job finished at: $(date)"
echo "Output saved to: ./out.wav"
echo "=================================================="
