#!/usr/bin/env bash
#SBATCH --job-name=yarn-ttt
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_6000:1
#SBATCH --output=/home/alufr/moshi_in_place_ttt/moshi_in_place/moshi-finetune/logs/yarn_ttt_%j.out
#SBATCH --error=/home/alufr/moshi_in_place_ttt/moshi_in_place/moshi-finetune/logs/yarn_ttt_%j.err

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

# Navigate to working directory
cd /home/alufr/moshi_in_place_ttt/moshi_in_place/moshi-finetune

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with YARN + TTT
echo "Starting YARN + TTT training..."
echo "Context Extension: 4x (3000 -> 12000 tokens)"
echo "TTT Layers: 3 (layers 10, 20, 30)"
echo "Base Model: Frozen (only TTT params trained)"
torchrun --nproc-per-node 1 --master_port $(shuf -i 30000-50000 -n 1) \
    train.py example/moshi_7B.yaml

echo "=================================================="
echo "Job finished at: $(date)"
echo "=================================================="
