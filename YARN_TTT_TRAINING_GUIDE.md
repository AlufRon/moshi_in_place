# YaRN + TTT Training Guide

## Overview

This guide explains how to train Moshi with **YaRN position scaling** and **TTT (Test-Time Training)** to extend context window while keeping the base model frozen.

## Key Concepts

### YaRN (Yet another RoPE extensioN)
- Extends context window by modifying RoPE frequency calculations
- **No model weight changes** - purely computational modification
- Efficient: 10x fewer tokens needed vs. other methods
- Paper: [Peng et al., 2023](https://arxiv.org/abs/2309.00071)

### In-Place TTT
- Adapts MLP fast weights during inference/training
- Works on different component than YaRN (MLP vs attention)
- Can be trained while base model is frozen
- Paper: TTT In-Place (ICLR 2026 submission)

### Why This Combination Works

1. **YaRN** scales position encodings in frozen attention layers
2. **TTT** learns to compress long-context information in MLP blocks
3. **Orthogonal modifications**: No parameter conflicts
4. **Frozen base model**: Preserve pre-trained knowledge, train only TTT

## Configuration

### Training Modes: Three Options

YaRN + TTT supports **three training modes** with different trade-offs:

#### Mode 1: TTT-Only (Most Efficient) ✨ **RECOMMENDED TO START**
```yaml
ttt:
  enabled: true
  unfreeze_ttt_layers: false  # Only train TTT params
full_finetuning: false
lora:
  enable: false

# Trains: ~4M params (0.1% of model)
# Memory: Low
# Speed: Fast
# Use when: Maximum efficiency, limited compute
```

#### Mode 2: TTT-Layer Unfreezing (Balanced) 🎯 **BEST FOR YARN**
```yaml
ttt:
  enabled: true
  unfreeze_ttt_layers: true   # Train entire layers with TTT
full_finetuning: false
lora:
  enable: false

# Trains: ~625M params (15.6% of model, ~5-6 layers)
# Memory: Medium
# Speed: Medium
# Use when: YaRN needs attention to co-adapt
```

#### Mode 3: Full Finetuning (Maximum Performance) 📊 **MATCHES TTT PAPER**
```yaml
full_finetuning: true
ttt:
  enabled: true
yarn:
  enabled: true

# Trains: ~4000M params (100% of model)
# Memory: High
# Speed: Slow
# Use when: Maximum performance needed
```

---

### Recommended Configuration (Mode 2: TTT-Layer Unfreezing)

```yaml
# YaRN Configuration
yarn:
  enabled: true
  scale: 4.0                      # Context extension factor
  original_max_seq_len: 3000      # Moshi's base context length
  beta_fast: 32                   # Default: handles low frequencies
  beta_slow: 1                    # Default: handles high frequencies
  mscale: 1.0                     # Default: attention scaling
  mscale_all_dim: 0.0             # Default: no extra scaling

# TTT Configuration
ttt:
  enabled: true
  layer_frequency: 6              # TTT every 6 layers
  start_layer: 5                  # Begin at layer 5
  chunk_size: 256                 # Update granularity
  learning_rate: 1e-3             # Fast weight LR
  conv_kernel_size: 2             # Target generator kernel
  unfreeze_ttt_layers: true       # KEY: Unfreeze full layers with TTT

# Training Mode
full_finetuning: false            # Not full finetuning
lora:
  enable: false                   # Disable LoRA

# Optimizer (adjusted for more parameters)
optim:
  lr: 5e-5                        # Lower LR for attention layers
  weight_decay: 0.1
```

## Progressive Training Strategy

The TTT paper used a **two-stage curriculum**:

### Stage 1: Warm-up (32k context)
```yaml
yarn:
  enabled: false                  # No scaling yet
  scale: 1.0
# Train for ~20B tokens
```

### Stage 2: Extended context (128k)
```yaml
yarn:
  enabled: true
  scale: 4.27                     # For 128k from 30k base
# Train for ~15B tokens
```

### Recommended for Moshi (base context=3000):

| Stage | Context | YaRN Scale | Duration |
|-------|---------|------------|----------|
| 1     | 3k      | 1.0        | 1-2k steps (warm-up) |
| 2     | 6k      | 2.0        | 2-3k steps |
| 3     | 12k     | 4.0        | 5-10k steps |
| 4     | 24k     | 8.0        | 10k+ steps (optional) |

## Training Command

```bash
torchrun --nproc_per_node=8 \\
  moshi-finetune/train.py \\
  configs/yarn_ttt_long_context_example.yaml
```

## What Gets Trained

Depends on the training mode:

### Mode 1: TTT-Only (unfreeze_ttt_layers=false)

**Frozen (No Gradients):**
- ✗ All attention layers (Q, K, V, O projections)
- ✗ All input/output embeddings
- ✗ All LayerNorms
- ✗ All base MLP projections

**Trained (Gradients Enabled):**
- ✓ TTT target generators only (`conv1d`, `W_target`)
- Total: ~4M params (0.1% of model)

**Modified at Runtime (Not Parameters):**
- → RoPE frequencies (via YaRN) - applied during forward pass

---

### Mode 2: TTT-Layer Unfreezing (unfreeze_ttt_layers=true) ⭐

**Frozen (No Gradients):**
- ✗ Layers WITHOUT TTT (e.g., layers 0-4, 6-10, 12-16, etc.)
- ✗ Input/output embeddings

**Trained (Gradients Enabled):**
- ✓ **Entire layers with TTT** (typically 5-6 layers):
  - Self-attention (Q, K, V, O projections)
  - LayerNorms
  - MLP (linear_in, linear_out, TTT)
  - TTT target generators
- Total: ~625M params (15.6% of model)

**Modified at Runtime (Not Parameters):**
- → RoPE frequencies (via YaRN) - applied during forward pass

**Why This Works Best with YaRN:**
- Attention in TTT layers can adapt to YaRN-scaled positions
- TTT and attention co-adapt for long-range dependencies
- Still preserves 84% of base model (no forgetting)
- Much more efficient than full finetuning

---

### Mode 3: Full Finetuning (full_finetuning=true)

**Frozen (No Gradients):**
- None

**Trained (Gradients Enabled):**
- ✓ Everything (all attention, embeddings, MLP, TTT)
- Total: ~4000M params (100% of model)

**Modified at Runtime (Not Parameters):**
- → RoPE frequencies (via YaRN) - applied during forward pass

## Monitoring Training

### Key Metrics to Watch

1. **Perplexity**: Should improve on long-context eval
2. **TTT Gradient Norms**: Logged automatically every `log_freq` steps
   ```
   [TTT] Step 100: grad_norm=0.0234 (12 params)
   ```
   - Should be stable (not NaN or exploding)
   - Typical range: 0.01 - 0.1

3. **Memory Usage**: Longer sequences use more memory
   - Reduce `batch_size` or `duration_sec` if OOM
   - Enable `gradient_checkpointing: true`

### Expected Logs

```
[YaRN] Enabled with scale=4.0, original_len=3000
[TTT] Initialized TTT layers at indices: [5, 11, 17, 23, 29]
[TTT] Initializing 15 TTT parameters (not in base checkpoint)
[TTT] Step 10: grad_norm=0.0456 (15 params)
```

## Validation

### Long-Context Benchmarks

After training, evaluate on:
- **RULER**: Standard long-context benchmark
- **Passkey retrieval**: Find information in long documents
- **Multi-document QA**: Reason across multiple texts

### Inference with YaRN

The trained checkpoint saves YaRN config automatically:
```python
# Load model - YaRN is applied automatically
model = checkpoint_info.get_moshi(device="cuda")

# Process long sequences (up to extended length)
output = model(long_input_tokens)  # Works with 12k tokens
```

## Troubleshooting

### Issue: OOM (Out of Memory)
**Solution**: Reduce batch size or sequence length
```yaml
batch_size: 1
duration_sec: 5.0  # Shorter audio segments
gradient_checkpointing: true
```

### Issue: TTT gradients are zero
**Solution**: Check that TTT layers are not frozen
```python
# Verify TTT params require gradients
for name, param in model.named_parameters():
    if 'target_generator' in name or 'w_down' in name:
        assert param.requires_grad, f"{name} should require gradients!"
```

### Issue: Poor long-context performance
**Potential causes**:
1. Not enough training steps at extended length
2. YaRN scale too aggressive (try smaller increments)
3. TTT chunk_size too large (reduces adaptation granularity)

**Solutions**:
- Train longer at each scale stage
- Use progressive scaling (2x → 4x → 8x)
- Reduce `chunk_size` to 128 or 64 for finer updates

## Technical Details

### How YaRN Modifies RoPE

Standard RoPE:
```python
freqs = 1.0 / (theta ** (dim_indices / dim))
```

YaRN (NTK-by-parts):
```python
# Low frequencies: scale down (longer wavelengths)
freqs_low = freqs / scale

# High frequencies: unchanged (short wavelengths)
freqs_high = freqs

# Blend with linear ramp
freqs_final = (1 - ramp) * freqs_high + ramp * freqs_low
```

This allows the model to:
- **Interpolate** positions it's seen (within original context)
- **Extrapolate** to longer positions (extended context)
- **Maintain** short-range relationships (high frequencies unchanged)

### Memory Complexity

| Component | Memory |
|-----------|--------|
| Base model (frozen) | O(params) |
| Activations | O(batch × seq_len × dim) |
| TTT fast weights | ~0.1% of base model |
| YaRN | 0 (no parameters) |

**Total**: Dominated by activations with long sequences

## References

1. **YaRN Paper**: [Peng et al., 2023 - Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
2. **TTT In-Place Paper**: Under review at ICLR 2026 (see `papers/ttt_in_place_paper.txt`)
3. **RoPE**: [Su et al., 2022 - Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

## Training Mode Comparison

| Feature | TTT-Only | TTT-Layer Unfreezing ⭐ | Full Finetuning |
|---------|----------|------------------------|-----------------|
| **Params Trained** | 0.1% (~4M) | 15.6% (~625M) | 100% (~4B) |
| **Memory Usage** | Low | Medium | High |
| **Training Speed** | Fast | Medium | Slow |
| **YaRN Compatibility** | Good | **Excellent** ✓ | Excellent |
| **Attention Adaptation** | ✗ Frozen | ✓ Co-adapts | ✓ Fully adapts |
| **Risk of Forgetting** | None | Low | Medium |
| **Recommended LR** | 1e-4 | 5e-5 | 1e-5 to 5e-5 |
| **Best For** | Quick baseline | **Production use** | Max performance |

### When to Use Each Mode:

**Use TTT-Only if:**
- Limited compute budget
- Need quick baseline results
- Don't want ANY risk of forgetting base model
- YaRN scale is modest (≤2x)

**Use TTT-Layer Unfreezing if:** ⭐ **RECOMMENDED**
- Using YaRN for context extension
- Want attention to adapt to long positions
- Have moderate compute (can train 15% of model)
- Need balance of performance and efficiency

**Use Full Finetuning if:**
- Maximum performance is critical
- Following TTT paper exactly
- Have sufficient compute budget
- Long training time is acceptable

## Example Training Scripts

### Mode 1: TTT-Only
```bash
# Fastest, most efficient
torchrun --nproc_per_node=8 moshi-finetune/train.py \
  configs/yarn_ttt_long_context_example.yaml
```
Config: [`configs/yarn_ttt_long_context_example.yaml`](configs/yarn_ttt_long_context_example.yaml)

### Mode 2: TTT-Layer Unfreezing ⭐ RECOMMENDED
```bash
# Best balance for YaRN
torchrun --nproc_per_node=8 moshi-finetune/train.py \
  configs/yarn_ttt_unfreeze_layers.yaml
```
Config: [`configs/yarn_ttt_unfreeze_layers.yaml`](configs/yarn_ttt_unfreeze_layers.yaml)

### Mode 3: Full Finetuning
```bash
# Maximum performance (matches paper)
# Create config with: full_finetuning: true
torchrun --nproc_per_node=8 moshi-finetune/train.py \
  configs/yarn_ttt_full_finetune.yaml
```

### Quick Start Workflow:
```bash
# 1. Prepare long-context dataset
# 2. Choose your mode (recommend Mode 2)
# 3. Edit config (set dataset paths, wandb project, etc.)
# 4. Run training
torchrun --nproc_per_node=8 moshi-finetune/train.py \
  configs/yarn_ttt_unfreeze_layers.yaml

# 5. Monitor with wandb or tensorboard
# 6. Evaluate on long-context benchmarks
```

## Conclusion

**YaRN + TTT** enables efficient context window extension:
- ✅ Frozen base model (no catastrophic forgetting)
- ✅ Minimal trainable parameters (fast training)
- ✅ Validated approach (TTT paper used YaRN)
- ✅ No architectural changes needed
- ✅ Works at inference without retraining

This is the recommended approach for adapting Moshi to long-context tasks.
