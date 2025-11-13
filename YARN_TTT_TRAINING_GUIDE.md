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

### Example: 4x Context Extension (3000 → 12000 tokens)

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

# Training Mode
full_finetuning: false            # Freeze base, train only TTT
lora:
  enable: false                   # Disable LoRA
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

### Frozen (No Gradients):
- ✗ Attention layers (Q, K, V, O projections)
- ✗ Input/output embeddings
- ✗ LayerNorms
- ✗ Base MLP projections (except where TTT is applied)

### Trained (Gradients Enabled):
- ✓ TTT fast weights (`w_down` at selected layers)
- ✓ TTT target generators (`conv1d`, `W_target`)

### Modified at Runtime (Not Parameters):
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

## Example Training Script

See complete example: [`configs/yarn_ttt_long_context_example.yaml`](configs/yarn_ttt_long_context_example.yaml)

Quick start:
```bash
# 1. Prepare long-context dataset
# 2. Edit config (set dataset paths, wandb project, etc.)
# 3. Run training
torchrun --nproc_per_node=8 moshi-finetune/train.py \\
  configs/yarn_ttt_long_context_example.yaml

# 4. Monitor with wandb or tensorboard
# 5. Evaluate on long-context benchmarks
```

## Conclusion

**YaRN + TTT** enables efficient context window extension:
- ✅ Frozen base model (no catastrophic forgetting)
- ✅ Minimal trainable parameters (fast training)
- ✅ Validated approach (TTT paper used YaRN)
- ✅ No architectural changes needed
- ✅ Works at inference without retraining

This is the recommended approach for adapting Moshi to long-context tasks.
