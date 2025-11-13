# YaRN + TTT-Only Finetuning: Feasibility Analysis

## Executive Summary

**Answer: YES**, YaRN position scaling CAN be applied with frozen base model + TTT-only finetuning.

## Background

### YaRN (Yet another RoPE extensioN)
- **Paper**: Peng et al., 2023 (arXiv:2309.00071)
- **Purpose**: Extends context window of LLMs efficiently
- **Method**: Modifies RoPE (Rotary Position Embedding) frequencies using:
  - NTK-by-parts interpolation
  - Temperature scaling (attention factor)
- **Training efficiency**: Requires 10x fewer tokens and 2.5x fewer training steps than previous methods

### TTT Paper Usage
From the In-Place TTT paper (lines 837-839, Table 5):
- Stage 1: ~20B tokens, 32k context, **no RoPE extension**
- Stage 2: ~15B tokens, 128k context, **YaRN applied**
- Both baseline and In-Place TTT models used the same training curriculum

## Technical Analysis

### 1. What YaRN Modifies

**YaRN ONLY modifies the frequency calculation in RoPE**, not model parameters:
- Changes how position indices are mapped to rotation frequencies
- Applied during forward pass in the RoPE embedding calculation
- No learnable parameters added
- No existing model weights changed

From HuggingFace Transformers implementation:
```python
# YaRN modifies inverse frequencies computation
# inv_freq_extrapolation and inv_freq_interpolation are computed differently
# attention_factor is applied to scale attention logits
# All modifications are computational, not parametric
```

### 2. Where YaRN Operates

**Location**: Attention layers via RoPE in Q/K projections
- Current codebase: `moshi/modules/rope.py` → `apply_rope()` function
- Applied in: `StreamingMultiheadAttention.forward()` (line 552-553)
- Affects: Query and Key tensors before attention computation

### 3. Where TTT Operates

**Location**: MLP blocks (feed-forward layers)
- Adapts final projection matrix (Wdown) as fast weights
- Completely separate from attention mechanism
- No interaction with position embeddings

## Compatibility Analysis

### Why YaRN Works with Frozen Base + TTT-Only Training

1. **Orthogonal Modifications**:
   - YaRN: Modifies position encoding (attention layers)
   - TTT: Modifies MLP fast weights
   - No parameter conflicts

2. **Frozen Attention Still Benefits from YaRN**:
   - Attention layers don't need retraining to use YaRN
   - YaRN scaling is applied during forward pass regardless of whether attention weights are frozen
   - The frozen attention mechanism automatically processes YaRN-scaled positions

3. **TTT Adapts to Extended Context**:
   - TTT parameters (target_generator, etc.) ARE trainable
   - These parameters learn to compress information from YaRN-extended context
   - No conflict with frozen base model

4. **Paper Precedent**:
   - TTT paper trained with YaRN in stage 2
   - Shows that YaRN + TTT training is validated
   - We're just adding the constraint of freezing non-TTT parameters

### Potential Considerations

1. **Optimal Performance**:
   - The TTT paper trained both baseline and TTT models (not frozen)
   - Freezing base model may slightly reduce adaptation capability
   - However, the paper's ablations show TTT provides most of the benefit

2. **Training Strategy**:
   - Should gradually increase sequence length (like paper: 32k → 128k)
   - YaRN parameters (scale, alpha, beta) need to be configured
   - May need to tune TTT learning rate for extended context

## Implementation Requirements

### 1. Extend RoPE Module (`rope.py`)
Add YaRN scaling support:
- NTK-by-parts frequency interpolation
- Attention factor computation
- Configurable scaling parameters

### 2. Add Training Configuration
```python
@dataclass
class YaRNConfig:
    enabled: bool = False
    scale: float = 1.0  # Context extension factor
    original_max_seq_len: int = 3000  # Base model's training length
    alpha: float = 1.0  # NTK alpha
    beta_fast: int = 32  # Low frequency boundary
    beta_slow: int = 1  # High frequency boundary
    mscale: float = 1.0  # Attention scaling
    mscale_all_dim: float = 0.0  # Additional scaling
```

### 3. Update Training Script
- Load YaRN config
- Apply to RoPE during model initialization
- Support progressive scaling (32k → 128k stages)

### 4. Freezing Strategy
```python
# Freeze base model
for name, param in model.named_parameters():
    if 'ttt' not in name and 'target_generator' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True  # Only train TTT params
```

## Recommended Implementation Plan

### Phase 1: Add YaRN Support
1. Implement YaRN-enhanced RoPE module
2. Add YaRN configuration to training args
3. Test with short sequences to verify correctness

### Phase 2: Long Sequence Training
1. Stage 1: Train on 32k context (warm-up, optional YaRN scale=1.0)
2. Stage 2: Train on 64k-128k context with YaRN (scale=4.0 for 128k from 32k base)
3. Monitor perplexity and TTT gradient norms

### Phase 3: Validation
1. Evaluate on RULER benchmark or similar long-context tasks
2. Compare frozen base + TTT vs. full finetuning
3. Measure extrapolation capability (test beyond training length)

## Conclusion

**YaRN is fully compatible with frozen base model + TTT-only finetuning**:
- ✅ YaRN requires no parameter updates
- ✅ Works through computational modification of position encodings
- ✅ TTT and YaRN operate on different model components
- ✅ Validated approach (TTT paper used YaRN for training)
- ✅ Frozen attention still benefits from YaRN-scaled positions

The main benefit of this approach:
- **Memory efficient**: Only train small subset of parameters (TTT modules)
- **Fast convergence**: Leverage pre-trained base model knowledge
- **Extended context**: YaRN enables processing much longer sequences
- **Maintains base model**: No risk of catastrophic forgetting

This is an optimal strategy for adapting pre-trained models to long-context tasks with limited compute.
