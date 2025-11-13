# TTT Weight Initialization Strategy

## Overview: What's New vs Pretrained in TTT

When adding TTT to Moshi, we add NEW learnable parameters to the existing pretrained model. Here's the breakdown:

---

## Weight Categories

### 1. **w_down (Fast Weight)** - NOT NEW!
**Source:** Copied from pretrained `linear_out.weight`
**Initialization:** Clone of existing pretrained weights
**Dtype:** float32 (for precision)
**Why:** This is the TTT "fast weight" that gets updated at test time. Start from pretrained projection to maintain model quality.

```python
# Training: wrapped_model.py:222
pretrained_weight = model_state_dict['gating.linear_out.weight'].clone().to(torch.float32)
module._parameters['w_down'] = torch.nn.Parameter(pretrained_weight)

# Inference: loaders.py:445
new_w_down = torch.empty_like(mlp.linear_out.weight, device=device, dtype=torch.float32)
new_w_down.copy_(mlp.linear_out.weight.to(torch.float32))
mlp.w_down = nn.Parameter(new_w_down)
```

**Rationale:** Since w_down starts identical to linear_out, the model initially produces the same outputs as pretrained. This ensures smooth warm-start without breaking existing behavior.

---

### 2. **linear_in (W_up and W_gate)** - ALREADY PRETRAINED!
**Source:** Existing pretrained weights
**Initialization:** N/A - uses checkpoint weights
**Dtype:** param_dtype (bfloat16/float16)
**Why:** These are "slow weights" (frozen during TTT) per the paper. They're the standard MLP input projection.

```python
# From ttt_module.py:96
self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
# This gets loaded from checkpoint - it's the existing MLP layer!
```

**Rationale:** The paper's key insight is to REUSE existing MLP weights. No new initialization needed.

---

### 3. **target_generator** - NEW WEIGHTS! ⚠️

This is the ONLY truly new component with randomly initialized weights.

#### 3a. **target_generator.conv1d.conv.weight** (Conv1D kernel)
**Shape:** `[d_model, d_model, kernel_size=2]`
**Initialization:** `normal_(mean=0.0, std=0.02)`
**Dtype:** param_dtype (bfloat16/float16)

```python
# wrapped_model.py:137-138
if "conv" in p_name or "target_generator" in p_name:
    torch.nn.init.normal_(param, mean=0.0, std=0.02)
```

**Rationale:** Small random initialization (std=0.02) is common for new layers. The conv kernel learns to extract temporal patterns from token embeddings.

#### 3b. **target_generator.W_target.weight** (Linear projection)
**Shape:** `[d_model, d_model]`
**Initialization:** `normal_(mean=0.0, std=0.02)`
**Dtype:** param_dtype (bfloat16/float16)

```python
# Same as above - wrapped_model.py:137-138
torch.nn.init.normal_(param, mean=0.0, std=0.02)
```

**Rationale:** Small random weights ensure the target generator starts with small, smooth outputs. As training proceeds, it learns to generate meaningful TTT targets V̂.

---

## Why This Initialization Strategy?

### Design Philosophy: Minimal Disruption

The paper's "In-Place TTT" philosophy is to add TTT **without breaking the pretrained model**:

1. **w_down = linear_out:** Initially, TTT forward pass computes the same output as pretrained MLP
2. **linear_in (W_up, W_gate):** Reuse existing weights - no disruption
3. **target_generator:** New but small (std=0.02) - starts with minimal impact

### During Training:
- Target generator learns to produce useful TTT targets V̂
- w_down learns to adapt via TTT updates (small ηV̂ᵀZ updates)
- Model gradually learns to leverage TTT while preserving pretrained knowledge

### During Inference:
- w_down updates at test time via: `w_down += η·V̂ᵀZ`
- This adapts the model to the current conversation context
- Target generator produces context-aware targets from token embeddings

---

## Initialization Values Summary

| Parameter | Source | Init Method | Dtype | New? |
|-----------|--------|-------------|-------|------|
| w_down | Pretrained linear_out | Clone | float32 | No |
| linear_in (W_up, W_gate) | Pretrained | Checkpoint | param_dtype | No |
| target_generator.conv1d | Random | N(0, 0.02) | param_dtype | **Yes** |
| target_generator.W_target | Random | N(0, 0.02) | param_dtype | **Yes** |

---

## Alternative Initialization Strategies

### Current: N(0, 0.02) for target_generator

**Pros:**
- Small initialization reduces disruption to pretrained model
- Standard practice for adding new layers to pretrained models
- Allows gradual learning during training

**Cons:**
- Random weights may produce suboptimal targets initially
- Requires some training before TTT becomes effective

### Alternative 1: Identity Initialization (not used)
Initialize conv1d as identity and W_target as identity:
```python
# Not implemented - just for comparison
torch.nn.init.eye_(W_target.weight)
```
**Why not:** Doesn't make semantic sense for TTT targets

### Alternative 2: Copy from Existing Layers (not used)
Initialize target_generator from other pretrained projections:
```python
# Not implemented - just for comparison
target_generator.W_target.weight = model.some_other_projection.weight.clone()
```
**Why not:** No obvious pretrained weight makes sense for temporal conv + target generation

### Alternative 3: Kaiming Initialization (not used)
Use Kaiming uniform like other linear layers:
```python
# Not implemented - we use normal(0, 0.02) instead
torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
```
**Why not:** Larger variance (~2/√fan_in) might disrupt pretrained outputs more

---

## Paper's Perspective

The paper doesn't explicitly specify initialization for target_generator. However:

1. **Paper emphasizes warm-start:** "Our In-Place TTT can warm start from a pretrained checkpoint"
2. **Key insight:** "Instead of replacing or adding components, we repurpose the MLP block"
3. **Implication:** Minimal new parameters, small initialization to preserve pretrained quality

Our choice of N(0, 0.02) aligns with this philosophy: small random weights that don't disrupt the model initially, but can be learned during training.

---

## Practical Impact

### Before Training:
- Model behaves almost identically to pretrained (target_generator has minimal effect due to small weights)
- No catastrophic forgetting of pretrained knowledge

### After Training:
- Target generator learns meaningful temporal patterns
- TTT mechanism becomes effective for context adaptation
- Model gains test-time adaptation capability

### During Inference:
- w_down updates accumulate in float32 (precision)
- Target generator produces learned targets from conversation context
- Model adapts to user's speaking style, topic, etc.

This initialization strategy enables the "in-place" nature of the approach: adding TTT without destroying pretrained capabilities.
