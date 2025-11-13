# Critical TTT Initialization Bug Fix

## Problem: Loss Starts at 18 Instead of 2-3

### Symptoms
- With TTT enabled: Initial training loss = **18** (near random)
- Without TTT or with LoRA: Initial training loss = **2-3** (pretrained quality)
- 6x worse starting point indicates model is producing garbage outputs

### Root Cause

**Random target_generator initialization corrupts pretrained model outputs**

The TTT forward pass:
```python
1. Z = activation(linear_in(x))              # pretrained weights ✅
2. V_hat = target_generator(token_embeddings) # RANDOM weights ❌
3. delta = V_hat^T @ Z                        # random V_hat → random delta
4. W_eff = w_down + ttt_lr * cumsum(delta)   # adds NOISE to pretrained w_down!
5. output = Z @ W_eff^T                       # corrupted output!
```

**The Corruption:**
- `target_generator.W_target` initialized with `N(0, 0.02)` (random)
- `target_generator.conv1d` initialized with `N(0, 0.02)` (random)
- `V_hat = W_target(Conv1D(embeddings))` produces **random values**
- Even small random V_hat creates significant perturbations when multiplied with Z
- w_down gets corrupted: `w_down_eff = w_down_pretrained + random_noise`
- Model outputs become randomized → loss = 18

### Why This Violates the Paper's Philosophy

**From the paper (Section 3, line 328-329):**
> "Moreover, introducing any new, randomly-initialized layer also creates a conflict with the billions of trained parameters of LLMs, necessitating costly and often impractical retraining to resolve this imbalance."

**The paper's approach:**
- Emphasizes "warm-start from pretrained checkpoint"
- Key insight: "repurpose existing MLP" to avoid introducing random components
- Expects **continual training** after adding TTT (Section 4.1, line 831-832)

**What we were doing wrong:**
1. Initialized target_generator with random weights `N(0, 0.02)`
2. Expected model to work immediately without training
3. Random weights corrupted pretrained outputs → catastrophic loss

---

## The Fix: Zero Initialization

### Mathematical Requirement

**Goal:** Make TTT have **zero effect** initially, so model behaves exactly like pretrained.

If `V_hat = 0`:
```
delta = V_hat^T @ Z = 0
W_eff = w_down + ttt_lr * cumsum(0) = w_down  (unchanged!)
output = Z @ w_down^T  (exactly pretrained MLP!)
→ Initial loss = 2-3 (pretrained quality) ✅
```

### Implementation

**Zero-initialize W_target, keep conv1d small random:**
```python
# Training: wrapped_model.py:137-142
elif "target_generator" in p_name:
    # Zero init for warm-start - ensures TTT has zero effect initially
    torch.nn.init.zeros_(param)
    logger.info(f"  ✓ Zero-initialized {m_name}.{p_name} for warm-start")
elif "conv" in p_name:
    # Small random init for conv (or could also be zero)
    torch.nn.init.normal_(param, mean=0.0, std=0.02)

# Inference: loaders.py:473-480
if hasattr(tg, 'W_target') and tg.W_target.weight.is_meta:
    # Zero init for warm-start
    tg.W_target.weight = nn.Parameter(
        torch.empty_like(tg.W_target.weight, device=device, dtype=dtype)
    )
    nn.init.zeros_(tg.W_target.weight)
    print(f"[TTT] Zero-initialized target_generator W_target at layer {idx} for warm-start")
```

**Why this works:**
- `V_hat = 0 @ Conv1D(embeddings) = 0` (W_target multiplication dominates)
- Even if Conv1D has small random weights, multiplying by zero W_target gives zero
- TTT has **zero effect** initially
- Model behaves exactly like pretrained → loss = 2-3 ✅

### During Training

**How TTT "activates" gradually:**
1. **Epoch 0:** W_target = 0 → V_hat = 0 → no TTT effect → loss ~2-3
2. **Early training:** Gradients flow into W_target → small non-zero values
3. **Mid training:** W_target grows → V_hat becomes meaningful → TTT starts helping
4. **Late training:** Fully learned target_generator → effective TTT adaptation

**Key properties:**
- **Smooth activation:** TTT effect grows gradually from zero
- **No catastrophic forgetting:** Pretrained quality preserved initially
- **Trainable from zero:** Standard practice (like initializing last layer of classifier to zero bias)
- **Minimal code change:** Only changed initialization, no architecture changes

---

## Alternative Considered: Freeze Target Generator

Could freeze target_generator initially:
```python
for param in model.target_generator.parameters():
    param.requires_grad = False
# ... train for N steps ...
# Unfreeze later
```

**Why zero init is better:**
- Simpler: no multi-stage training logic
- Elegant: single mathematical property (V_hat=0)
- Standard: learning from zero is well-established practice

---

## Verification

### Before Fix:
```
Initial loss with TTT: 18.0
Model outputs: near random (huge perturbations from random target_generator)
```

### After Fix:
```
Initial loss with TTT: 2-3 (same as pretrained!)
Model outputs: identical to pretrained initially (V_hat=0 → no TTT effect)
During training: TTT gradually learns and activates
```

### Files Modified:
1. `moshi-finetune/finetune/wrapped_model.py:137-142` - Zero init target_generator for training
2. `moshi/moshi/moshi/models/loaders.py:473-480` - Zero init target_generator for inference

---

## Impact

✅ **Warm-start:** Model starts with pretrained quality (loss 2-3, not 18)
✅ **Trainable:** TTT can learn from zero during training
✅ **Stable:** No random corruption of pretrained outputs
✅ **Aligned with paper:** Matches "warm-start from pretrained" philosophy
✅ **Minimal:** Only 2 lines changed (zero init instead of random)

**Critical for:**
- Fine-tuning pretrained Moshi with TTT
- Continual training experiments
- Any scenario starting from pretrained weights

**This fix is REQUIRED** before any serious TTT training or evaluation.
