# TTT Gradient Debug Investigation

## The Mystery: `grad_norm=0`

Your training logs show:
```
[TTT] Step 6: grad_norm=0.0000, param_norm=1.2215, delta_norm=0.000021
```

**What this means:**
- `delta_norm > 0` → TTT in-place updates ARE working ✅
- `grad_norm = 0` → target_generator NOT getting gradients from backprop ❌

## What Should Happen (Per Paper)

According to the In-Place TTT paper:
1. Target V̂ = Conv1D(X0)·Wtarget contains future token information
2. TTT updates: W_down ← W_down + η·V̂^T·Z
3. Main NTP loss provides gradients to learn Wtarget
4. NO separate reconstruction loss needed

**Gradient flow should be:**
```
NTP Loss → Output → Z @ W_eff^T → W_eff → deltas → V̂ → Wtarget
```

## Debug Code Added

Added to [train.py:393-409](../moshi-finetune/train.py#L393-L409):

```python
# DEBUG: Check target_generator parameters
if state.step % args.log_freq == 0 and state.step < 5 and args.ttt.enabled:
    main_logger_info("\n=== TTT Parameter Debug ===")
    for name, p in model.named_parameters():
        if 'target_generator' in name:
            main_logger_info(f"{name}:")
            main_logger_info(f"  requires_grad: {p.requires_grad}")
            main_logger_info(f"  has grad: {p.grad is not None}")
            if p.grad is not None:
                main_logger_info(f"  grad norm: {p.grad.norm().item():.6f}")
```

**This will run on steps 0-4 and tell us:**
1. Do target_generator parameters exist?
2. Do they have `requires_grad=True`?
3. Are they receiving gradients?

## Possible Root Causes

### 1. Parameters Not Trainable
```python
# Expected to see (in logs):
#   requires_grad: True
#
# If you see False, the fix is in wrapped_model.py:448-454
```

### 2. Gradient Checkpointing Breaking Flow
Your config has:
```yaml
gradient_checkpointing: true
```

This can break gradient flow if TTT operations aren't included in checkpointed blocks.

### 3. Parameters Not in Optimizer
Check if target_generator parameters are included when creating the optimizer.

### 4. Computation Graph Broken
Something (detach, no_grad context, etc.) is breaking the computational graph.

## What to Look For in New Training Logs

When you run training with the debug code, look for:

### ✅ Good Output (Gradients Working):
```
=== TTT Parameter Debug ===
transformer.layers.10.gating.target_generator.conv1d.conv.weight:
  requires_grad: True
  has grad: True
  grad norm: 0.000123
transformer.layers.10.gating.target_generator.W_target.weight:
  requires_grad: True
  has grad: True
  grad norm: 0.000456
```

### ❌ Bad Output (Problem 1 - Not Trainable):
```
=== TTT Parameter Debug ===
transformer.layers.10.gating.target_generator.conv1d.conv.weight:
  requires_grad: False  ← PROBLEM!
  has grad: False
```

### ❌ Bad Output (Problem 2 - No Gradients):
```
=== TTT Parameter Debug ===
transformer.layers.10.gating.target_generator.conv1d.conv.weight:
  requires_grad: True
  has grad: False  ← PROBLEM! Graph broken somewhere
  grad: None
```

### ❌ Bad Output (Problem 3 - No Parameters):
```
=== TTT Parameter Debug ===
WARNING: No target_generator parameters found!
```

## Next Steps Based on Output

1. **If `requires_grad=False`**: Check [wrapped_model.py:448-454](../moshi-finetune/finetune/wrapped_model.py#L448-L454)
2. **If `has grad: False` but `requires_grad=True`**: Gradient flow is broken - likely gradient_checkpointing issue
3. **If no parameters found**: Initialization issue - TTT layers not created properly

## Files Modified

1. [train.py](../moshi-finetune/train.py) - Added debug logging
2. [args.py](../moshi-finetune/finetune/args.py) - Fixed Python 3.10 compatibility for type hints

## Run Training

```bash
cd moshi-finetune
./run_yarn_ttt_training.sh
```

Check the first 5 steps of logs for the debug output!
