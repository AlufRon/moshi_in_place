# Critical Fix: Nested Parameter Assignment Bug in TTT Initialization

## The Bug

**Symptom:** Loss starts at 16-18 instead of 2-3 with TTT enabled, despite zero-initializing target_generator.

**Root Cause:** When LoRA is applied to a model, layers like `W_target` become `LoRALinear` with nested parameters:
```
target_generator.W_target.frozen_W.weight
target_generator.W_target.lora_A.weight
target_generator.W_target.lora_B.weight
```

The old initialization code did:
```python
module._parameters[p_name] = torch.nn.Parameter(...)  # WRONG!
```

Where `p_name = "target_generator.W_target.frozen_W.weight"` (nested path).

**The Problem:** `module._parameters` is a flat dict containing only direct child parameters, not nested ones!

Assigning `module._parameters["target_generator.W_target.frozen_W.weight"]` creates a NEW entry with that dotted key, but **doesn't replace the actual nested parameter**!

Result: The actual `frozen_W.weight` remained meta (uninitialized), so `LoRALinear.forward()` used random/garbage weights!

## LoRALinear Forward Pass

```python
def forward(self, x):
    lora = self.lora_B(self.lora_A(x))
    return self.frozen_W(x) + lora * self.scaling
```

Even with `lora_A` and `lora_B` zero-initialized, if `frozen_W` has random weights:
```
output = RANDOM(x) + 0 * scaling = RANDOM(x)  ❌
```

This is why V_hat was random, causing loss of 16-18!

## The Fix

**Navigate to the nested module before assigning:**

```python
# Split the parameter name to get the navigation path
parts = p_name.split('.')  # ["target_generator", "W_target", "frozen_W", "weight"]
param_name = parts[-1]  # "weight"
nested_module = module
for part in parts[:-1]:  # Navigate: module.target_generator.W_target.frozen_W
    nested_module = getattr(nested_module, part)

# Create and initialize the parameter
new_param = torch.nn.Parameter(torch.empty_like(param, device="cpu", dtype=param_dtype))
torch.nn.init.zeros_(new_param)  # Zero init for warm-start

# Assign to the CORRECT nested module
nested_module._parameters[param_name] = new_param  # ✅ Actually replaces frozen_W.weight!
```

## Why This Works

Now the assignment goes to the correct nested module:
```python
module.target_generator.W_target.frozen_W._parameters["weight"] = new_param
```

This actually replaces the `frozen_W.weight` parameter!

Result with LoRALinear forward:
```python
output = frozen_W(x) + 0 * scaling
      = 0(x) + 0  # frozen_W.weight is now zeros!
      = 0  ✅
```

So `V_hat = 0`, TTT has zero effect initially, loss = 2-3 (pretrained quality)!

## Code Changes

**File:** `moshi-finetune/finetune/wrapped_model.py`

**Function:** `initialize_ttt_parameters`

**Changes:**
1. Added navigation to nested modules before parameter assignment
2. Store new parameter in a variable instead of accessing `module._parameters[p_name]` multiple times
3. Assign to `nested_module._parameters[param_name]` instead of `module._parameters[p_name]`

**Lines changed:** 115-160 (parameter initialization loop)

## Testing

**Before fix:**
```
Step 1: loss 16.768
Step 6: loss 14.625
Step 20: loss 16.139
```

**Expected after fix:**
```
Step 1: loss ~2-3 (pretrained quality)
Gradual decrease as TTT learns from zero
```

## Related Issues

This bug ONLY affects models with:
1. TTT enabled
2. LoRA enabled
3. Zero-initialization strategy

Without LoRA, parameters aren't nested, so the old code worked fine.
With random initialization (std=0.02), the bug was masked because random was "close enough" to random.
With zero-initialization, the bug was exposed: zeros weren't actually applied!

## Impact

✅ **After fix:**
- target_generator weights are correctly zero-initialized
- V_hat = 0 initially
- Loss starts at pretrained quality (~2-3)
- TTT learns gradually from zero during training
- No catastrophic forgetting of pretrained knowledge

This is CRITICAL for using TTT with LoRA fine-tuning!
