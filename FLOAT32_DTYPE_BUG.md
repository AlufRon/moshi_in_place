# Critical Float32 Dtype Bug in TTT

**Status**: 🔴 **CRITICAL BUG FOUND**
**Impact**: Dtype mismatch during inference, potential crashes or wrong results

---

## The Problem

### What the Code Claims

**ttt_module.py:115** (comment):
```python
# IMPORTANT: Keep w_down in float32 for precise gradient updates during inference
```

**ttt_module.py:234-237** (runtime logic):
```python
# If w_down is already float32, use it directly; otherwise convert
if self.w_down.dtype == torch.float32:
    W_down_init = self.w_down
else:
    W_down_init = self.w_down.to(torch.float32)
```

**ttt_module.py:257** (inference update):
```python
# Keep w_down in float32 during inference for efficiency
# No conversion needed since final_state is already float32
self.w_down.data.copy_(final_state)  # final_state is float32
```

### What Actually Happens

**wrapped_model.py:209-210** (training initialization):
```python
for k, v in model_state_dict.items():
    model_state_dict[k] = v.to(param_dtype)  # Converts ALL to param_dtype!
```

**wrapped_model.py:222-223**:
```python
pretrained_weight = model_state_dict[ckpt_key].clone()  # Is param_dtype now!
module._parameters['w_down'] = torch.nn.Parameter(pretrained_weight)
```

**loaders.py:445-448** (inference initialization):
```python
new_w_down = torch.empty_like(mlp.linear_out.weight, device=device, dtype=dtype)
# ^^^^^^^^^^^^^^ dtype is param_dtype (bfloat16/float16), NOT float32!
new_w_down.copy_(mlp.linear_out.weight)
mlp.w_down = nn.Parameter(new_w_down)
```

**loaders.py:454**:
```python
mlp.w_down_pretrained = torch.empty_like(mlp.w_down.data, device=device, dtype=dtype)
# ^^^^ Also uses param_dtype!
```

---

## The Bug Sequence

### Initialization:
1. ✅ w_down created as meta: `torch.empty(dim, hidden, device='meta')`
2. ❌ w_down initialized as **param_dtype** (bfloat16), not float32
3. ✅ w_down_pretrained buffer created with `dtype=torch.float32` (our fix)
4. ❌ w_down_pretrained filled with **param_dtype** data from w_down

### During Forward Pass:
5. ✅ Converts w_down to float32 for computation: `W_down_init = self.w_down.to(torch.float32)`
6. ✅ All TTT math in float32
7. ✅ final_state computed in float32
8. ❌ **DTYPE MISMATCH**: `w_down.data.copy_(final_state)` - copying float32 into bfloat16 parameter!

### Result:
- **Silent dtype conversion** on every inference step
- Loss of precision benefits (defeats the purpose!)
- Potential crashes if copy_ doesn't allow dtype mismatch
- w_down_pretrained has wrong dtype after initialization

---

## Evidence of the Bug

### From loaders.py:445
```python
def get_moshi(..., dtype=torch.bfloat16):
    ...
    new_w_down = torch.empty_like(..., dtype=dtype)  # Uses bfloat16!
```

### From wrapped_model.py:209-210
```python
# param_dtype is bfloat16 by default
for k, v in model_state_dict.items():
    model_state_dict[k] = v.to(param_dtype)  # Everything becomes bfloat16
```

### Proof of Mismatch
```python
# After initialization:
mlp.w_down.dtype                 # bfloat16 (wrong!)
mlp.w_down_pretrained.dtype      # float32 (our fix, but filled with bfloat16 data!)

# After first inference forward:
# final_state is float32
mlp.w_down.data.copy_(final_state)  # Converts float32 → bfloat16 (defeats purpose!)
```

---

## Why This Matters

### Precision Loss
- Paper uses float32 for small gradient updates (η·V̂^T·Z with η=1e-3)
- bfloat16 has only 7 bits of mantissa vs float32's 23 bits
- Small updates (1e-3 * small_gradient) get rounded to zero in bfloat16
- **TTT fast weight adaptation severely degraded**

### Comment Says:
> "bfloat16 loses precision on small gradient updates during inference"

**But code actually uses bfloat16 for w_down!**

---

## The Correct Fix

### Fix 1: Training Initialization (wrapped_model.py)

**Current (WRONG)**:
```python
for k, v in model_state_dict.items():
    model_state_dict[k] = v.to(param_dtype)  # Everything to bfloat16

if args.ttt and args.ttt.enabled:
    for m_name, module in model.named_modules():
        if "gating" in m_name and hasattr(module, 'w_down'):
            pretrained_weight = model_state_dict[ckpt_key].clone()  # bfloat16!
            module._parameters['w_down'] = torch.nn.Parameter(pretrained_weight)
```

**Fixed**:
```python
for k, v in model_state_dict.items():
    model_state_dict[k] = v.to(param_dtype)

if args.ttt and args.ttt.enabled:
    for m_name, module in model.named_modules():
        if "gating" in m_name and hasattr(module, 'w_down'):
            # Keep w_down in float32 for TTT precision!
            pretrained_weight = model_state_dict[ckpt_key].clone().to(torch.float32)
            module._parameters['w_down'] = torch.nn.Parameter(pretrained_weight)
            print(f"  ✓ {m_name}.w_down initialized as float32 (shape: {pretrained_weight.shape})")
```

### Fix 2: Inference Initialization (loaders.py)

**Current (WRONG)**:
```python
new_w_down = torch.empty_like(mlp.linear_out.weight, device=device, dtype=dtype)  # bfloat16!
new_w_down.copy_(mlp.linear_out.weight)
mlp.w_down = nn.Parameter(new_w_down)

mlp.w_down_pretrained = torch.empty_like(mlp.w_down.data, device=device, dtype=dtype)  # bfloat16!
mlp.w_down_pretrained.copy_(mlp.w_down.data)
```

**Fixed**:
```python
# Always use float32 for TTT fast weights, regardless of model dtype
new_w_down = torch.empty_like(mlp.linear_out.weight, device=device, dtype=torch.float32)
new_w_down.copy_(mlp.linear_out.weight.to(torch.float32))
mlp.w_down = nn.Parameter(new_w_down)
print(f"[TTT] Initialized w_down at layer {idx} as float32")

# w_down_pretrained should match w_down dtype (float32)
mlp.w_down_pretrained = torch.empty_like(mlp.w_down.data, device=device, dtype=torch.float32)
mlp.w_down_pretrained.copy_(mlp.w_down.data)
```

---

## Impact Assessment

### Current State:
- ❌ w_down is bfloat16 (should be float32)
- ❌ w_down_pretrained buffer has float32 dtype but contains bfloat16-precision data
- ❌ Every inference forward converts float32 → bfloat16 (silent precision loss)
- ❌ TTT updates lose precision due to bfloat16 accumulation

### After Fix:
- ✅ w_down is float32 (matches intent)
- ✅ w_down_pretrained is float32 with full precision
- ✅ No dtype conversions during forward (efficient)
- ✅ TTT updates preserve full float32 precision

---

## Testing the Fix

### Check dtype after initialization:
```python
# After loading model:
for name, module in model.named_modules():
    if hasattr(module, 'w_down'):
        print(f"{name}.w_down.dtype = {module.w_down.dtype}")  # Should be float32
        print(f"{name}.w_down_pretrained.dtype = {module.w_down_pretrained.dtype}")  # Should be float32
```

### Check no conversions during inference:
```python
# Before fix: Logs "Converting w_down from bfloat16 to float32"
# After fix: No conversion needed (already float32)
```

---

## Priority

**🔴 CRITICAL** - This defeats the entire purpose of the float32 comment and degrades TTT quality.

Should be fixed before any training or serious inference testing.
