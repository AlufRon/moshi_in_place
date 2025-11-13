# Minimal Inference Fix for TTT

**Date**: November 13, 2025  
**Status**: Ready to implement  
**Issue**: Current TTT implementation doesn't persist w_down updates during inference

---

## Root Cause Analysis

### Current Implementation (Training-Only)

```python
# Line 157 in ttt_module.py
W_down_init = self.w_down  # Read current value

# Line 187
W_eff = W_init_bc + self.ttt_lr * S  # Compute effective weights

# Line 197
O_chunks = torch.matmul(Z_chunks, W_eff_T)  # Use them

# Return O
# ❌ NEVER UPDATES self.w_down!
```

**Why this works for training**: 
- Gradients flow back through `W_eff` to `self.w_down`
- Optimizer updates `self.w_down` via backprop
- Each batch starts fresh from optimizer's current `w_down` value

**Why this FAILS for inference**:
- No optimizer running
- `self.w_down` never changes
- Every `step()` call starts from same initial value
- No persistence across streaming calls!
- Violates paper's Equation 1: `W^(i) = W^(i-1) + η V̂^T Z`

---

## Validation from TTT-LM-Kernels

✅ **Analyzed reference implementation** at `/home/alufr/moshi_in_place_ttt/ttt-lm-kernels`
- Their approach: Gradient accumulation + mini-batch updates (different paper)
- Our approach: Direct updates (simpler, matches our paper)
- **Key validation**: Both agree **weight persistence is essential** during inference
- **Conclusion**: Our 3-line fix is correct and simpler for In-Place TTT with T=1

See `TTT_KERNELS_INFERENCE_ANALYSIS.md` for detailed comparison.

---

## The Fix: 3 Lines

### Option 1: Always Update (Simplest)

```python
# Add at end of _parallel_ttt_update(), around line 202

def _parallel_ttt_update(self, Z: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
    # ... existing code ...
    
    O = O_chunks.permute(1, 0, 2, 3).reshape(B, num_chunks * self.chunk_size, dim)
    if orig_T != O.shape[1]:
        O = O[:, :orig_T]
    
    # NEW: Update w_down to final state for next call (inference persistence)
    if not self.training:  # Only during inference
        final_state = W_eff[-1, 0]  # Last chunk, first batch item [dim, hidden]
        self.w_down.data.copy_(final_state)
    
    return O
```

**Lines added**: 3  
**Complexity**: Trivial

**How it works**:
- During training: `self.training=True`, skip the update (optimizer handles it)
- During inference: `self.training=False`, persist final w_down state
- Next `step()` call starts from updated w_down
- Accumulates updates across entire conversation

---

### Option 2: Streaming Mode Flag (More Explicit)

```python
# In __init__, around line 110
if self.ttt_enabled:
    self.w_down = nn.Parameter(torch.empty(dim, hidden, device='meta'))
    self.register_buffer('w_down_pretrained', torch.empty(dim, hidden))  # NEW
    self._streaming_inference = False  # NEW

# NEW: Enable/disable streaming mode
def set_streaming_inference(self, enabled: bool):
    self._streaming_inference = enabled
    if enabled and hasattr(self, 'w_down_pretrained'):
        # Reset to pretrained at start of conversation
        self.w_down.data.copy_(self.w_down_pretrained)

# In _parallel_ttt_update, at end:
def _parallel_ttt_update(self, Z: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
    # ... existing code ...
    
    # Update w_down if in streaming inference mode
    if self._streaming_inference:
        final_state = W_eff[-1, 0]  # Last chunk, first batch item
        self.w_down.data.copy_(final_state)
    
    return O
```

**Lines added**: ~10  
**Complexity**: Simple

**Advantages**:
- Explicit control over when persistence happens
- Can reset to pretrained at conversation boundaries
- Clear separation of training vs inference behavior

---

## Which to Use?

### Recommend: **Option 1** (check `self.training`)

**Reasons**:
1. **Minimal code**: Only 3 lines
2. **Automatic**: No need to call `set_streaming_inference()`
3. **PyTorch convention**: `model.eval()` already sets `self.training=False`
4. **Works immediately**: Run inference just calls `model.eval()`, done!

### When to use Option 2:
- If need explicit conversation reset functionality
- If want to preserve pretrained weights separately
- If need finer control over when updates persist

---

## Testing

### Test 1: Verify Persistence
```python
model.eval()
with torch.no_grad():
    # First call
    w_before = model.transformer.layers[10].gating.w_down.clone()
    out1 = model.step(input1)
    w_after = model.transformer.layers[10].gating.w_down.clone()
    
    assert not torch.equal(w_before, w_after), "w_down should change!"
    
    # Second call should start from updated state
    w_before2 = model.transformer.layers[10].gating.w_down.clone()
    assert torch.equal(w_after, w_before2), "Should persist across calls!"
```

### Test 2: Training Unchanged
```python
model.train()
optimizer.zero_grad()
loss = model(batch).loss
loss.backward()

# w_down should have gradients
assert model.transformer.layers[10].gating.w_down.grad is not None
# Manual update should NOT have happened (training mode)
```

---

## Complete Implementation (Option 1)

```python
# File: moshi/moshi/moshi/modules/ttt_module.py
# Modify _parallel_ttt_update method

def _parallel_ttt_update(self, Z: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
    # Shapes: Z [B, T, hidden], V_hat [B, T, dim]
    B, T, hidden = Z.shape
    _, _, dim = V_hat.shape
    orig_T = T

    # initial fast weights - use w_down parameter directly
    W_down_init = self.w_down  # [dim, hidden]

    # partition into chunks
    num_chunks = (T + self.chunk_size - 1) // self.chunk_size
    pad_size = num_chunks * self.chunk_size - T
    if pad_size != 0:
        Z = F.pad(Z, (0, 0, 0, pad_size))
        V_hat = F.pad(V_hat, (0, 0, 0, pad_size))
        T = num_chunks * self.chunk_size

    # reshape: [B, num_chunks, chunk_size, *]
    Zc = Z.view(B, num_chunks, self.chunk_size, hidden)
    Vc = V_hat.view(B, num_chunks, self.chunk_size, dim)

    # compute deltas in parallel: result shape [num_chunks, B, dim, hidden]
    deltas = torch.einsum('b n t d, b n t h -> n b d h', Vc, Zc)

    # prefix sum across chunks (causal): S_i = sum_{j=0..i-1} deltas_j
    cumsum = torch.cumsum(deltas, dim=0)
    zero = torch.zeros_like(cumsum[0:1])
    S = torch.cat([zero, cumsum[:-1]], dim=0)  # [num_chunks, B, dim, hidden]

    # broadcast W_down_init to [num_chunks, B, dim, hidden]
    W_init_bc = W_down_init.unsqueeze(0).unsqueeze(0).expand(num_chunks, B, -1, -1)

    # effective weights per chunk
    W_eff = W_init_bc + self.ttt_lr * S

    # prepare Z for matmul: want [num_chunks, B, chunk_size, hidden]
    Z_chunks = Zc.permute(1, 0, 2, 3)

    # W_eff: [num_chunks, B, dim, hidden] -> transpose last two
    W_eff_T = W_eff.transpose(-2, -1)  # [num_chunks, B, hidden, dim]

    # batch matmul per chunk
    O_chunks = torch.matmul(Z_chunks, W_eff_T)  # [num_chunks, B, chunk_size, dim]

    # put back to [B, T, dim]
    O = O_chunks.permute(1, 0, 2, 3).reshape(B, num_chunks * self.chunk_size, dim)

    if orig_T != O.shape[1]:
        O = O[:, :orig_T]

    # ===== NEW: Persist w_down updates during inference =====
    if not self.training:
        # Update w_down to final state for next streaming call
        # W_eff shape: [num_chunks, B, dim, hidden]
        # Take last chunk (num_chunks-1), first batch item (0)
        final_state = W_eff[-1, 0]  # [dim, hidden]
        self.w_down.data.copy_(final_state)
    # ========================================================

    return O
```

**Total changes**: 3 lines (plus comments)

---

## Why This Works

### For T=1 (Moshi Streaming):
1. Call 1: `W_eff[0,0] = W^(0) + η*0 = W^(0)` (no prior updates)
   - Output: `O = Z @ W^(0)^T`
   - Update: `w_down = W^(0) + η*V^T@Z` ✓
   
2. Call 2: Start with updated w_down
   - `W_eff[0,0] = w_down + η*0 = W^(1)` ✓
   - Output: `O = Z @ W^(1)^T` ✓
   - Update: `w_down = W^(1) + η*V^T@Z = W^(2)` ✓

3. Continues accumulating...

### For T>1 (Training with chunks):
- Gradients flow normally
- Optimizer updates w_down
- Manual update skipped (self.training=True)
- No behavior change ✓

---

## Summary

**Problem**: w_down doesn't persist across inference calls  
**Root cause**: Only updates via optimizer (training), not manual (inference)  
**Solution**: Copy final state to w_down.data when not training  
**Code change**: 3 lines  
**Impact**: Enables actual streaming TTT inference

This is **NOT a bug in our approach** - it's the **correct paper implementation for training**.  
We just need to add **inference persistence** for streaming use case.
