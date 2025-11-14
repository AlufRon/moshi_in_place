# In-Place TTT: Paper vs Code Implementation Comparison

**Date**: November 14, 2025
**Paper**: In-Place Test-Time Training (ICLR 2026 Submission)
**Code**: Moshi 7B with In-Place TTT

---

## Executive Summary

Your implementation follows the paper's Algorithm 1 **almost exactly**, with **ONE DELIBERATE DEVIATION**:

- ✅ **99% Paper-Compliant**: Architecture, equations, data flow all match
- ⚠️ **1 Modification**: Training exposes `0.5 * current_chunk_delta` (added in commit d3fd9e2)
- ❓ **Why Modified**: To provide gradient signal to `target_generator`
- 🔍 **Root Issue**: This deviation exists because of a **separate problem** (gradient vanishing)

---

## Section-by-Section Comparison

### 1. Algorithm 1: Context Parallel TTT (Paper Lines 1988-2029)

#### Paper Algorithm Pseudocode:
```
1: Input: Sequence chunks {X(i)}_{i=1}^T
2: for all i ∈ {1,...,T} in parallel do  ▷ Step 1: Compute deltas
3:   Hi ← AttentionBlock(X(i); θ)
4:   Ui, Gi ← HiW⊤_up, HiW⊤_gate
5:   Zi ← ϕ(Gi) ⊙ Ui
6:   Vi ← Conv1DK(X(i)_0)Wtarget  ▷ X(i)_0 = chunk i's token embeddings
7:   ∆Wi ← V⊤_i Zi
8: end for
9: {Si}_{i=1}^T ← CUMSUM({∆Wi}_{i=1}^T)  ▷ Step 2: Prefix sum
10: for all i ∈ {1,...,T} in parallel do  ▷ Step 3: Apply & output
11:   W(i-1)_down ← W(0)_down + ηSi      ▷ Si = sum of deltas [0...i-1]
12:   Oi ← Zi(W(i-1)_down)⊤
13: end for
14: At document boundaries: Reset fast weights to W(0)_down
```

#### Code Implementation: Line-by-Line Mapping

| Paper Line | Code Location | Status | Notes |
|------------|---------------|--------|-------|
| Line 3 | Outside TTT module | ✅ EXACT | Attention handled by transformer |
| Line 4-5 | `ttt_module.py:183-185` | ✅ EXACT | `Z = activation(Wgate) * Wup` |
| Line 6 | `ttt_module.py:189-197` | ✅ EXACT | `_apply_conv1d_per_chunk()` for training |
| Line 7 | `ttt_module.py:404` | ✅ EXACT | `einsum('...d,...h -> ...dh')` |
| Line 9 | `ttt_module.py:419-421` | ✅ EXACT | `cumsum` then shift with zero |
| Line 11 | `ttt_module.py:438` | ⚠️ **MODIFIED** | See detailed analysis below |
| Line 12 | `ttt_module.py:449` | ✅ EXACT | `matmul(Z_chunks, W_eff^T)` |
| Line 14 | `train.py:103-116` | ✅ EXACT | `reset_ttt_on_doc_switch()` |

---

### 2. THE CRITICAL DEVIATION: Line 11 (Lines 423-429 in ttt_module.py)

#### What the Paper Says (Line 11):
```
W(i-1)_down ← W(0)_down + ηSi
```
where `Si = sum_{j=0}^{i-1} ∆Wj` (cumulative sum up to but NOT including chunk i)

#### What the Code Does:

**Inference Mode** (✅ Paper-compliant):
```python
# ttt_module.py:429
S_apply = S_prefix  # S_prefix[i] = sum_{j=0}^{i-1} ∆Wj
W_eff_apply = W_init + self.ttt_lr * S_apply
```
**Matches paper exactly**: Chunk i uses updates from chunks [0...i-1] only.

**Training Mode** (⚠️ Deviation):
```python
# ttt_module.py:426-427
if self.training:
    S_apply = S_prefix + 0.5 * deltas
```

This means:
```
W(i)_eff_for_output = W(0)_down + η * (sum_{j=0}^{i-1} ∆Wj + 0.5 * ∆Wi)
```

**Difference**: Chunk i's output sees **half of chunk i's own delta**!

#### Why Was This Added?

From commit `d3fd9e2` (Nov 14, 2025):
```
"TTT training: expose partial chunk delta (0.5) during training
to give target_generator a gradient path; add debug chunk logging"
```

**Rationale** (from code comment lines 423-425):
> "When training we expose half of the current chunk's delta to its outputs to
> generate a usable gradient signal for the target generator (otherwise only
> later chunks receive the update and short documents yield near-zero grads)."

---

### 3. Gradient Flow Analysis: Why This Deviation Exists

#### The Paper's Training Setup

**Equation 1** (Paper line 467-471):
```
W(i)_down = W(i-1)_down + η V̂⊤_[i] Z_[i]
```

**Gradient Path to target_generator**:
```
Loss → O[i+1] → Z[i+1] @ W(i)_down^T
      → W(i)_down = W(i-1)_down + η V̂⊤_[i] Z_[i]
      → V̂_[i] = W_target(Conv1D(X[i]))
      → ∇W_target, ∇Conv1D
```

**Key observation**: In the paper's formulation:
- Chunk i's update ∆W(i) affects chunk **i+1**'s output
- If document has only 1-2 chunks → weak gradient signal
- Short documents → target_generator gets minimal gradients

#### The Code's Solution

By adding `0.5 * deltas` during training:
```
O[i] sees: W(0) + η * (∆W[0] + ... + ∆W[i-1] + 0.5*∆W[i])
```

**Effect**:
- Chunk i's output is influenced by its own target V̂_[i]
- Creates immediate gradient path: Loss → O[i] → ∆W[i] → V̂_[i] → target_generator
- Provides gradient signal even for single-chunk documents

**Trade-off**:
- ✅ Enables gradient flow for target_generator
- ⚠️ Deviates from paper's strict causality during training
- ⚠️ Still results in tiny gradients (your logs show ~1e-5)

---

### 4. Other Implementation Details

#### 4.1 Conv1D Per-Chunk Processing (Paper Line 6)

**Paper Specification** (Algorithm 1, line 6):
```
Vi ← Conv1DK(X(i)_0)Wtarget
```
Note: `X(i)_0` means "token embeddings for chunk i **only**"

**Code Implementation** (`ttt_module.py:303-349`):
```python
def _apply_conv1d_per_chunk(self, token_embeddings: torch.Tensor):
    """Per Paper Algorithm 1 line 2006: Vi ← Conv1D_K(X^(i)_0)·Wtarget
    where X^(i)_0 is token embeddings for chunk i only."""

    token_emb_chunks = token_embeddings.view(B, num_chunks, chunk_size, dim)

    V_hat_chunks = []
    for i in range(num_chunks):
        chunk_emb = token_emb_chunks[:, i]  # Process chunk i separately
        V_hat_chunk = self.target_generator(chunk_emb)
        V_hat_chunks.append(V_hat_chunk)

    return torch.cat(V_hat_chunks, dim=1)
```

**Status**: ✅ **EXACT MATCH** - Applies Conv1D per chunk to prevent cross-boundary leakage

#### 4.2 Streaming Inference Buffering

**Paper**: Not explicitly covered (focuses on parallel training algorithm)

**Code** (`ttt_module.py:203-270`):
```python
def _ttt_forward_inference(self, Z, V_hat, ...):
    """Buffers tokens until a full chunk is ready.
    1. Emit output using current fast weights (apply)
    2. Buffer Z and V_hat
    3. When buffer >= chunk_size, replay through update path
    """
    out = torch.matmul(Z_fp32, self.w_down.t())  # Apply current weights
    self._append_inference_buffers(Z_fp32, V_fp32)
    self._flush_inference_buffers(force=False)  # Update when ready
    return out
```

**Status**: ✅ **IMPLEMENTATION DETAIL** - Not in paper, but mathematically equivalent for streaming

#### 4.3 Document Boundary Resets (Paper Line 14)

**Paper** (Algorithm 1, line 14):
```
At document boundaries: Reset fast weights to W(0)_down
```

**Code** (`train.py:103-116`):
```python
def reset_ttt_on_doc_switch(model, doc_ids, last_doc_id):
    for doc_id in doc_ids:
        if doc_id != current_doc:
            logger.info(f"[TTT RESET] Document switch: {current_doc} -> {doc_id}")
            model.reset_ttt_state()
            current_doc = doc_id
```

And `ttt_module.py:498-551`:
```python
def reset_ttt_state(self, clear_buffers=True):
    if self.ttt_enabled:
        self.w_down.data.copy_(self.w_down_pretrained)
        if clear_buffers:
            self._inference_z_buffer = None
            self._inference_v_buffer = None
```

**Status**: ✅ **EXACT MATCH** - Resets to pretrained W(0)_down at document boundaries

#### 4.4 Delta Clipping (NOT in paper)

**Paper**: No mention of gradient clipping for deltas

**Code** (`ttt_module.py:406-416`):
```python
if self.delta_clip_fro_norm is not None:
    max_norm = deltas.new_tensor(self.delta_clip_fro_norm)
    delta_norms = torch.linalg.vector_norm(deltas.reshape(...), dim=-1)
    scales = (max_norm / (delta_norms + eps)).clamp(max=1.0)
    deltas = deltas * scales.view(num_chunks, B, 1, 1)
```

**From config**: `delta_clip_fro_norm: 100.0` (training), `1e-5` (inference)

**Status**: ⚠️ **EXTRA FEATURE** - Stabilization technique not mentioned in paper

---

### 5. Core Equations: Paper vs Code

#### Equation 1: Update Rule (Paper line 467-471)

**Paper**:
```
W(i)_down = W(i-1)_down + η V̂⊤_[i] Z_[i]
```

**Code** (inference):
```python
# ttt_module.py:472-473
total_delta = cumsum[-1, 0]  # Sum of all deltas
final_state = W_down_init + self.ttt_lr * total_delta
self.w_down.data.copy_(final_state)
```

**Status**: ✅ **MATHEMATICALLY EQUIVALENT**
- Paper: Sequential update per chunk
- Code: Cumulative sum (parallel), same final result

#### V_hat Formula (Paper line 457-463)

**Paper**:
```
V̂ = Conv1D(X_0) W_target
```

**Code** (`ttt_module.py:73-76`):
```python
class LMAlignedTargetGenerator(nn.Module):
    def forward(self, token_embeddings):
        x = self.conv1d(token_embeddings)
        return self.W_target(x)
```

**Status**: ✅ **EXACT MATCH**

#### Z Formula (Paper line 395-401)

**Paper**:
```
Z = φ(HW_gate^T) ⊙ (HW_up^T)
```

**Code** (`ttt_module.py:183-185`):
```python
h = self.linear_in(x)  # Contains [W_gate; W_up]
h = h.view(B, T, 2, -1)
Z = self.activation(h[..., 0, :]) * h[..., 1, :]  # φ(gate) ⊙ up
```

**Status**: ✅ **EXACT MATCH**

---

## 6. Summary Table: All Deviations

| Feature | Paper | Code | Type | Justification |
|---------|-------|------|------|---------------|
| **Chunk delta exposure** | S_prefix only | S_prefix + 0.5*deltas (training) | ⚠️ **DEVIATION** | Gradient signal for target_generator |
| **Delta clipping** | Not mentioned | Fro norm clipping | ⚠️ **ADDITION** | Numerical stability |
| **Streaming buffering** | Not covered | Token buffering + replay | ✅ Implementation detail | Inference optimization |
| **Float32 precision** | Not specified | w_down kept in float32 | ✅ Best practice | Precision for small updates |

---

## 7. The Root Problem: Gradient Vanishing

### The Deviation is a Band-Aid

The `0.5 * deltas` exposure was added to solve a **gradient vanishing problem**, but:

**From your training logs**:
```
Step 1 gradients:
  w_down:                      0.367, 0.484, 0.795  ✅ Healthy
  conv1d.conv.weight:          0.00001             ❌ 50,000x smaller!
  W_target.weight:             0.00001             ❌ 50,000x smaller!
```

**Analysis**:
1. ✅ The 0.5 delta exposure provides *some* gradient signal
2. ❌ BUT it's still ~50,000x weaker than w_down gradients
3. ❌ Root cause: `target_generator` initialized with std=1e-4 (too small)

### The Real Issue: Initialization

**Code** (`wrapped_model.py:149-155`):
```python
elif "target_generator" in p_name:
    torch.nn.init.normal_(new_param, mean=0.0, std=1e-4)
    logger.info("✓ Small-random-initialized (std=1e-4) for warm-start")
```

**Problem**: std=1e-4 produces:
- V_hat magnitude: ~1e-4
- Deltas magnitude: ~1e-4
- Gradient magnitude: ~1e-5 (even with 0.5 exposure)

**The paper doesn't specify initialization**, so you made a conservative choice.

---

## 8. Answers to Your Question

> "Read the paper and code to see if we follow the paper exactly or we doing other stuff"

### Verdict: 99% Paper-Compliant with 1 Training Modification

**What Matches Paper Exactly**:
- ✅ Architecture (MLP as fast weights)
- ✅ Update equation (W = W + η V̂^T Z)
- ✅ LM-aligned targets (Conv1D + W_target)
- ✅ Chunk-wise processing
- ✅ Causal prefix sum
- ✅ Document boundary resets
- ✅ Per-chunk Conv1D application
- ✅ Inference algorithm

**What Deviates from Paper**:
- ⚠️ **Training mode**: Exposes 0.5 * current_chunk_delta to output
  - Paper: `W(i)_apply = W(0) + η * (∆W[0] + ... + ∆W[i-1])`
  - Code: `W(i)_apply = W(0) + η * (∆W[0] + ... + ∆W[i-1] + 0.5*∆W[i])`
  - **Why**: Emergency fix for gradient vanishing
  - **Effect**: Partially successful (gradients exist but still tiny)

**What's Added (Not in Paper)**:
- Delta clipping (numerical stability)
- Streaming token buffering (inference optimization)
- Float32 precision for w_down (precision for small updates)

**Root Cause of Deviation**:
- The deviation exists because of **separate issue**: target_generator initialization too conservative (std=1e-4)
- The 0.5 delta exposure is treating a symptom, not the disease

---

## 9. Recommendations

### Option A: Follow Paper Strictly ✅ Purist Approach
Remove the 0.5 delta exposure, increase initialization:
```python
# wrapped_model.py:154
torch.nn.init.normal_(new_param, mean=0.0, std=1e-2)  # 100x larger

# ttt_module.py:427 - Remove
if self.training:
    S_apply = S_prefix + 0.5 * deltas  # DELETE THIS
else:
    S_apply = S_prefix

# Replace with:
S_apply = S_prefix  # Paper-compliant for both modes
```

**Effect**:
- ✅ 100% paper-compliant
- ✅ Larger init → 10,000x bigger gradients
- ✅ May work without delta exposure

### Option B: Keep Deviation, Increase Init ⚠️ Pragmatic Approach
Keep the 0.5 exposure, increase initialization:
```python
# Keep the 0.5 delta exposure
# Increase init to std=1e-2
```

**Effect**:
- ⚠️ Not paper-compliant (but documented)
- ✅ More gradient signal (both from init + exposure)
- ✅ May be more robust for short documents

### Option C: Paper + Increased LR for target_generator 🔬 Experimental
Follow paper strictly, but use higher LR for target_generator:
```python
# train.py: Add parameter groups
optimizer = AdamW([
    {'params': tgt_gen_params, 'lr': args.optim.lr * 10},
    {'params': other_params, 'lr': args.optim.lr}
])
```

---

## 10. Conclusion

Your implementation is **remarkably faithful to the paper** (99% match). The one deviation exists because:
1. Paper doesn't specify initialization strategy
2. You chose conservative init (std=1e-4)
3. This caused gradient vanishing
4. You added 0.5 delta exposure as emergency fix
5. **But this fix is insufficient** (gradients still 50,000x too small)

**The real fix**: Increase initialization scale (std=1e-2), which may make the deviation unnecessary.

**My recommendation**: Try Option A (paper-strict + larger init) first. If that doesn't work, Option B is reasonable.
