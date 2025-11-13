# Moshi TTT Reality Check: Training vs Inference

**Date**: 2025-11-13
**Analysis**: Deep dive into how TTT actually works in Moshi's streaming architecture

---

## Key Discovery: Streaming Inference Uses T=1

### Moshi's Streaming Architecture

**Server mode** (`server.py`):
```python
# Line 52, 59-60
batch_size = 1
self.mimi.streaming_forever(1)
self.lm_gen.streaming_forever(1)
```

**Batch inference** (`run_inference.py`):
```python
# Line 266: DEFAULT batch_size=8!
parser.add_argument("--batch-size", type=int, default=8)

# Line 295: Override for some models
if lm.dep_q == 0:
    args.batch_size = 1
```

**Streaming step** (`lm.py:700`):
```python
assert S == 1, "Only support being given steps one by one."
```

### What This Means for TTT

**Each forward pass**:
- Input shape: `[B, K=17, 1]` where B=1-8, K=codebooks, **T=1**
- After embedding: `[B, 1, d_model]` - **ONE token position only**
- TTT receives: `x=[B,1,d_model]`, `token_embeddings=[B,1,d_model]`

**TTT chunking with T=1**:
```python
# ttt_module.py:185
effective_chunk_size = min(T, self.chunk_size)  # min(1, 256) = 1
num_chunks = 1  # Always!
```

**Result**: No chunking happens - processes single token at a time.

---

## Critical Analysis

### 1. Conv1D "Causality Bug" - Is It Actually a Bug?

**With T=1 (streaming inference)**:
- Conv1D input: `[B, 1, d_model]`
- kernel_size=2 pads right by 1: `[token_t, PAD]`
- Cannot see next token because there IS no next token in the batch
- **This is CORRECT for streaming!**

**With T>chunk_size (training)**:
- Conv1D input: `[B, T, d_model]` where T could be 1024+
- If applied to full sequence before chunking:
  - Position 255 sees [255, 256] across chunk boundary
  - **This IS a bug for training!**

**Conclusion**: Conv1D behavior is correct for inference, wrong for training.

---

### 2. Multi-Batch Bug - Confirmed Real Issue

**The bug** (`ttt_module.py:238`):
```python
final_state = W_eff[-1, 0] + self.ttt_lr * deltas[-1, 0]  # Hardcoded [0]!
```

**Impact**:
- `batch_size=1` (server.py): ✅ Works fine
- `batch_size=8` (default run_inference.py): ❌ **BROKEN**
  - Only batch element 0 gets TTT updates
  - Elements 1-7 use stale weights

**With T=1 and num_chunks=1**:
```python
W_eff.shape = [1, B, dim, hidden]  # [num_chunks=1, batch_size, ...]
deltas.shape = [1, B, dim, hidden]

# Should be:
final_state = W_eff[-1, :] + self.ttt_lr * deltas[-1, :]  # All batches [:]
```

**This is a CONFIRMED BUG** regardless of chunking.

---

### 3. Paper vs Moshi: Fundamental Mismatch?

**Paper assumption**:
- Process sequences with T >> chunk_size
- Example: T=1024, chunk_size=256 → 4 chunks
- Update fast weights per-chunk in parallel

**Moshi streaming reality**:
- Process T=1 token at a time (autoregressive streaming)
- num_chunks = 1 always
- Fast weights update every single token

**Questions**:
1. Did the paper's authors test on streaming inference?
2. Is TTT meant for batch training only, not streaming inference?
3. Is the chunking optimization irrelevant for streaming?

---

## Training vs Inference Analysis

### During TRAINING (need to verify):

**Likely scenario**:
- Sequences processed in longer chunks (T=512, T=1024, etc.)
- True chunking with num_chunks > 1
- Conv1D per-chunk matters
- Batch updates with B > 1

**If this is true**:
- Conv1D bug DOES matter (sees across chunk boundaries)
- Multi-batch bug DOES matter (only saves batch[0])
- Both bugs affect training quality

### During INFERENCE (confirmed):

**Streaming mode** (server.py):
- T=1, B=1 per forward pass
- Conv1D sees [current, PAD] - correct
- Multi-batch bug doesn't trigger (B=1)
- **Works correctly!**

**Batch inference** (run_inference.py default):
- T=1, B=8 per forward pass
- Conv1D sees [current, PAD] - correct
- Multi-batch bug DOES trigger (B=8)
- **BROKEN for batch elements 1-7!**

---

## The Correct Fixes

### Fix 1: Multi-Batch Bug (MUST FIX)

**Current** (`ttt_module.py:238`):
```python
if not self.training:
    final_state = W_eff[-1, 0] + self.ttt_lr * deltas[-1, 0]  # ❌ Only [0]
    self.w_down.data.copy_(final_state)
```

**Fixed**:
```python
if not self.training:
    # Handle all batch elements
    final_state = W_eff[-1, :] + self.ttt_lr * deltas[-1, :]  # ✅ All [:]

    # Option A: Enforce batch_size=1
    if final_state.shape[0] != 1:
        raise ValueError("TTT inference only supports batch_size=1")
    self.w_down.data.copy_(final_state[0])

    # Option B: Maintain per-batch states (complex, requires architecture change)
    # Would need self.w_down to be [B, dim, hidden] not [dim, hidden]
```

**Recommended**: Option A - enforce batch_size=1 in streaming inference mode.

---

### Fix 2: Conv1D Chunking (FOR TRAINING)

**Only needed if training with T > chunk_size**.

**Current** (`ttt_module.py:144-156`):
```python
def _ttt_forward(self, x, token_embeddings):
    Z = self.activation(h[..., 0, :]) * h[..., 1, :]
    V_hat = self.target_generator(token_embeddings)  # ❌ Full sequence
    return self._parallel_ttt_update(Z, V_hat)
```

**Fixed for training**:
```python
def _ttt_forward(self, x, token_embeddings):
    B, T, _ = x.shape
    Z = self.activation(h[..., 0, :]) * h[..., 1, :]

    # For streaming (T=1), apply Conv1D normally
    if T == 1:
        V_hat = self.target_generator(token_embeddings)

    # For training (T > chunk_size), apply per-chunk
    else:
        # Chunk first, then apply Conv1D per chunk
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        # Pad to chunk boundary
        pad_size = num_chunks * self.chunk_size - T
        if pad_size > 0:
            token_embeddings = F.pad(token_embeddings, (0, 0, 0, pad_size))

        # Reshape into chunks
        token_emb_chunks = token_embeddings.view(B, num_chunks, self.chunk_size, -1)

        # Apply Conv1D per chunk
        V_hat_chunks = []
        for i in range(num_chunks):
            chunk_emb = token_emb_chunks[:, i]  # [B, chunk_size, d_model]
            V_hat_chunk = self.target_generator(chunk_emb)
            V_hat_chunks.append(V_hat_chunk)

        V_hat = torch.cat(V_hat_chunks, dim=1)  # [B, T_padded, d_model]
        if pad_size > 0:
            V_hat = V_hat[:, :T]

    return self._parallel_ttt_update(Z, V_hat)
```

---

## Recommendations

### For Your Use Case (Streaming Inference)

**You said**: "I am using moshi inference for one conversation at a time"

**Implications**:
- Always `batch_size=1` ✅
- Always `T=1` per forward pass ✅
- Multi-batch bug doesn't affect you ✅
- Conv1D behavior is correct ✅

**What you need**:
1. ✅ **Nothing!** Your usage is fine.
2. ⚠️ Optional: Add assertion to catch if batch_size > 1

**Simple safety check**:
```python
# Add to ttt_module.py __init__ or forward
if not self.training and B > 1:
    raise ValueError(
        "TTT streaming inference only supports batch_size=1. "
        "Current batch_size={B}. This is a known limitation."
    )
```

---

### For Paper Compliance (Training)

**If you ever train with TTT**:
1. ❌ **MUST FIX**: Conv1D per-chunk (not full sequence)
2. ❌ **MUST FIX**: Multi-batch handling (all batch elements)
3. ⚠️ Verify training sequences use T > chunk_size

---

## Summary Table

| Issue | Affects Inference? | Affects Training? | Your Use Case? |
|-------|-------------------|-------------------|----------------|
| Conv1D chunking | ❌ No (T=1) | ✅ Yes | ❌ Not affected |
| Multi-batch bug | ✅ Yes (if B>1) | ✅ Yes | ❌ Not affected (B=1) |
| No reset | ✅ Design choice | ✅ Design choice | ✅ Correct (single convo) |

---

## Final Verdict

**For your streaming inference with batch_size=1**:
- ✅ Implementation works correctly
- ✅ No bugs affect your use case
- ✅ Conv1D behavior is appropriate for streaming
- ✅ No chunking needed (T=1)

**For general Moshi usage**:
- ❌ Batch inference (batch_size>1) is BROKEN
- ⚠️ Training may have issues if used

**For paper compliance**:
- ⚠️ Deviates from paper in streaming mode (by necessity)
- ❌ Has bugs that violate paper if used for training

**Recommendation**: Add batch_size=1 assertion for safety, otherwise your usage is correct.
