# Agent 2 Verification Summary - LM-Aligned Objective

## 🚨 CRITICAL BUG FOUND: Conv1D Causality Violation

### The Problem

**Paper says (Algorithm 1, line 2006):**
```
Vi ← Conv1D_K(X^(i)_0)·Wtarget
```
where X^(i)_0 means "token embeddings **for chunk i only**"

**Paper says (Section 3.4, lines 736-738):**
> "This **isolates each delta calculation to its respective chunk**"

**Code does:**
```python
# Line 153 in ttt_module.py
V_hat = self.target_generator(token_embeddings)  # FULL SEQUENCE [B, T, dim]
return self._parallel_ttt_update(Z, V_hat)       # Then chunked
```

Conv1D is applied to the **ENTIRE sequence** before chunking!

---

### Why This is a Bug

At chunk boundaries (e.g., position 255 with chunk_size=256):

**Current Implementation:**
```
Position 255 (last of chunk 0):
  Conv1D receptive field: [255, 256]  ← Position 256 is in Chunk 1!
  V_hat[255] = f(token[255], token[256])  ← Information leakage!

Delta_0 = Σ(V_hat[0:256]^T @ Z[0:256])
        → Includes V_hat[255] which depends on token 256 (Chunk 1)
        → Delta_0 contains future information ❌
```

**Paper's Intent:**
```
Position 255 (last of chunk 0):
  Conv1D receptive field: [255, PAD]  ← Padding, not next chunk!
  V_hat[255] = f(token[255], PAD)  ← No leakage!

Delta_0 = Σ(V_hat[0:256]^T @ Z[0:256])
        → All V_hat values only depend on Chunk 0
        → Delta_0 contains NO future information ✅
```

---

### Impact

1. **Breaks causality** at every chunk boundary
2. **Violates paper's core requirement**: "isolates each delta calculation to its respective chunk"
3. **Information leakage**: For 4096 tokens with chunk_size=256, there are 15 boundaries with leakage
4. **Theoretical guarantee broken**: Parallel algorithm is no longer mathematically equivalent to sequential

---

### What's Correct ✅

1. ✅ Formula V̂ = Conv1D(X₀)·Wₜₐᵣ𝓰ₑₜ
2. ✅ Conv1D semantics for next-token prediction (right padding)
3. ✅ Loss function L(·,·) = -⟨·,·⟩_F and gradient derivation
4. ✅ Kernel size (kernel_size=2)
5. ✅ Efficient einsum delta computation

---

### Recommendation

**Priority: HIGH** - Fix before production use

**Fix:** Apply Conv1D per-chunk, not to full sequence

```python
# Chunk token_embeddings FIRST
token_emb_chunks = chunk(token_embeddings, chunk_size)

# Apply Conv1D to each chunk independently
V_hat_chunks = [self.target_generator(chunk) for chunk in token_emb_chunks]

V_hat = concat(V_hat_chunks)
return self._parallel_ttt_update(Z, V_hat)
```

---

## Full Report

See: `/home/user/moshi_in_place/AGENT2_LM_ALIGNED_OBJECTIVE_VERIFICATION.md`

