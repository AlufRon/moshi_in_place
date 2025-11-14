# In-Place TTT Implementation Insights

**Author:** GitHub Copilot Agent  
**Date:** November 14, 2025

This note distills the key findings from auditing the current Moshi In-Place TTT implementation against the ICLR 2026 paper (`papers/ttt_in_place_paper.txt`) and supporting code paths (`moshi/moshi/moshi/models/lm.py`, `moshi/moshi/moshi/modules/transformer.py`, `moshi/moshi/moshi/modules/ttt_module.py`, and `moshi/tests/ttt/test_plan_suite.py`). It is meant as a quick reference for reviewers and future contributors.

---

## 1. Architectural Alignment

| Paper Requirement | Implementation Evidence | Status |
| --- | --- | --- |
| Repurpose gated MLP `W_down` as fast weights while keeping `W_up`, `W_gate` frozen (Sec. 3.1) | `TTTGating.linear_in` holds `[W_gate; W_up]` (slow), `w_down` created as the only fast parameter and copied from pretrained checkpoints in `finetune/wrapped_model.py` | ✅
| Apply-then-update chunk loop (Figure 1 & lines 418–428) | `_parallel_ttt_update` computes chunk deltas, takes prefix sums to obtain causal states, then matmuls `Z[i]` with `W_down^(i)` before adding the next update | ✅
| LM-aligned target generator `V̂ = Conv1D(X₀) W_target` (Eq. 1) | `LMAlignedTargetGenerator` runs causal conv (right padding for future tokens) followed by linear projection; token embeddings are plumbed from `LMModel.forward_text` → `StreamingTransformer` → TTT layers | ✅
| Chunk-wise parallelism with causal prefix scan (Sec. 3.4) | `torch.einsum` + `torch.cumsum` implement the three-stage algorithm exactly (delta calc, scan, apply) | ✅
| Reset fast weights at document boundaries (Sec. 3.4, "Boundary Handling") | **Not implemented yet**; `TTTGating.reset_ttt_state` exists but is never invoked by `LMModel`/`LMGen` when sequences reset | ⚠️

---

## 2. Fast Weight Lifecycle Observations

1. **Initialization:**
   - `w_down` is hydrated directly from the pretrained `linear_out.weight` before `load_state_dict`, ensuring drop-in equivalence at step zero.
   - Buffers such as `w_down_pretrained` mirror the loaded weights, enabling manual resets.

2. **Precision:**
   - Fast weights remain in float32 regardless of model dtype, which is consistent with the paper’s need for small lr stability when running in bfloat16.

3. **Runtime Updates:**
   - During training/inference, `w_down.data.copy_(final_state)` persists the last chunk’s state if `self.training == False`. As a result, autoregressive decoding accumulates fast-weight state across streaming chunks.

4. **Missing Reset Hook:**
   - `LMGen._reset_callback` only resets transformer streaming buffers; it never calls into the gating modules. Consequently, separate documents currently leak fast weights unless the client manually triggers `reset_ttt_state()`.

---

## 3. Objective & Target Pathing Checks

- **Token Embedding Sum:** `LMModel.forward_text` explicitly sums all audio codebook embeddings plus the text embedding, then passes this tensor as `token_embeddings` through every transformer layer. This ensures the target generator sees the same features described in Sec. 3.2.
- **Chunked Conv Handling:** `_apply_conv1d_per_chunk` guards against cross-chunk leakage by slicing the sequence into non-overlapping windows before applying Conv1D, matching Algorithm 1 (Appendix C).
- **Delta Clipping:** Configurable Frobenius-norm clipping prevents unbounded growth during long contexts; tests cover both clipped and unclipped regimes.

---

## 4. Test Coverage Snapshot (`moshi/tests/ttt/test_plan_suite.py`)

| Scenario | Paper Tie-In | Status |
| --- | --- | --- |
| Fast-weight delta clipping & telemetry | Sec. 3.1 runtime stability | ✅ Tests ensure norms stay under cap and counters increment |
| LM-aligned target chunking vs. manual reference | Algorithm 1 chunk semantics | ✅ |
| Learning-rate / chunk-size sweeps | Ablation Sec. 4.3 | ✅ Basic grid sanity |
| Perplexity proxy improvement after updates | Theorem 1 intuition | ✅ |
| Boundary reset behavior | Sec. 3.4 boundary handling | ⚠️ Missing automated test |
| Conv/Projection ablations | Figure 3c | ⚠️ Not yet encoded |

Actionable follow-ups for the tests are already tracked in `todoList` items #1–#2.

---

## 5. Gaps & Recommendations

1. **Document Boundary Reset (Highest Priority):**
   - Plumbing needed so that `LMModel.reset_streaming()` (and `LMGen` callbacks) call `reset_ttt_state()` on every TTT-enabled gating layer.
   - Add regression test simulating back-to-back documents to guarantee no leakage.
   - Practical note: today’s inference harness spins up a fresh process per conversation, so the missing reset hasn’t bitten us yet; this becomes critical only if we ever reuse a model instance across sessions.

2. **Paper-Aligned Ablation Tests:**
   - Implement toggles for Conv1D and `W_target` to match Figure 3(c) experiments and ensure performance degradation when either component is removed.

3. **Telemetry Surfacing:**
   - Consider surfacing `ttt_clip_event_counter`, `w_down.norm`, and `_update_count` through the public API for easier monitoring during long-context inference.

4. **Documentation Sync:**
   - Update `IMPLEMENTATION_VERIFICATION.md` once resets/tests land so the “⚠️” rows can be flipped to green.

---

## 6. Quick Pointers

- **Core files:**
  - `moshi/moshi/moshi/modules/ttt_module.py` — LM-aligned target generator and chunked fast-weight updates
  - `moshi/moshi/moshi/modules/transformer.py` — TTT gating integration points and layer scheduling
  - `moshi/moshi/moshi/models/lm.py` — Token embedding plumbing and inference hooks
  - `moshi/tests/ttt/test_plan_suite.py` — Regression coverage for clipping, chunking, telemetry, etc.

- **Current Config Hooks:** Set under `ttt_config` (chunk size, lr, kernel size, layer placement) inside LM construction or finetune scripts.

---

## 7. Next Steps Checklist

- [ ] Wire fast-weight resets into `LMModel.reset_streaming` and `LMGen._reset_callback`.
- [ ] Extend regression suite with boundary-reset test plus Conv/Projection ablation cases.
- [ ] Re-run `pytest moshi/tests/ttt/test_plan_suite.py` inside `moshi_ttt_fixed` after changes.
- [ ] Update `IMPLEMENTATION_VERIFICATION.md` to reflect the resolved gaps.

This document will evolve alongside the remaining TODO items; feel free to append additional findings as investigations continue.
