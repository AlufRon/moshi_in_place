# TTT Validation & Observability Test Plan

This note captures every test we should run (or build) to understand whether Moshi's In-Place TTT stack truly works under long-running inference without resets. The focus is on proving stability, catching regressions before production, and making failures obvious through instrumentation.

## 1. Goals & Success Signals
- **Stability:** Fast weights (`w_down`) stay bounded and do not explode or drift to NaNs during hours-long conversations.
- **Correctness:** Token embeddings used for LM-aligned targets match the transformer inputs; text/audio alignment remains consistent.
- **Responsiveness:** TTT updates improve perplexity and subjective audio quality relative to TTT-disabled baselines.
- **Observability:** Clipping, update counts, and saturation events are surfaced via logs/metrics so we can diagnose issues quickly.
- **Regression safety:** Short prompts, zero-shot runs, and complex conditioning flows behave exactly as before when TTT is off.

## 2. High-Level Test Matrix
| # | Category | Test Name | Purpose | Key Signals |
|---|----------|-----------|---------|-------------|
| 1 | Unit | Fast-weight delta clipping | Ensure Frobenius-norm clipping triggers correctly and stays CUDA-graph safe | Max delta norm \<= τ, clip counter increments |
| 2 | Unit | Target generator chunking | Conv1D per chunk matches streaming math | Chunk boundaries show no leakage, outputs match reference |
| 3 | Unit | Token embedding plumbing | Summed audio+text embeddings reach TTT layers | Transformer receives identical tensor as gating input, gradients consistent |
| 4 | Component | Single-layer TTT A/B | Compare outputs of TTT-enabled layer vs frozen baseline on synthetic data | Lower reconstruction loss, deterministic replay |
| 5 | Integration | Short-form inference sanity | 2–3 minute convo with CUDA graph capture | No regressions vs baseline, clip counter near zero |
| 6 | Integration | Long-form (30–60 min) streaming | Reproduce historic gibberish scenario | Observe clip events, fast-weight norms plateau |
| 7 | Integration | YaRN long-context sweep | 8k, 16k, 32k tokens with and without TTT | Same logits as baseline pre-TTT, monitor throughput |
| 8 | Stress | Learning-rate grid | Vary TTT LR (1e-4–1e-2) and chunk sizes | Identify divergence boundaries, best LR |
| 9 | Regression | TTT disabled mode | Run full inference/test suite with `ttt_config.enabled=False` | Bitwise-equal logits to pre-TTT commit |
|10 | Observability | Telemetry export test | Force clipping and verify logs/metrics | Log line structure stable, counter monotonic |
|11 | QA | Human listening panel | Rate audio from long conversations before/after fixes | MOS gap \> 0 vs corrupted baseline |
|12 | Tooling | Replay harness for `w_down` | Serialize fast weights mid-run and reload | Reproducible state, reset path optional |

## 3. Detailed Test Descriptions

### 3.1 Unit & Component Tests
1. **Fast-weight delta clipping**
   - *Setup:* Isolate `TTTGating._parallel_ttt_update` with deterministic tensors; sweep delta norms above/below τ.
   - *Assertions:* output deltas capped to τ, `ttt_clip_event_counter` increments per clipped chunk, no `.item()` usage (CUDA-graph friendly).
   - *Edge cases:* `delta_clip_fro_norm=None`, mixed precision inputs, `chunk_size=1` streaming case.

2. **Target-generator chunking**
   - *Setup:* Feed >chunk_size sequence with known pattern; compare `_apply_conv1d_per_chunk` vs reference implementation that zeroes future tokens.
   - *Assertions:* No leakage across chunk boundaries; streaming (T\<=chunk_size) matches full Conv1D.

3. **Token embedding plumbing**
   - *Setup:* Hook `transformer.forward` to capture `token_embeddings` argument and compare to manually recomputed sum of text+audio embeddings before conditioning.
   - *Assertions:* Tensors are identical, dtype conversions consistent, gradient flows preserved (finite differences).

4. **Single-layer TTT A/B**
   - *Setup:* Build tiny transformer (one TTT layer) and run teacher-forcing on synthetic batch.
   - *Assertions:* With TTT enabled, reconstruction loss drops vs disabled baseline; disable clipping to confirm difference, then re-enable to show stability.

### 3.2 Integration & System Tests
5. **Short-form inference sanity**
   - *Scenario:* 120-second scripted conversation using production inference flags (CUDA graphs, chunk_size=256, τ=1e-5).
   - *Metrics:* Logits continuity, audio MOS (quick listen), `ttt_clip_event_counter` ideally zero, `w_down` norm stable.

6. **Long-form streaming endurance**
   - *Scenario:* 30–60 minute interactive session replicating the “gibberish after minutes” repro.
   - *Metrics:* Sliding perplexity, clip counter trend, `||w_down||_F` vs time, captured audio transcript quality.
   - *Goal:* Prove gibberish disappears and drift stops with clipping + correct embeddings.

7. **YaRN context sweep**
   - *Scenario:* Prompts at 8k/16k/32k tokens with heavy conditioning. Compare TTT-on vs TTT-off logits for first N tokens.<br>
   - *Metrics:* Relative error \< 1e-4 for overlapping region, throughput regression \<3%.

8. **Learning-rate grid / chunk ablation**
   - *Scenario:* Sweep LR ∈ {3e-4, 1e-3, 3e-3, 1e-2} and chunk sizes {128, 256, 512}.
   - *Metrics:* Track best LR per scenario, record divergence boundaries (NaNs, clip saturations).

9. **TTT disabled regression guard**
   - *Scenario:* Run full `tests/` suite and inference smoke tests with `enabled=False`.
   - *Metrics:* Exact match with pre-TTT checkpoint, ensuring no unintended perturbation.

10. **Telemetry export & logging**
    - *Scenario:* Force clipping by injecting adversarial deltas; ensure structured log line/metric (e.g., `TTT_CLIP chunk=12 norm=...`).
    - *Metrics:* Logs emitted once per chunk (or aggregated), compatible with CUDA graph capture (no `.item()`).

11. **Human listening QA**
    - *Scenario:* Prepare paired audio (TTT fix vs old gibberish) and run MOS-style blind test.
    - *Goal:* Validate that technical metrics translate to audible quality improvements.

12. **Fast-weight replay harness**
    - *Scenario:* Snapshot `w_down` mid-conversation, reload, and continue streaming to ensure deterministic continuation without resets.
    - *Use:* Helps debug if gibberish still appears, isolates state serialization bugs.

### 3.3 Observability Hooks to Add Before Running Tests
- Log `ttt_clip_event_counter` every N tokens plus `max_delta_norm` for the window.
- Optional histogram/EMA of `||w_down||_F` to spot drift.
- TTT layer-wise update counters to ensure only intended layers run TTT (respecting `start_layer` + `layer_frequency`).
- Exposure of chunk-level LR, τ, and chunk_size in experiment metadata for reproducibility.

## 4. Tooling & Automation Suggestions
- Extend `test_ttt_minimal.py` into a pytest module that covers items 1–4.
- Create a `scripts/run_ttt_longform_eval.sh` that orchestrates tests 5–8 with standardized logging and optional cloud bucket upload of artifacts.
- Integrate telemetry assertions into CI (fail if clip counter exceeds threshold or logs missing).
- Provide a lightweight notebook/markdown template for manual QA notes (link to MOS panel results).

## 5. Coverage Gaps & Follow-ups
- Need Python ≥3.10 in CI so type-hinted modules run (current local failure blocks automated tests).
- Missing automatic comparison versus historical logs (add diffing script).
- Consider synthetic adversarial prompts to stress TTT (rapid topic shifts, code-switching) once baseline tests pass.

This plan should serve as the backlog for Todo #8 (“Add instrumentation + tests”). Prioritize unit tests + logging hooks first, then scale to long-form runs with human evaluation.
