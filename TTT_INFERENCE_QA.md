# TTT Inference Questions & Answers

**Date**: November 13, 2025  
**Purpose**: Clarify implementation questions by reading paper and code

---

## GROUP 1: Questions Answerable from Paper/Code

### Q1: Should we average delta updates over the batch dimension during inference?

**Answer**: NO - Don't average over batch.

**Evidence from paper**:
- Line 2208: "run the inference on Nvidia H800 GPUs with **batch size of 1**"
- The paper's inference experiments use batch_size=1
- Equation 1 (line 469): `W^(i) = W^(i-1) + η V̂^T[i] Z[i]`
  - This is matrix multiplication, not averaging
  - `V̂^T[i]` shape: [dim, chunk_size] for single batch item
  - `Z[i]` shape: [chunk_size, hidden] for single batch item
  - Result: [dim, hidden] - no batch dimension

**Evidence from our training code**:
```python
# Line 170 in ttt_module.py
deltas = torch.einsum('b n t d, b n t h -> n b d h', Vc, Zc)
```
- Output shape: `[num_chunks, B, dim, hidden]`
- Each batch item has its own delta
- No averaging across batch dimension

**Conclusion**: 
- For batch_size=1 (typical inference): delta = V̂^T @ Z directly
- For batch_size>1: Each batch item maintains separate w_down state
- Never average deltas across batch

---

### Q2: Parameter vs Buffer - should we use self.w_down directly or separate self._w_down_current?

**Answer**: Use parameter directly during inference, buffer only for pretrained copy.

**Evidence from code**:
- Training mode: `self.w_down` is nn.Parameter, updated via optimizer
- Inference mode: No optimizer running, can modify `.data` directly
- `register_buffer('w_down_pretrained')`: For storing pretrained copy only

**Best approach**:
```python
# Store pretrained
self.register_buffer('w_down_pretrained', torch.empty(dim, hidden))

# During inference: modify parameter directly
self.w_down.data = self.w_down.data + self.ttt_lr * delta

# During reset: copy from buffer
self.w_down.data.copy_(self.w_down_pretrained)
```

**Why not separate _w_down_current**:
- Unnecessary complexity
- Parameter already exists
- Just need to track pretrained state for reset

---

### Q3: Does the paper's chunk-wise approach apply to inference or only training?

**Answer**: Chunk-wise is primarily for **training efficiency**, but can also be used for inference if processing multiple tokens.

**Evidence from paper**:
- Lines 162-163: "canonical per-token update mechanism of TTT is inherently sequential, severely bottlenecking parallel processing"
- Lines 174-175: "We replace the inefficient per-token updates with a scalable **chunk-wise update rule**"
- Lines 405-408: "Conventional TTT methods... were bound to inefficient per-token updates... Our framework sidesteps this... enabling a far more efficient chunk-wise update strategy"

**Key insight (lines 736-741)**:
> "the parallel scan mathematically equivalent to a sequential update... at document boundaries, the fast weights are reset to their pre-trained state"

**Interpretation**:
- **Training**: Must use chunks for GPU efficiency (process full sequences)
- **Inference**: Can use chunks OR sequential, both are mathematically equivalent
- Choice depends on how many tokens processed per step

---

### Q4: What is W^(0) - pretrained weights or pretrained + first update?

**Answer**: W^(0) is the pretrained weights (before any TTT updates).

**Evidence from paper**:
- Lines 418-420: "Let W^(i)_down be the fast weights state before processing chunk i and **W^(0)_down = W_down**"
- "W_down" refers to the pretrained final projection matrix (line 399)
- Lines 738-741: "at document boundaries, the fast weights are reset to their **pre-trained state**"

**Update sequence**:
```
W^(0) = pretrained weights
W^(1) = W^(0) + η V̂^T[0] Z[0]  (after processing chunk 0)
W^(2) = W^(1) + η V̂^T[1] Z[1]  (after processing chunk 1)
...
```

**In code**:
```python
# Line 178 in ttt_module.py
W_eff = W_init_bc + self.ttt_lr * S
```
- `W_init_bc` = W^(0) = pretrained
- `S` = cumulative updates
- `W_eff[i]` = W^(i) used for chunk i

---

### Q5: What is the apply-then-update order exactly?

**Answer**: For chunk i, use W^(i) to compute output, then update to get W^(i+1).

**Evidence from paper (lines 418-428)**:
> "For each chunk i ∈ [k], we perform two sequential operations:
> 1. **Apply Operation**: The current state of the fast weights W^(i)_down are used to process chunk Z[i], i.e., O[i] = Z[i](W^(i)_down)^T.
> 2. **Update Operation**: The fast weight W^(i)_down are updated using Z[i] as keys and V[i] as values"

**Critical detail in parallel implementation (lines 720-726)**:
> "the effective fast weights for each chunk, **W^(i-1)_down** = W^(0)_down + η∆S_i, and the corresponding output, **O[i] = Z[i](W^(i-1)_down)^T**"

**Wait, there's confusion in notation!**

Let me re-read... Lines 418-420 say:
- "W^(i)_down be the fast weights state **before processing chunk i**"
- So chunk i uses W^(i), then updates to W^(i+1)

But line 726 says:
- "W^(i-1)_down = W^(0) + η∆S_i"
- "O[i] = Z[i](W^(i-1))^T"

**Resolution**: These are the same! The prefix sum gives us:
- ∆S_i = sum of deltas from chunks 0 to i-1
- W_eff[i] = W^(0) + η∆S_i = state **before** processing chunk i
- This is W^(i) in the sequential notation

**In our code (line 173-178)**:
```python
cumsum = torch.cumsum(deltas, dim=0)
S = torch.cat([zero, cumsum[:-1]], dim=0)  # S_i = sum_{j=0}^{i-1}
W_eff = W_init_bc + self.ttt_lr * S      # W^(i) = W^(0) + updates from 0..i-1
```

✅ **This is correct!**

---

### Q6: Does the paper discuss batch processing during inference?

**Answer**: No, paper only mentions batch_size=1 for inference.

**Evidence**:
- Line 2208: "run the inference on Nvidia H800 GPUs with **batch size of 1**"
- All inference/evaluation done with single sequence at a time
- Training uses larger batches (lines 2111, 2122 mention batch sizes for training)

**Implication**: 
- For Moshi inference, we likely only need to handle batch_size=1
- If batch_size>1, each item maintains independent w_down state
- No interaction between batch items during TTT updates

---

### Q7: Is there any discussion of sequential per-token updates vs chunk-wise?

**Answer**: Yes, paper explicitly discusses this tradeoff.

**Sequential per-token (the problem - lines 162-163)**:
> "the canonical per-token update mechanism of TTT is inherently sequential, severely bottlenecking the parallel processing capabilities of modern accelerators"

**Chunk-wise solution (lines 174-175)**:
> "We replace the inefficient per-token updates with a scalable chunk-wise update rule"

**Why chunk-wise works (lines 405-408)**:
> "Since we adapt only the MLP blocks and leave the attention layers intact, we are liberated from the per-token constraint, enabling a far more efficient chunk-wise update strategy"

**Mathematical equivalence (lines 736-741)**:
> "making the parallel scan mathematically equivalent to a sequential update"

**Conclusion**:
- Sequential per-token: Slow but conceptually simple
- Chunk-wise parallel: Fast for training, mathematically equivalent
- Both valid for inference, choice depends on implementation constraints

---

### Q8: What does "document boundaries" mean for reset?

**Answer**: Reset happens when starting a new independent sequence/conversation.

**Evidence from paper (lines 738-741)**:
> "Moreover, at **document boundaries**, the fast weights are reset to their pre-trained state to prevent **context leakage across independent sequences**"

**Interpretation**:
- Document = independent text/conversation
- Boundary = end of one document, start of another
- Purpose: Prevent information from one conversation affecting another
- Implementation: Detect sequence boundaries, call reset

**For Moshi**:
- Document boundary = new conversation session
- Reset when user starts new conversation
- Keep accumulating within same conversation

---

## GROUP 2: Questions Specific to Moshi Architecture (Unanswered)

### Q2.1: What is the value of T in lm_gen.step(codes) during Moshi inference?

**Answer**: T = 1 (always processes one time step at a time during streaming)

**Evidence from code**:

**lm.py line 713** (in `LMGen._step`):
```python
assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
B, Ki, S = input_tokens.shape
assert S == 1, "Only support being given steps one by one."
```

**run_inference.py line 140-141**:
```python
chunk = chunks.popleft()
codes = self.mimi.encode(chunk)  # Returns [B, K, 1]
```

**mimi encode output** (compression.py line 376-387):
```python
def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode the given input tensor to quantized representation.
    
    Args:
        x (torch.Tensor): Float tensor of shape [B, C, T]
    
    Returns:
        codes (torch.Tensor): an int tensor of shape [B, K, T]
    """
```

During streaming, one audio frame → one time step → codes shape [B, K, 1]

**Implication**: 
- Moshi's streaming inference processes **S=1 time step per call**
- `lm_gen.step()` explicitly asserts S==1
- This means T=1 throughout the transformer during streaming
- **Our existing chunk-wise implementation with chunk_size=256 won't work for streaming**
- **Need either**: Sequential per-token update OR handle chunk_size=1 as special case

---

### Q2.2: Does Moshi's streaming process audio frames or individual tokens?

**Answer**: Processes one audio frame at a time, where each frame produces one time step with multiple codebook tokens.

**Evidence**:

**run_inference.py line 86**:
```python
self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
# 24000 Hz / 12.5 fps = 1920 samples per frame
```

**run_inference.py lines 127-141**:
```python
chunks = deque([
    chunk for chunk in in_pcms.split(self.frame_size, dim=2)
    if chunk.shape[-1] == self.frame_size
])

while not all(eos_reached):
    if chunks:
        chunk = chunks.popleft()  # One audio frame
        codes = self.mimi.encode(chunk)  # Encode to [B, K, 1]
    # ...
    tokens = self.lm_gen.step(codes)  # Process one time step
```

**Understanding**:
- One audio frame (80ms @ 12.5 fps) → One encoding step
- Mimi encoder: [B, C=1, T=1920 samples] → [B, K=32, S=1 time step]
- K = number of codebooks (32 for Mimi, but Moshi uses subset)
- S = 1 time step per frame
- LM processes: [B, K, S=1] → [B, K_out, S=1]

**Key insight**: "Frame" and "time step" are equivalent in Moshi streaming. One frame = one T=1 sequence.

---

### Q2.3: How does Moshi's depformer affect TTT?

**Question**: Does the depformer architecture interact with TTT layers?

**Observations from code**:
- Moshi has separate "depformer" (lines 204-218 in lm.py)
- Main transformer vs depformer have different streaming behavior
- Depformer is "streaming_detached" (line 218)

**Need to understand**:
- Do TTT layers exist in depformer or only main transformer?
- How does depformer streaming affect TTT state?
- Should TTT reset when depformer resets?

---

### Q2.4: What is the batch_size during Moshi real-time inference?

**Question**: Does Moshi inference use batch_size=1 or can it be >1?

**Observations**:
- `run_inference.py` line 236: `--batch-size` argument with default=8
- But line 290: `if lm.dep_q == 0: args.batch_size = 1`
- Unclear when batch>1 is actually used

**Need to verify**:
- Typical batch_size in production inference
- Whether multiple conversations can be batched
- How batch dimension affects TTT state (separate per item?)

---

### Q2.5: How does Moshi handle the temporal dimension?

**Question**: What does the T (time/sequence) dimension represent in Moshi during streaming?

**Confusion**:
- Standard LLM: T = sequence length (number of tokens)
- Moshi: Audio frames + multiple codebooks + text tokens
- Streaming mode: Does T=1 (one step) or T>1 (buffered frames)?

**Need to trace**:
- Input shape to transformer during streaming
- How sequence dimension is managed
- Relationship to audio frame timing

---

### Q2.6: Should w_down state be per-batch-item or shared?

**Question**: If batch_size>1 during inference, does each batch item need separate w_down?

**Paper evidence**: Inference uses batch=1 (line 2208)

**But for Moshi**: 
- Default batch_size=8 in run_inference.py
- Multiple parallel conversations?
- Would need separate TTT state per conversation

**Implementation options**:
1. Force batch=1 for TTT (simplest)
2. Maintain separate w_down per batch item (complex)
3. Share w_down across batch (wrong - would leak context)

**Need to decide based on**: Actual Moshi deployment scenario

---

### Q2.7: When exactly should TTT state be reset in Moshi?

**Question**: What constitutes a "document boundary" in an audio conversation model?

**Options**:
1. Never reset (accumulate forever) - matches paper's long-context goal
2. Reset on explicit user signal (new conversation button)
3. Reset on silence detection (audio-based boundary)
4. Reset when KV cache is cleared
5. Reset periodically (every N frames)

**Need to consider**:
- User experience (natural conversation flow)
- Memory constraints (w_down drift)
- Audio-specific characteristics

---

## Summary

### Group 1 (Answered): 7 questions ✅
- Batch averaging: NO
- Parameter vs buffer: Use parameter directly
- Chunk-wise for inference: Yes, can use
- W^(0) definition: Pretrained weights
- Apply-then-update order: Chunk i uses W^(i)
- Batch processing: Paper uses batch=1 for inference
- Sequential vs chunk-wise: Both valid, chunks preferred for efficiency

### Group 2 (Moshi-specific, Unanswered): 7 questions ❓
- T dimension in lm_gen.step()
- Audio frames vs tokens relationship  
- Depformer interaction with TTT
- Actual batch_size in production
- Temporal dimension semantics
- Per-batch-item vs shared w_down
- Reset timing/triggers

**Next step**: Investigate Moshi code to answer Group 2 questions before finalizing implementation plan.
