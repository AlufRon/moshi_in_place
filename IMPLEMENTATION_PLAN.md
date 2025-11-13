# In-Place TTT Implementation Plan for Moshi

## Overview
This document outlines the minimal code changes needed to implement In-Place Test-Time Training (TTT) in Moshi, following the ICLR 2026 paper exactly.

## Key Design Decisions (Confirmed)
1. **Token Embeddings**: Pass original embeddings through all layers (Option A)
2. **Chunk Size**: 256 tokens (configurable)
3. **TTT Layers**: Every 6th layer (layers 5, 11, 17, 23, 29 in 32-layer transformer)
4. **No Resets**: Fast weights persist across conversation (no document boundary resets)
5. **Conv1D**: Configurable kernel size, learnable weights
6. **Gradient Flow**: Let gradients flow through TTT updates (no detach)

## Critical Understanding from Paper

### Chunk-wise Update Rule (Equation 1)
```
W_down^(i) = W_down^(i-1) + η * V̂[i]^T @ Z[i]
```
Where:
- `Z[i]` = intermediate activations from gated MLP
- `V̂[i]` = Conv1D(X_0[i]) @ W_target (target from token embeddings)
- `η` = TTT learning rate (separate from main optimizer)

### Parallel Context Parallelism Algorithm
From Algorithm 1 in paper:
```
Step 1: Compute all deltas in parallel
  ΔW_i = V̂[i]^T @ Z[i]  for all i

Step 2: Prefix sum (causal cumsum)
  S_i = Σ(j=1 to i-1) ΔW_j
  Note: Chunk i uses sum of chunks BEFORE it (not including i)

Step 3: Apply in parallel
  W_down^(i-1) = W_down^(0) + η * S_i
  O[i] = Z[i] @ W_down^(i-1)^T
```

The key is: **S_i contains updates from chunks 0 to i-1**, so chunk i uses the state BEFORE processing chunk i.

---

## Implementation Structure

### New Files to Create

#### 1. `moshi/moshi/modules/ttt_module.py` (NEW)
Core TTT components:

```python
class LMAlignedTargetGenerator(nn.Module):
    """Generates targets V̂ from token embeddings using Conv1D + projection.
    
    V̂ = Conv1D(X_0) @ W_target
    """
    def __init__(self, d_model, kernel_size=2, device=None, dtype=None):
        # Causal 1D convolution
        self.conv1d = CausalConv1D(d_model, d_model, kernel_size)
        # Learnable projection
        self.W_target = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, token_embeddings):
        # token_embeddings: [B, T, d_model]
        # Apply causal conv1d, then projection
        return self.W_target(self.conv1d(token_embeddings))


class CausalConv1D(nn.Module):
    """Causal 1D convolution that looks at future tokens.
    
    Uses padding to ensure causality while allowing future token info.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.kernel_size = kernel_size
    
    def forward(self, x):
        # x: [B, T, C]
        # Transpose to [B, C, T] for Conv1d
        x = x.transpose(1, 2)
        # Causal padding: pad on the right to see "future" tokens
        x = F.pad(x, (0, self.kernel_size - 1))
        x = self.conv(x)
        # Remove extra positions, transpose back
        x = x[:, :, :-(self.kernel_size - 1)]
        return x.transpose(1, 2)


class TTTGating(nn.Module):
    """Gated MLP with In-Place TTT on the final projection (W_down).
    
    Standard flow:
      Z = activation(H @ W_gate^T) ⊙ (H @ W_up^T)
      O = Z @ W_down^T
    
    TTT flow:
      1. Compute Z (same as above)
      2. Generate V̂ from token embeddings
      3. Chunk-wise update W_down using Z and V̂
      4. Apply updated W_down to get O
    """
    def __init__(
        self,
        dim: int,
        dim_feedforward: int,
        activation,
        ttt_config: dict,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # Same hidden size calculation as ActivationGating
        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3
        
        # Slow weights (frozen during TTT updates)
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)
        
        # Fast weights (W_down) - updated during TTT
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)
        
        self.activation = activation
        
        # TTT-specific components
        self.ttt_enabled = ttt_config.get('enabled', True)
        self.chunk_size = ttt_config.get('chunk_size', 256)
        self.ttt_lr = ttt_config.get('learning_rate', 1e-3)
        
        # Target generator (learns W_target and Conv1D)
        self.target_generator = LMAlignedTargetGenerator(
            dim,
            kernel_size=ttt_config.get('conv_kernel_size', 2),
            **factory_kwargs
        )
    
    def forward(self, x: torch.Tensor, token_embeddings: torch.Tensor):
        """
        Args:
            x: Hidden states [B, T, dim]
            token_embeddings: Original token embeddings [B, T, dim]
        """
        if not self.ttt_enabled or not self.training:
            # Standard gating without TTT
            return self._standard_forward(x)
        
        # TTT forward with chunk-wise updates
        return self._ttt_forward(x, token_embeddings)
    
    def _standard_forward(self, x):
        """Standard gated MLP forward (no TTT)."""
        x = self.linear_in(x)
        B, T, _ = x.shape
        x = x.view(B, T, 2, -1)
        x = self.activation(x[..., 0, :]) * x[..., 1, :]
        x = self.linear_out(x)
        return x
    
    def _ttt_forward(self, x, token_embeddings):
        """TTT forward with chunk-wise W_down updates."""
        B, T, dim = x.shape
        
        # Step 1: Compute Z (intermediate activations)
        h = self.linear_in(x)
        h = h.view(B, T, 2, -1)
        Z = self.activation(h[..., 0, :]) * h[..., 1, :]  # [B, T, hidden]
        
        # Step 2: Generate targets V̂ from token embeddings
        V_hat = self.target_generator(token_embeddings)  # [B, T, dim]
        
        # Step 3: Chunk-wise TTT update (parallel context parallelism)
        O = self._parallel_ttt_update(Z, V_hat)
        
        return O
    
    def _parallel_ttt_update(self, Z, V_hat):
        """Parallel context parallelism implementation.
        
        Following Algorithm 1 from the paper.
        """
        B, T, hidden = Z.shape
        _, _, dim = V_hat.shape
        
        # Get initial fast weights
        W_down_init = self.linear_out.weight  # [dim, hidden]
        
        # Partition into chunks
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        
        # Pad to make divisible by chunk_size
        if T % self.chunk_size != 0:
            pad_size = num_chunks * self.chunk_size - T
            Z = F.pad(Z, (0, 0, 0, pad_size))
            V_hat = F.pad(V_hat, (0, 0, 0, pad_size))
        
        # Reshape into chunks: [B, num_chunks, chunk_size, d]
        Z_chunks = Z.view(B, num_chunks, self.chunk_size, hidden)
        V_hat_chunks = V_hat.view(B, num_chunks, self.chunk_size, dim)
        
        # Step 1: Compute all deltas in parallel
        # ΔW_i = V̂[i]^T @ Z[i]
        # V_hat_chunks[i]: [B, chunk_size, dim]
        # Z_chunks[i]: [B, chunk_size, hidden]
        # Result: [B, num_chunks, dim, hidden]
        deltas = torch.einsum('bncd,bnch->nbdh', V_hat_chunks, Z_chunks)
        # Rearrange: [num_chunks, B, dim, hidden]
        
        # Step 2: Prefix sum (causal cumsum)
        # S_i = Σ(j=0 to i-1) ΔW_j
        cumsum = torch.cumsum(deltas, dim=0)  # [num_chunks, B, dim, hidden]
        
        # Shift to make causal: chunk i uses sum of chunks 0..i-1
        # S_0 = 0, S_1 = ΔW_0, S_2 = ΔW_0 + ΔW_1, ...
        S = torch.cat([
            torch.zeros_like(cumsum[0:1]),  # S_0 = 0
            cumsum[:-1]  # S_1, S_2, ..., S_{n-1}
        ], dim=0)  # [num_chunks, B, dim, hidden]
        
        # Step 3: Apply updates and compute outputs in parallel
        # W_down^(i-1) = W_down^(0) + η * S_i
        # O[i] = Z[i] @ W_down^(i-1)^T
        
        # Broadcast W_down_init: [dim, hidden] -> [num_chunks, B, dim, hidden]
        W_down_init_bc = W_down_init.unsqueeze(0).unsqueeze(0).expand(
            num_chunks, B, -1, -1
        )
        
        # Effective weights for each chunk
        W_down_eff = W_down_init_bc + self.ttt_lr * S  # [num_chunks, B, dim, hidden]
        
        # Compute outputs: Z_chunks @ W_down_eff^T
        # Z_chunks: [B, num_chunks, chunk_size, hidden]
        # W_down_eff: [num_chunks, B, dim, hidden]
        # Output: [B, num_chunks, chunk_size, dim]
        
        # Rearrange for batch matrix multiply
        Z_chunks_flat = Z_chunks.transpose(0, 1)  # [num_chunks, B, chunk_size, hidden]
        W_down_eff_T = W_down_eff.transpose(-2, -1)  # [num_chunks, B, hidden, dim]
        
        # Batch matmul: [num_chunks, B, chunk_size, hidden] @ [num_chunks, B, hidden, dim]
        O_chunks = torch.matmul(Z_chunks_flat, W_down_eff_T)  # [num_chunks, B, chunk_size, dim]
        
        # Reshape back: [B, num_chunks, chunk_size, dim] -> [B, T, dim]
        O = O_chunks.transpose(0, 1).reshape(B, -1, dim)
        
        # Remove padding
        if T % self.chunk_size != 0:
            O = O[:, :T]
        
        return O
```

---

### Files to Modify

#### 2. `moshi/moshi/modules/transformer.py` (MODIFY)

**Changes needed:**

##### A. Modify `StreamingTransformerLayer.__init__`
Add TTT configuration support:

```python
def __init__(
    self,
    d_model: int,
    num_heads: int,
    dim_feedforward: int | list[int] = 2048,
    # ... existing params ...
    ttt_config: dict | None = None,  # NEW PARAMETER
    layer_idx: int | None = None,     # NEW PARAMETER
    device=None,
    dtype=None,
):
    # ... existing code ...
    
    # Replace gating initialization
    if gating == "none":
        # ... existing code unchanged ...
    else:
        # NEW: Check if this layer should use TTT
        use_ttt = False
        if ttt_config is not None and layer_idx is not None:
            ttt_frequency = ttt_config.get('layer_frequency', 6)
            ttt_start_layer = ttt_config.get('start_layer', 0)
            if layer_idx >= ttt_start_layer and (layer_idx - ttt_start_layer) % ttt_frequency == 0:
                use_ttt = True
        
        if weights_per_step:
            # ... existing code unchanged ...
        else:
            assert isinstance(dim_feedforward, int)
            
            if use_ttt:
                # NEW: Use TTT-enabled gating
                from .ttt_module import TTTGating
                self.gating = TTTGating(
                    gating, d_model, dim_feedforward, ttt_config, **factory_kwargs
                )
                self.use_ttt = True
            else:
                # Standard gating
                self.gating = make_gating(
                    gating, d_model, dim_feedforward, **factory_kwargs
                )
                self.use_ttt = False
```

##### B. Modify `StreamingTransformerLayer._ff_block`
Pass token embeddings to TTT layers:

```python
def _ff_block(self, x: torch.Tensor, token_embeddings: torch.Tensor | None = None) -> torch.Tensor:
    state = self._streaming_state
    offset = 0
    if state is not None:
        offset = state.offset_cpu
    x_orig = x
    x = self.norm2(x)
    if self.gating is None:
        # ... existing code unchanged ...
    else:
        if self.weights_per_step:
            # ... existing code unchanged ...
        else:
            # NEW: Pass token embeddings if TTT is enabled
            if hasattr(self, 'use_ttt') and self.use_ttt and token_embeddings is not None:
                update = self.gating(x, token_embeddings)
            else:
                update = self.gating(x)
    return x_orig.to(update) + self.layer_scale_2(update)
```

##### C. Modify `StreamingTransformerLayer.forward`
Accept and pass token embeddings:

```python
def forward(
    self,
    x: torch.Tensor,
    cross_attention_src: torch.Tensor | None = None,
    token_embeddings: torch.Tensor | None = None,  # NEW PARAMETER
):
    with ExitStack() as stack:
        if x.device.type != 'cuda':
            stack.enter_context(no_compile())
        x = self._sa_block(x)
        if self.cross_attention is not None:
            # ... existing code unchanged ...
        else:
            assert cross_attention_src is None
        x = self._ff_block(x, token_embeddings)  # MODIFIED: pass token_embeddings
        state = self._streaming_state
        if state:
            state.offset_cpu += x.shape[1]
        return x
```

##### D. Modify `StreamingTransformer.__init__`
Accept and propagate TTT config:

```python
def __init__(
    self,
    d_model: int,
    num_heads: int,
    num_layers: int,
    # ... existing params ...
    ttt_config: dict | None = None,  # NEW PARAMETER
    **kwargs,
):
    # ... existing code ...
    
    self.layers = nn.ModuleList()
    for layer_idx in range(num_layers):
        self.layers.append(
            layer_class(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                causal=causal,
                context=context,
                rope=self.rope,
                device=device,
                dtype=dtype,
                ttt_config=ttt_config,      # NEW
                layer_idx=layer_idx,         # NEW
                **kwargs,
            )
        )
        # ... existing quantization code ...
```

##### E. Modify `StreamingTransformer.forward`
Accept and pass token embeddings through layers:

```python
def forward(
    self,
    x: torch.Tensor,
    token_embeddings: torch.Tensor | None = None,  # NEW PARAMETER
    *args,
    **kwargs
):
    B, T, C = x.shape
    
    # ... existing positional embedding code ...
    
    for layer in self.layers:
        if self.checkpointing:
            y = torch_checkpoint(
                layer, x, token_embeddings, *args,  # MODIFIED: pass token_embeddings
                use_reentrant=False,
                determinism_check='none',
                preserve_rng_state=False,
                **kwargs
            )
            assert isinstance(y, torch.Tensor)
            x = y
        else:
            x = layer(x, token_embeddings=token_embeddings, *args, **kwargs)  # MODIFIED
    
    # ... rest unchanged ...
```

---

#### 3. `moshi/moshi/models/lm.py` (MODIFY)

##### A. Modify `LMModel.__init__`
Add TTT configuration:

```python
def __init__(
    self,
    delays: tp.List[int] = [0],
    # ... existing params ...
    ttt_enabled: bool = False,           # NEW
    ttt_layer_frequency: int = 6,        # NEW
    ttt_chunk_size: int = 256,           # NEW
    ttt_learning_rate: float = 1e-3,     # NEW
    ttt_conv_kernel_size: int = 2,       # NEW
    **kwargs,
):
    super().__init__()
    # ... existing code ...
    
    # NEW: TTT configuration
    ttt_config = None
    if ttt_enabled:
        ttt_config = {
            'enabled': True,
            'layer_frequency': ttt_layer_frequency,
            'chunk_size': ttt_chunk_size,
            'learning_rate': ttt_learning_rate,
            'conv_kernel_size': ttt_conv_kernel_size,
            'start_layer': 5,  # Start from layer 5 (0-indexed)
        }
    
    self.transformer = StreamingTransformer(
        d_model=dim,
        num_heads=num_heads,
        dim_feedforward=int(hidden_scale * dim),
        norm=norm,
        device=device,
        dtype=dtype,
        quantize=quantize,
        context=context,
        causal=causal,
        checkpointing=gradient_checkpointing,
        ttt_config=ttt_config,  # NEW
        **main_kwargs,
    )
    # ... rest unchanged ...
```

##### B. Modify `LMModel.forward_text`
Extract and pass token embeddings:

```python
def forward_text(
    self,
    sequence: torch.Tensor,
    sum_condition: torch.Tensor | None = None,
    cross_attention_src: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, K, S = sequence.shape
    assert K == self.num_codebooks
    
    input_sequence = sequence
    input_ = None
    
    # Compute embeddings
    for cb_index in range(self.num_audio_codebooks):
        audio_emb = self.emb[cb_index](
            input_sequence[:, cb_index + self.audio_offset]
        )
        input_ = audio_emb if input_ is None else input_ + audio_emb
    
    text_emb = self.text_emb(input_sequence[:, 0])
    input_ = text_emb if input_ is None else input_ + text_emb
    
    # NEW: Save token embeddings for TTT
    token_embeddings = input_.clone()
    
    if sum_condition is not None:
        input_ = input_ + sum_condition.to(input_)
    if cross_attention_src is not None:
        cross_attention_src = cross_attention_src.to(input_)
    
    # NEW: Pass token embeddings to transformer
    transformer_out = self.transformer(
        input_,
        token_embeddings=token_embeddings,  # NEW
        cross_attention_src=cross_attention_src
    )
    
    # ... rest unchanged ...
```

---

## Gradient Flow Strategy

Based on `ttt-lm-pytorch`, the dual form naturally allows gradients to flow:

1. **Forward pass**: TTT updates are computed using current parameters
2. **Backward pass**: Gradients flow through the entire computational graph
3. **No detach needed**: The updates are part of the differentiable computation

The key insight: `W_down_eff = W_down_init + eta * cumsum(deltas)` is fully differentiable, allowing:
- Gradients to W_down (the base fast weight)
- Gradients to W_target and Conv1D (in target generator)
- Gradients through Z (the gated activations)

---

## Configuration Example

```python
# In model config or training script:
model = LMModel(
    # ... existing params ...
    ttt_enabled=True,
    ttt_layer_frequency=6,      # Every 6th layer: 5, 11, 17, 23, 29
    ttt_chunk_size=256,         # Process 256 tokens per chunk
    ttt_learning_rate=1e-3,     # Fast weight learning rate
    ttt_conv_kernel_size=2,     # Look at next token
)
```

---

## Testing Plan

### Phase 1: Unit Tests
1. Test `CausalConv1D` with known inputs
2. Test `LMAlignedTargetGenerator` output shapes
3. Test `TTTGating` without TTT (standard mode)
4. Test `TTTGating` with TTT on small sequences

### Phase 2: Integration Tests
1. Test single TTT layer in transformer
2. Test multiple TTT layers (5, 11, 17, 23, 29)
3. Test with token embedding flow
4. Test gradient flow (backward pass)

### Phase 3: Training Tests
1. Small dataset overfitting test
2. Compare with/without TTT on same dataset
3. Monitor TTT learning (W_target convergence)

---

## File Summary

### New Files (1)
- `moshi/moshi/modules/ttt_module.py` (~300 lines)
  - `CausalConv1D`
  - `LMAlignedTargetGenerator`
  - `TTTGating`

### Modified Files (2)
- `moshi/moshi/modules/transformer.py` (~30 lines changed)
  - `StreamingTransformerLayer.__init__` (+15 lines)
  - `StreamingTransformerLayer._ff_block` (+5 lines)
  - `StreamingTransformerLayer.forward` (+1 line)
  - `StreamingTransformer.__init__` (+2 lines)
  - `StreamingTransformer.forward` (+3 lines)

- `moshi/moshi/models/lm.py` (~20 lines changed)
  - `LMModel.__init__` (+12 lines)
  - `LMModel.forward_text` (+5 lines)

**Total**: ~350 lines of new code, ~50 lines modified

---

## Key Implementation Notes

1. **Minimal Changes**: Reuses existing `ActivationGating` architecture, only adding TTT logic
2. **Backward Compatible**: TTT disabled by default, no breaking changes
3. **Configurable**: All TTT parameters exposed in model config
4. **Efficient**: Parallel context parallelism from the start
5. **Paper-Aligned**: Follows Algorithm 1 exactly, including causal cumsum

---

## Next Steps

1. ✅ Review and approve this plan
2. Create `ttt_module.py` with core components
3. Modify `transformer.py` with minimal changes
4. Modify `lm.py` to wire everything together
5. Write unit tests
6. Run integration tests
7. Begin training experiments
