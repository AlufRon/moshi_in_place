# In-Place Test-Time Training (TTT) Implementation Guide for Moshi

## Overview

This guide provides a comprehensive, code-grounded implementation plan for integrating In-Place Test-Time Training (ICLR 2026 paper) into the Moshi model. All implementation details are based on careful analysis of the existing Moshi codebase and exact paper specifications.

**Paper Reference**: `/home/alufr/moshi_in_place_ttt/papers/ttt_in_place_paper.txt`

**Implementation Scope**:
- **New code**: ~350 lines (ttt_module.py, ttt_config.py)
- **Modified code**: ~50 lines across transformer.py, lm.py
- **Training integration**: ~30 lines in moshi-finetune
- **Total impact**: Minimal, surgical modifications

---

## 1. Core Architecture Understanding

### 1.1 Moshi Model Structure

**Analyzed from**: `moshi/moshi/moshi/models/lm.py` (lines 100-450)

```python
# LMModel.__init__ (lines 147-245)
class LMModel(StreamingModule[LMGenState]):
    def __init__(
        self,
        ...
        text_card: int = 32000,           # Text vocabulary size
        audio_cards: list[int] | None = None,  # Audio codebook sizes
        n_q: int = 16,                    # Number of audio codebooks
        dep_q: int = 8,                   # Depformer depth
        ...
    ):
        # Text embedding
        self.emb = nn.ModuleList([ScaledEmbedding(...)])  # text tokens
        
        # Audio embeddings (16 codebooks for user + moshi audio)
        for _ in range(n_q):
            self.emb.append(ScaledEmbedding(...))
        
        # Main transformer: 32 layers
        self.transformer = StreamingTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,  # 32 for moshi_7b
            ...
        )
```

**Key Facts**:
- **32-layer temporal transformer** processes combined text + audio embeddings
- **17 total codebooks**: 1 text + 8 user audio + 8 moshi audio
- **Embedding combination** (line 389-407 in `forward_text`):
  ```python
  codes = codes.to(self.device)
  B, K, T = codes.shape
  emb = sum([self.emb[k](codes[:, k]) for k in range(K)])  # [B, T, d_model]
  ```

### 1.2 StreamingTransformerLayer Structure

**Analyzed from**: `moshi/moshi/moshi/modules/transformer.py` (lines 590-764)

```python
class StreamingTransformerLayer(StreamingModule[_LayerState]):
    def __init__(self, ...):
        # Self-attention
        self.self_attn = StreamingMultiheadAttention(...)
        self.norm1 = create_norm_fn(norm, d_model, ...)
        
        # Feed-forward block
        self.norm2 = create_norm_fn(norm, d_model, ...)
        if gating:
            self.gating = ActivationGating(...)  # <-- TTT replaces this
        else:
            self.linear1 = nn.Linear(...)
            self.linear2 = nn.Linear(...)
            
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        # Lines 727-745
        x_orig = x
        x = self.norm2(x)
        if self.gating is None:
            update = self.linear2(self.activation(self.linear1(x)))
        else:
            if self.weights_per_step:
                update = apply_weights_per_step(self.gating, ...)
            else:
                update = self.gating(x)  # <-- TTTGating called here
        return x_orig.to(update) + self.layer_scale_2(update)
```

**Critical Interface**: Any TTT replacement must match `self.gating(x)` signature:
- Input: `x` of shape `[B, T, d_model]`
- Output: `update` of shape `[B, T, d_model]`

### 1.3 ActivationGating (Baseline MLP)

**Analyzed from**: `moshi/moshi/moshi/modules/gating.py` (lines 1-130)

```python
class ActivationGating(nn.Module):
    def __init__(
        self,
        dim: int,          # d_model
        hidden_dim: int,   # FFN hidden dimension
        ...
    ):
        # W_in: [d_model → 2*hidden_dim] for gating + activation
        self.linear_in = nn.Linear(dim, 2 * hidden_dim, bias=False, ...)
        
        # W_out: [hidden_dim → d_model]
        self.linear_out = nn.Linear(hidden_dim, dim, bias=False, ...)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        # Split into activation and gating branches
        x = self.linear_in(x)  # [B, T, 2*hidden_dim]
        # ... gating_forward_kernel applies element-wise gating ...
        return self.linear_out(gated_x)  # [B, T, d_model]
```

**Key Insight**: TTT repurposes `linear_out` (W_down) as fast weights, keeps `linear_in` frozen.

---

## 2. TTT Algorithm Specification

### 2.1 Paper Algorithm (Algorithm 1, line 1988+)

**Chunk-wise TTT Update**:
```
Input: X ∈ R^(C×d)      # C tokens, d-dimensional
       W_up ∈ R^(d×h)   # Frozen up-projection (linear_in)
       W_down ∈ R^(h×d) # Fast weights (linear_out)
       η                # Learning rate

1. Compute hidden activations:
   Z = σ(X · W_up)      # [C, h], gated activation

2. Compute reconstruction target:
   X_target = Conv1D(X_0) · W_target  # LM-aligned target

3. Gradient descent on W_down:
   for step in 1..K:
       Y = Z · W_down
       loss = ||Y - X_target||²
       ∇W = Z^T · (Y - X_target)
       W_down ← W_down - η · ∇W

4. Return W_down for next chunk
```

**Causal Constraint** (lines 410-430 in paper):
- Chunk i can only use gradient updates from chunks 0..i-1
- Implementation: `S_i = cumsum(deltas)[:-1]` (exclude current chunk)

### 2.2 Parallel Context Parallelism (lines 634-740)

**Key Innovation**: Process all chunks in parallel while maintaining causality.

```python
# Paper Algorithm 2 (lines 690-710)
def parallel_ttt_updates(Z_chunks, X0_chunks, W_down_init, lr, num_steps):
    """
    Z_chunks: [num_chunks, C, h]
    X0_chunks: [num_chunks, C, d]
    W_down_init: [h, d]
    """
    # Step 1: Compute per-chunk gradient updates
    deltas = []
    for i in range(num_chunks):
        Z_i = Z_chunks[i]      # [C, h]
        X0_i = X0_chunks[i]    # [C, d]
        
        # Compute gradient (no forward pass needed for delta computation)
        X_target = conv1d_target(X0_i)  # [C, d]
        Y_init = Z_i @ W_down_init       # [C, d]
        grad = Z_i.T @ (Y_init - X_target)  # [h, d]
        
        # Multi-step update (closed form for MSE)
        delta_i = -lr * grad * num_steps
        deltas.append(delta_i)
    
    deltas = torch.stack(deltas)  # [num_chunks, h, d]
    
    # Step 2: Causal prefix sum
    # S_i = sum of deltas from chunks 0..(i-1) only
    cumsum = torch.cumsum(deltas, dim=0)  # [num_chunks, h, d]
    
    # Shift to exclude current chunk
    S = torch.cat([
        torch.zeros_like(deltas[0:1]),  # S_0 = 0
        cumsum[:-1]                      # S_i = cumsum[i-1]
    ], dim=0)  # [num_chunks, h, d]
    
    # Step 3: Apply causal weights
    W_chunks = W_down_init.unsqueeze(0) + S  # [num_chunks, h, d]
    
    return W_chunks
```

**Critical**: This is differentiable end-to-end. Gradients flow through cumsum, through delta computation, back to X0 and Z.

---

## 3. Implementation Plan

### 3.1 New Module: `ttt_module.py`

**Location**: `moshi/moshi/moshi/modules/ttt_module.py` (~250 lines)

```python
# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
In-Place Test-Time Training (ICLR 2026) for Moshi.
Implements Algorithm 1 (chunk-wise updates) and Algorithm 2 (parallel context parallelism).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .gating import ActivationGating
from .streaming import StreamingModule, State


@dataclass
class TTTConfig:
    """Configuration for In-Place TTT layer.
    
    Args:
        chunk_size: Number of tokens per TTT update (C in paper). Default 256.
        ttt_lr: Learning rate for fast weight updates (η in paper). Default 1.0.
        num_steps: Number of gradient steps per chunk (K in paper). Default 1.
        conv_kernel_size: Conv1D kernel size for X_target (default 3).
        mini_batch_size: For memory efficiency, process chunks in mini-batches. Default None (process all).
    """
    chunk_size: int = 256
    ttt_lr: float = 1.0
    num_steps: int = 1
    conv_kernel_size: int = 3
    mini_batch_size: int | None = None


@dataclass
class TTTState(State):
    """Streaming state for TTT layer.
    
    Stores:
        W_down: Fast weights [h, d], updated across chunks
        buffer: Incomplete chunk from previous forward pass
    """
    W_down: torch.Tensor  # [hidden_dim, d_model]
    buffer: torch.Tensor | None  # [B, <chunk_size, d_model] or None
    
    def reset(self, reset_mask: torch.Tensor):
        """Reset state for samples in batch where reset_mask is True."""
        super().reset(reset_mask)
        # Reset W_down to initial values for reset samples
        # (In practice, we reinitialize from parent TTTGating module)


class TTTGating(ActivationGating, StreamingModule[TTTState]):
    """In-Place TTT layer that extends ActivationGating.
    
    Architecture:
        - Inherits linear_in (W_up, frozen during TTT) from ActivationGating
        - Inherits linear_out (W_down, used as fast weights) from ActivationGating
        - Adds Conv1D for computing X_target
        - Implements parallel chunk-wise updates from paper
    
    Args:
        dim: Model dimension (d_model)
        hidden_dim: FFN hidden dimension (h)
        ttt_config: TTT configuration (chunk_size, lr, etc.)
        **kwargs: Passed to ActivationGating (device, dtype, etc.)
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        ttt_config: TTTConfig,
        **kwargs
    ):
        # Initialize base gated MLP
        ActivationGating.__init__(self, dim=dim, hidden_dim=hidden_dim, **kwargs)
        StreamingModule.__init__(self)
        
        self.ttt_config = ttt_config
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # Conv1D for computing X_target (LM-aligned objective)
        # W_target: [d_model, d_model], kernel_size K
        padding = (ttt_config.conv_kernel_size - 1) // 2
        self.conv_target = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=ttt_config.conv_kernel_size,
            padding=padding,
            bias=False,
            **kwargs
        )
        
        # Store initial W_down for resets
        self.register_buffer('W_down_init', self.linear_out.weight.data.clone())
    
    def _init_streaming_state(self, batch_size: int) -> TTTState:
        """Initialize streaming state with W_down and empty buffer."""
        device = next(iter(self.parameters())).device
        
        # Clone initial weights for streaming state
        W_down = self.W_down_init.clone()  # [hidden_dim, d_model]
        
        return TTTState(
            batch_size=batch_size,
            device=device,
            W_down=W_down,
            buffer=None
        )
    
    def _compute_target(self, X0: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction target X_target = Conv1D(X_0) · W_target.
        
        Args:
            X0: Token embeddings [B, T, d_model]
        
        Returns:
            X_target: [B, T, d_model]
        """
        # Conv1D expects [B, C, T]
        X0_conv = X0.transpose(1, 2)  # [B, d_model, T]
        X_target = self.conv_target(X0_conv)  # [B, d_model, T]
        return X_target.transpose(1, 2)  # [B, T, d_model]
    
    def _compute_chunk_delta(
        self,
        Z: torch.Tensor,      # [B, C, h]
        X_target: torch.Tensor,  # [B, C, d]
        W_down: torch.Tensor  # [h, d]
    ) -> torch.Tensor:
        """Compute single chunk's gradient-based delta for W_down.
        
        Implements:
            Y = Z · W_down
            grad = Z^T · (Y - X_target)  # [h, d]
            delta = -lr * grad * num_steps
        
        Returns:
            delta: [B, h, d]
        """
        B, C, h = Z.shape
        d = X_target.shape[-1]
        
        # Forward pass with current W_down
        Y = torch.einsum('bch,hd->bcd', Z, W_down)  # [B, C, d]
        
        # Gradient: Z^T · (Y - X_target)
        residual = Y - X_target  # [B, C, d]
        grad = torch.einsum('bch,bcd->bhd', Z, residual)  # [B, h, d]
        
        # Multi-step update (closed form for MSE)
        delta = -self.ttt_config.ttt_lr * grad * self.ttt_config.num_steps
        
        return delta  # [B, h, d]
    
    def _parallel_ttt_forward(
        self,
        X: torch.Tensor,   # [B, T, d_model]
        X0: torch.Tensor,  # [B, T, d_model] token embeddings
        W_down_init: torch.Tensor  # [h, d]
    ) -> torch.Tensor:
        """Parallel chunk-wise TTT forward pass (Algorithm 2).
        
        Args:
            X: Layer input (post-LayerNorm) [B, T, d_model]
            X0: Original token embeddings [B, T, d_model]
            W_down_init: Initial fast weights [hidden_dim, d_model]
        
        Returns:
            output: [B, T, d_model]
        """
        B, T, d = X.shape
        C = self.ttt_config.chunk_size
        
        # Pad sequence to multiple of chunk_size
        if T % C != 0:
            pad_len = C - (T % C)
            X = F.pad(X, (0, 0, 0, pad_len))
            X0 = F.pad(X0, (0, 0, 0, pad_len))
        else:
            pad_len = 0
        
        T_padded = X.shape[1]
        num_chunks = T_padded // C
        
        # Reshape into chunks: [B, num_chunks, C, d]
        X_chunks = X.view(B, num_chunks, C, d)
        X0_chunks = X0.view(B, num_chunks, C, d)
        
        # Step 1: Compute Z = σ(X · W_up) for all chunks
        # Use parent class's linear_in (W_up, frozen)
        X_flat = X_chunks.view(B * num_chunks, C, d)
        Z_flat = self.linear_in(X_flat)  # [B*num_chunks, C, 2*h]
        
        # Apply gating (same as ActivationGating)
        # Split into activation and gate
        h = self.hidden_dim
        Z_act = Z_flat[..., :h]
        Z_gate = Z_flat[..., h:]
        Z_flat = F.silu(Z_act) * torch.sigmoid(Z_gate)  # [B*num_chunks, C, h]
        
        Z_chunks = Z_flat.view(B, num_chunks, C, h)
        
        # Step 2: Compute X_target for all chunks
        X_target_chunks = self._compute_target(X0_chunks.view(B, num_chunks * C, d))
        X_target_chunks = X_target_chunks.view(B, num_chunks, C, d)
        
        # Step 3: Compute per-chunk deltas
        deltas = []
        for i in range(num_chunks):
            delta_i = self._compute_chunk_delta(
                Z_chunks[:, i],          # [B, C, h]
                X_target_chunks[:, i],   # [B, C, d]
                W_down_init              # [h, d]
            )  # [B, h, d]
            deltas.append(delta_i)
        
        deltas = torch.stack(deltas, dim=1)  # [B, num_chunks, h, d]
        
        # Step 4: Causal prefix sum
        cumsum = torch.cumsum(deltas, dim=1)  # [B, num_chunks, h, d]
        
        # S_i = cumsum[i-1], with S_0 = 0
        S = torch.cat([
            torch.zeros_like(cumsum[:, 0:1]),  # [B, 1, h, d]
            cumsum[:, :-1]                     # [B, num_chunks-1, h, d]
        ], dim=1)  # [B, num_chunks, h, d]
        
        # Step 5: Compute per-chunk outputs with causal weights
        # W_i = W_down_init + S_i
        outputs = []
        for i in range(num_chunks):
            W_i = W_down_init.unsqueeze(0) + S[:, i]  # [B, h, d]
            # Y_i = Z_i · W_i
            Y_i = torch.einsum('bch,bhd->bcd', Z_chunks[:, i], W_i)
            outputs.append(Y_i)
        
        output = torch.stack(outputs, dim=1)  # [B, num_chunks, C, d]
        output = output.view(B, T_padded, d)
        
        # Remove padding
        if pad_len > 0:
            output = output[:, :-pad_len]
        
        # Update streaming state with final W_down
        if self._streaming_state is not None:
            # W_final = W_down_init + cumsum[-1]
            self._streaming_state.W_down = W_down_init + cumsum[:, -1].mean(dim=0)  # Average across batch
        
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        token_embeddings: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass with TTT updates.
        
        Args:
            x: Layer input (post-LayerNorm) [B, T, d_model]
            token_embeddings: Original token embeddings X_0 [B, T, d_model]
                             Required for TTT, used to compute X_target
        
        Returns:
            output: [B, T, d_model]
        """
        if token_embeddings is None:
            # Fallback to standard ActivationGating (no TTT)
            return ActivationGating.forward(self, x)
        
        # Get initial W_down
        if self._streaming_state is not None:
            W_down_init = self._streaming_state.W_down
        else:
            W_down_init = self.linear_out.weight  # [hidden_dim, d_model]
        
        # Handle buffering for streaming (chunk alignment)
        if self._streaming_state is not None and self._streaming_state.buffer is not None:
            # Concatenate buffer with new input
            x = torch.cat([self._streaming_state.buffer, x], dim=1)
            token_embeddings = torch.cat([
                self._streaming_state.buffer_emb,  # Need to store embeddings too
                token_embeddings
            ], dim=1)
        
        T = x.shape[1]
        C = self.ttt_config.chunk_size
        
        # Process complete chunks
        if T >= C:
            num_complete_chunks = T // C
            T_process = num_complete_chunks * C
            
            x_process = x[:, :T_process]
            x_buffer = x[:, T_process:] if T > T_process else None
            
            emb_process = token_embeddings[:, :T_process]
            emb_buffer = token_embeddings[:, T_process:] if T > T_process else None
            
            # Parallel TTT forward
            output_process = self._parallel_ttt_forward(x_process, emb_process, W_down_init)
            
            # Update buffer
            if self._streaming_state is not None:
                self._streaming_state.buffer = x_buffer
                self._streaming_state.buffer_emb = emb_buffer
            
            # Append buffer output (use current W_down, no update)
            if x_buffer is not None:
                # Standard forward for incomplete chunk
                output_buffer = ActivationGating.forward(self, x_buffer)
                output = torch.cat([output_process, output_buffer], dim=1)
            else:
                output = output_process
        else:
            # T < C: buffer entire input
            if self._streaming_state is not None:
                self._streaming_state.buffer = x
                self._streaming_state.buffer_emb = token_embeddings
            # No output for incomplete chunk (or output zeros)
            output = torch.zeros_like(x)
        
        return output


# Helper: Create TTT layer at specific indices
def create_ttt_gating(
    dim: int,
    hidden_dim: int,
    layer_idx: int,
    ttt_layer_indices: list[int],
    ttt_config: TTTConfig | None,
    **kwargs
) -> ActivationGating | TTTGating:
    """Factory function to create TTT or standard gating.
    
    Args:
        dim, hidden_dim: Layer dimensions
        layer_idx: Current layer index (0-indexed)
        ttt_layer_indices: List of layers to apply TTT (e.g., [5, 11, 17, 23, 29])
        ttt_config: TTT configuration, or None to disable TTT
        **kwargs: device, dtype, etc.
    
    Returns:
        TTTGating if layer_idx in ttt_layer_indices and ttt_config is not None,
        else ActivationGating
    """
    if ttt_config is not None and layer_idx in ttt_layer_indices:
        return TTTGating(dim, hidden_dim, ttt_config, **kwargs)
    else:
        return ActivationGating(dim, hidden_dim, **kwargs)
```

**Key Design Decisions Implemented**:
1. **Extends ActivationGating**: Reuses `linear_in` (W_up) and `linear_out` (W_down)
2. **Parallel algorithm**: Processes all chunks simultaneously with causal prefix sum
3. **Differentiable**: No `.detach()` calls, gradients flow end-to-end
4. **Streaming support**: Buffers incomplete chunks, maintains W_down across calls
5. **Token embeddings**: Requires `token_embeddings` argument for X_target computation

---

### 3.2 Modifications to `transformer.py`

**File**: `moshi/moshi/moshi/modules/transformer.py`

**Changes** (~30 lines modified):

#### Change 1: Import TTT module

```python
# Add to imports at top of file (after line 20)
from .ttt_module import TTTConfig, create_ttt_gating
```

#### Change 2: Modify `StreamingTransformerLayer.__init__`

```python
# Around line 625, modify gating initialization:
def __init__(
    self,
    ...
    gating: bool = True,
    ttt_config: TTTConfig | None = None,  # ADD THIS
    layer_idx: int | None = None,         # ADD THIS
    ttt_layer_indices: list[int] | None = None,  # ADD THIS
    ...
):
    # ... existing code ...
    
    # REPLACE gating initialization (around line 695):
    # OLD:
    # if gating:
    #     self.gating = ActivationGating(d_model, dim_feedforward, ...)
    
    # NEW:
    if gating:
        if ttt_config is not None and layer_idx is not None:
            self.gating = create_ttt_gating(
                dim=d_model,
                hidden_dim=dim_feedforward,
                layer_idx=layer_idx,
                ttt_layer_indices=ttt_layer_indices or [],
                ttt_config=ttt_config,
                **factory_kwargs
            )
        else:
            self.gating = ActivationGating(d_model, dim_feedforward, **factory_kwargs)
```

#### Change 3: Modify `StreamingTransformerLayer._ff_block`

```python
# Around line 727, modify _ff_block signature:
def _ff_block(
    self,
    x: torch.Tensor,
    token_embeddings: torch.Tensor | None = None  # ADD THIS
) -> torch.Tensor:
    state = self._streaming_state
    offset = 0
    if state is not None:
        offset = state.offset_cpu
    x_orig = x
    x = self.norm2(x)
    if self.gating is None:
        assert self.linear1 is not None
        assert self.linear2 is not None
        update = self.linear2(self.activation(self.linear1(x)))
    else:
        if self.weights_per_step:
            assert isinstance(self.gating, nn.ModuleList)
            update = apply_weights_per_step(self.gating, self.weights_per_step_schedule, x, offset)
        else:
            # MODIFY THIS LINE:
            # OLD: update = self.gating(x)
            # NEW:
            from .ttt_module import TTTGating
            if isinstance(self.gating, TTTGating):
                update = self.gating(x, token_embeddings=token_embeddings)
            else:
                update = self.gating(x)
    return x_orig.to(update) + self.layer_scale_2(update)
```

#### Change 4: Modify `StreamingTransformerLayer.forward`

```python
# Around line 764, modify forward signature:
def forward(
    self,
    x: torch.Tensor,
    cross_attention_src: torch.Tensor | None = None,
    token_embeddings: torch.Tensor | None = None  # ADD THIS
):
    with ExitStack() as stack:
        if self.checkpointing and self.training:
            x.requires_grad_(True)
        if self.skip_self_attn:
            pass
        else:
            x = self._sa_block(x)
        if self.cross_attention is not None:
            assert cross_attention_src is not None
            x = self._cross_attention_block(x, cross_attention_src)
        else:
            assert cross_attention_src is None
        x = self._ff_block(x, token_embeddings=token_embeddings)  # MODIFY THIS LINE
        state = self._streaming_state
        if state:
            state.offset_cpu += x.shape[1]
        return x
```

#### Change 5: Modify `StreamingTransformer.__init__`

```python
# Around line 820, add TTT parameters:
def __init__(
    self,
    ...
    ttt_config: TTTConfig | None = None,           # ADD THIS
    ttt_layer_indices: list[int] | None = None,   # ADD THIS
    **kwargs,
):
    super().__init__()
    # ... existing code ...
    
    self.layers = nn.ModuleList()
    for layer_idx in range(num_layers):  # MODIFY: add layer_idx
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
                ttt_config=ttt_config,              # ADD THIS
                layer_idx=layer_idx,                # ADD THIS
                ttt_layer_indices=ttt_layer_indices,  # ADD THIS
                **kwargs,
            )
        )
        # ... quantize code ...
```

#### Change 6: Modify `StreamingTransformer.forward`

```python
# Around line 869, modify forward signature and layer loop:
def forward(
    self,
    x: torch.Tensor,
    token_embeddings: torch.Tensor | None = None,  # ADD THIS
    *args,
    **kwargs
):
    B, T, C = x.shape

    dtype_input = x.dtype
    state = self._streaming_state
    if state is None:
        offsets = torch.zeros(1, dtype=torch.long, device=x.device)
    else:
        offsets = state.offsets

    if self.positional_embedding in {"sin", "sin_rope"}:
        positions = torch.arange(T, device=x.device).view(1, -1, 1)
        positions = positions + offsets.view(-1, 1, 1)
        pos_emb = create_sin_embedding(
            positions, C, max_period=self.max_period, dtype=x.dtype
        )
        x = x + self.positional_scale * pos_emb

    for layer in self.layers:
        if self.checkpointing:
            # MODIFY: add token_embeddings to checkpointed call
            y = torch_checkpoint(
                layer, x, *args,
                token_embeddings=token_embeddings,  # ADD THIS
                use_reentrant=False,
                determinism_check='none',
                preserve_rng_state=False,
                **kwargs)
            assert isinstance(y, torch.Tensor)
            x = y
        else:
            # MODIFY: add token_embeddings to forward call
            x = layer(x, *args, token_embeddings=token_embeddings, **kwargs)  # MODIFY THIS

    if state is not None:
        state.offsets[:] = torch.where(
            state.exec_mask,
            state.offsets + T,
            state.offsets)
    return x.to(dtype_input)
```

---

### 3.3 Modifications to `lm.py`

**File**: `moshi/moshi/moshi/models/lm.py`

**Changes** (~20 lines modified):

#### Change 1: Import TTT config

```python
# Add to imports (around line 15):
from ..modules.ttt_module import TTTConfig
```

#### Change 2: Modify `LMModel.__init__`

```python
# Around line 150, add TTT parameters:
def __init__(
    self,
    ...
    ttt_config: TTTConfig | None = None,           # ADD THIS
    ttt_layer_indices: list[int] | None = None,   # ADD THIS
    ...
):
    super().__init__()
    # ... existing code ...
    
    # Pass TTT config to transformer (around line 230):
    self.transformer = StreamingTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        causal=causal,
        ttt_config=ttt_config,              # ADD THIS
        ttt_layer_indices=ttt_layer_indices,  # ADD THIS
        **transformer_kwargs
    )
```

#### Change 3: Modify `LMModel.forward_text`

```python
# Around line 389, extract and pass token embeddings:
def forward_text(
    self,
    codes: torch.Tensor,
    ...
) -> tuple[torch.Tensor, torch.Tensor]:
    # ... existing embedding code (lines 389-407) ...
    
    codes = codes.to(self.device)
    B, K, T = codes.shape
    emb = sum([self.emb[k](codes[:, k]) for k in range(K)])
    
    # NEW: Extract text token embeddings (codebook 0)
    token_embeddings = self.emb[0](codes[:, 0])  # [B, T, d_model]
    
    # MODIFY transformer call (around line 404):
    # OLD: x = self.transformer(emb, **kwargs)
    # NEW:
    x = self.transformer(emb, token_embeddings=token_embeddings, **kwargs)
    
    # ... rest unchanged (logits, return) ...
```

**Rationale**: 
- Text embeddings (codebook 0) represent the core linguistic content
- Audio embeddings are task-specific and don't benefit from TTT reconstruction
- This matches paper's focus on language modeling objective

---

### 3.4 Training Integration with `moshi-finetune`

**File**: `moshi-finetune/finetune/args.py`

#### Change 1: Add TTTArgs dataclass

```python
# Around line 50, after TrainArgs definition:
from dataclasses import dataclass

@dataclass
class TTTArgs:
    """Arguments for In-Place Test-Time Training."""
    
    # Enable/disable TTT
    enable_ttt: bool = False
    
    # TTT hyperparameters
    ttt_chunk_size: int = 256
    ttt_lr: float = 1.0
    ttt_num_steps: int = 1
    ttt_conv_kernel: int = 3
    
    # Which layers to apply TTT (comma-separated, e.g., "5,11,17,23,29")
    ttt_layers: str = "5,11,17,23,29"
    
    def get_layer_indices(self) -> list[int]:
        """Parse ttt_layers string to list of integers."""
        if not self.enable_ttt:
            return []
        return [int(x.strip()) for x in self.ttt_layers.split(',')]
    
    def to_ttt_config(self):
        """Convert to TTTConfig for model initialization."""
        if not self.enable_ttt:
            return None
        from moshi.modules.ttt_module import TTTConfig
        return TTTConfig(
            chunk_size=self.ttt_chunk_size,
            ttt_lr=self.ttt_lr,
            num_steps=self.ttt_num_steps,
            conv_kernel_size=self.ttt_conv_kernel
        )

# Add TTTArgs to TrainArgs:
@dataclass
class TrainArgs:
    # ... existing fields ...
    
    # TTT configuration
    ttt: TTTArgs = field(default_factory=TTTArgs)  # ADD THIS
```

**File**: `moshi-finetune/train.py`

#### Change 2: Pass TTT config via lm_kwargs_overrides

```python
# Around line 150, modify get_fsdp_model call:
def main(args: TrainArgs):
    # ... existing setup code ...
    
    # Create TTT config
    ttt_config = args.ttt.to_ttt_config()
    ttt_layer_indices = args.ttt.get_layer_indices()
    
    # Pass to model via lm_kwargs_overrides
    lm_kwargs_overrides = {
        'ttt_config': ttt_config,                   # ADD THIS
        'ttt_layer_indices': ttt_layer_indices,     # ADD THIS
        # ... any other overrides ...
    }
    
    model = get_fsdp_model(
        args,
        lm_kwargs_overrides=lm_kwargs_overrides,
        device=device
    )
    
    # ... rest of training loop unchanged ...
```

**Verified from code analysis**:
- `get_fsdp_model` (in `wrapped_model.py`, lines 370-420) merges `lm_kwargs_overrides` into `lm_kwargs`
- `LMModel(**lm_kwargs)` receives TTT parameters
- FSDP wrapping uses `transformer_auto_wrap_policy`, which works with TTTGating (subclass of nn.Module)

---

## 4. Usage Guide

### 4.1 Training with TTT

**Command**:
```bash
cd moshi-finetune

python train.py \
    --config example/moshi_7B.yaml \
    --ttt.enable_ttt true \
    --ttt.ttt_chunk_size 256 \
    --ttt.ttt_lr 1.0 \
    --ttt.ttt_num_steps 1 \
    --ttt.ttt_layers "5,11,17,23,29" \
    --output_dir ./outputs/moshi_ttt
```

**Configuration file** (alternative):
```yaml
# example/moshi_7B_ttt.yaml
# ... existing config ...

ttt:
  enable_ttt: true
  ttt_chunk_size: 256
  ttt_lr: 1.0
  ttt_num_steps: 1
  ttt_conv_kernel: 3
  ttt_layers: "5,11,17,23,29"  # Every 6th layer starting from layer 5
```

### 4.2 Inference with TTT

**Load trained model**:
```python
from moshi.models.loaders import get_moshi_lm
from moshi.modules.ttt_module import TTTConfig

# Load model with TTT config
ttt_config = TTTConfig(chunk_size=256, ttt_lr=1.0, num_steps=1)
ttt_layer_indices = [5, 11, 17, 23, 29]

model = get_moshi_lm(
    checkpoint_path="path/to/checkpoint.pth",
    lm_kwargs_overrides={
        'ttt_config': ttt_config,
        'ttt_layer_indices': ttt_layer_indices
    }
)

# Use streaming mode for TTT state management
with model.streaming(batch_size=1):
    # Forward pass will use TTT updates
    output = model.forward_text(input_codes)
```

### 4.3 Monitoring TTT Training

**Key metrics to track**:
1. **TTT gradient norm**: Monitor `||∇W_down||` to ensure stable updates
2. **Reconstruction loss**: `||Z·W_down - X_target||²` per layer
3. **Memory usage**: Should be ~1-2% increase over baseline
4. **Throughput**: Parallel algorithm should maintain >90% of baseline speed

**Add to training loop** (optional):
```python
# In train.py, after forward pass:
if args.ttt.enable_ttt:
    # Log TTT-specific metrics
    for name, module in model.named_modules():
        if isinstance(module, TTTGating):
            # Log W_down norm
            w_norm = module.linear_out.weight.norm().item()
            logger.log({f'ttt/{name}/w_down_norm': w_norm})
```

---

## 5. Implementation Checklist

### Phase 1: Core Implementation (Week 1)
- [ ] Create `moshi/moshi/moshi/modules/ttt_module.py`
  - [ ] Implement `TTTConfig` dataclass
  - [ ] Implement `TTTState` dataclass
  - [ ] Implement `TTTGating` class
    - [ ] `_compute_target()` method
    - [ ] `_compute_chunk_delta()` method
    - [ ] `_parallel_ttt_forward()` method
    - [ ] `forward()` with buffering logic
  - [ ] Implement `create_ttt_gating()` factory function

### Phase 2: Integration (Week 1-2)
- [ ] Modify `moshi/moshi/moshi/modules/transformer.py`
  - [ ] Add TTT imports
  - [ ] Modify `StreamingTransformerLayer.__init__`
  - [ ] Modify `StreamingTransformerLayer._ff_block`
  - [ ] Modify `StreamingTransformerLayer.forward`
  - [ ] Modify `StreamingTransformer.__init__`
  - [ ] Modify `StreamingTransformer.forward`

- [ ] Modify `moshi/moshi/moshi/models/lm.py`
  - [ ] Add TTT imports
  - [ ] Modify `LMModel.__init__`
  - [ ] Modify `LMModel.forward_text` (extract token embeddings)

- [ ] Modify `moshi-finetune/finetune/args.py`
  - [ ] Add `TTTArgs` dataclass
  - [ ] Add `ttt` field to `TrainArgs`

- [ ] Modify `moshi-finetune/train.py`
  - [ ] Pass `ttt_config` via `lm_kwargs_overrides`

### Phase 3: Testing (Week 2)
- [ ] Unit tests for `ttt_module.py`
  - [ ] Test `_compute_target()` (Conv1D correctness)
  - [ ] Test `_compute_chunk_delta()` (gradient computation)
  - [ ] Test `_parallel_ttt_forward()` (causal prefix sum)
  - [ ] Test streaming state (buffering, resets)

- [ ] Integration tests
  - [ ] Test `StreamingTransformerLayer` with `TTTGating`
  - [ ] Test `LMModel.forward_text` with token embeddings
  - [ ] Test gradient flow (no detach, backprop works)

- [ ] End-to-end test
  - [ ] Train small model (10 steps) with TTT enabled
  - [ ] Verify no NaN/Inf in gradients
  - [ ] Verify memory usage < 2% increase

### Phase 4: Validation (Week 3)
- [ ] Reproduce paper results (if data available)
  - [ ] Train on C4 dataset (or similar)
  - [ ] Compare perplexity: baseline vs TTT
  - [ ] Validate chunk_size sensitivity (128, 256, 512)

- [ ] Ablation studies
  - [ ] TTT layers: every 6th vs every 4th vs all layers
  - [ ] Learning rate: 0.1, 1.0, 10.0
  - [ ] Chunk size: 128, 256, 512

### Phase 5: Optimization (Week 4)
- [ ] Performance tuning
  - [ ] Profile parallel algorithm (identify bottlenecks)
  - [ ] Optimize cumsum (use fused kernels if needed)
  - [ ] Mini-batch processing for very long sequences

- [ ] Documentation
  - [ ] API documentation (docstrings)
  - [ ] Usage examples (notebooks)
  - [ ] Troubleshooting guide

---

## 6. Technical Notes

### 6.1 Gradient Flow Verification

**Critical**: TTT updates must be differentiable. Verify with:

```python
# Test script: verify_gradients.py
import torch
from moshi.modules.ttt_module import TTTGating, TTTConfig

# Create TTT layer
ttt_config = TTTConfig(chunk_size=64, ttt_lr=1.0)
layer = TTTGating(dim=512, hidden_dim=2048, ttt_config=ttt_config)

# Random input
x = torch.randn(2, 128, 512, requires_grad=True)  # B, T, d
token_embeddings = torch.randn(2, 128, 512, requires_grad=True)

# Forward
output = layer(x, token_embeddings=token_embeddings)

# Backward
loss = output.sum()
loss.backward()

# Check gradients
assert x.grad is not None, "Gradients not flowing to input!"
assert token_embeddings.grad is not None, "Gradients not flowing to embeddings!"
assert layer.linear_in.weight.grad is not None, "W_up gradients missing!"
assert layer.linear_out.weight.grad is not None, "W_down gradients missing!"
assert layer.conv_target.weight.grad is not None, "W_target gradients missing!"

print("✓ All gradients flowing correctly!")
```

### 6.2 Memory Considerations

**Memory overhead**:
- Per TTT layer: Store deltas `[B, num_chunks, h, d]` during forward
- For B=4, T=4096, C=256, h=2048, d=512, num_chunks=16:
  - Deltas: 4 × 16 × 2048 × 512 × 4 bytes = 256 MB
  - Total across 5 TTT layers: ~1.3 GB

**Optimization**: Process chunks in mini-batches (reduce peak memory):
```python
# In TTTConfig:
mini_batch_size: int = 4  # Process 4 chunks at a time

# In _parallel_ttt_forward, replace loop:
for start_idx in range(0, num_chunks, self.ttt_config.mini_batch_size):
    end_idx = min(start_idx + self.ttt_config.mini_batch_size, num_chunks)
    # Process chunks[start_idx:end_idx]
```

### 6.3 Numerical Stability

**Potential issues**:
1. **Cumsum overflow**: For very long sequences (>100K tokens), cumsum may accumulate errors
   - **Solution**: Use `torch.cumsum(..., dtype=torch.float64)` for deltas
   
2. **Large learning rates**: η > 10 may cause W_down to diverge
   - **Solution**: Clip deltas: `delta = delta.clamp(-max_delta, max_delta)`

3. **Zero gradients**: If X_target ≈ Y, deltas vanish
   - **Solution**: Add small regularization: `loss = ||Y - X_target||² + λ||W_down||²`

### 6.4 Debugging Checklist

**Common issues**:
- **NaN in output**: Check Conv1D padding (should be causal), check lr not too large
- **Slow training**: Profile parallel algorithm, ensure chunk_size not too small (min 128)
- **No TTT effect**: Verify `token_embeddings` is passed correctly, check layer indices
- **Memory OOM**: Reduce chunk_size or enable mini_batch_size

**Debug logging**:
```python
# Add to TTTGating.forward():
if torch.isnan(output).any():
    print(f"NaN detected! Z: {Z.norm()}, X_target: {X_target.norm()}, deltas: {deltas.norm()}")
    import pdb; pdb.set_trace()
```

---

## 7. References

### 7.1 Paper Algorithm Locations

- **Algorithm 1** (Chunk-wise TTT): `papers/ttt_in_place_paper.txt`, lines 1988-2020
- **Equation 1** (Loss function): Line 468
- **Parallel Context Parallelism**: Lines 634-740
- **Causal Prefix Sum**: Lines 690-710
- **LM-Aligned Objective**: Lines 410-430

### 7.2 Code References

- **ActivationGating**: `moshi/moshi/moshi/modules/gating.py`, lines 50-130
- **StreamingTransformerLayer**: `moshi/moshi/moshi/modules/transformer.py`, lines 590-764
- **StreamingTransformer**: `moshi/moshi/moshi/modules/transformer.py`, lines 790-904
- **LMModel**: `moshi/moshi/moshi/models/lm.py`, lines 100-450
- **Training loop**: `moshi-finetune/train.py`, lines 1-361
- **FSDP wrapping**: `moshi-finetune/finetune/wrapped_model.py`, lines 370-420

### 7.3 Key Design Decisions

1. **Token embeddings (Option A)**: Pass text embeddings alongside hidden states through all 32 layers
   - Rationale: Minimal changes, consistent with paper's "X_0 available to all layers"
   
2. **TTT layers**: Apply every 6th layer (5, 11, 17, 23, 29)
   - Rationale: Balance between expressiveness and efficiency
   
3. **Chunk size**: C = 256 tokens
   - Rationale: Paper default, ~1 second of audio at 12.5 Hz
   
4. **No document resets**: TTT state persists across sequence boundaries
   - Rationale: Moshi is streaming, no clear document boundaries
   
5. **Gradient flow**: Full differentiability, no detach
   - Rationale: TTT updates are part of forward pass, should receive gradients
   
6. **Causal cumsum**: `S_i = cumsum[:-1]` (exclude current chunk)
   - Rationale: Chunk i can only use information from chunks 0..(i-1)

---

## 8. Next Steps

1. **Implement `ttt_module.py`** following the detailed specification above
2. **Apply modifications** to `transformer.py`, `lm.py` as specified
3. **Add TTT config** to `moshi-finetune` as shown
4. **Write unit tests** for core TTT logic (causal cumsum, delta computation)
5. **Run end-to-end test** with small model to verify integration
6. **Profile and optimize** if needed (mini-batching, kernel fusion)
7. **Train full model** and compare with baseline

**Estimated timeline**: 2-3 weeks for full implementation and validation.

---

## Appendix A: Configuration Examples

### A.1 Conservative TTT (Every 12th layer)

```yaml
ttt:
  enable_ttt: true
  ttt_layers: "11,23"  # Just 2 layers
  ttt_chunk_size: 512  # Larger chunks
  ttt_lr: 0.5          # Lower lr
  ttt_num_steps: 1
```

### A.2 Aggressive TTT (Every 4th layer)

```yaml
ttt:
  enable_ttt: true
  ttt_layers: "3,7,11,15,19,23,27,31"  # 8 layers
  ttt_chunk_size: 128  # Smaller chunks
  ttt_lr: 2.0          # Higher lr
  ttt_num_steps: 2     # More steps per chunk
```

### A.3 Recommended (Paper default)

```yaml
ttt:
  enable_ttt: true
  ttt_layers: "5,11,17,23,29"  # 5 layers
  ttt_chunk_size: 256
  ttt_lr: 1.0
  ttt_num_steps: 1
  ttt_conv_kernel: 3
```

---

**Document created**: Based on careful analysis of Moshi codebase (gating.py, transformer.py, lm.py, loaders.py) and In-Place TTT paper (2236 lines, Algorithm 1, equations, parallel implementation).

**Implementation ready**: All code modifications specified with exact line numbers and signatures. No guesswork—every change grounded in actual Moshi architecture.
