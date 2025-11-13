"""In-Place TTT helpers: causal conv target generator and TTT-enabled gating.

This module implements LMAlignedTargetGenerator, CausalConv1D and a
TTTGating class intended to be a drop-in replacement for the existing
ActivationGating. It's guarded by configuration so it remains backward
compatible with models that don't enable TTT.
"""
from __future__ import annotations

import math
import typing as tp
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalConv1D(nn.Module):
    """Causal 1D convolution that can look at future tokens via right padding.

    Behavior: input [B, T, C] -> conv over time with padding on the right so
    each output position can incorporate a fixed number of "future" tokens.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = int(kernel_size)
        # use bias=False to match linear-like behaviour
        # Conv1d expects [B, C, T]
        self.conv = nn.Conv1d(in_channels, out_channels, self.kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if x.dim() != 3:
            raise ValueError(f"CausalConv1D expects 3D input [B,T,C], got {x.shape}")
        x = x.transpose(1, 2)
        # pad on the right so conv window can see future tokens
        pad = (0, self.kernel_size - 1)
        x_padded = F.pad(x, pad)
        x_conv = self.conv(x_padded)
        # sanity check: temporal length should be the same as original
        if x_conv.shape[-1] != x.shape[-1]:
            raise RuntimeError(
                f"CausalConv1D produced wrong temporal length: padded_conv_len={x_conv.shape[-1]} vs input_len={x.shape[-1]}, "
                f"kernel_size={self.kernel_size}, pad={pad}"
            )
        x = x_conv.transpose(1, 2)
        return x


class LMAlignedTargetGenerator(nn.Module):
    """Generates V_hat targets from token embeddings.

    V_hat = W_target( CausalConv1D(token_embeddings) )
    returns tensor [B, T, d_model]
    """

    def __init__(self, d_model: int, kernel_size: int = 2, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {k: v for k, v in {"device": device, "dtype": dtype}.items() if v is not None}
        self.conv1d = CausalConv1D(d_model, d_model, kernel_size=kernel_size)
        # learnable projection (slow weight)
        self.W_target = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        # token_embeddings: [B, T, d_model]
        x = self.conv1d(token_embeddings)
        return self.W_target(x)


class TTTGating(nn.Module):
    """Gated MLP with optional In-Place TTT updates on the final projection.

    This class intentionally keeps the `linear_in` and `linear_out` attributes so
    it can be used as a drop-in to existing code expecting that API. When
    `ttt_enabled` is False it behaves like the standard ActivationGating.
    """

    def __init__(
        self,
        activation: tp.Callable,
        dim: int,
        dim_feedforward: int,
        ttt_config: dict | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {k: v for k, v in {"device": device, "dtype": dtype}.items() if v is not None}

        if dim_feedforward == 4 * dim:
            hidden = (21 * dim) // 8
        else:
            hidden = (2 * dim_feedforward) // 3

        # slow weights
        self.linear_in = nn.Linear(dim, 2 * hidden, bias=False, **factory_kwargs)

        # fast weights (read from, but not permanently mutated in forward)
        self.linear_out = nn.Linear(hidden, dim, bias=False, **factory_kwargs)

        self.activation = activation

        # TTT-specific config
        self.ttt_config = {} if ttt_config is None else dict(ttt_config)
        self.ttt_enabled = self.ttt_config.get("enabled", False)
        self.chunk_size = int(self.ttt_config.get("chunk_size", 256))
        self.ttt_lr = float(self.ttt_config.get("learning_rate", 1e-3))
        self.conv_kernel_size = int(self.ttt_config.get("conv_kernel_size", 2))

        # TTT fast weights - these are the "test-time trainable" parameters
        # w_down: [dim, hidden] - projection down from hidden to dim (final projection)
        # NOTE: W_up and W_gate (in linear_in) are FROZEN slow weights per the paper
        # Only W_down is the adaptable fast weight that gets updated via TTT
        # Create as meta tensor so it can be initialized later with the correct dtype
        # IMPORTANT: Keep w_down in float32 for precise gradient updates during inference
        if self.ttt_enabled:
            self.w_down = nn.Parameter(torch.empty(dim, hidden, device='meta'))
            # Store pretrained weights for optional reset (e.g., conversation boundaries)
            # Keep pretrained in float32 for precision (matches w_down precision)
            self.register_buffer('w_down_pretrained', torch.empty(dim, hidden, dtype=torch.float32))
            print(f"[TTT] Enabled TTT gating: chunk_size={self.chunk_size}, lr={self.ttt_lr}, dim={dim}, hidden={hidden}")

        # target generator (slow weights)
        self.target_generator = LMAlignedTargetGenerator(dim, kernel_size=self.conv_kernel_size, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        # Keep API compatible: if TTT is disabled or token_embeddings not provided,
        # fall back to standard gating behavior
        if not self.ttt_enabled or token_embeddings is None:
            if self.ttt_enabled and token_embeddings is None and not hasattr(self, '_warned_no_embeddings'):
                self._warned_no_embeddings = True
                print(f"[TTT WARNING] TTT enabled but token_embeddings is None - falling back to standard gating")
            return self._standard_forward(x)
        return self._ttt_forward(x, token_embeddings)

    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.linear_in.weight)
        B, T, _ = h.shape
        h = h.view(B, T, 2, -1)
        z = self.activation(h[..., 0, :]) * h[..., 1, :]
        out = F.linear(z, self.linear_out.weight)
        return out

    def _ttt_forward(self, x: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        # x: [B, T, dim]
        B, T, _ = x.shape
        # compute Z - use the module directly (handles both Linear and LoRALinear)
        h = self.linear_in(x)
        h = h.view(B, T, 2, -1)
        Z = self.activation(h[..., 0, :]) * h[..., 1, :]  # [B, T, hidden]

        # V_hat from token embeddings
        # For streaming (T <= chunk_size), apply Conv1D to full sequence - no chunk boundaries
        # For training (T > chunk_size), apply Conv1D per-chunk to prevent cross-boundary leakage
        if T <= self.chunk_size:
            # Streaming case: no chunking needed, apply Conv1D normally
            V_hat = self.target_generator(token_embeddings)  # [B, T, dim]
        else:
            # Training case: apply Conv1D per-chunk to maintain causality at chunk boundaries
            # Paper Algorithm 1 line 2006: Vi ← Conv1D_K(X^(i)_0)·Wtarget
            # X^(i)_0 means token embeddings for chunk i only (not full sequence)
            V_hat = self._apply_conv1d_per_chunk(token_embeddings)  # [B, T, dim]

        # chunk-wise parallel update
        return self._parallel_ttt_update(Z, V_hat)

    def _apply_conv1d_per_chunk(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply Conv1D per-chunk to maintain causality at chunk boundaries.

        Per Paper Algorithm 1 line 2006: Vi ← Conv1D_K(X^(i)_0)·Wtarget
        where X^(i)_0 is token embeddings for chunk i only.

        This prevents Conv1D from seeing across chunk boundaries, which would
        violate causality (chunk i's update should not use info from chunk i+1).

        Args:
            token_embeddings: [B, T, dim] where T > chunk_size

        Returns:
            V_hat: [B, T, dim] with Conv1D applied per-chunk
        """
        B, T, dim = token_embeddings.shape

        # Calculate chunks and padding
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        pad_size = num_chunks * self.chunk_size - T

        # Pad if needed
        if pad_size > 0:
            token_embeddings = F.pad(token_embeddings, (0, 0, 0, pad_size))
            T_padded = num_chunks * self.chunk_size
        else:
            T_padded = T

        # Reshape into chunks: [B, num_chunks, chunk_size, dim]
        token_emb_chunks = token_embeddings.view(B, num_chunks, self.chunk_size, dim)

        # Apply Conv1D + projection per chunk
        V_hat_chunks = []
        for i in range(num_chunks):
            chunk_emb = token_emb_chunks[:, i]  # [B, chunk_size, dim]
            # Apply target generator (Conv1D + W_target) to this chunk only
            V_hat_chunk = self.target_generator(chunk_emb)  # [B, chunk_size, dim]
            V_hat_chunks.append(V_hat_chunk)

        # Concatenate chunks back: [B, T_padded, dim]
        V_hat = torch.cat(V_hat_chunks, dim=1)

        # Remove padding if added
        if pad_size > 0:
            V_hat = V_hat[:, :T]

        return V_hat

    def _parallel_ttt_update(self, Z: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
        # Shapes: Z [B, T, hidden], V_hat [B, T, dim]
        B, T, hidden = Z.shape
        _, _, dim = V_hat.shape
        orig_T = T
        
        # Store input dtype to convert back at the end
        input_dtype = Z.dtype
        
        # Convert to float32 for TTT operations - critical for precision
        # bfloat16 loses precision on small gradient updates during inference
        # Only convert if not already float32 to avoid redundant operations
        if Z.dtype != torch.float32:
            Z = Z.to(torch.float32)
        if V_hat.dtype != torch.float32:
            V_hat = V_hat.to(torch.float32)

        # initial fast weights - use w_down parameter directly
        # If w_down is already float32, use it directly; otherwise convert
        if self.w_down.dtype == torch.float32:
            W_down_init = self.w_down
        else:
            W_down_init = self.w_down.to(torch.float32)

        # partition into chunks
        # For short sequences (T < chunk_size), treat entire sequence as one chunk
        # This avoids wasteful padding for streaming inference (T=1)
        effective_chunk_size = min(T, self.chunk_size)
        num_chunks = (T + effective_chunk_size - 1) // effective_chunk_size
        pad_size = num_chunks * effective_chunk_size - T
        
        if pad_size != 0:
            Z = F.pad(Z, (0, 0, 0, pad_size))
            V_hat = F.pad(V_hat, (0, 0, 0, pad_size))
            T_padded = num_chunks * effective_chunk_size
        else:
            T_padded = T

        # reshape: [B, num_chunks, effective_chunk_size, *]
        Zc = Z.view(B, num_chunks, effective_chunk_size, hidden)
        Vc = V_hat.view(B, num_chunks, effective_chunk_size, dim)

        # compute deltas in parallel: result shape [num_chunks, B, dim, hidden]
        # einsum to reorder directly
        deltas = torch.einsum('b n t d, b n t h -> n b d h', Vc, Zc)

        # prefix sum across chunks (causal): S_i = sum_{j=0..i-1} deltas_j
        cumsum = torch.cumsum(deltas, dim=0)
        zero = torch.zeros_like(cumsum[0:1])
        S = torch.cat([zero, cumsum[:-1]], dim=0)  # [num_chunks, B, dim, hidden]

        # broadcast W_down_init to [num_chunks, B, dim, hidden]
        W_init_bc = W_down_init.unsqueeze(0).unsqueeze(0).expand(num_chunks, B, -1, -1)

        device = W_down_init.device
        dtype = W_down_init.dtype

        # effective weights per chunk
        W_eff = W_init_bc + self.ttt_lr * S

        # prepare Z for matmul: want [num_chunks, B, effective_chunk_size, hidden]
        Z_chunks = Zc.permute(1, 0, 2, 3)

        # W_eff: [num_chunks, B, dim, hidden] -> transpose last two
        W_eff_T = W_eff.transpose(-2, -1)  # [num_chunks, B, hidden, dim]

        # batch matmul per chunk
        O_chunks = torch.matmul(Z_chunks, W_eff_T)  # [num_chunks, B, effective_chunk_size, dim]

        # put back to [B, T_padded, dim]
        O = O_chunks.permute(1, 0, 2, 3).reshape(B, T_padded, dim)

        if orig_T != O.shape[1]:
            O = O[:, :orig_T]

        # Persist final state for inference (training uses optimizer)
        if not self.training:
            # Streaming inference only supports batch_size=1 due to w_down being [dim, hidden]
            # For batch_size > 1, each sample would need separate fast weights
            if B > 1:
                raise ValueError(
                    f"TTT inference only supports batch_size=1, got batch_size={B}. "
                    f"This is because w_down [dim, hidden] can only store one batch's state. "
                    f"For batched inference, set batch_size=1 or disable TTT."
                )

            # For the final state, we need to include the delta from the last chunk
            # W_eff[-1] only has prefix sum up to (but not including) the last chunk
            # So we need to add the delta from the last chunk: W_final = W_eff[-1] + lr * deltas[-1]
            # Shape: W_eff[-1, 0] is [dim, hidden], deltas[-1, 0] is [dim, hidden]
            final_state = W_eff[-1, 0] + self.ttt_lr * deltas[-1, 0]  # [dim, hidden]

            # Store update count for debugging (no .item() calls - breaks CUDA graphs)
            if not hasattr(self, '_update_count'):
                self._update_count = 0
            self._update_count += 1

            # Keep w_down in float32 during inference for efficiency
            # No conversion needed since final_state is already float32
            self.w_down.data.copy_(final_state)

        # Convert output back to input dtype only if needed
        if O.dtype != input_dtype:
            return O.to(input_dtype)
        return O

    def reset_ttt_state(self):
        """Reset TTT fast weights to pretrained state (e.g., at conversation boundaries).
        
        This is useful for inference when starting a new conversation or document
        to prevent context leakage from previous inputs.
        """
        if self.ttt_enabled and hasattr(self, 'w_down_pretrained'):
            self.w_down.data.copy_(self.w_down_pretrained)
