"""In-Place TTT helpers: causal conv target generator and TTT-enabled gating.

This module implements LMAlignedTargetGenerator, CausalConv1D and a
TTTGating class intended to be a drop-in replacement for the existing
ActivationGating. It's guarded by configuration so it remains backward
compatible with models that don't enable TTT.
"""
from __future__ import annotations

import logging
import math
import typing as tp
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from ..utils.ttt_monitor import TTTInferenceMonitor


logger = logging.getLogger(__name__)


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
        self._ttt_monitor = None
        self._ttt_monitor_layer = None
        self._ttt_chunk_counter = 0

        # TTT-specific config
        self.ttt_config = {} if ttt_config is None else dict(ttt_config)
        self.ttt_enabled = self.ttt_config.get("enabled", False)
        self.chunk_size = int(self.ttt_config.get("chunk_size", 256))
        self.ttt_lr = float(self.ttt_config.get("learning_rate", 1e-3))
        self.conv_kernel_size = int(self.ttt_config.get("conv_kernel_size", 2))
        self.delta_clip_fro_norm: float | None = self.ttt_config.get("delta_clip_fro_norm", 1e-5)
        if self.delta_clip_fro_norm is not None:
            self.delta_clip_fro_norm = float(self.delta_clip_fro_norm)
            if self.delta_clip_fro_norm <= 0:
                self.delta_clip_fro_norm = None
        self.delta_clip_epsilon = float(self.ttt_config.get("delta_clip_epsilon", 1e-12))

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
            self.register_buffer('ttt_clip_event_counter', torch.zeros(1, dtype=torch.float32), persistent=False)
            self._inference_z_buffer = None
            self._inference_v_buffer = None
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

    def set_ttt_monitor(self, monitor: "TTTInferenceMonitor" | None, layer_name: str | None = None) -> None:
        """Register a monitor that consumes chunk-level TTT statistics."""

        self._ttt_monitor = monitor
        self._ttt_monitor_layer = layer_name or self.__class__.__name__
        if monitor is not None:
            metadata = {
                "chunk_size": self.chunk_size,
                "learning_rate": self.ttt_lr,
                "hidden_dim": self.linear_out.in_features,
                "output_dim": self.linear_out.out_features,
            }
            monitor.register_layer(self._ttt_monitor_layer, metadata)

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

        if self.training:
            return self._parallel_ttt_update(Z, V_hat)
        return self._ttt_forward_inference(Z, V_hat, input_dtype=x.dtype)

    def _ttt_forward_inference(self, Z: torch.Tensor, V_hat: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        """Inference path that buffers tokens until a full chunk is ready.

        We must emit activations one token at a time for streaming, but the paper's
        chunk-wise rule requires applying fast-weight updates only after seeing a
        complete chunk. We therefore:
          1. Produce the current output using the latest fast weights without
             mutating them (identical to the "apply" step).
          2. Append the token's statistics (Z, V_hat) to a buffer.
          3. Once the buffer accumulates >= chunk_size tokens, replay that chunk
             through `_parallel_ttt_update` to perform the "update" step. The
             replayed outputs are discarded—the real-time outputs have already
             been emitted—but the fast weights now include the chunk delta.
        """

        if not self.ttt_enabled:
            return self._standard_forward(Z)

        if Z.dtype != torch.float32:
            Z_fp32 = Z.to(torch.float32)
        else:
            Z_fp32 = Z

        if V_hat.dtype != torch.float32:
            V_fp32 = V_hat.to(torch.float32)
        else:
            V_fp32 = V_hat

        out = torch.matmul(Z_fp32, self.w_down.t())
        if out.dtype != input_dtype:
            out = out.to(input_dtype)

        self._append_inference_buffers(Z_fp32, V_fp32)
        self._flush_inference_buffers(force=False)

        return out

    def _append_inference_buffers(self, Z: torch.Tensor, V_hat: torch.Tensor) -> None:
        if self._inference_z_buffer is None:
            self._inference_z_buffer = Z.detach().clone()
            self._inference_v_buffer = V_hat.detach().clone()
            return
        self._inference_z_buffer = torch.cat([self._inference_z_buffer, Z.detach()], dim=1)
        self._inference_v_buffer = torch.cat([self._inference_v_buffer, V_hat.detach()], dim=1)

    def _flush_inference_buffers(self, force: bool) -> None:
        if not self.ttt_enabled or self._inference_z_buffer is None:
            return
        buffer_len = self._inference_z_buffer.shape[1]
        if buffer_len < self.chunk_size and not force:
            return

        flush_len = buffer_len if force else (buffer_len // self.chunk_size) * self.chunk_size
        if flush_len == 0:
            return

        Z_chunk = self._inference_z_buffer[:, :flush_len, :]
        V_chunk = self._inference_v_buffer[:, :flush_len, :]
        # Replay chunk through the full update path (outputs discarded).
        _ = self._parallel_ttt_update(Z_chunk, V_chunk)

        if flush_len == buffer_len:
            self._inference_z_buffer = None
            self._inference_v_buffer = None
        else:
            self._inference_z_buffer = self._inference_z_buffer[:, flush_len:, :].detach()
            self._inference_v_buffer = self._inference_v_buffer[:, flush_len:, :].detach()

    def _record_ttt_monitor_events(
        self,
        deltas: torch.Tensor,
        w_eff: torch.Tensor,
        scales: torch.Tensor,
        effective_chunk_size: int,
    ) -> None:
        if self._ttt_monitor is None or self.training:
            return
        chunk_count, batch = deltas.shape[:2]
        for chunk_idx in range(chunk_count):
            for batch_idx in range(batch):
                self._ttt_chunk_counter += 1
                delta = deltas[chunk_idx, batch_idx]
                pre = w_eff[chunk_idx, batch_idx]
                post = pre + self.ttt_lr * delta
                delta_norm = torch.linalg.vector_norm(delta).item()
                pre_norm = torch.linalg.vector_norm(pre).item()
                post_norm = torch.linalg.vector_norm(post).item()
                scale_val = scales[chunk_idx, batch_idx].item()
                event = {
                    "chunk_index": self._ttt_chunk_counter,
                    "tokens": effective_chunk_size,
                    "delta_norm": float(delta_norm),
                    "pre_norm": float(pre_norm),
                    "post_norm": float(post_norm),
                    "relative_delta": float(delta_norm / (pre_norm + self.delta_clip_epsilon)),
                    "clip_applied": bool(scale_val < 0.9999),
                    "lr": float(self.ttt_lr),
                }
                self._ttt_monitor.record_event(self._ttt_monitor_layer or "ttt_layer", event)

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

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[TTT][parallel_update] seq_len=%d effective_chunk_size=%d num_chunks=%d (configured_chunk_size=%d)",
                orig_T,
                effective_chunk_size,
                num_chunks,
                self.chunk_size,
            )
        
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

        if self.delta_clip_fro_norm is not None:
            max_norm = deltas.new_tensor(self.delta_clip_fro_norm)
            eps = deltas.new_tensor(self.delta_clip_epsilon)
            delta_norms = torch.linalg.vector_norm(deltas.reshape(num_chunks, B, -1), dim=-1)
            scales = (max_norm / (delta_norms + eps)).clamp(max=1.0)
            deltas = deltas * scales.view(num_chunks, B, 1, 1)
            if hasattr(self, 'ttt_clip_event_counter'):
                clip_events = (scales < (1.0 - 1e-6)).to(deltas.dtype)
                self.ttt_clip_event_counter.add_(clip_events.reshape(-1).sum())
        else:
            scales = torch.ones((num_chunks, B), dtype=deltas.dtype, device=deltas.device)

        # Memory-optimized prefix sum computation
        # Key mathematical insight for training with delta exposure:
        #   S_apply[i] = S_prefix[i] + deltas[i]
        #              = sum(deltas[0:i-1]) + deltas[i]
        #              = sum(deltas[0:i])
        #              = cumsum[i]
        # Therefore: S_apply == cumsum (saves creating S_prefix and S_apply separately!)

        if self.training:
            # Training mode: use cumsum directly (equals S_prefix + deltas)
            # Memory savings: ~6 GB per layer (avoids S_prefix and S_apply copies)
            cumsum = torch.cumsum(deltas, dim=0)
            S_apply = cumsum  # Direct reuse - no extra allocation!
        else:
            # Inference mode: S_apply = S_prefix only (paper Algorithm 1 line 11)
            cumsum = torch.cumsum(deltas, dim=0)
            zero = torch.zeros_like(cumsum[0:1])
            S_apply = torch.cat([zero, cumsum[:-1]], dim=0)

        # broadcast W_down_init to [num_chunks, B, dim, hidden]
        W_init_bc = W_down_init.unsqueeze(0).unsqueeze(0).expand(num_chunks, B, -1, -1)

        device = W_down_init.device
        dtype = W_down_init.dtype

        # effective weights per chunk used for the forward pass
        W_eff_apply = W_init_bc + self.ttt_lr * S_apply

        self._record_ttt_monitor_events(deltas, W_eff_apply, scales, effective_chunk_size)

        # prepare Z for matmul: want [num_chunks, B, effective_chunk_size, hidden]
        Z_chunks = Zc.permute(1, 0, 2, 3)

        # W_eff: [num_chunks, B, dim, hidden] -> transpose last two
        W_eff_T = W_eff_apply.transpose(-2, -1)  # [num_chunks, B, hidden, dim]

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
            total_delta = cumsum[-1, 0]
            final_state = W_down_init + self.ttt_lr * total_delta  # [dim, hidden]

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

    def flush_ttt_buffers(self):
        """Force flush any remaining tokens in TTT inference buffers.

        This should be called at the end of inference to ensure all buffered
        tokens are processed and contribute to TTT updates.
        """
        if self.ttt_enabled:
            self._flush_inference_buffers(force=True)

    def reset_ttt_state(self, clear_buffers: bool = True):
        """Reset TTT fast weights to pretrained state (e.g., at conversation boundaries).

        This is useful for inference when starting a new conversation or document
        to prevent context leakage from previous inputs.

        Args:
            clear_buffers: If True, also clear inference buffers. Set to False during
                streaming resets to preserve accumulated tokens for the next update.
        """
        if self.ttt_enabled and hasattr(self, 'w_down_pretrained'):
            def _safe_norm(tensor: torch.Tensor):
                if getattr(tensor, "is_meta", False):
                    return None
                return torch.linalg.norm(tensor).item()

            def _target_generator_norm():
                total = 0.0
                counted = 0
                for param in self.target_generator.parameters():
                    if getattr(param, "is_meta", False):
                        continue
                    total += torch.linalg.norm(param).item()
                    counted += 1
                return total if counted > 0 else None

            norm_before = _safe_norm(self.w_down.data)

            self.w_down.data.copy_(self.w_down_pretrained)
            if clear_buffers:
                self._inference_z_buffer = None
                self._inference_v_buffer = None
                self._ttt_chunk_counter = 0

            norm_after = _safe_norm(self.w_down.data)

            if self.training:
                tg_norm = _target_generator_norm()
                w_down_norm = _safe_norm(self.w_down.data)
                if tg_norm is not None and w_down_norm is not None:
                    logger.info(
                        f"[TTT RESET][train] target_generator norm {tg_norm:.6f}, w_down norm {w_down_norm:.6f} (dtype: {self.w_down.dtype}); "
                        "both learn via backprop during training"
                    )
                else:
                    logger.info("[TTT RESET][train] TTT parameters on meta device (norm unavailable)")
                return

            if norm_before is not None and norm_after is not None:
                logger.info(
                    f"[TTT RESET] Layer reset: w_down norm {norm_before:.6f} -> {norm_after:.6f}"
                )
            else:
                logger.info("[TTT RESET] Layer reset on meta device (norm unavailable)")
