# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import nn
import math
import torch
from typing import Optional, Tuple
from ..utils.compile import torch_compile_lazy


def _compute_yarn_parameters(
    dim: int,
    max_period: float,
    scale: float,
    original_max_seq_len: int,
    beta_fast: int = 32,
    beta_slow: int = 1,
    mscale: float = 1.0,
    mscale_all_dim: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Compute YaRN-scaled inverse frequencies and attention factor.

    Based on the YaRN paper (Peng et al., 2023): https://arxiv.org/abs/2309.00071

    Args:
        dim: Dimension of the rotary embeddings (must be even)
        max_period: Base frequency (theta) for RoPE
        scale: Context extension factor
        original_max_seq_len: Original sequence length model was trained on
        beta_fast: Low frequency boundary (default 32)
        beta_slow: High frequency boundary (default 1)
        mscale: Attention scaling factor
        mscale_all_dim: Additional dimension-wise scaling
        device: Device to create tensors on

    Returns:
        inv_freq: Scaled inverse frequencies [dim/2]
        attention_factor: Scaling factor for attention logits
    """
    assert dim % 2 == 0, "Dimension must be even for YaRN"

    # Compute base inverse frequencies
    dim_half = dim // 2
    indices = torch.arange(dim_half, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (max_period ** (indices / dim_half))

    if scale <= 1.0:
        # No scaling needed
        return inv_freq, 1.0

    # Compute wavelengths
    wavelengths = 2 * math.pi * max_period ** (indices / dim_half)

    # NTK-by-parts: divide frequencies into 3 ranges
    # Low freq (long wavelength): wavelength > beta_fast * original_len
    # High freq (short wavelength): wavelength < beta_slow * original_len
    # Mid freq: interpolate between the two

    low_freq_wavelen = beta_fast * original_max_seq_len
    high_freq_wavelen = beta_slow * original_max_seq_len

    # Compute linear ramp factor for each frequency
    linear_ramp = (wavelengths - high_freq_wavelen) / (low_freq_wavelen - high_freq_wavelen)
    linear_ramp = torch.clamp(linear_ramp, 0.0, 1.0)

    # Apply NTK scaling to low frequencies, linear interpolation to mid frequencies
    # inv_freq_extrapolation: NTK-scaled (for low freq)
    # inv_freq_interpolation: linearly interpolated (for mid freq)
    inv_freq_extrapolation = inv_freq / scale
    inv_freq_interpolation = inv_freq  # High freq unchanged

    # Blend based on ramp
    inv_freq = (1 - linear_ramp) * inv_freq_interpolation + linear_ramp * inv_freq_extrapolation

    # Compute attention factor (mscale)
    if mscale != 1.0:
        def get_mscale(scale_factor, mscale_param):
            if scale_factor <= 1.0:
                return 1.0
            return 0.1 * mscale_param * math.log(scale_factor) + 1.0

        if mscale_all_dim > 0:
            attention_factor = get_mscale(scale, mscale) / get_mscale(scale, mscale_all_dim)
        else:
            attention_factor = get_mscale(scale, mscale)
    else:
        attention_factor = 1.0

    return inv_freq, attention_factor


@torch_compile_lazy
def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000,
    time_before_heads: bool = False,
    inv_freq: Optional[torch.Tensor] = None,
    attention_factor: float = 1.0,
):
    """Apply Rotary Position Embedding with optional YaRN scaling.

    Args:
        q (torch.Tensor): queries, shape `[B, T, H, D]`.
        k (torch.Tensor): keys, shape `[B, T, H, D]`.
        offset (torch.Tensor): current offset, e.g. when streaming.
        max_period (float): maximum period for the cos and sin.
        time_before_heads (bool): if True, expected [B, T, H, D], else [B, H, T, D]
        inv_freq (torch.Tensor, optional): Precomputed inverse frequencies (for YaRN)
        attention_factor (float): Attention scaling factor (for YaRN mscale)
    """

    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape
    assert k.shape == q.shape
    assert D > 0
    assert D % 2 == 0
    assert max_period > 0

    # Use provided inverse frequencies (YaRN) or compute standard ones
    if inv_freq is None:
        ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    else:
        freqs = inv_freq.to(q.device)

    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)
    if time_before_heads:
        ts = ts.view(B, -1, 1, 1)
    else:
        ts = ts.view(B, 1, -1, 1)

    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)

    # convention is `r` suffix is real part, `i` is imaginary.
    qr = q[..., 0].float()
    qi = q[..., 1].float()

    kr = k[..., 0].float()
    ki = k[..., 1].float()

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    # Apply rotation
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr

    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    # Apply attention factor (YaRN mscale)
    if attention_factor != 1.0:
        qor = qor * attention_factor
        qoi = qoi * attention_factor

    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo.view(*dims, D), ko.view(*dims, D)


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Supports YaRN scaling from [Peng et al 2023](https://arxiv.org/abs/2309.00071).

    Args:
        max_period (float): Maximum period of the rotation frequencies.
        dim (int, optional): Dimension of embeddings (required for YaRN).
        yarn_scale (float): YaRN scaling factor (1.0 = no scaling).
        original_max_seq_len (int): Original training sequence length (for YaRN).
        beta_fast (int): YaRN low frequency boundary.
        beta_slow (int): YaRN high frequency boundary.
        mscale (float): YaRN attention scaling factor.
        mscale_all_dim (float): YaRN additional dimension-wise scaling.
    """

    def __init__(
        self,
        max_period: float = 10000.0,
        dim: Optional[int] = None,
        yarn_scale: float = 1.0,
        original_max_seq_len: int = 3000,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.yarn_scale = yarn_scale
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        # Precompute YaRN parameters if scaling is enabled
        if yarn_scale > 1.0 and dim is not None:
            inv_freq, attention_factor = _compute_yarn_parameters(
                dim=dim,
                max_period=max_period,
                scale=yarn_scale,
                original_max_seq_len=original_max_seq_len,
                beta_fast=beta_fast,
                beta_slow=beta_slow,
                mscale=mscale,
                mscale_all_dim=mscale_all_dim,
                device=None,  # Will be moved to correct device on first forward
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.attention_factor = attention_factor
        else:
            self.inv_freq = None
            self.attention_factor = 1.0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: torch.Tensor,
        time_before_heads: bool = False,
    ):
        """Apply rope rotation to query or key tensor."""
        return apply_rope(
            q,
            k,
            offset,
            self.max_period,
            time_before_heads,
            inv_freq=self.inv_freq,
            attention_factor=self.attention_factor,
        )
