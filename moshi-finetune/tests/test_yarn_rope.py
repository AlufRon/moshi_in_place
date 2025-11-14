import torch
from moshi.modules.rope import _compute_yarn_parameters


def test_compute_yarn_parameters_shape_and_types():
    dim = 512
    max_period = 10000.0
    scale = 4.0
    original_max_seq_len = 3000

    inv_freq, attention_factor = _compute_yarn_parameters(
        dim=dim,
        max_period=max_period,
        scale=scale,
        original_max_seq_len=original_max_seq_len,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        mscale_all_dim=0.0,
        device=torch.device("cpu"),
    )

    # inv_freq should be a 1D float32 tensor of length dim/2
    assert isinstance(inv_freq, torch.Tensor)
    assert inv_freq.ndim == 1
    assert inv_freq.numel() == dim // 2
    assert inv_freq.dtype == torch.float32

    # attention_factor should be a float
    assert isinstance(attention_factor, float) or (isinstance(attention_factor, torch.Tensor) and attention_factor.numel() == 1)


class DummyRope:
    def __init__(self, dim, max_period, yarn_scale, original_max_seq_len, beta_fast, beta_slow, mscale, mscale_all_dim):
        self.dim = dim
        self.max_period = max_period
        self.yarn_scale = yarn_scale
        self.original_max_seq_len = original_max_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        # create a meta tensor to emulate unloaded buffer
        self.inv_freq = torch.empty(dim // 2, device="meta")
        # simulate _buffers dict as in nn.Module
        self._buffers = {"inv_freq": self.inv_freq}
        self.attention_factor = 1.0


def test_wrapped_model_like_rope_initialization_replaces_meta_and_sets_attention():
    # use smaller dim to speed the test
    dim = 64
    max_period = 10000.0
    yarn_scale = 4.0
    original_max_seq_len = 3000

    rope = DummyRope(dim, max_period, yarn_scale, original_max_seq_len, beta_fast=32, beta_slow=1, mscale=1.2, mscale_all_dim=0.0)

    assert rope.inv_freq.is_meta

    inv_freq, attention_factor = _compute_yarn_parameters(
        dim=rope.dim,
        max_period=rope.max_period,
        scale=rope.yarn_scale,
        original_max_seq_len=rope.original_max_seq_len,
        beta_fast=rope.beta_fast,
        beta_slow=rope.beta_slow,
        mscale=rope.mscale,
        mscale_all_dim=rope.mscale_all_dim,
        device=torch.device("cpu"),
    )

    # emulate the initialization done in wrapped_model.py
    rope._buffers['inv_freq'] = inv_freq
    rope.attention_factor = attention_factor

    # Now inv_freq should not be meta
    assert not rope._buffers['inv_freq'].is_meta
    # attention factor should be applied
    assert rope.attention_factor == attention_factor
