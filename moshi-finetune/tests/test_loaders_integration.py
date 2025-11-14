import pytest
import torch
from moshi.modules.rope import _compute_yarn_parameters
from moshi.models.loaders import get_moshi_lm, _lm_kwargs as DEFAULT_LM_KWARGS


@pytest.mark.cpu
def test_get_moshi_lm_with_yarn_on_cpu_matches_compute():
    # instantiate model on CPU without loading weights (filename=None)
    yarn_cfg = {
        'enabled': True,
        'scale': 4.0,
        'original_max_seq_len': 3000,
        'beta_fast': 32,
        'beta_slow': 1,
        'mscale': 1.0,
        'mscale_all_dim': 0.0,
    }

    # Start from default lm kwargs and override only the relevant entries
    lm_kwargs = dict(DEFAULT_LM_KWARGS)
    lm_kwargs.update({
        'context': 3000,
        'max_period': 10000,
        'num_heads': 32,
        'dim': 4096,
        'positional_embedding': 'rope',
        'yarn_config': yarn_cfg,
    })

    model = get_moshi_lm(None, lm_kwargs=lm_kwargs, device='cpu', dtype=torch.float32, lora_weights=None)
    # check first layer rope
    rope = model.transformer.layers[0].self_attn.rope
    assert hasattr(rope, 'inv_freq')
    assert rope.inv_freq.ndim == 1

    # compute expected inv_freq and attention_factor
    inv_freq_expected, attention_factor_expected = _compute_yarn_parameters(
        dim=rope.dim,
        max_period=rope.max_period,
        scale=yarn_cfg['scale'],
        original_max_seq_len=yarn_cfg['original_max_seq_len'],
        beta_fast=yarn_cfg['beta_fast'],
        beta_slow=yarn_cfg['beta_slow'],
        mscale=yarn_cfg['mscale'],
        mscale_all_dim=yarn_cfg['mscale_all_dim'],
        device='cpu',
    )

    # shapes/dtypes
    assert inv_freq_expected.shape == rope.inv_freq.shape
    # attention factor may be a float or tensor
    assert hasattr(rope, 'attention_factor')
    # If attention_factor is a tensor, compare numerically
    if isinstance(rope.attention_factor, torch.Tensor):
        assert torch.allclose(torch.tensor(attention_factor_expected, dtype=rope.attention_factor.dtype), rope.attention_factor)
    else:
        assert float(rope.attention_factor) == float(attention_factor_expected)
