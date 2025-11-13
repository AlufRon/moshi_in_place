import torch
from torch import nn

from moshi.moshi.moshi.modules.ttt_module import CausalConv1D, LMAlignedTargetGenerator, TTTGating


def test_causal_conv1d_shape():
    B, T, C = 2, 10, 16
    x = torch.randn(B, T, C)
    conv = CausalConv1D(C, C, kernel_size=3)
    y = conv(x)
    assert y.shape == (B, T, C)


def test_lm_aligned_target_generator_shape():
    B, T, C = 2, 12, 16
    emb = torch.randn(B, T, C)
    gen = LMAlignedTargetGenerator(C, kernel_size=2)
    vhat = gen(emb)
    assert vhat.shape == (B, T, C)


def test_tttgating_standard_and_ttt():
    B, T, D = 2, 20, 32
    x = torch.randn(B, T, D)
    emb = torch.randn(B, T, D)

    # standard (ttt disabled)
    g0 = TTTGating(nn.functional.gelu, D, 4 * D, ttt_config={'enabled': False})
    out0 = g0(x)
    assert out0.shape == (B, T, D)

    # ttt enabled but token_embeddings required
    g1 = TTTGating(nn.functional.gelu, D, 4 * D, ttt_config={'enabled': True, 'chunk_size': 8, 'learning_rate': 1e-3})
    out1 = g1(x, token_embeddings=emb)
    assert out1.shape == (B, T, D)
