"""Comprehensive TTT regression tests covering the scenarios outlined in
TTT_TEST_PLAN.md.
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import pytest
import torch
from torch import nn
import torch.nn.functional as F

import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "moshi" / "moshi"
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from moshi.models.lm import LMModel
from moshi.modules.ttt_module import TTTGating
from moshi.modules.transformer import StreamingTransformer


def _materialize_fast_weights(gating: TTTGating, fill_value: float = 0.0) -> None:
    dim, hidden = gating.linear_out.weight.shape
    param = torch.full((dim, hidden), fill_value, dtype=torch.float32)
    gating.w_down = nn.Parameter(param)
    if hasattr(gating, "w_down_pretrained"):
        gating.w_down_pretrained.resize_as_(param)
        gating.w_down_pretrained.copy_(param)


def _build_ttt_gating(
    *,
    dim: int = 4,
    chunk_size: int = 1,
    lr: float = 1.0,
    delta_clip: float | None = 1e-1,
    dim_feedforward: int | None = None,
) -> TTTGating:
    config = {
        "enabled": True,
        "chunk_size": chunk_size,
        "learning_rate": lr,
        "layer_frequency": 1,
        "start_layer": 0,
        "delta_clip_fro_norm": delta_clip,
        "delta_clip_epsilon": 1e-12,
    }
    dim_ff = 4 * dim if dim_feedforward is None else dim_feedforward
    gating = TTTGating(F.silu, dim, dim_ff, ttt_config=config)
    _materialize_fast_weights(gating, 0.0)
    gating.training = False
    return gating


def _collect_ttt_layers(model: LMModel) -> list[TTTGating]:
    layers: list[TTTGating] = []
    for layer in model.transformer.layers:
        if getattr(layer, "use_ttt", False):
            assert isinstance(layer.gating, TTTGating)
            layers.append(layer.gating)
    return layers


def _build_small_model(
    *,
    ttt_enabled: bool = True,
    delta_clip: float | None = 1e3,
    num_layers: int = 2,
    context: int = 64,
) -> LMModel:
    torch.manual_seed(0)
    ttt_config = {
        "enabled": ttt_enabled,
        "chunk_size": 2,
        "learning_rate": 5e-4,
        "layer_frequency": 1,
        "start_layer": 0,
        "delta_clip_fro_norm": delta_clip,
        "delta_clip_epsilon": 1e-12,
    }
    if not ttt_enabled:
        ttt_config["enabled"] = False
    model = LMModel(
        dim=16,
        text_card=64,
    n_q=2,
    dep_q=2,
        num_heads=2,
        hidden_scale=2,
        norm="layer_norm",
    delays=[0, 0, 0],
        context=context,
        causal=True,
        num_layers=num_layers,
        gating="silu",
        depformer_num_heads=1,
        depformer_num_layers=1,
        ttt_config=ttt_config,
        positional_embedding="rope",
        yarn_scale=1.0,
        original_max_seq_len=1024,
    )
    for layer in model.transformer.layers:
        if getattr(layer, "use_ttt", False):
            assert isinstance(layer.gating, TTTGating)
            _materialize_fast_weights(layer.gating, 0.0)
    return model


def _make_codes(model: LMModel, T: int) -> torch.Tensor:
    return torch.randint(0, 10, (1, model.num_codebooks, T))


def test_fast_weight_delta_clipping_enforces_bound():
    gating = _build_ttt_gating(dim=4, chunk_size=1, lr=1.0, delta_clip=0.1)
    Z = torch.ones(1, 1, gating.linear_out.in_features)
    V_hat = torch.ones(1, 1, gating.linear_out.out_features)

    gating._parallel_ttt_update(Z, V_hat)

    # delta norm should match the configured cap (within numerical tolerance)
    norm = gating.w_down.norm().item()
    assert math.isclose(norm, gating.ttt_lr * gating.delta_clip_fro_norm, rel_tol=1e-2)
    assert float(gating.ttt_clip_event_counter) >= 1.0


def test_target_generator_chunking_matches_manual():
    gating = _build_ttt_gating(dim=8, chunk_size=2, lr=1e-3, delta_clip=None)
    token_embeddings = torch.arange(0, 8 * 6, dtype=torch.float32).view(1, 6, 8)

    with torch.no_grad():
        manual_chunks = []
        for i in range(0, token_embeddings.shape[1], gating.chunk_size):
            chunk = token_embeddings[:, i : i + gating.chunk_size]
            manual_chunks.append(gating.target_generator(chunk))
        manual = torch.cat(manual_chunks, dim=1)

    chunked = gating._apply_conv1d_per_chunk(token_embeddings)
    assert torch.allclose(chunked, manual, atol=1e-6)


def test_token_embeddings_match_transformer_input():
    model = _build_small_model(ttt_enabled=False)
    sequence = torch.randint(0, 10, (1, model.num_codebooks, 12))

    with mock.patch.object(
        model.transformer,
        "forward",
        wraps=model.transformer.forward,
    ) as wrapped_forward:
        model.forward_text(sequence)

    assert wrapped_forward.call_args is not None
    passed = wrapped_forward.call_args.kwargs["token_embeddings"]

    # recompute expected embeddings: sum audio + text (no conditioners)
    audio_sum = None
    for cb_index in range(model.num_audio_codebooks):
        emb = model.emb[cb_index](sequence[:, cb_index + model.audio_offset])
        audio_sum = emb if audio_sum is None else audio_sum + emb
    text_emb = model.text_emb(sequence[:, 0])
    expected = audio_sum + text_emb

    assert torch.allclose(passed, expected)


def test_ttt_adaptation_reduces_reconstruction_error_second_chunk():
    gating = _build_ttt_gating(dim=2, chunk_size=2, lr=1.0, delta_clip=None, dim_feedforward=3)
    hidden = gating.linear_out.in_features
    dim = gating.linear_out.out_features
    base = torch.eye(hidden).unsqueeze(0)
    Z = torch.cat([base, base], dim=1)
    timesteps = Z.shape[1]
    V_hat = torch.ones(1, timesteps, dim)

    output = gating._parallel_ttt_update(Z, V_hat)
    ttt_chunk = output[:, gating.chunk_size :]
    target_chunk = V_hat[:, gating.chunk_size :]
    baseline_chunk = torch.zeros_like(ttt_chunk)

    mse_ttt = F.mse_loss(ttt_chunk, target_chunk)
    mse_baseline = F.mse_loss(baseline_chunk, target_chunk)

    assert mse_ttt < mse_baseline


def test_short_form_inference_is_stable_without_clipping():
    model = _build_small_model(ttt_enabled=True, delta_clip=1e6)
    codes = _make_codes(model, T=12)
    with torch.no_grad():
        output = model.forward(codes)

    assert torch.isfinite(output.text_logits).all()
    for gating in _collect_ttt_layers(model):
        assert float(gating.ttt_clip_event_counter) == 0.0
        assert torch.isfinite(gating.w_down).all()


def test_long_form_inference_triggers_clipping_and_bounds_norm():
    model = _build_small_model(ttt_enabled=True, delta_clip=1e-3, context=128)
    codes = _make_codes(model, T=64)
    with torch.no_grad():
        _ = model.forward(codes)

    for gating in _collect_ttt_layers(model):
        assert float(gating.ttt_clip_event_counter) > 0.0
        assert gating.w_down.norm().item() < 1e3


def test_yarn_context_sweep_consistency():
    model = _build_small_model(ttt_enabled=False, num_layers=1, context=128)
    codes_long = _make_codes(model, T=32)
    codes_short = codes_long[:, :, :16]

    with torch.no_grad():
        out_short = model.forward(codes_short)
        out_long = model.forward(codes_long)

    prefix = out_long.text_logits[:, :, : out_short.text_logits.shape[2]]
    assert torch.allclose(out_short.text_logits, prefix, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("lr", [3e-4, 1e-3, 3e-3, 1e-2])
@pytest.mark.parametrize("chunk", [1, 4])
def test_learning_rate_grid_remains_stable(lr: float, chunk: int):
    gating = _build_ttt_gating(dim=4, chunk_size=chunk, lr=lr, delta_clip=None)
    T = chunk * 2
    Z = torch.randn(1, T, gating.linear_out.in_features)
    V_hat = torch.randn(1, T, gating.linear_out.out_features)

    gating._parallel_ttt_update(Z, V_hat)

    assert torch.isfinite(gating.w_down).all()
    assert gating.w_down.norm().item() < 1e3


def test_ttt_disabled_matches_plain_model():
    torch.manual_seed(123)
    model_plain = _build_small_model(ttt_enabled=False)
    torch.manual_seed(123)
    model_disabled_flag = _build_small_model(ttt_enabled=False)
    codes = _make_codes(model_plain, T=10)

    with torch.no_grad():
        out_plain = model_plain.forward(codes)
        out_disabled = model_disabled_flag.forward(codes)

    assert torch.allclose(out_plain.text_logits, out_disabled.text_logits)
    assert torch.allclose(out_plain.logits, out_disabled.logits)


def test_ttt_telemetry_payload_is_structured():
    model = _build_small_model(ttt_enabled=True, delta_clip=1e-4)
    codes = _make_codes(model, T=32)
    with torch.no_grad():
        model.forward(codes)

    telemetry = []
    for idx, gating in enumerate(_collect_ttt_layers(model)):
        telemetry.append(
            {
                "layer": idx,
                "clip_events": float(gating.ttt_clip_event_counter),
                "w_down_norm": gating.w_down.norm().item(),
                "update_count": getattr(gating, "_update_count", 0),
            }
        )

    assert telemetry, "expected at least one TTT layer"
    for entry in telemetry:
        assert entry["clip_events"] >= 0
        assert entry["w_down_norm"] >= 0
        assert isinstance(entry["update_count"], int)


def test_perplexity_proxy_improves_after_ttt_updates():
    gating = _build_ttt_gating(dim=2, chunk_size=2, lr=0.5, delta_clip=None, dim_feedforward=3)
    hidden = gating.linear_out.in_features
    dim = gating.linear_out.out_features
    base = torch.eye(hidden).unsqueeze(0)
    Z = torch.cat([base, base], dim=1)
    V_hat = torch.ones(1, 2 * gating.chunk_size, dim)

    gating._parallel_ttt_update(Z[:, :gating.chunk_size], V_hat[:, :gating.chunk_size])
    out_after_update = gating._parallel_ttt_update(Z[:, gating.chunk_size :], V_hat[:, gating.chunk_size :])

    mse = F.mse_loss(out_after_update, V_hat[:, gating.chunk_size :])
    mse_baseline = F.mse_loss(torch.zeros_like(V_hat[:, gating.chunk_size :]), V_hat[:, gating.chunk_size :])
    assert mse < mse_baseline


def test_fast_weight_replay_and_reset_behaviour():
    gating = _build_ttt_gating(dim=2, chunk_size=1, lr=1.0, delta_clip=None)
    Z = torch.randn(1, 2, gating.linear_out.in_features)
    V_hat = torch.randn(1, 2, gating.linear_out.out_features)

    gating._parallel_ttt_update(Z[:, :1], V_hat[:, :1])
    snapshot = gating.w_down.detach().clone()

    gating._parallel_ttt_update(Z[:, 1:], V_hat[:, 1:])
    mutated = gating.w_down.detach().clone()
    assert not torch.allclose(mutated, snapshot)

    gating.reset_ttt_state()
    assert torch.allclose(gating.w_down, gating.w_down_pretrained)

    # restore snapshot manually to emulate replay harness
    gating.w_down.data.copy_(snapshot)
    assert torch.allclose(gating.w_down, snapshot)


def test_streaming_reset_triggers_fast_weight_reset_only_when_masked():
    model = _build_small_model(ttt_enabled=True)
    ttt_layers = _collect_ttt_layers(model)
    assert ttt_layers, "expected at least one TTT-enabled layer"
    gating = ttt_layers[0]

    with model.streaming(batch_size=2):
        mutated = gating.w_down.detach().clone()
        mutated += 1.0
        gating.w_down.data.copy_(mutated)
        assert not torch.allclose(gating.w_down, gating.w_down_pretrained)

        # Mask with all False should leave fast weights untouched
        mask_keep = torch.zeros(2, dtype=torch.bool, device=model.device)
        model.reset_streaming(mask_keep)
        assert torch.allclose(gating.w_down, mutated)

        # Any True entry should trigger a global fast-weight reset
        mask_reset = torch.tensor([False, True], dtype=torch.bool, device=model.device)
        model.reset_streaming(mask_reset)
        assert torch.allclose(gating.w_down, gating.w_down_pretrained)


def test_inference_chunk_buffer_updates_only_after_full_chunk():
    gating = _build_ttt_gating(dim=4, chunk_size=3, lr=0.5, delta_clip=None)
    x = torch.randn(1, 1, 4)
    tokens = torch.randn(1, 1, 4)
    baseline = gating.w_down.detach().clone()

    for _ in range(2):
        gating(x, tokens)
        assert torch.allclose(gating.w_down, baseline)

    gating(x, tokens)
    assert not torch.allclose(gating.w_down, baseline)


def test_reset_clears_inference_buffer():
    gating = _build_ttt_gating(dim=4, chunk_size=2, lr=0.5, delta_clip=None)
    x = torch.randn(1, 1, 4)
    tokens = torch.randn(1, 1, 4)
    baseline = gating.w_down.detach().clone()

    gating(x, tokens)
    assert torch.allclose(gating.w_down, baseline)

    gating.reset_ttt_state()
    gating(x, tokens)
    assert torch.allclose(gating.w_down, baseline)


def test_ttt_target_generator_receives_gradients_in_training_mode():
    gating = _build_ttt_gating(dim=4, chunk_size=2, lr=0.5, delta_clip=None)
    gating.train()
    gating.zero_grad()

    # Use at least two chunks so target_generator gradients have signal.
    # With T == chunk_size the replayed chunk never influences another chunk
    # and gradients stay zero, which masked regressions in training runs.
    x = torch.randn(1, 4, 4)
    tokens = torch.randn(1, 4, 4)

    out = gating(x, token_embeddings=tokens)
    loss = out.pow(2).mean()
    loss.backward()

    target_gen_grad_norms = [
        param.grad.norm().item()
        for name, param in gating.named_parameters()
        if "target_generator" in name and param.grad is not None
    ]

    w_down_grad = gating.w_down.grad
    assert w_down_grad is not None, "w_down should have gradient during training"
    w_down_grad_norm = w_down_grad.norm().item()

    assert target_gen_grad_norms, "Expected gradients on target_generator params"
    assert any(grad > 0 for grad in target_gen_grad_norms), "target_generator gradients should be non-zero"
    assert w_down_grad_norm > 0, "w_down gradient should be non-zero during training"
