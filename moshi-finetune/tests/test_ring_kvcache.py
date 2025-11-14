import torch
from moshi.modules.transformer import RingKVCache


def create_random_kv(B, H, T, D, device='cpu', dtype=torch.float32):
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    return k, v


def test_ring_kvcache_no_wrap_when_within_capacity():
    B, H, D = 2, 4, 8
    capacity = 16
    cache = RingKVCache(batch_size=B, num_heads=H, dim_per_head=D, capacity=capacity, device='cpu', dtype=torch.float32)

    # write T=5 then T=6 -> total 11 < capacity
    k1, v1 = create_random_kv(B, H, 5, D)
    exec_mask1 = torch.ones(B, dtype=torch.bool)
    res1 = cache.complete(k1, v1, exec_mask1)
    # After first write, there should be exactly 5 valid positions per batch
    assert (res1.positions >= 0).sum().item() == B * 5

    k2, v2 = create_random_kv(B, H, 6, D)
    exec_mask2 = torch.ones(B, dtype=torch.bool)
    res2 = cache.complete(k2, v2, exec_mask2)
    assert (res2.positions >= 0).all()


def test_ring_kvcache_wrap_marks_invalid_positions():
    B, H, D = 1, 2, 4
    capacity = 8
    cache = RingKVCache(batch_size=B, num_heads=H, dim_per_head=D, capacity=capacity, device='cpu', dtype=torch.float32)

    # write total of 6 then 4 -> second write will overflow capacity
    k1, v1 = create_random_kv(B, H, 6, D)
    res1 = cache.complete(k1, v1, torch.ones(B, dtype=torch.bool))
    assert (res1.positions >= 0).all()

    k2, v2 = create_random_kv(B, H, 4, D)
    res2 = cache.complete(k2, v2, torch.ones(B, dtype=torch.bool))

    # After overflow some positions should be -1 (invalid) because entries were overwritten
    assert (res2.positions == -1).any()
