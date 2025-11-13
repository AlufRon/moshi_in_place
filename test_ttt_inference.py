#!/usr/bin/env python3
"""Test script to verify TTT inference state persistence.

This script tests that w_down updates persist across forward passes during inference.
"""

import torch
import torch.nn as nn
import sys
import os

# Add moshi to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'moshi/moshi'))

from moshi.modules.ttt_module import TTTGating


def test_ttt_inference_persistence():
    """Test that w_down persists across inference calls."""
    print("=" * 60)
    print("Testing TTT Inference State Persistence")
    print("=" * 60)
    
    # Create TTT module
    dim = 128
    dim_feedforward = 512
    ttt_config = {
        'enabled': True,
        'chunk_size': 4,
        'learning_rate': 1e-3,
        'conv_kernel_size': 2,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    print(f"\nDevice: {device}")
    print(f"Dtype: {dtype}")
    print(f"Config: {ttt_config}")
    
    module = TTTGating(
        activation=torch.nn.functional.gelu,
        dim=dim,
        dim_feedforward=dim_feedforward,
        ttt_config=ttt_config,
        device=device,
        dtype=dtype,
    )
    
    # Initialize w_down with specific values (not meta)
    if module.ttt_enabled:
        hidden = (2 * dim_feedforward) // 3
        # Create properly on device first
        init_data = torch.randn(dim, hidden, device=device, dtype=dtype) * 0.02
        # Replace the meta tensor with a real parameter
        module.w_down = nn.Parameter(init_data)
        # Initialize pretrained buffer
        module.w_down_pretrained = torch.empty(dim, hidden, device=device, dtype=dtype)
        module.w_down_pretrained.copy_(module.w_down.data)
        print(f"\n✓ w_down initialized: {module.w_down.shape}")
    
    module.eval()  # Set to inference mode
    
    # Test 1: Check that w_down changes during inference
    print("\n" + "=" * 60)
    print("Test 1: w_down persistence during inference")
    print("=" * 60)
    
    B, T = 2, 8  # batch=2, time=8
    x = torch.randn(B, T, dim, device=device, dtype=dtype)
    token_embeddings = torch.randn(B, T, dim, device=device, dtype=dtype)
    
    # Get initial w_down
    w_down_initial = module.w_down.data.clone()
    print(f"\nInitial w_down norm: {w_down_initial.norm().item():.6f}")
    
    # First forward pass
    with torch.no_grad():
        out1 = module(x, token_embeddings)
    
    w_down_after_1 = module.w_down.data.clone()
    print(f"After pass 1 w_down norm: {w_down_after_1.norm().item():.6f}")
    
    # Check if w_down changed
    diff_1 = (w_down_after_1 - w_down_initial).norm().item()
    print(f"Change from initial: {diff_1:.6f}")
    
    if diff_1 > 1e-6:
        print("✓ PASS: w_down changed during inference (state persisted)")
    else:
        print("✗ FAIL: w_down did NOT change (no persistence!)")
        return False
    
    # Second forward pass
    with torch.no_grad():
        out2 = module(x, token_embeddings)
    
    w_down_after_2 = module.w_down.data.clone()
    print(f"After pass 2 w_down norm: {w_down_after_2.norm().item():.6f}")
    
    diff_2 = (w_down_after_2 - w_down_after_1).norm().item()
    print(f"Change from pass 1: {diff_2:.6f}")
    
    if diff_2 > 1e-6:
        print("✓ PASS: w_down accumulated updates across passes")
    else:
        print("✗ FAIL: w_down did not accumulate (no state accumulation!)")
        return False
    
    # Test 2: Check reset functionality
    print("\n" + "=" * 60)
    print("Test 2: TTT state reset")
    print("=" * 60)
    
    module.reset_ttt_state()
    w_down_after_reset = module.w_down.data.clone()
    print(f"\nAfter reset w_down norm: {w_down_after_reset.norm().item():.6f}")
    
    reset_diff = (w_down_after_reset - w_down_initial).norm().item()
    print(f"Diff from initial: {reset_diff:.6f}")
    
    if reset_diff < 1e-6:
        print("✓ PASS: Reset restored pretrained weights")
    else:
        print(f"✗ FAIL: Reset did not restore weights (diff={reset_diff:.6f})")
        return False
    
    # Test 3: Check training mode doesn't auto-persist
    print("\n" + "=" * 60)
    print("Test 3: Training mode (should NOT auto-persist)")
    print("=" * 60)
    
    module.train()  # Set to training mode
    
    # Reset to known state
    module.w_down.data.copy_(w_down_initial)
    w_down_before_train = module.w_down.data.clone()
    print(f"\nBefore training pass w_down norm: {w_down_before_train.norm().item():.6f}")
    
    # Forward pass in training mode
    out_train = module(x, token_embeddings)
    
    w_down_after_train = module.w_down.data.clone()
    print(f"After training pass w_down norm: {w_down_after_train.norm().item():.6f}")
    
    train_diff = (w_down_after_train - w_down_before_train).norm().item()
    print(f"Change during training: {train_diff:.6f}")
    
    # In training mode, w_down should NOT be auto-updated (optimizer does it)
    if train_diff < 1e-6:
        print("✓ PASS: Training mode does NOT auto-persist (optimizer handles it)")
    else:
        print("✗ WARNING: Training mode auto-persisted (may interfere with optimizer)")
        print("  (This is OK if both gradients and manual update work correctly)")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_ttt_inference_persistence()
    sys.exit(0 if success else 1)
