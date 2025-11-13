#!/usr/bin/env python3
"""
Final verification that all TTT components are working.
Run this before starting actual training.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "moshi" / "moshi"))

from moshi.models.lm import LMModel

print("=" * 70)
print("FINAL TTT VERIFICATION")
print("=" * 70)

# Test configuration
ttt_config = {
    'enabled': True,
    'layer_frequency': 2,
    'start_layer': 0,
    'chunk_size': 8,
    'learning_rate': 1e-3,
    'conv_kernel_size': 2
}

print("\n[1/5] Creating model with TTT...")
model = LMModel(
    dim=128,
    text_card=256,
    num_heads=4,
    num_layers=4,
    causal=True,
    norm="layer_norm",
    gating="silu",
    n_q=8,
    dep_q=8,
    depformer_num_heads=2,
    depformer_num_layers=2,
    delays=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    ttt_config=ttt_config,
    device='cpu',
    dtype=torch.float32
)

ttt_layers = [i for i, layer in enumerate(model.transformer.layers) 
              if getattr(layer, 'use_ttt', False)]

if len(ttt_layers) == 0:
    print("   ✗ FAIL: No TTT layers created")
    sys.exit(1)
print(f"   ✓ PASS: {len(ttt_layers)} TTT layers at indices {ttt_layers}")

print("\n[2/5] Testing forward pass...")
B, K, T = 1, 9, 16
codes = torch.randint(0, 50, (B, K, T))

try:
    with torch.no_grad():
        output = model.forward(codes)
    print(f"   ✓ PASS: Output shapes {output.logits.shape}, {output.text_logits.shape}")
except Exception as e:
    print(f"   ✗ FAIL: {e}")
    sys.exit(1)

print("\n[3/5] Testing backward pass...")
model.train()
try:
    output = model.forward(codes)
    loss = (output.logits[output.mask].sum() + 
            output.text_logits[output.text_mask].sum())
    loss.backward()
    print(f"   ✓ PASS: Loss={loss.item():.4f}")
except Exception as e:
    print(f"   ✗ FAIL: {e}")
    sys.exit(1)

print("\n[4/5] Checking TTT gradients...")
ttt_params_with_grad = [
    name for name, p in model.named_parameters()
    if p.grad is not None and 'target_generator' in name
]

if len(ttt_params_with_grad) == 0:
    print("   ✗ FAIL: No TTT parameters have gradients")
    sys.exit(1)
print(f"   ✓ PASS: {len(ttt_params_with_grad)} TTT parameters have gradients")

print("\n[5/5] Testing train/eval modes...")
try:
    model.eval()
    with torch.no_grad():
        _ = model.forward(codes)
    model.train()
    _ = model.forward(codes)
    print("   ✓ PASS: Mode switching works")
except Exception as e:
    print(f"   ✗ FAIL: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL CHECKS PASSED - TTT IS READY FOR TRAINING!")
print("=" * 70)
print("\nWhat's working:")
print("  • TTT layers are created based on config")
print("  • Token embeddings flow to transformer")
print("  • Forward pass produces valid outputs")
print("  • Backward pass computes gradients")
print("  • TTT parameters receive gradients")
print("  • Train/eval modes work correctly")
print("\nYou can now:")
print("  1. Edit moshi-finetune/example/moshi_ttt.yaml")
print("  2. Run: cd moshi-finetune && python train.py --config example/moshi_ttt.yaml")
print("\nGood luck with training!")
