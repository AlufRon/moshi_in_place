#!/usr/bin/env python3
"""Minimal TTT test - creates small model and verifies TTT layers."""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "moshi" / "moshi"))

from moshi.models.lm import LMModel

print("Creating small Moshi model with TTT...")

ttt_config = {
    'enabled': True,
    'layer_frequency': 2,
    'start_layer': 0,
    'chunk_size': 8,
    'learning_rate': 1e-3,
    'conv_kernel_size': 2
}

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

print(f"✓ Model created with {len(model.transformer.layers)} layers")

# Check TTT
ttt_layers = [i for i, layer in enumerate(model.transformer.layers) 
              if getattr(layer, 'use_ttt', False)]
print(f"✓ TTT layers: {ttt_layers}")

if len(ttt_layers) == 0:
    print("✗ NO TTT LAYERS CREATED!")
    sys.exit(1)

# Test forward
B, K, T = 1, 9, 16
codes = torch.randint(0, 50, (B, K, T))

with torch.no_grad():
    output = model.forward(codes)

print(f"✓ Forward OK: logits={output.logits.shape}, text_logits={output.text_logits.shape}")
print(f"\n✅ TTT SETUP WORKING!")
print(f"   {len(ttt_layers)} TTT layers created at indices {ttt_layers}")
