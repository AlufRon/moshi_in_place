#!/usr/bin/env python3
"""Quick test to verify TTT buffer flushing works correctly."""

import torch
from moshi.moshi.moshi.modules.ttt_module import TTTGating

# Create a TTT gating module
ttt_config = {
    'enabled': True,
    'chunk_size': 256,
    'learning_rate': 1e-3,
    'conv_kernel_size': 2,
}

d_model = 512
hidden = 2048
gating = TTTGating(
    torch.nn.SiLU(),
    d_model,
    hidden,
    ttt_config,
    device='cpu',
    dtype=torch.float32,
)

# Set to eval mode to trigger inference path
gating.eval()

# Initialize w_down_pretrained for proper setup
gating.w_down_pretrained.copy_(gating.w_down.data)

# Store initial norm
initial_norm = torch.norm(gating.w_down.data).item()
print(f"Initial w_down norm: {initial_norm:.6f}")

# Simulate streaming inference with small batches (< chunk_size)
# This mimics what happens during actual inference where tokens are processed one at a time
batch_size = 1
seq_len = 100  # Less than chunk_size=256
num_batches = 3  # Total 300 tokens across batches

update_count_before = getattr(gating, '_update_count', 0)
print(f"Initial update count: {update_count_before}")

# Process tokens in small batches
for i in range(num_batches):
    x = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
    token_embeddings = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)

    # Forward pass (should buffer but not update yet)
    with torch.no_grad():
        _ = gating(x, token_embeddings)

    if hasattr(gating, '_inference_z_buffer') and gating._inference_z_buffer is not None:
        buffer_len = gating._inference_z_buffer.shape[1]
        print(f"After batch {i+1}: buffer length = {buffer_len}")
    else:
        print(f"After batch {i+1}: no buffer yet")

update_count_mid = getattr(gating, '_update_count', 0)
print(f"Update count after {num_batches} batches: {update_count_mid}")

# Now flush the buffers
print("\nFlushing TTT buffers...")
gating.flush_ttt_buffers()

update_count_after = getattr(gating, '_update_count', 0)
final_norm = torch.norm(gating.w_down.data).item()
norm_change = final_norm - initial_norm

print(f"Update count after flush: {update_count_after}")
print(f"Final w_down norm: {final_norm:.6f}")
print(f"Norm change: {norm_change:.6f} ({100*norm_change/initial_norm:.4f}%)")

# Verify buffers are cleared
if hasattr(gating, '_inference_z_buffer'):
    if gating._inference_z_buffer is None:
        print("✓ Buffers cleared successfully")
    else:
        print(f"✗ Buffers not cleared! Remaining: {gating._inference_z_buffer.shape[1]} tokens")

if update_count_after > update_count_before:
    print(f"\n✓ SUCCESS: TTT updates applied ({update_count_after - update_count_before} updates)")
else:
    print(f"\n✗ FAILED: No TTT updates applied")
