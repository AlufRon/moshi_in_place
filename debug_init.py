"""Debug script to check TTT initialization"""
import torch
from moshi.modules.ttt_module import TTTGating

# Create a TTTGating module with TTT enabled
ttt_config = {
    'enabled': True,
    'chunk_size': 256,
    'learning_rate': 1e-3,
    'conv_kernel_size': 2,
}

# Small dims for testing
dim = 128
dim_feedforward = 512

gating = TTTGating(
    activation=torch.nn.functional.silu,
    dim=dim,
    dim_feedforward=dim_feedforward,
    ttt_config=ttt_config,
    device='cpu',
    dtype=torch.float32,
)

print("TTTGating created")
print(f"linear_in shape: {gating.linear_in.weight.shape}")
print(f"linear_out shape: {gating.linear_out.weight.shape}")
print(f"w_down shape: {gating.w_down.shape}")
print(f"w_down is_meta: {gating.w_down.is_meta}")
print(f"linear_out.weight is_meta: {gating.linear_out.weight.is_meta}")

# Check if w_up exists
if hasattr(gating, 'w_up'):
    print(f"ERROR: w_up still exists! Shape: {gating.w_up.shape}")
else:
    print("✓ w_up removed correctly")

# Now simulate initialization
if gating.w_down.is_meta:
    print("\nSimulating initialization from linear_out.weight...")
    if hasattr(gating, 'linear_out') and hasattr(gating.linear_out, 'weight'):
        pretrained_weight = gating.linear_out.weight.data.transpose(0, 1)
        print(f"Copying weight shape {pretrained_weight.shape} to w_down")
        gating._parameters['w_down'] = torch.nn.Parameter(pretrained_weight.clone())
        print(f"✓ w_down initialized from pretrained weight")
        print(f"w_down is_meta after init: {gating.w_down.is_meta}")
    else:
        print("ERROR: Cannot access linear_out.weight")
