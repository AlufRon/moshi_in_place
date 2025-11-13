# Minimal Changes Needed for TTT Inference Support

**Date**: November 13, 2025  
**Purpose**: Document minimal code changes required to support TTT checkpoint loading and inference  
**Status**: Planning document for implementation

---

## ⚠️ CRITICAL INSIGHT (Updated After Paper Re-Review)

**Initial assumption (WRONG)**: "Sequential processing already works, just need state management"

**Corrected understanding (RIGHT)**: We need to implement **two separate TTT execution modes**:

1. **Training Mode (✅ Already Implemented)**:
   - Chunk-wise parallel updates
   - Processes full sequences in chunks (e.g., 256 tokens)
   - Optimized for GPU parallelism
   - Used in `_parallel_ttt_update()`

2. **Inference Mode (❌ NOT Implemented)**:
   - Sequential per-token updates
   - Processes tokens one at a time during generation
   - Follows paper's Equation 1: `W^(i) = W^(i-1) + η V̂^T[i] Z[i]`
   - Apply-then-update per token
   - **Currently missing from our code!**

**Why both are needed**:
- Paper's chunk-wise approach (lines 174-175) is for **training efficiency**
- Autoregressive generation is inherently sequential (one token at a time)
- Both follow same update equation, different execution strategies

**Impact**: Estimated implementation increased from ~58 lines to ~150 lines due to sequential update logic.

---

## Executive Summary

To enable inference with TTT-enhanced Moshi checkpoints, we need **5 categories of changes**:

1. **Add sequential per-token TTT update** for autoregressive generation (~70 lines)
2. **Add streaming mode control** to switch between training/inference (~35 lines)
3. **Modify checkpoint loader** to handle TTT parameters and config (~20 lines)
4. **Hook streaming mode** into LMGen and reset callbacks (~10 lines)
5. **Add CLI argument** to run_inference.py for TTT config (~15 lines)

Total estimated lines of code: **~150 lines** across 5 files.

---

## Paper Requirements for Inference

### 1. Sequential Token-by-Token Processing for Generation ⚠️
**Paper context**: 
- Lines 162-163: "the canonical per-token update mechanism of TTT is inherently sequential"
- Lines 174-175: "We replace the inefficient per-token updates with a scalable chunk-wise update rule"
- **KEY INSIGHT**: Chunk-wise updates are for **training efficiency only**

**For Inference/Generation**:
- Autoregressive generation is inherently sequential (generate one token at a time)
- Per-token TTT updates are natural for streaming inference
- The paper's chunk-wise approach is for parallel **training**, not sequential **generation**

**Current Implementation Gap**: 
- ✅ Moshi uses `streaming_forever()` mode (sequential token processing)
- ❌ Our TTT module only implements chunk-wise updates (training mode)
- ❌ No per-token sequential update path for streaming inference

**Required**: Implement sequential per-token TTT update for inference mode

---

### 2. Fast Weight State Management
**Paper requirement**: Fast weights (w_down) must:
- Start from pretrained values at conversation start
- Accumulate updates during conversation
- Reset to pretrained values at conversation boundaries

**Current Gap**: No TTT state tracking in streaming mode

**Required Changes**:

#### Change 1: Add sequential inference mode to TTTGating
**File**: `moshi/moshi/moshi/modules/ttt_module.py`

```python
# Modify around line 95-120

class TTTGating(ActivationGating):
    def __init__(self, ...):
        # ... existing code ...
        
        if self.ttt_enabled:
            self.w_down = nn.Parameter(torch.empty(dim, hidden, device='meta'))
            
            # NEW: Store pretrained weights for reset
            self.register_buffer('w_down_pretrained', torch.empty(dim, hidden))
            
            # NEW: Track if we're in streaming inference mode
            self._streaming_mode = False
            self._w_down_current = None  # Current state during streaming
    
    # NEW: Enable streaming mode
    def enable_streaming_mode(self):
        """Switch to sequential per-token update mode for inference."""
        if self.ttt_enabled:
            self._streaming_mode = True
            # Initialize current state from pretrained
            self._w_down_current = self.w_down_pretrained.clone()
    
    # NEW: Disable streaming mode (back to training)
    def disable_streaming_mode(self):
        """Switch back to chunk-wise parallel mode for training."""
        self._streaming_mode = False
        self._w_down_current = None
    
    # NEW: Reset to pretrained state
    def reset_ttt_state(self):
        """Reset w_down to pretrained state (e.g., at conversation boundaries)."""
        if self.ttt_enabled and hasattr(self, 'w_down_pretrained'):
            if self._streaming_mode and self._w_down_current is not None:
                self._w_down_current.copy_(self.w_down_pretrained)
            else:
                # Training mode: reset parameter directly
                with torch.no_grad():
                    self.w_down.copy_(self.w_down_pretrained)
    
    # MODIFIED: Update forward to handle streaming vs training
    def _ttt_forward(self, x: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        # compute Z
        h = self.linear_in(x)
        h = h.view(B, T, 2, -1)
        Z = self.activation(h[..., 0, :]) * h[..., 1, :]  # [B, T, hidden]

        # V_hat from token embeddings
        V_hat = self.target_generator(token_embeddings)  # [B, T, dim]

        # Choose update path based on mode
        if self._streaming_mode:
            # Sequential per-token update for inference
            return self._sequential_ttt_update(Z, V_hat)
        else:
            # Parallel chunk-wise update for training
            return self._parallel_ttt_update(Z, V_hat)
    
    # NEW: Sequential update for streaming inference
    def _sequential_ttt_update(self, Z: torch.Tensor, V_hat: torch.Tensor) -> torch.Tensor:
        """
        Sequential per-token TTT update for autoregressive generation.
        Processes tokens one at a time, updating w_down after each token.
        
        Z: [B, T, hidden] - typically T=1 during streaming
        V_hat: [B, T, dim] - target values
        """
        B, T, hidden = Z.shape
        dim = V_hat.shape[-1]
        
        # Use current streaming state
        W_current = self._w_down_current  # [dim, hidden]
        
        outputs = []
        for t in range(T):
            # 1. Apply: O_t = Z_t @ W_current^T
            z_t = Z[:, t, :]  # [B, hidden]
            o_t = torch.matmul(z_t, W_current.t())  # [B, dim]
            outputs.append(o_t)
            
            # 2. Update: W_current = W_current + η * V_hat_t^T @ Z_t
            v_t = V_hat[:, t, :]  # [B, dim]
            # Average over batch dimension for update
            delta = torch.einsum('bd,bh->dh', v_t, z_t) / B  # [dim, hidden]
            W_current = W_current + self.ttt_lr * delta
        
        # Update streaming state for next call
        self._w_down_current = W_current
        
        # Stack outputs
        O = torch.stack(outputs, dim=1)  # [B, T, dim]
        return O
```

**Lines added**: ~70 lines (including streaming mode support)

**Key changes**:
- Add streaming mode flag and current state tracking
- Implement sequential per-token update (Equation 1 from paper)
- Apply-then-update per token (not per chunk)
- Maintain w_down_current separately from parameter during inference

---

#### Change 2: Enable streaming mode in LMModel
**File**: `moshi/moshi/moshi/models/lm.py`

```python
# Add around line 180 (in __init__ after transformer creation)

def __init__(self, ...):
    # ... existing code ...
    
    self.transformer = StreamingTransformer(...)
    
    # NEW: Setup TTT streaming mode control
    if ttt_config is not None and ttt_config.get('enabled', False):
        self._setup_ttt_callbacks()
        self.ttt_layers = self._get_ttt_layers()

# NEW: Get all TTT layers
def _get_ttt_layers(self):
    """Collect all TTT-enabled layers."""
    ttt_layers = []
    for module in self.transformer.modules():
        if hasattr(module, 'reset_ttt_state') and hasattr(module, 'enable_streaming_mode'):
            ttt_layers.append(module)
    return ttt_layers

# NEW: Setup TTT callbacks
def _setup_ttt_callbacks(self):
    """Register callbacks for TTT state management."""
    # This will be called by LMGen when entering/exiting streaming mode
    pass

# NEW: Enable TTT streaming mode (called when entering inference)
def enable_ttt_streaming(self):
    """Enable sequential per-token TTT updates for inference."""
    if hasattr(self, 'ttt_layers'):
        for layer in self.ttt_layers:
            layer.enable_streaming_mode()

# NEW: Disable TTT streaming mode (called when back to training)
def disable_ttt_streaming(self):
    """Disable streaming mode, back to chunk-wise parallel updates."""
    if hasattr(self, 'ttt_layers'):
        for layer in self.ttt_layers:
            layer.disable_streaming_mode()

# NEW: Reset TTT state
def reset_ttt_state(self):
    """Reset TTT fast weights to pretrained state."""
    if hasattr(self, 'ttt_layers'):
        for layer in self.ttt_layers:
            layer.reset_ttt_state()
```

**Lines added**: ~35 lines

---

### 3. Checkpoint Loading with TTT Parameters

**Current Gap**: `loaders.py` doesn't know about TTT config or w_down_pretrained buffer

**Required Changes**:

#### Change 3: Extend CheckpointInfo to handle TTT
**File**: `moshi/moshi/moshi/models/loaders.py`

```python
# Modify around line 165 (CheckpointInfo dataclass)

@dataclass
class CheckpointInfo:
    moshi_weights: Path
    mimi_weights: Path
    tokenizer: Path
    lm_config: dict | None = None
    raw_config: dict | None = None
    mimi_config: dict | None = None
    model_type: str = "moshi"
    lora_weights: Path | None = None
    lm_gen_config: dict = field(default_factory=dict)
    tts_config: dict = field(default_factory=dict)
    stt_config: dict = field(default_factory=dict)
    model_id: dict = field(default_factory=dict)
    ttt_config: dict = field(default_factory=dict)  # NEW: TTT configuration

# Modify from_hf_repo() around line 230

@staticmethod
def from_hf_repo(...) -> "CheckpointInfo":
    # ... existing config loading ...
    
    if config_path is None:
        # ... existing defaults ...
        ttt_config = {}  # NEW
    else:
        raw_config = json.loads(Path(config_path).read_text())
        lm_config = dict(raw_config)
        # ... existing pops ...
        ttt_config = lm_config.pop("ttt_config", {})  # NEW: Extract TTT config
    
    return CheckpointInfo(
        # ... existing parameters ...
        ttt_config=ttt_config,  # NEW
    )

# Modify get_moshi() around line 290

def get_moshi(self, device, dtype=torch.bfloat16, **kwargs) -> LMModel:
    return get_moshi_lm(
        self.moshi_weights,
        lm_kwargs=self.lm_config,
        device=device,
        dtype=dtype,
        lora_weights=self.lora_weights,
        ttt_config=self.ttt_config,  # NEW: Pass TTT config
        **kwargs,
    )
```

**Lines added**: ~10 lines

---

#### Change 4: Handle TTT in get_moshi_lm()
**File**: `moshi/moshi/moshi/models/loaders.py`

```python
# Modify around line 365 (get_moshi_lm function signature and body)

def get_moshi_lm(
    filename: str | Path | None,
    lm_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    lora_weights: str | Path | None = None,
    fuse_lora: bool = False,
    lm_kwargs_overrides={},
    ttt_config: dict | None = None,  # NEW: TTT configuration
) -> LMModel:
    # ... existing code ...
    
    # NEW: Add TTT config to lm_kwargs if provided
    if ttt_config is not None and ttt_config.get('enabled', False):
        lm_kwargs['ttt_config'] = ttt_config
    
    model = LMModel(
        device=init_device,
        dtype=dtype,
        **lm_kwargs)

    if filename is not None:
        if _is_safetensors(filename):
            state = load_file(filename, device=str(device))
            # ... existing dtype conversion ...
            model.load_state_dict(state, assign=True)
            
            # NEW: Initialize w_down_pretrained buffers after loading
            if ttt_config is not None and ttt_config.get('enabled', False):
                for name, module in model.named_modules():
                    if hasattr(module, 'w_down') and hasattr(module, 'w_down_pretrained'):
                        module.w_down_pretrained.copy_(module.w_down.data)
```

**Lines added**: ~12 lines

---

### 4. CLI Support for TTT Inference

**Required Changes**:

#### Change 5: Hook TTT streaming into LMGen and run_inference.py
**File**: `moshi/moshi/moshi/models/lm.py` (LMGen class)

```python
# Around line 560 in LMGen class

class LMGen(StreamingContainer):
    # ... existing code ...
    
    # MODIFY: streaming_forever to enable TTT streaming mode
    def streaming_forever(self, batch_size: int):
        """Enter streaming mode for inference."""
        super().streaming_forever(batch_size)
        
        # NEW: Enable TTT streaming mode
        if hasattr(self.lm_model, 'enable_ttt_streaming'):
            self.lm_model.enable_ttt_streaming()
    
    # NEW: Override reset to include TTT reset
    def reset(self, reset_mask: torch.Tensor) -> None:
        super().reset(reset_mask)
        
        # Reset TTT state for conversations
        if hasattr(self.lm_model, 'reset_ttt_state'):
            self.lm_model.reset_ttt_state()
```

**File**: `moshi/moshi/moshi/run_inference.py`

```python
# Add around line 260 (in argument parser)

def main():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...
    parser.add_argument(
        "--config",
        "--lm-config",
        dest="config",
        type=str,
        help="The config as a json file.",
    )
    # NEW: TTT support
    parser.add_argument(
        "--ttt-config",
        type=str,
        help="Path to TTT configuration JSON file. If the checkpoint was trained with TTT, "
             "this should match the training configuration.",
    )
    parser.add_argument("--cfg-coef", type=float, default=1.0, help="CFG coefficient.")
    # ... rest of arguments ...
    
    args = parser.parse_args()
    seed_all(4242)

    log("info", "retrieving checkpoint")
    
    # NEW: Load TTT config if provided
    ttt_config = None
    if args.ttt_config:
        import json
        ttt_config = json.loads(Path(args.ttt_config).read_text())
        log("info", f"loaded TTT config: {ttt_config}")
    
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, args.config
    )
    
    # NEW: Override TTT config if provided via CLI
    if ttt_config is not None:
        checkpoint_info.ttt_config = ttt_config
```

**Lines added**: ~30 lines total

---

## Summary of Changes

| File | Change | Lines | Complexity |
|------|--------|-------|------------|
| `ttt_module.py` | Add streaming mode + sequential updates | ~70 | **Medium** |
| `lm.py` (LMModel) | Add streaming mode control | ~35 | Simple |
| `lm.py` (LMGen) | Hook streaming + reset | ~10 | Simple |
| `loaders.py` (CheckpointInfo) | Add TTT config field | ~5 | Trivial |
| `loaders.py` (from_hf_repo) | Extract TTT config | ~3 | Trivial |
| `loaders.py` (get_moshi_lm) | Pass TTT config, init buffers | ~12 | Simple |
| `run_inference.py` | Add CLI argument | ~15 | Simple |
| **TOTAL** | | **~150 lines** | **Medium** |

**Key Complexity Increase**: Sequential per-token update implementation in TTTGating (~40 lines of new logic)

---

## Key Design Decisions

### 1. **Leverage Existing Reset Mechanism**
- Moshi already has `reset_callback` in `LMGen` (line 541)
- We hook into this rather than creating new infrastructure
- Minimal invasiveness to existing code

### 2. **Store Pretrained Weights as Buffer**
- Use `register_buffer()` for `w_down_pretrained`
- Automatically saved/loaded with checkpoint
- Zero overhead during forward pass

### 3. **Backward Compatibility**
- All changes are additive (no breaking changes)
- If `ttt_config` is None or disabled, code paths unchanged
- Existing checkpoints work without modification

### 4. **Configuration via JSON**
- TTT config stored in checkpoint's `config.json`
- Can be overridden via CLI for experimentation
- Same format as training config

---

## What We Don't Need to Change

### ✅ **Parallel Chunk-wise Training**
- Already implemented in `_parallel_ttt_update()`
- Works correctly for training
- No changes needed to training code path

### ✅ **Token Embeddings Flow**
- Already passing through all layers
- Verified in `IMPLEMENTATION_VERIFICATION.md`
- No changes needed

### ✅ **Checkpoint Format**
- FSDP checkpoints already contain `w_down` parameters
- Standard safetensors loading works
- Just need to initialize `w_down_pretrained` buffer after loading

### ❌ **What We Were Wrong About**
**Previous assumption**: "Sequential processing already works, no changes needed"

**Reality**: Our implementation only has chunk-wise parallel updates (for training). For autoregressive **inference**, we need sequential per-token updates because:
1. Generation produces one token at a time (can't batch into chunks)
2. Each new token needs the updated w_down from previous token
3. This is the "canonical per-token update" the paper mentions (line 162)
4. Chunk-wise is optimization for parallel **training**, not for sequential **generation**

---

## Testing Plan

### Test 1: Load TTT Checkpoint
```bash
python -m moshi.run_inference \
    --moshi-weight /path/to/ttt_checkpoint.safetensors \
    --ttt-config /path/to/ttt_config.json \
    input.wav output.wav
```

**Expected**: 
- Model loads without errors
- w_down_pretrained initialized from checkpoint
- Streaming mode enabled automatically
- Sequential updates active during generation

### Test 2: Verify Sequential Updates
- Add debug logging to `_sequential_ttt_update()`
- Run inference on short input
- Check logs show:
  - Token-by-token processing (T=1 typically)
  - w_down_current updating after each token
  - Apply-then-update order preserved

### Test 3: Verify TTT State Reset
- Add debug logging to `reset_ttt_state()`
- Simulate conversation boundary
- Verify w_down_current resets to w_down_pretrained

### Test 4: Compare Training vs Inference Updates
- Create test with known input sequence
- Run same sequence through:
  1. Training mode (chunk-wise, batch processing)
  2. Inference mode (sequential, token-by-token)
- Outputs should be **identical** (both follow Equation 1)
- This validates mathematical equivalence

### Test 5: Long Conversation
- Run extended conversation (many tokens)
- Monitor w_down_current drift from pretrained
- Verify model maintains coherence
- Test reset functionality mid-conversation

---

## Example TTT Config JSON

For use with `--ttt-config`:

```json
{
    "enabled": true,
    "layer_indices": [10, 20, 30],
    "chunk_size": 256,
    "ttt_lr": 0.001,
    "conv_kernel": 2
}
```

This would be saved alongside the checkpoint or passed explicitly.

---

## Implementation Priority

1. **First**: Sequential per-token update in ttt_module.py
   - Most critical: enables inference to work at all
   - Implement `_sequential_ttt_update()`
   - Add streaming mode flag
   - Tests: Verify update equation matches paper

2. **Second**: Streaming mode control in lm.py
   - Enables switching between training/inference modes
   - Add enable/disable streaming methods
   - Tests: Mode switching works correctly

3. **Third**: Checkpoint loading in loaders.py
   - Enables loading TTT checkpoints
   - Initialize w_down_pretrained buffer
   - Tests: Can load without errors

4. **Fourth**: LMGen integration
   - Hook streaming mode into LMGen.streaming_forever()
   - Add reset callback
   - Tests: Streaming activates automatically

5. **Fifth**: CLI in run_inference.py
   - Enables user-facing functionality
   - Tests: End-to-end inference works

**Critical path**: Steps 1-2 are essential. Without sequential updates, inference cannot work.

---

## Potential Issues and Mitigations

### Issue 1: When to Reset?
**Problem**: Paper says "at document boundaries" but we have continuous conversation

**Solutions**:
- Option A: Never reset (let w_down accumulate indefinitely)
  - Pro: Simplest, matches paper's long-context goal
  - Con: May drift too far from pretrained
  
- Option B: Reset on explicit user signal
  - Pro: User controls conversation boundaries
  - Con: Requires UI/API changes
  
- Option C: Reset when context window is cleared
  - Pro: Automatic, aligns with attention KV cache
  - Con: May reset too frequently

**Recommendation**: Start with Option A for inference, add Option B later if needed

### Issue 2: Batch Size > 1
**Problem**: Different items in batch may need different reset times

**Solution**: 
- `reset_mask` is already a tensor (line 544)
- Can reset individual batch items independently
- Already supported in existing infrastructure

### Issue 3: Memory for w_down_pretrained
**Problem**: Doubles memory for TTT parameters

**Analysis**:
- w_down: ~4096 * 16896 * 4 bytes = 277 MB per layer
- w_down_pretrained: same size
- 3 TTT layers → ~1.7 GB total overhead
- Acceptable for inference on GPU

**Mitigation**: For memory-constrained scenarios, could reload from checkpoint on reset

---

## Conclusion

**Updated assessment**: ~150 lines across 5 files (not ~58 as initially estimated)

**Critical realization**: We need to implement sequential per-token TTT updates for inference. The paper's chunk-wise approach is a **training optimization** to enable parallelism, but autoregressive generation is inherently sequential.

**Key insight**: 
- **Training**: Chunk-wise parallel updates (already implemented ✅)
- **Inference**: Sequential per-token updates (needs implementation ❌)

The sequential update follows the paper's Equation 1:
```
W_down^(i) = W_down^(i-1) + η V̂^T[i] Z[i]
```

For each token during generation:
1. **Apply**: `O_t = Z_t @ W_current^T` (use current weights)
2. **Update**: `W_current = W_current + η * V̂_t^T @ Z_t` (update for next token)

**Next steps**:
1. Implement streaming mode flag in TTTGating
2. Implement `_sequential_ttt_update()` method
3. Add mode switching in LMModel and LMGen
4. Test with TTT checkpoint
5. Validate sequential updates match paper's formulation

The implementation is more involved than initially thought because we need **two separate code paths**:
- ✅ Parallel chunk-wise (training) - already have this
- ❌ Sequential per-token (inference) - need to implement this

Both follow the same update equation, just different execution strategies!
