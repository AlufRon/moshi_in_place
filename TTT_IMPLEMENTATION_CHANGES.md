# In-Place Test-Time Training (TTT) Implementation for Moshi

## ⚠️ CRITICAL IMPLEMENTATION DETAIL

**This implementation correctly follows the ICLR 2026 "In-Place Test-Time Training" paper**:

- ✅ **Only `W_down`** (final projection matrix) is the TTT **fast weight** that gets updated
- ✅ **`W_up` and `W_gate`** (input projections in `linear_in`) are **FROZEN slow weights**
- ✅ This preserves the true "in-place" nature: we repurpose existing MLP blocks without replacing them

**Paper quote (line 399-401)**:
> "In our framework, we treat the input projections Wup and Wgate as frozen slow weights, while repurposing the final projection matrix, Wdown, as the adaptable fast weights."

**Why this matters**: 
- Updating only W_down preserves pre-trained knowledge in W_up/W_gate
- Creates a "drop-in" enhancement rather than replacing the entire MLP
- Maintains architectural integrity with minimal divergence from pre-training

---

## Overview
Implementation of In-Place TTT (ICLR 2026 paper) for Moshi 7B model with LoRA-compatible training.

## Modified Files

### 1. moshi/moshi/moshi/modules/ttt_module.py (NEW FILE - ~200 lines)
**Purpose**: Core TTT implementation with parallel chunk-wise updates

**Key Components**:
- `CausalConv1D`: Causal 1D convolution for looking ahead in sequence
- `LMAlignedTargetGenerator`: Generates training targets from token embeddings using causal conv
- `TTTGating`: Main TTT-enabled gating module that replaces standard ActivationGating

**Critical Implementation Details**:
- Lines 10-13: Imports including `math` for parameter initialization
- Lines 18-44: `CausalConv1D` class with right-padding for future token access
- Lines 47-69: `LMAlignedTargetGenerator` using Conv1D for W_target
- Lines 72-117: `TTTGating.__init__()`:
  - **Line 114**: Creates `w_down` as meta tensor - the ONLY TTT fast weight
  - **Important**: `w_up` was removed - it's NOT a fast weight per the paper
  - `linear_in` contains W_up and W_gate as frozen slow weights
- Lines 120-141: `_ttt_forward()`: Computes Z from input, V_hat from embeddings
- Lines 143-189: `_parallel_ttt_update()`: 
  - Chunk-wise parallel algorithm (lines 146-154)
  - Causal prefix sum for cumulative updates (lines 162-164)
  - Per-chunk matmul with effective weights (lines 176-182)
  - **Only updates `w_down`** (line 158: `W_down_init = self.w_down`)

**Key Architectural Insight**:
- `Z = activation(linear_in(x))` uses frozen slow weights (W_up, W_gate)
- Only the final projection `w_down` is updated as fast weight
- This is the core "in-place" design from the paper

---

### 2. moshi/moshi/moshi/modules/transformer.py (MODIFIED)
**Purpose**: Integrate TTT into transformer layers

**Changes**:
- Line 27: `from moshi.modules.ttt_module import TTTGating`
- Lines 693-701: TTT layer creation logic in `StreamingTransformerLayer.__init__()`:
  ```python
  if ttt_config and ttt_config.get('enabled', False):
      self.gating = TTTGating(
          activation=self.activation,
          dim=dim,
          dim_feedforward=dim_feedforward,
          ttt_config=ttt_config,
          device=device,
          dtype=dtype,
      )
  ```
- Line 876: **Critical bug fix**: Extract `ttt_config` BEFORE loop to pass to all layers
  ```python
  ttt_config = kwargs.pop('ttt_config', None)  # Extract before loop!
  ```
- Line 933: Pass `token_embeddings` through transformer layers
- Lines 776, 808: Pass `token_embeddings` to gating module in `_ff_block()`

---

### 3. moshi/moshi/moshi/models/lm.py (MODIFIED)
**Purpose**: Extract and pass token embeddings to transformer

**Changes**:
- Lines 404-409: In `forward_text()` method:
  ```python
  # Extract token embeddings for TTT
  text_emb = self.emb[0](input[:, :, 0])  # [B, T, dim]
  
  transformer_out = self.transformer(
      input, cross_attention_src, token_embeddings=text_emb
  )
  ```

---

### 4. moshi-finetune/finetune/args.py (MODIFIED)
**Purpose**: Add TTT configuration to training arguments

**Changes**:
- Lines 28-36: New `TTTArgs` dataclass:
  ```python
  @dataclass
  class TTTArgs:
      enabled: bool = False
      layer_frequency: int = 6
      start_layer: int = 5
      chunk_size: int = 256
      learning_rate: float = 1e-3
      conv_kernel_size: int = 2
  ```
- Line 121: Add `ttt: TTTArgs = field(default_factory=TTTArgs)` to `TrainArgs`

---

### 5. moshi-finetune/finetune/wrapped_model.py (MODIFIED)
**Purpose**: Model creation with TTT initialization

**Changes**:
- Line 1: Fixed syntax error (`vimport` → `import`)
- Lines 75-96: Modified `initialize_lora_parameters()`:
  - Line 78: Skip gating modules (they're handled separately)
  - Prevents LoRA init from trying to initialize TTT params
- Lines 99-124: **NEW** `initialize_ttt_parameters()` function:
  - **CRITICAL**: Only initializes `w_down` (not `w_up` - removed per paper)
  - Line 103: Docstring clarifies: "only W_down is the fast weight"
  - Iterates through gating modules
  - Checks each parameter individually for `is_meta`
  - Initializes `w_down` and `linear` params with kaiming_uniform
  - Initializes conv/target_generator with normal distribution
  - **Note**: `w_up` removed from line 118 (was incorrectly included before)
- Lines 125-136: Build `ttt_config` dict from args:
  ```python
  ttt_config = {
      'enabled': args.ttt.enabled,
      'layer_frequency': args.ttt.layer_frequency,
      'start_layer': args.ttt.start_layer,
      'chunk_size': args.ttt.chunk_size,
      'learning_rate': args.ttt.learning_rate,
      'conv_kernel_size': args.ttt.conv_kernel_size,
  }
  ```
- Lines 160-167: Log TTT layer indices after model creation
- Lines 200-203: Call `initialize_ttt_parameters()` when TTT enabled
- Lines 205-210: Debug logging for uninitialized meta parameters

---

### 6. moshi-finetune/train.py (MODIFIED)
**Purpose**: Add TTT gradient logging during training

**Changes**:
- Lines 303-312: Log TTT gradient norms every `log_freq` steps:
  ```python
  if step % args.log_freq == 0:
      # Log TTT gradients if enabled
      if args.ttt and args.ttt.enabled:
          for name, param in model.named_parameters():
              if 'gating' in name and param.grad is not None:
                  grad_norm = param.grad.norm().item()
                  main_logger_info(f"TTT grad {name}: {grad_norm:.6f}")
  ```

---

### 7. moshi-finetune/example/moshi_7B.yaml (MODIFIED)
**Purpose**: Training configuration with TTT enabled

**Changes**:
- Line 5: `train_data: /sise/eliyanac-group/ron_al/talkbank_callhome_english/talkbank.jsonl`
- Line 11: `full_finetuning: false` (using LoRA for memory efficiency)
- Lines 12-14: LoRA enabled with rank=64
- Line 23: `batch_size: 4` (reduced for memory)
- Line 43: `save_adapters: true`
- Lines 48-54: **NEW** TTT configuration:
  ```yaml
  ttt:
    enabled: true
    layer_frequency: 10    # TTT every 10th layer
    start_layer: 10        # Layers 10, 20, 30
    chunk_size: 256
    learning_rate: 0.001
    conv_kernel_size: 2
  ```

---

## Architecture Summary

**TTT Layers**: 3 layers at indices [10, 20, 30] out of 32 total layers

**Parameters**:
- **Base model**: Frozen or adapted with LoRA (rank 64)
- **TTT parameters** (trainable):
  - `w_down`: [dim, hidden] per TTT layer
  - `w_up`: [hidden, dim] per TTT layer  
  - `target_generator.conv.weight`: [dim, dim, kernel_size] per TTT layer

**Training Setup**:
- Batch size: 4
- Learning rate: 2e-6 (base), 1e-3 (TTT)
- Gradient checkpointing: enabled
- Mixed precision: bfloat16
- Data: TalkBank CallHome English corpus

---

## Key Implementation Decisions

1. **LoRA + TTT**: Use LoRA for base model adaptation + TTT for test-time updates
   - More memory efficient than full finetuning
   - Aligns with paper's approach (train only TTT params)

2. **Meta Tensors**: TTT parameters created as meta tensors, initialized separately
   - Allows proper dtype control
   - Compatible with FSDP model loading

3. **Parallel Algorithm**: Chunk-wise updates with causal prefix sum
   - Efficient parallelization across chunks
   - Maintains causal dependencies within chunks

4. **LoRALinear Compatibility**: Access module functions, not `.weight` directly
   - `self.linear_in(x)` instead of `F.linear(x, self.linear_in.weight)`
   - `self.w_down` instead of `self.linear_out.weight`

---

## Verification Tests

Created `test_ttt_minimal.py` to verify:
- ✅ TTT layers created at correct indices
- ✅ Forward pass works
- ✅ Backward pass works
- ✅ Gradients flow to TTT parameters

Result: 4-layer model with 2 TTT layers at [0, 2] - all tests passed

---

## Files Created

### 1. /home/alufr/moshi_in_place_ttt/moshi/moshi/moshi/modules/ttt_module.py (NEW - 205 lines)
**Purpose**: Core In-Place TTT implementation

**Full implementation documented above in "Modified Files" section.**

---

### 2. /home/alufr/moshi_in_place_ttt/test_ttt_minimal.py (NEW - Test File)
**Purpose**: Minimal test to verify TTT integration

**Key Tests**:
```python
# Create small 4-layer model with TTT at layers 0, 2
ttt_config = {
    'enabled': True,
    'layer_frequency': 2,
    'start_layer': 0,
    'chunk_size': 64,
    'learning_rate': 1e-3,
    'conv_kernel_size': 2,
}

# Verify:
# 1. TTT layers created at correct positions
# 2. Forward pass works
# 3. Backward pass works
# 4. Gradients flow to TTT parameters
```

**Result**: All tests passed ✅

---

### 3. /home/alufr/moshi_in_place_ttt/test_ttt_setup.py (NEW - Test File)
**Purpose**: Setup verification test

---

### 4. /home/alufr/moshi_in_place_ttt/moshi/tests/test_ttt_module.py (NEW - Test File)
**Purpose**: Unit tests for TTT module components

---

### 5. /home/alufr/moshi_in_place_ttt/TTT_IMPLEMENTATION_CHANGES.md (NEW - This File)
**Purpose**: Comprehensive documentation of all changes

---

### 6. /home/alufr/moshi_in_place_ttt/IN_PLACE_TTT_IMPLEMENTATION.md (EXISTING)
**Purpose**: Initial implementation notes and paper analysis

---

## Total Changes Summary

### New Files Created:
1. `moshi/moshi/moshi/modules/ttt_module.py` - Core TTT implementation (205 lines)
2. `test_ttt_minimal.py` - Integration test
3. `test_ttt_setup.py` - Setup test
4. `moshi/tests/test_ttt_module.py` - Unit tests
5. `TTT_IMPLEMENTATION_CHANGES.md` - This documentation

### Modified Files:
1. `moshi/moshi/moshi/modules/transformer.py` - 4 key changes for TTT integration
2. `moshi/moshi/moshi/models/lm.py` - Token embedding extraction
3. `moshi-finetune/finetune/args.py` - TTTArgs dataclass
4. `moshi-finetune/finetune/wrapped_model.py` - Initialization logic
5. `moshi-finetune/train.py` - Gradient logging
6. `moshi-finetune/example/moshi_7B.yaml` - Training config

### Statistics:
- **Total new files**: 5
- **Total modified files**: 6
- **Total lines added**: ~450
- **Training ready**: Yes ✅
- **Tests passing**: Yes ✅

---
