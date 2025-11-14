# TTT Buffer Flush Fix

## Problem

TTT weights were not updating during inference (updates: 0, change: 0.000000).

## Root Cause

Recent changes introduced a separate inference path for TTT that buffers tokens and only processes them when a full chunk (256 tokens) is accumulated:

```python
# In ttt_module.py
if self.training:
    return self._parallel_ttt_update(Z, V_hat)
return self._ttt_forward_inference(Z, V_hat, input_dtype=x.dtype)
```

The inference path calls:
```python
self._append_inference_buffers(Z_fp32, V_fp32)
self._flush_inference_buffers(force=False)  # ← Always force=False!
```

This means:
1. Tokens are buffered during inference
2. Updates only happen when buffer reaches 256 tokens
3. **At the end of inference, any remaining tokens < 256 are never flushed**
4. **No final flush occurs, so the last partial chunk is lost**

For a 1820.5 second audio file, there should be many updates, but if tokens are processed in a way that leaves a significant remainder, or if the entire sequence is shorter than 256 tokens, NO updates occur.

## Solution

### 1. Added `flush_ttt_buffers()` method to TTTGating

**File:** `moshi/modules/ttt_module.py`

```python
def flush_ttt_buffers(self):
    """Force flush any remaining tokens in TTT inference buffers.

    This should be called at the end of inference to ensure all buffered
    tokens are processed and contribute to TTT updates.
    """
    if self.ttt_enabled:
        self._flush_inference_buffers(force=True)
```

### 2. Called flush at end of inference

**File:** `moshi/run_inference.py` (after line 428)

```python
out_items = state.run(in_pcms)

# Flush any remaining TTT buffers to ensure all tokens contribute to updates
if ttt_enabled_layers:
    for idx in ttt_enabled_layers:
        mlp = getattr(lm.transformer.layers[idx], 'mlp', None) or getattr(lm.transformer.layers[idx], 'gating', None)
        if hasattr(mlp, 'flush_ttt_buffers'):
            mlp.flush_ttt_buffers()
```

## Testing

Run your inference again:
```bash
cd moshi-finetune
./run_yarn_ttt_inference.sh
```

You should now see:
- **updates: > 0** (number of chunks processed)
- **change: > 0.000000** (w_down norm should change)

## Why This Happened

The buffering logic was added to handle streaming inference correctly (processing tokens one at a time while only updating every 256 tokens as per the TTT paper). However, the final flush was missing, causing any partial chunks to be discarded.

## Files Modified

1. `moshi/modules/ttt_module.py` - Added `flush_ttt_buffers()` method
2. `moshi/run_inference.py` - Added flush call at end of inference
