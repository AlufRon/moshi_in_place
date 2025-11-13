# In-Place Test-Time Training (In-Place TTT) Implementation Guide

This document outlines the technical implementation plan for the "In-Place Test-Time Training" (In-Place TTT) framework, based on the ICLR 2026 paper.

## 1. Core Concepts

-   **In-Place Adaptation**: Instead of adding new layers, this method repurposes the final projection matrix (`W_down`) of existing MLP blocks in a Transformer as adaptable "fast weights".
-   **Chunk-wise Updates**: To ensure computational efficiency, updates are performed on non-overlapping segments (chunks) of the input sequence, not per token.
-   **LM-Aligned Objective**: The update rule is not a simple reconstruction. It uses a target (`V_hat`) derived from *future* tokens, aligning the test-time adaptation process with the Next-Token Prediction (NTP) goal of language models.

## 2. Implementation Steps

### Step 1: Modify the Transformer MLP Block

The core of the implementation lies in modifying the Multi-Layer Perceptron (MLP) or Feed-Forward Network (FFN) block.

1.  **Isolate MLP Weights**:
    -   The paper assumes a gated MLP architecture: `Output = ((ϕ(H * W_gate^T) ⊙ (H * W_up^T)) * W_down^T`.
    -   The final projection matrix, `W_down`, must be designated as the **fast weight**.
    -   All other parameters (`W_gate`, `W_up`, attention matrices, etc.) are **slow weights** and remain frozen during inference.

2.  **Manage Fast Weight State**:
    -   Implement a state manager for `W_down`. Its state must be tracked for each sequence being processed.
    -   **Crucially**, the state of `W_down` must be reset to its initial, pre-trained value at the beginning of every new document or independent sequence to prevent context leakage.

### Step 2: Implement the Chunk-wise "Apply-then-Update" Mechanism

This mechanism governs how the model processes a sequence chunk by chunk.

1.  **Partition Sequence**: Divide the input token sequence into non-overlapping chunks of size `C`.

2.  **Create the Loop**: For each chunk `i` in the sequence:
    -   **Apply**: Compute the MLP output for the current chunk `Z_i` using the current fast weight state `W_down^(i)`.
      ```
      Output_i = Z_i * (W_down^(i))^T
      ```
    -   **Update**: Calculate the next state of the fast weights, `W_down^(i+1)`, which will be used for chunk `i+1`. The calculation is detailed in the next section.

### Step 3: Implement the LM-Aligned Objective and Update Rule

This is the learning rule for the fast weights.

1.  **Generate the Target `V_hat`**:
    -   The target for the update is `V_hat = Conv1D(X_0) * W_target`.
    -   `X_0`: The input token embeddings for the current chunk.
    -   `Conv1D`: A 1D convolutional layer whose kernel is designed to look at future tokens. To capture the next token, the kernel would be `[0, 1]` with appropriate padding to ensure causality.
    -   `W_target`: A trainable `d_model x d_model` projection matrix. This is a slow weight, learned during the model's main training or fine-tuning phase.

2.  **Implement the Fast Weight Update Rule**:
    -   The update is a simple, efficient, Hebbian-like rule:
      ```
      W_down^(i+1) = W_down^(i) + eta * V_hat_i^T * Z_i
      ```
    -   `Z_i`: The intermediate activation from the gated MLP for chunk `i`.
    -   `eta`: The fast weight learning rate, a crucial hyperparameter.

### Step 4: (Advanced) Implement Context Parallelism for Efficiency

To avoid a slow, sequential loop over chunks, the paper proposes a parallel scan implementation.

1.  **Parallel Delta Calculation**: For all chunks `i` in parallel, compute:
    -   Intermediate activations `Z_i`.
    -   Fast weight update deltas `delta_W_i = V_hat_i^T * Z_i`.

2.  **Prefix Sum (Scan)**: Perform a prefix sum over the sequence of deltas to get the cumulative update for each position:
    -   `S_i = sum_{j=1}^{i-1} delta_W_j`

3.  **Parallel Output Computation**: For all chunks `i` in parallel, compute the final output:
    -   Calculate the effective fast weights for the chunk: `W_down_eff_i = W_down_initial + eta * S_i`.
    -   Calculate the MLP output: `O_i = Z_i * (W_down_eff_i)^T`.

## 3. Training and Fine-tuning

-   **Continual Training is Required**: After modifying a pre-trained model, it must be fine-tuned (the paper calls this "continual training").
-   **Learning the Objective**: This training phase is essential for the model to learn the weights of the `W_target` projection matrix, teaching it *how* to adapt.
-   **Loss Function**: The overall model is trained with a standard Next-Token Prediction (cross-entropy) loss. The In-Place TTT mechanism is an internal component that helps the model minimize this loss over long sequences.
