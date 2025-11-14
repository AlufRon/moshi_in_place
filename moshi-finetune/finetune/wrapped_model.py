import functools
import logging
import math
from typing import Callable, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from moshi.models.lm import LMModel
from moshi.models.loaders import CheckpointInfo, _is_safetensors
from moshi.modules.transformer import StreamingTransformerLayer
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .args import TrainArgs
from .distributed import get_rank, get_world_size

logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy(has_mixed_grad: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single
      Transformer block is sharded at a time
    - If LoRA or TTT-only training is enabled, we additionally create separate FSDP
      sub-groups for every trainable and non-trainable parameter group since this is a
      requirement for mixed requires_grad=True/False training. See:
      https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )

    if not has_mixed_grad:
        return transformer_block_wrap_policy

    def fsdp_mixed_grad_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    # For partial training (LoRA or TTT-only), trainable and non-trainable parameters
    # need to be put into different FSDP groups
    fsdp_mixed_grad_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_mixed_grad_policy_fn
    )

    policies = [fsdp_mixed_grad_policy, transformer_block_wrap_policy]

    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel]):
    world_size = get_world_size()

    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize LoRA layers with Kaiming uniform and zeros.
    See original paper for more info: https://arxiv.org/abs/2106.09685 and
    original github repo:
    https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        # Skip TTT gating modules - they have their own initialization
        if "gating" in m_name:
            continue
            
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise ValueError(f"Only Lora layers should be randomly initialized. Got: {m_name}")


def initialize_ttt_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize TTT gating module parameters and buffers.
    
    Per the In-Place TTT paper: only W_down is the fast weight.
    W_up and W_gate (in linear_in) are frozen slow weights.
    
    NOTE: w_down should already be copied from checkpoint in get_fsdp_model().
    This function only initializes if w_down is still meta (fallback).
    """
    for m_name, module in model.named_modules():
        # Only initialize TTT gating modules
        if "gating" in m_name:
            # Initialize parameters
            for p_name, param in module.named_parameters():
                # Only initialize if this specific parameter is meta
                if param.is_meta:
                    # CRITICAL FIX: Navigate to nested module for LoRA layers
                    # p_name can be like "target_generator.W_target.frozen_W.weight"
                    # We need to navigate to the actual parent module before assigning
                    parts = p_name.split('.')
                    param_name = parts[-1]  # e.g., "weight"
                    nested_module = module
                    for part in parts[:-1]:  # Navigate through nested path
                        nested_module = getattr(nested_module, part)

                    # Determine dtype based on parameter type
                    # For w_down: should have been copied from checkpoint already
                    if "w_down" in p_name and "w_down_pretrained" not in p_name:
                        # Fallback: This shouldn't happen if checkpoint loading worked
                        # Keep w_down in float32 for TTT precision (not param_dtype)
                        logger.warning(f"w_down {m_name}.{p_name} still meta - using random init as fallback (float32)")
                        new_param = torch.nn.Parameter(
                            torch.empty_like(param, device="cpu", dtype=torch.float32)
                        )
                        torch.nn.init.kaiming_uniform_(new_param, a=math.sqrt(5))
                        # Assign to the nested module
                        nested_module._parameters[param_name] = new_param
                    else:
                        # For all other parameters (linear_in, conv, target_generator)
                        new_param = torch.nn.Parameter(
                            torch.empty_like(param, device="cpu", dtype=param_dtype)
                        )

                        # Initialize based on parameter type
                        if "linear" in p_name:
                            torch.nn.init.kaiming_uniform_(new_param, a=math.sqrt(5))
                        elif "target_generator" in p_name:
                            # Initialize target_generator with std=1e-2 for proper gradient flow
                            # Previous std=1e-4 was too conservative, causing 50,000x gradient vanishing
                            # std=1e-2 provides V_hat with sufficient magnitude for backprop learning
                            # while still being small enough for warm-start from pretrained model
                            torch.nn.init.normal_(new_param, mean=0.0, std=1e-2)
                            logger.info(f"  ✓ Small-random-initialized {m_name}.{p_name} (std=1e-2) for proper gradient flow")
                        elif "conv" in p_name:
                            # Initialize conv layers (part of target_generator) with std=1e-2
                            # Matches target_generator initialization for consistent gradient flow
                            torch.nn.init.normal_(new_param, mean=0.0, std=1e-2)
                            logger.info(f"  ✓ Initialized {m_name}.{p_name} with small random values (std=1e-2)")
                        else:
                            # Default initialization for other TTT params
                            torch.nn.init.kaiming_uniform_(new_param, a=math.sqrt(5))
                        
                        # Assign to the nested module
                        nested_module._parameters[param_name] = new_param
            
            # Initialize buffers (e.g., w_down_pretrained)
            for b_name, buffer in module.named_buffers():
                if buffer.is_meta:
                    # Initialize w_down_pretrained buffer to match w_down
                    if "w_down_pretrained" in b_name:
                        # Copy from w_down if it exists and is initialized
                        if hasattr(module, 'w_down') and not module.w_down.is_meta:
                            module._buffers[b_name] = module.w_down.data.clone().detach()
                            logger.info(f"  ✓ Initialized {m_name}.{b_name} from w_down")
                        else:
                            # Fallback: create empty buffer on CPU in float32
                            logger.warning(f"Buffer {m_name}.{b_name} still meta - initializing as zeros (float32)")
                            module._buffers[b_name] = torch.zeros_like(buffer, device="cpu", dtype=torch.float32)
                    else:
                        # Other buffers: initialize on CPU with appropriate dtype
                        logger.warning(f"Buffer {m_name}.{b_name} still meta - initializing as zeros")
                        module._buffers[b_name] = torch.zeros_like(buffer, device="cpu")


def get_fsdp_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> FullyShardedDataParallel | LMModel:
    """
    Initializes and returns a FullyShardedDataParallel (FSDP) LMModel or a non sharded LMModel if one GPU available.
    Args:
        args (TrainArgs): A configuration object containing training arguments
            and settings. Key attributes include:
            - param_dtype: The data type for model parameters (e.g., "bfloat16", "float32").
            - gradient_checkpointing: Whether to enable gradient checkpointing.
            - lora: Configuration for LoRA fine-tuning, including enabling, rank, and scaling.
            - full_finetuning: Whether to enable full model fine-tuning or only LoRA fine-tuning.
        checkpointer_info: provide the initial checkpoint to train from.
    Notes:
        - The function uses meta-device initialization for memory efficiency.
        - Then parameters are initialized on the first GPU (rank=0) only.
    """

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32

    with torch.device("meta"):
        # Build TTT config if enabled
        ttt_config = None
        if args.ttt.enabled:
            ttt_config = {
                'enabled': True,
                'layer_frequency': args.ttt.layer_frequency,
                'start_layer': args.ttt.start_layer,
                'chunk_size': args.ttt.chunk_size,
                'learning_rate': args.ttt.learning_rate,
                'conv_kernel_size': args.ttt.conv_kernel_size,
            }
            logger.info("=" * 70)
            logger.info("TTT (Test-Time Training) ENABLED")
            logger.info("=" * 70)
            logger.info(f"  Layer frequency: {args.ttt.layer_frequency}")
            logger.info(f"  Start layer: {args.ttt.start_layer}")
            logger.info(f"  Chunk size: {args.ttt.chunk_size}")
            logger.info(f"  Learning rate: {args.ttt.learning_rate}")
            logger.info(f"  Conv kernel: {args.ttt.conv_kernel_size}")
            logger.info("=" * 70)
        
        # Build YARN config if enabled
        yarn_config = None
        if hasattr(args, 'yarn') and args.yarn.enabled:
            yarn_config = {
                'yarn_scale': args.yarn.scale,
                'original_max_seq_len': args.yarn.original_max_seq_len,
                'yarn_beta_fast': args.yarn.beta_fast,
                'yarn_beta_slow': args.yarn.beta_slow,
                'yarn_mscale': args.yarn.mscale,
                'yarn_mscale_all_dim': args.yarn.mscale_all_dim,
            }
            logger.info("=" * 70)
            logger.info("YaRN (Context Window Extension) ENABLED")
            logger.info("=" * 70)
            logger.info(f"  Scale: {args.yarn.scale}x")
            logger.info(f"  Original max seq len: {args.yarn.original_max_seq_len}")
            logger.info(f"  Beta fast: {args.yarn.beta_fast}")
            logger.info(f"  Beta slow: {args.yarn.beta_slow}")
            logger.info("=" * 70)
        
        lm_overrides = {
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora": args.lora.enable,
            "lora_rank": args.lora.rank,
            "lora_scaling": args.lora.scaling,
            "ttt_config": ttt_config,
        }
        if yarn_config:
            lm_overrides.update(yarn_config)
        
        model = checkpointer_info.get_moshi(
            device="meta",
            dtype=param_dtype,
            lm_kwargs_overrides=lm_overrides,
            load_weight=False,
        )

    if get_rank() == 0:
        moshi_weight = checkpointer_info.moshi_weights

        assert _is_safetensors(moshi_weight), "Model is not safetensors"
        model_state_dict = safetensors.torch.load_file(moshi_weight)

        logger.info(f"Converting model to dtype {param_dtype} ...")

        for k, v in model_state_dict.items():
            if "w_down_pretrained" in k:
                # Keep pretrained buffer in float32 to mirror fast weight precision
                model_state_dict[k] = v.to(torch.float32)
            else:
                model_state_dict[k] = v.to(param_dtype)

        # Initialize TTT w_down from checkpoint: add w_down keys in float32 to state_dict
        # BEFORE load_state_dict so assign=True will use float32 tensors
        if args.ttt and args.ttt.enabled:
            logger.info("Initializing TTT w_down from pretrained checkpoint...")
            for m_name, module in model.named_modules():
                if "gating" in m_name and hasattr(module, 'w_down'):
                    # Find the corresponding checkpoint key
                    ckpt_key = f"{m_name}.linear_out.weight"
                    if ckpt_key in model_state_dict:
                        # Add w_down key to state_dict in float32 (not bfloat16)
                        w_down_key = f"{m_name}.w_down"
                        pretrained_weight = model_state_dict[ckpt_key].clone().to(torch.float32)
                        model_state_dict[w_down_key] = pretrained_weight
                        logger.info(f"  ✓ {w_down_key} <- {ckpt_key} (shape: {pretrained_weight.shape}, dtype: float32)")
                    else:
                        logger.warning(f"  ✗ Checkpoint key {ckpt_key} not found!")

        model.load_state_dict(model_state_dict, strict=False, assign=True)

        if args.lora.enable and not args.full_finetuning:
            logger.info("Initializing lora layers ...")
            # initialize LoRA layers
            initialize_lora_parameters(model, param_dtype)

        # Initialize TTT parameters if TTT is enabled
        if args.ttt and args.ttt.enabled:
            logger.info("Initializing TTT layers ...")
            initialize_ttt_parameters(model, param_dtype)

        # Initialize YaRN inv_freq buffer if YARN is enabled
        if hasattr(args, 'yarn') and args.yarn.enabled:
            logger.info("Initializing YaRN RoPE buffers ...")
            from moshi.modules.rope import _compute_yarn_parameters
            for name, module in model.named_modules():
                if hasattr(module, 'rope') and module.rope is not None:
                    rope = module.rope
                    if hasattr(rope, 'inv_freq') and rope.inv_freq is not None and rope.inv_freq.is_meta:
                        # Recompute inv_freq on the correct device
                        inv_freq, attention_factor = _compute_yarn_parameters(
                            dim=rope.dim,
                            max_period=rope.max_period,
                            scale=rope.yarn_scale,
                            original_max_seq_len=rope.original_max_seq_len,
                            beta_fast=rope.beta_fast,
                            beta_slow=rope.beta_slow,
                            mscale=rope.mscale,
                            mscale_all_dim=rope.mscale_all_dim,
                            device="cuda",
                        )
                        rope._buffers['inv_freq'] = inv_freq.to(dtype=param_dtype)
                        rope.attention_factor = attention_factor
                        logger.info(f"  ✓ Initialized {name}.rope.inv_freq (shape: {inv_freq.shape}, scale: {rope.yarn_scale}x)")

        # Debug: Check which parameters and buffers are still meta
        meta_params = [(name, param.shape) for name, param in model.named_parameters() if param.is_meta]
        meta_buffers = [(name, buf.shape) for name, buf in model.named_buffers() if buf.is_meta]
        
        if meta_params:
            logger.error(f"Found {len(meta_params)} meta parameters that were not initialized:")
            for name, shape in meta_params[:10]:  # Show first 10
                logger.error(f"  {name}: {shape}")
        
        if meta_buffers:
            logger.error(f"Found {len(meta_buffers)} meta buffers that were not initialized:")
            for name, shape in meta_buffers[:10]:  # Show first 10
                logger.error(f"  {name}: {shape}")

        assert not any(p.is_meta for p in model.parameters()), (
            "All parameters should be initialized by now"
        )
        
        assert not any(b.is_meta for b in model.buffers()), (
            "All buffers should be initialized by now"
        )
        
        # Ensure TTT fast weights stay in float32 and opt-out from mixed precision casting
        if args.ttt and args.ttt.enabled:
            ttt_fp32_params = 0
            for m_name, module in model.named_modules():
                if hasattr(module, "w_down"):
                    module.w_down.data = module.w_down.data.to(torch.float32)
                    setattr(module.w_down, "_ttt_keep_fp32", True)
                    ttt_fp32_params += 1
                    if hasattr(module, "w_down_pretrained"):
                        module.w_down_pretrained.data = module.w_down_pretrained.data.to(torch.float32)
            if ttt_fp32_params > 0:
                logger.info(
                    f"Pinned {ttt_fp32_params} w_down parameters to float32 for TTT precision"
                )

        # Check dtype, but allow w_down to be float32 for TTT precision
        non_compliant = []
        for name, p in model.named_parameters():
            # w_down is intentionally kept in float32 for TTT precision
            if 'w_down' in name and args.ttt and args.ttt.enabled:
                if p.dtype != torch.float32:
                    non_compliant.append((name, p.dtype))
            elif p.dtype != param_dtype:
                non_compliant.append((name, p.dtype))
        
        if non_compliant:
            logger.error(f"Found {len(non_compliant)} parameters with wrong dtype:")
            for name, dtype in non_compliant[:20]:
                logger.error(f"  {name}: {dtype}")
            raise AssertionError(f"All parameters should be on {param_dtype} (except w_down which should be float32)")

        logger.info("Finished initialization!")
        
        # Log TTT status
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            ttt_layers = [i for i, layer in enumerate(model.transformer.layers) 
                         if getattr(layer, 'use_ttt', False)]
            if ttt_layers:
                logger.info("=" * 70)
                logger.info(f"TTT ACTIVE: {len(ttt_layers)} layers enabled")
                logger.info(f"TTT layer indices: {ttt_layers}")
                logger.info("=" * 70)
            else:
                logger.warning("TTT config provided but NO TTT layers created!")
        
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in model.parameters()), (
            "All parameters should be on meta"
        )

    torch.distributed.barrier()

    # Set requires_grad based on training mode
    if args.full_finetuning:
        # Full fine-tuning: train all parameters
        for param in model.parameters():
            param.requires_grad = True
    elif args.lora.enable:
        # LoRA fine-tuning: train only LoRA parameters (and optionally embeddings)
        # TTT parameters will also be trained if TTT is enabled
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif args.lora.ft_embed and "emb" in name:
                param.requires_grad = True
            elif args.ttt and args.ttt.enabled and "target_generator" in name:
                # Train TTT target_generator when both LoRA and TTT are enabled
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.ttt and args.ttt.enabled:
        if args.ttt.unfreeze_ttt_layers:
            # TTT-layer finetuning: train entire layers where TTT is applied
            # This allows attention to co-adapt with TTT and YaRN scaling

            # Identify layers with TTT
            ttt_layer_indices = set()
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                for idx, layer in enumerate(model.transformer.layers):
                    # Check if this layer has TTT enabled
                    mlp = getattr(layer, 'mlp', None) or getattr(layer, 'gating', None)
                    if mlp and hasattr(mlp, 'ttt_enabled') and mlp.ttt_enabled:
                        ttt_layer_indices.add(idx)

            logger.info("=" * 70)
            logger.info("TTT-LAYER UNFREEZING ENABLED")
            logger.info(f"  Unfreezing entire layers at indices: {sorted(ttt_layer_indices)}")
            logger.info(f"  Total layers to train: {len(ttt_layer_indices)}/{len(model.transformer.layers)}")
            logger.info("=" * 70)

            # Freeze all by default, then unfreeze TTT layers
            for name, param in model.named_parameters():
                # Check if parameter belongs to a TTT layer
                is_ttt_layer = False
                for idx in ttt_layer_indices:
                    if f"transformer.layers.{idx}." in name:
                        is_ttt_layer = True
                        break

                if is_ttt_layer:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # TTT-only fine-tuning: train target_generator + w_down (fast weights)
            # The paper trains target_generator (slow weights that produce V_hat) plus
            # w_down (which starts as linear_out and learns via backprop during training,
            # then gets test-time updates via the delta rule during inference).
            for name, param in model.named_parameters():
                if "target_generator" in name or "w_down" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    else:
        # Fallback: should not reach here due to assertion in train.py
        raise ValueError("Either full_finetuning, lora, or ttt must be enabled")

    if get_world_size() == 1:
        return model.cuda()

    # Use mixed grad policy for partial training (LoRA or TTT-only)
    has_mixed_grad = not args.full_finetuning
    auto_wrap_policy = get_fsdp_policy(has_mixed_grad)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        use_orig_params=True,
    )

    main_logger_info("Model sharded!")

    log_train_params(wrapped_model)

    return wrapped_model
