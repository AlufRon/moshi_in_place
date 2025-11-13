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


def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single
      Transformer block is sharded at a time
    - If LoRA is enabled, we additionally create separate FSDP sub-groups for
      every trainable and non-trainable parameter group since this is a
      requirement for mixed requires_grad=True/False training. See:
      https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer,),
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def fsdp_lora_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    # For LoRA training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_lora_policy_fn
    )

    policies = [fsdp_lora_policy, transformer_block_wrap_policy]

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
                    # For w_down: should have been copied from checkpoint already
                    if "w_down" in p_name:
                        # Fallback: This shouldn't happen if checkpoint loading worked
                        # Keep w_down in float32 for TTT precision (not param_dtype)
                        logger.warning(f"w_down {m_name}.{p_name} still meta - using random init as fallback (float32)")
                        module._parameters[p_name] = torch.nn.Parameter(
                            torch.empty_like(param, device="cpu", dtype=torch.float32)
                        )
                        torch.nn.init.kaiming_uniform_(module._parameters[p_name], a=math.sqrt(5))
                    else:
                        # For all other parameters (linear_in, conv, target_generator)
                        module._parameters[p_name] = torch.nn.Parameter(
                            torch.empty_like(param, device="cpu", dtype=param_dtype)
                        )
                        param = module._parameters[p_name]

                        # Initialize based on parameter type
                        if "linear" in p_name:
                            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                        elif "target_generator" in p_name:
                            # CRITICAL: Zero init for warm-start - ensures TTT has zero effect initially
                            # This prevents random target_generator from corrupting pretrained outputs
                            # During training, weights learn from zero via gradients
                            torch.nn.init.zeros_(param)
                            logger.info(f"  ✓ Zero-initialized {m_name}.{p_name} for warm-start")
                        elif "conv" in p_name:
                            # Small random init for conv layers (could also be zero)
                            torch.nn.init.normal_(param, mean=0.0, std=0.02)
                        else:
                            # Default initialization for other TTT params
                            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            
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
        
        model = checkpointer_info.get_moshi(
            device="meta",
            dtype=param_dtype,
            lm_kwargs_overrides={
                "gradient_checkpointing": args.gradient_checkpointing,
                "lora": args.lora.enable,
                "lora_rank": args.lora.rank,
                "lora_scaling": args.lora.scaling,
                "ttt_config": ttt_config,
            },
            load_weight=False,
        )

    if get_rank() == 0:
        moshi_weight = checkpointer_info.moshi_weights

        assert _is_safetensors(moshi_weight), "Model is not safetensors"
        model_state_dict = safetensors.torch.load_file(moshi_weight)

        logger.info(f"Converting model to dtype {param_dtype} ...")

        for k, v in model_state_dict.items():
            model_state_dict[k] = v.to(param_dtype)

        # Initialize TTT w_down from checkpoint BEFORE load_state_dict
        # because assign=True keeps meta tensors as meta
        if args.ttt and args.ttt.enabled:
            logger.info("Initializing TTT w_down from pretrained checkpoint...")
            for m_name, module in model.named_modules():
                if "gating" in m_name and hasattr(module, 'w_down'):
                    # Find the corresponding checkpoint key
                    ckpt_key = f"{m_name}.linear_out.weight"
                    if ckpt_key in model_state_dict:
                        # Copy from checkpoint dict directly and convert to float32 for TTT precision
                        pretrained_weight = model_state_dict[ckpt_key].clone().to(torch.float32)
                        module._parameters['w_down'] = torch.nn.Parameter(pretrained_weight)
                        logger.info(f"  ✓ {m_name}.w_down <- {ckpt_key} (shape: {pretrained_weight.shape}, dtype: float32)")
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

    # only finetune LoRA parameters and freeze before wrapping
    if args.lora.enable and not args.full_finetuning:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif args.lora.ft_embed and "emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    if get_world_size() == 1:
        return model.cuda()

    auto_wrap_policy = get_fsdp_policy(args.lora.enable)

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
