import dataclasses
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

# from torch.profiler import ProfilerActivity, profile

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import Batch, InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

logger = logging.getLogger("train")


def _is_rank_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def main_logger_info(message: str) -> None:
    if _is_rank_zero():
        logger.info(message)


def _summarize_doc_runs(doc_ids: list[str], segment_indices: list[int | None]) -> list[str]:
    runs: list[str] = []
    current_doc: str | None = None
    start_seg: int | None = None
    last_seg: int | None = None
    for doc_id, seg_idx in zip(doc_ids, segment_indices):
        if current_doc != doc_id:
            if current_doc is not None:
                runs.append(_format_doc_run(current_doc, start_seg, last_seg))
            current_doc = doc_id
            start_seg = seg_idx
        last_seg = seg_idx
    if current_doc is not None:
        runs.append(_format_doc_run(current_doc, start_seg, last_seg))
    return runs


def _format_doc_run(doc_id: str, start_seg: int | None, end_seg: int | None) -> str:
    doc_name = Path(doc_id).name
    if start_seg is None and end_seg is None:
        return f"{doc_name}[segments=?]"
    if start_seg == end_seg:
        return f"{doc_name}[segment={start_seg}]"
    return f"{doc_name}[segments={start_seg}-{end_seg}]"


def log_doc_stream(step: int, microbatch_idx: int, batch: Batch) -> None:
    if batch.doc_ids is None:
        return
    segment_indices = batch.segment_indices
    if segment_indices is None:
        segment_indices = [None] * len(batch.doc_ids)
    runs = _summarize_doc_runs(batch.doc_ids, segment_indices)
    unique_docs = len(set(batch.doc_ids))
    preview = ", ".join(runs[:5])
    if len(runs) > 5:
        preview += ", ..."
    main_logger_info(
        f"[DocStream] step={step} microbatch={microbatch_idx} samples={len(batch.doc_ids)} "
        f"unique_docs={unique_docs} runs={preview}"
    )


def reset_ttt_on_doc_switch(model, doc_ids: list[str], last_doc_id: str | None) -> str | None:
    if not hasattr(model, "reset_ttt_state"):
        return last_doc_id

    current_doc = last_doc_id
    for doc_id in doc_ids:
        if doc_id is None:
            continue
        if doc_id != current_doc:
            main_logger_info(f"[TTT RESET] Document switch detected: {current_doc} -> {doc_id}")
            model.reset_ttt_state()
            current_doc = doc_id
    return current_doc


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.DEBUG)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        # Partial fine-tuning: require either LoRA or TTT (or both)
        assert args.lora.enable or (args.ttt and args.ttt.enabled), \
            "For partial finetuning, either LoRA or TTT must be enabled"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling
    
    # Add TTT config to saved checkpoint config
    if args.ttt.enabled:
        lm_config["ttt_config"] = {
            'enabled': True,
            'layer_frequency': args.ttt.layer_frequency,
            'start_layer': args.ttt.start_layer,
            'chunk_size': args.ttt.chunk_size,
            'learning_rate': args.ttt.learning_rate,
            'conv_kernel_size': args.ttt.conv_kernel_size,
        }

    # Add YaRN config to saved checkpoint config
    if args.yarn.enabled:
        lm_config["yarn_config"] = {
            'enabled': True,
            'scale': args.yarn.scale,
            'original_max_seq_len': args.yarn.original_max_seq_len,
            'beta_fast': args.yarn.beta_fast,
            'beta_slow': args.yarn.beta_slow,
            'mscale': args.yarn.mscale,
            'mscale_all_dim': args.yarn.mscale_all_dim,
        }

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    model = get_fsdp_model(args, checkpoint_info)

    spm = checkpoint_info.get_text_tokenizer()

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # 5. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
    )

    if args.do_eval:
        eval_data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 8. Initialize checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    model.train()
    torch.cuda.empty_cache()

    last_doc_id: str | None = None

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes

            if batch.doc_ids is not None and state.step % args.log_freq == 0:
                log_doc_stream(state.step, i, batch)

            if batch.doc_ids is not None:
                last_doc_id = reset_ttt_on_doc_switch(model, batch.doc_ids, last_doc_id)

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # forward / backward
            output = model(codes=codes, condition_tensors=condition_tensors)
            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            mb_loss = text_loss + audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_real_tokens += (
                torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
            )

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # DEBUG: Check target_generator + w_down parameters
        if state.step % args.log_freq == 0 and state.step < 5 and args.ttt.enabled:
            main_logger_info("\n=== TTT Parameter Debug ===")
            ttt_param_count = 0
            for name, p in model.named_parameters():
                if 'target_generator' in name or 'w_down' in name:
                    ttt_param_count += 1
                    main_logger_info(f"{name}:")
                    main_logger_info(f"  requires_grad: {p.requires_grad}")
                    main_logger_info(f"  has grad: {p.grad is not None}")
                    if p.grad is not None:
                        main_logger_info(f"  grad norm: {p.grad.norm().item():.6f}")
                    else:
                        main_logger_info(f"  grad: None")
            if ttt_param_count == 0:
                main_logger_info("WARNING: No TTT parameters (target_generator or w_down) found!")
            main_logger_info("=========================\n")

        # Log TTT gradient norms and track parameter changes
        ttt_stats = {}
        if state.step % args.log_freq == 0 and args.ttt.enabled:
            # Collect TTT parameter stats BEFORE optimizer step (target_generator + w_down)
            ttt_params_before = {}
            ttt_grad_norm = 0.0
            ttt_param_norm = 0.0
            ttt_param_count = 0
            
            for name, p in model.named_parameters():
                if ('target_generator' in name or 'w_down' in name) and p.grad is not None:
                    ttt_grad_norm += p.grad.norm().item() ** 2
                    ttt_param_norm += p.data.norm().item() ** 2
                    ttt_param_count += 1
                    # Store parameter snapshot for computing delta
                    ttt_params_before[name] = p.data.clone()
            
            if ttt_param_count > 0:
                ttt_grad_norm = (ttt_grad_norm ** 0.5)
                ttt_param_norm = (ttt_param_norm ** 0.5)
                ttt_stats['grad_norm'] = ttt_grad_norm
                ttt_stats['param_norm_before'] = ttt_param_norm
                ttt_stats['param_count'] = ttt_param_count

        # optimizer step
        optimizer.step()
        
        # Log TTT parameter changes AFTER optimizer step
        if state.step % args.log_freq == 0 and args.ttt.enabled and ttt_stats:
            ttt_delta_norm = 0.0
            ttt_param_norm_after = 0.0
            
            for name, p in model.named_parameters():
                if name in ttt_params_before:
                    delta = (p.data - ttt_params_before[name]).norm().item()
                    ttt_delta_norm += delta ** 2
                    ttt_param_norm_after += p.data.norm().item() ** 2
            
            ttt_delta_norm = (ttt_delta_norm ** 0.5)
            ttt_param_norm_after = (ttt_param_norm_after ** 0.5)
            
            # Calculate relative change
            relative_change = (ttt_delta_norm / ttt_param_norm_after * 100) if ttt_param_norm_after > 0 else 0.0
            
            # Store for logging
            ttt_stats['param_norm'] = ttt_param_norm_after
            ttt_stats['delta_norm'] = ttt_delta_norm
            ttt_stats['relative_change'] = relative_change
            
            logger.info(
                f"[TTT] Step {state.step}: "
                f"grad_norm={ttt_stats['grad_norm']:.3e}, "
                f"param_norm={ttt_param_norm_after:.4f}, "
                f"delta_norm={ttt_delta_norm:.3e}, "
                f"relative_change={relative_change:.4f}% "
                f"({ttt_stats['param_count']} params)"
            )

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            # write perplexity to state
            evaluate(model, eval_data_loader, state, args)

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            
            # Add TTT statistics to logs if available
            if args.ttt.enabled and ttt_stats:
                train_logs['ttt/grad_norm'] = ttt_stats.get('grad_norm', 0.0)
                train_logs['ttt/param_norm'] = ttt_stats.get('param_norm', 0.0)
                train_logs['ttt/delta_norm'] = ttt_stats.get('delta_norm', 0.0)
                train_logs['ttt/relative_change_pct'] = ttt_stats.get('relative_change', 0.0)
                train_logs['ttt/param_count'] = ttt_stats.get('param_count', 0)
            
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
