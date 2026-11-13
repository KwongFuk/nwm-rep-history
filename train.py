# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY-NC 4.0 license found in
# LICENSES/CC-BY-NC-4.0-NWM.txt. Derived from facebookresearch/nwm and
# modified by the authors; see THIRD_PARTY.md.
# --------------------------------------------------------
# References:
# NoMaD, GNM, ViNT: https://github.com/robodhruv/visualnav-transformer
# --------------------------------------------------------

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os, sys
import contextlib
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAE_SRC = os.path.join(PROJECT_ROOT, "RAE", "src")
if RAE_SRC not in sys.path:
    sys.path.append(RAE_SRC)
import matplotlib
matplotlib.use('Agg')
from collections import OrderedDict
from copy import deepcopy
from time import time
import argparse
import json
import logging
import math
import random
import matplotlib.pyplot as plt 
import wandb
import numpy as np
import torch.nn.functional as F


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, Sampler as TorchSampler
from torch.utils.data.distributed import DistributedSampler
torch.backends.cuda.enable_math_sdp(False)
from RAE.src.stage2.transport.transport import Transport, ModelType, PathType, WeightType, Sampler
from nwm.utils.distributed import init_distributed
from nwm.data.datasets import TrainingDataset
from nwm.models.alignment import (
    clean_latent_prediction_loss,
    local_spatial_relation_loss,
    representation_alignment_loss,
)
from nwm.utils.metrics import DINOImageDistance
from nwm.data.misc import transform
from nwm.utils.config import assert_config_not_deprecated, load_config
from nwm.models.factory import build_nwm_model
from nwm.tokenizers.build import build_stage1_tokenizer
from nwm.utils.training import (
    build_microbatch_slices,
    parse_epoch_milestones,
    resolve_time_dist_shift,
    sample_step_context_length,
    validate_training_data_schedule,
    validate_transition_batch_contract,
)
from nwm.utils.checkpointing import (
    capture_rng_state,
    config_fingerprint,
    is_complete_checkpoint,
    latest_readable_complete_checkpoint,
    publish_checkpoint_alias,
    publish_latest_checkpoint,
    restore_rng_state,
    validate_training_checkpoint,
    write_complete_checkpoint,
    write_complete_evaluation_snapshot,
)
from nwm.utils.storage import require_durable_path

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

_FLASH_ATTN_STATUS_PRINTED = False


def _get_sdpa_backend_flags() -> dict:
    flags = {}
    try:
        flags["flash"] = bool(torch.backends.cuda.flash_sdp_enabled())
    except Exception:
        flags["flash"] = None
    try:
        flags["mem_efficient"] = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    except Exception:
        flags["mem_efficient"] = None
    try:
        flags["math"] = bool(torch.backends.cuda.math_sdp_enabled())
    except Exception:
        flags["math"] = None
    return flags


def _maybe_print_flash_attn_status_once(*, device: torch.device, dtype: torch.dtype, num_heads: int, head_dim: int, seqlen: int, rank: int) -> None:
    global _FLASH_ATTN_STATUS_PRINTED
    if _FLASH_ATTN_STATUS_PRINTED or rank != 0:
        return
    if os.environ.get("FLASH_ATTN_MONITOR", "1").strip().lower() in ("0", "false", "no", "off"):
        return

    _FLASH_ATTN_STATUS_PRINTED = True

    gpu_name = None
    try:
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
    except Exception:
        gpu_name = None

    flags = _get_sdpa_backend_flags()

    probe_ok = False
    probe_err = None
    try:
        cm = contextlib.nullcontext()
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            cm = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
        else:
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                cm = sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
            except Exception:
                cm = contextlib.nullcontext()

        with cm:
            q = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, num_heads, seqlen, head_dim, device=device, dtype=dtype)
            _ = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        probe_ok = True
    except Exception as e:
        probe_err = str(e).split("\n")[0]

    print(
        f"[SDPA] torch={torch.__version__} cuda={torch.version.cuda} gpu={gpu_name} dtype={str(dtype).replace('torch.', '')} "
        f"flags={flags} flash_only_probe={{'ok': {probe_ok}, 'heads': {num_heads}, 'head_dim': {head_dim}, 'seqlen': {seqlen}, 'err': {probe_err!r}}}"
    )


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace('_orig_mod.', '')
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


class BalancedDistributedSampler(TorchSampler):
    def __init__(
        self,
        datasets,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        desired_total_size=None,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        if any(l <= 0 for l in self.lengths):
            raise ValueError(f"All datasets must be non-empty, got lengths={self.lengths}")

        self.num_datasets = len(self.datasets)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

        offsets = [0]
        for l in self.lengths[:-1]:
            offsets.append(offsets[-1] + int(l))
        self.offsets = offsets

        if desired_total_size is None:
            desired_total_size = int(sum(self.lengths))
        self.desired_total_size = int(desired_total_size)

        self.samples_per_dataset = int(math.ceil(self.desired_total_size / float(self.num_datasets)))
        base_total = self.samples_per_dataset * self.num_datasets

        self.total_size = int(math.ceil(base_total / float(self.num_replicas)) * self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        per_dataset = []
        for ds_idx, ds_len in enumerate(self.lengths):
            if self.shuffle:
                base = torch.randperm(ds_len, generator=g).tolist()
            else:
                base = list(range(ds_len))

            if self.samples_per_dataset <= ds_len:
                chosen = base[: self.samples_per_dataset]
            else:
                remaining = self.samples_per_dataset - ds_len
                extra = torch.randint(high=ds_len, size=(remaining,), generator=g).tolist() if remaining > 0 else []
                chosen = base + extra

            per_dataset.append([i + self.offsets[ds_idx] for i in chosen])

        indices = []
        for j in range(self.samples_per_dataset):
            for i in range(self.num_datasets):
                indices.append(per_dataset[i][j])

        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[k] for k in perm]

        indices = indices[: self.desired_total_size]

        if len(indices) < self.total_size:
            indices += indices[: (self.total_size - len(indices))]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new CDiT model.
    """
    config = load_config(args.config, defaults_path="config/eval_config.yaml")
    assert_config_not_deprecated(config, config_path=args.config)
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    _, rank, gpu, _ = init_distributed()
    device = torch.device(f"cuda:{gpu}")
    # rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    if args.epochs is None:
        args.epochs = int(config.get("epochs", 300))
    args.eval_at_epochs = parse_epoch_milestones(
        args.eval_at_epochs,
        total_epochs=args.epochs,
    )
    tp = config.get('transport', {})
    exact_resume = bool(config.get("exact_resume", False))
    optimizer_batch_size = int(config["batch_size"])
    train_microbatch_size = int(
        config.get("train_microbatch_size", optimizer_batch_size)
    )
    build_microbatch_slices(optimizer_batch_size, train_microbatch_size)
    batch_contract = validate_transition_batch_contract(
        config,
        dist.get_world_size(),
    )
    if bool(config.get("require_scratch_output", False)):
        require_durable_path(
            config["results_dir"],
            label="training output",
        )

    config_identity = deepcopy(config)
    for operational_key in (
        "from_checkpoint",
        "allow_legacy_checkpoint",
        "allow_config_mismatch",
    ):
        config_identity.pop(operational_key, None)
    config_sha256 = config_fingerprint(config_identity)

    termination_requested = {"signal": None}
    termination_request_file = os.environ.get(
        "RAENWM_TERMINATION_REQUEST_FILE", ""
    ).strip()

    def _request_termination(signum, _frame):
        termination_requested["signal"] = int(signum)
        print(
            f"Received signal {signum}; will save a complete checkpoint after the current optimizer step.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _request_termination)
    signal.signal(signal.SIGINT, _request_termination)

    # Setup an experiment folder:
    os.makedirs(config['results_dir'], exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_dir = f"{config['results_dir']}/{config['run_name']}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores exact-resume state
    eval_snapshot_dir = f"{experiment_dir}/eval_snapshots"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(eval_snapshot_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        wandb_config = config.get('wandb', {})
        use_wandb_new = wandb_config.get('enabled', True)
        use_wandb_old = config.get('use_wandb', True)
        
        if 'wandb' in config:
            should_use_wandb = use_wandb_new
        else:
            should_use_wandb = use_wandb_old
            
        if should_use_wandb:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = wandb_config.get('run_name') or f"{config['run_name']}_{timestamp}"
            
            wandb.init(
                project=wandb_config.get('project', 'nwm-training'),
                name=run_name,
                entity=wandb_config.get('entity', None),
                config=config,
                dir=experiment_dir,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', '')
            )
            logger.info(f"WandB initialized successfully with run name: {run_name}")
    else:
        logger = create_logger(None)

    # Create the frozen stage-1 tokenizer through the same audited path used
    # by inference.  PAE and RAE arms differ only through this config surface.
    tokenizer, tokenizer_info = build_stage1_tokenizer(
        config,
        device=device,
        project_root=PROJECT_ROOT,
    )
    rae = tokenizer
    latent_size_value = config.get("latent_size")
    if latent_size_value is None:
        if int(config["image_size"]) % 14 != 0:
            raise ValueError(
                "Set latent_size explicitly for encoders that do not use the "
                "DINOv2 patch size 14"
            )
        latent_size_value = int(config["image_size"]) // 14
    latent_size = int(latent_size_value)
    if latent_size <= 0:
        raise ValueError(f"latent_size must be positive, got {latent_size}")
    num_cond = config['context_size']
    train_context_min = int(config.get("train_context_min", num_cond))
    train_context_max = int(config.get("train_context_max", num_cond))
    if not 1 <= train_context_min <= train_context_max <= int(num_cond):
        raise ValueError(
            "Training context range must satisfy "
            f"1 <= min <= max <= context_size, got "
            f"[{train_context_min},{train_context_max}] with "
            f"context_size={num_cond}"
        )
    
    learn_sigma_cfg = config.get('learn_sigma', False)
    if isinstance(learn_sigma_cfg, str):
        learn_sigma_cfg = learn_sigma_cfg.strip().lower() == 'true'
    model, paired_init_report = build_nwm_model(
        config,
        latent_channels=int(rae.latent_dim),
        latent_size=latent_size,
    )
    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    parameter_budget = config.get('parameter_budget_trainable', None)
    parameter_tolerance = float(config.get('parameter_tolerance', 0.005))
    if parameter_budget is not None:
        parameter_budget = int(parameter_budget)
        relative_error = abs(trainable_params - parameter_budget) / parameter_budget
        if relative_error > parameter_tolerance:
            raise ValueError(
                "Trainable parameter budget violation: "
                f"actual={trainable_params:,}, budget={parameter_budget:,}, "
                f"relative_error={relative_error:.4%}, "
                f"tolerance={parameter_tolerance:.4%}"
            )
    if rank == 0:
        logger.info(
            "Tokenizer: "
            f"{tokenizer_info.to_dict()}, latent_size={latent_size}, "
            f"learn_sigma={learn_sigma_cfg}"
        )
        logger.info(f"Paired initialization: {paired_init_report.to_dict()}")
        logger.info(
            f"CDiT in_channels={model.in_channels}, out_channels={model.out_channels}, patch_size={model.patch_size}, head_width={getattr(model, 'head_width', 'NA')}"
        )
        logger.info(
            "Model config: "
            f"backbone_mode={model.backbone_mode}, "
            f"attention_mode={model.attention_mode}, repa_mode={model.repa_mode}, "
            f"repa_target={config.get('repa_target', 'state')}, "
            f"repa_target_dim={model.repa_target_dim}, "
            f"mlp_ratio={float(config.get('mlp_ratio', 4.0))}, "
            f"trainable_params={trainable_params:,}, total_params={total_params:,}, "
            f"budget={parameter_budget}"
        )
        logger.info(
            "Batch contract: "
            f"optimizer_batch_per_rank={optimizer_batch_size}, "
            f"train_microbatch_per_rank={train_microbatch_size}, "
            f"backward_microbatches="
            f"{len(build_microbatch_slices(optimizer_batch_size, train_microbatch_size))}, "
            f"effective_global_observation_batch="
            f"{batch_contract['global_observations']}, "
            f"goals_per_observation={batch_contract['goals_per_observation']}, "
            f"effective_global_transition_batch="
            f"{batch_contract['global_transition_targets']}"
        )
        logger.info(
            "History sampling contract: one shared H per optimizer batch, "
            f"H~Uniform{{{train_context_min},...,{train_context_max}}}; "
            f"dataset/model maximum={num_cond}"
        )

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    # Resolve the optimizer from the experiment config.  Paper-aligned configs
    # are validated above, so no launcher-side value can silently override it.
    base_lr = float(config.get('lr', 2e-4))
    betas = tuple(float(value) for value in config.get('betas', (0.9, 0.999)))
    weight_decay = float(config.get("weight_decay", 0.0))
    adam_eps = float(config.get("adam_eps", 1.0e-8))
    override_lr_on_resume = bool(config.get('override_lr_on_resume', True))
    if exact_resume and override_lr_on_resume:
        raise ValueError(
            "exact_resume=true requires override_lr_on_resume=false so optimizer "
            "and scheduler state remain unchanged"
        )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )

    bfloat_enable = bool(hasattr(args, 'bfloat16') and args.bfloat16)

    try:
        attn0 = model.blocks[0].attn
        num_heads = int(getattr(attn0, "num_heads", getattr(model, "num_heads", 0)) or 0)
        head_dim = int(getattr(attn0, "head_dim", 0) or 0)
        seqlen = int(getattr(getattr(model, "x_embedder", None), "num_patches", 0) or 0)
        if num_heads <= 0 or head_dim <= 0 or seqlen <= 0:
            raise ValueError("invalid attention shape")
    except Exception:
        num_heads, head_dim, seqlen = 8, 64, 256

    probe_dtype = torch.bfloat16 if bfloat_enable else torch.float16
    _maybe_print_flash_attn_status_once(
        device=device,
        dtype=probe_dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        seqlen=seqlen,
        rank=rank,
    )

    if bfloat_enable:
        scaler = torch.amp.GradScaler()

    # Load the newest complete full-state checkpoint.  A file without its
    # atomic ``.complete`` marker is deliberately ignored after preemption.
    explicit_checkpoint = config.get('from_checkpoint', None)
    latest_path = None
    if explicit_checkpoint:
        explicit_checkpoint = os.path.realpath(str(explicit_checkpoint))
        if is_complete_checkpoint(explicit_checkpoint):
            latest_path = explicit_checkpoint
        elif bool(config.get("allow_legacy_checkpoint", False)) and os.path.isfile(explicit_checkpoint):
            latest_path = explicit_checkpoint
        else:
            raise ValueError(
                f"Explicit checkpoint is incomplete or missing: {explicit_checkpoint}"
            )
    else:
        complete_checkpoint = latest_readable_complete_checkpoint(checkpoint_dir)
        if complete_checkpoint is not None:
            latest_path = str(complete_checkpoint)
    print('Searching for a complete model checkpoint in ', checkpoint_dir)
    start_epoch = 0
    resume_batch_in_epoch = 0
    train_steps = 0
    resume_rng_state = None
    latest_checkpoint = None
    if latest_path is not None:
        print("Loading model from ", latest_path)
        latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False) 

        if exact_resume:
            validate_training_checkpoint(latest_checkpoint)
            saved_fingerprint = str(latest_checkpoint["config_sha256"])
            if (
                saved_fingerprint != config_sha256
                and not bool(config.get("allow_config_mismatch", False))
            ):
                raise ValueError(
                    "Resolved config does not match checkpoint: "
                    f"current={config_sha256}, saved={saved_fingerprint}"
                )
            saved_world_size = int(latest_checkpoint.get("world_size", dist.get_world_size()))
            if saved_world_size != dist.get_world_size():
                raise ValueError(
                    "Exact resume requires the same DDP world size: "
                    f"current={dist.get_world_size()}, saved={saved_world_size}"
                )

        if "model" in latest_checkpoint:
            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['model'].items()}
            res = model.load_state_dict(model_ckp, strict=True)
            print("Loading model weights", res)

            model_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['ema'].items()}
            res = ema.load_state_dict(model_ckp, strict=True)
            print("Loading EMA model weights", res)
        else:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

        if "opt" in latest_checkpoint:
            opt_ckp = {k.replace('_orig_mod.', ''):v for k,v in latest_checkpoint['opt'].items()}
            opt.load_state_dict(opt_ckp)
            print("Loading optimizer params")

            if override_lr_on_resume:
                for pg in opt.param_groups:
                    pg["lr"] = base_lr
                opt.defaults["lr"] = base_lr
                print(f"Override optimizer lr on resume: lr={base_lr}")

        if "epoch" in latest_checkpoint:
            if exact_resume and "batch_in_epoch" in latest_checkpoint:
                start_epoch = int(latest_checkpoint['epoch'])
                resume_batch_in_epoch = int(latest_checkpoint['batch_in_epoch'])
            else:
                start_epoch = int(latest_checkpoint['epoch']) + 1
        
        if "train_steps" in latest_checkpoint:
            train_steps = latest_checkpoint["train_steps"]
        
        if "scaler" in latest_checkpoint and bfloat_enable:
            scaler.load_state_dict(latest_checkpoint["scaler"])
        elif "scaler" in latest_checkpoint and exact_resume:
            raise ValueError("Checkpoint uses AMP scaling but --bfloat16 is disabled")

        if exact_resume:
            saved_args = latest_checkpoint.get("args")
            for argument_name in ("epochs", "global_seed", "bfloat16"):
                if saved_args is not None and getattr(saved_args, argument_name, None) != getattr(args, argument_name, None):
                    raise ValueError(
                        f"Exact resume argument mismatch for {argument_name}: "
                        f"current={getattr(args, argument_name, None)!r}, "
                        f"saved={getattr(saved_args, argument_name, None)!r}"
                    )
            rank_rng_states = latest_checkpoint["rng_states_by_rank"]
            if len(rank_rng_states) != dist.get_world_size():
                raise ValueError(
                    "Checkpoint RNG rank count does not match current world size"
                )
            resume_rng_state = rank_rng_states[rank]
        
    # ~40% speedup but might leads to worse performance depending on pytorch version
    if args.torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    # Compute the RAE dimension shift, clamped to the transport's valid domain.
    # The clamp is active only for natural low-dimensional latents (for example
    # C4 at 16x16); native C768 retains the released sqrt(C*H*W/base) value.
    time_dist_shift = resolve_time_dist_shift(
        latent_channels=int(rae.latent_dim),
        latent_size=latent_size,
        transport_config=tp,
    )
    if rank == 0:
        logger.info(
            "Time distribution shift contract: "
            f"latent_channels={int(rae.latent_dim)}, latent_size={latent_size}, "
            f"base={float(tp.get('time_dist_shift_base', 4096)):.8g}, "
            f"resolved={time_dist_shift:.8g}, minimum=1"
        )
    transport = Transport(
        model_type=getattr(ModelType, str(tp.get('model_type', 'velocity')).upper()),
        path_type=getattr(PathType, str(tp.get('path_type', 'linear')).upper()),
        loss_type=getattr(WeightType, str(tp.get('loss_type', 'velocity')).upper()),
        time_dist_type=str(tp.get('time_dist_type', 'uniform')),
        # time_dist_shift=float(tp.get('time_dist_shift', 1.0)),

        time_dist_shift=time_dist_shift,
        train_eps=1e-3,
        sample_eps=1e-3,
    )
    sampler_transport = Sampler(transport)
    logger.info(f"CDiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_datasets = []
    train_dataset_names = []
    test_datasets = []

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    goals_per_obs = int(data_config["goals_per_obs"])
                    if data_split_type == 'test':
                        goals_per_obs = 4 # standardize testing
                    
                    if "distance" in data_config:
                        min_dist_cat=data_config["distance"]["min_dist_cat"]
                        max_dist_cat=data_config["distance"]["max_dist_cat"]
                    else:
                        min_dist_cat=config["distance"]["min_dist_cat"]
                        max_dist_cat=config["distance"]["max_dist_cat"]

                    if "len_traj_pred" in data_config:
                        len_traj_pred=data_config["len_traj_pred"]
                    else:
                        len_traj_pred=config["len_traj_pred"]

                    dataset = TrainingDataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        min_dist_cat=min_dist_cat,
                        max_dist_cat=max_dist_cat,
                        len_traj_pred=len_traj_pred,
                        context_size=config["context_size"],
                        normalize=config["normalize"],
                        goals_per_obs=goals_per_obs,
                        transform=transform,
                        predefined_index=None,
                        traj_stride=1,
                        deterministic_sampling=bool(
                            config.get("deterministic_data_sampling", exact_resume)
                        ),
                        sample_seed=(
                            int(config.get("eval_data_seed", args.global_seed))
                            if data_split_type == "test"
                            else args.global_seed
                        ),
                        goal_offset_bins=(
                            None
                            if data_split_type == "test"
                            else config.get("goal_offset_bins")
                        ),
                        index_cache_folder=os.path.join(
                            str(config.get("dataset_index_root", data_config[data_split_type])),
                            dataset_name,
                            data_split_type,
                        ),
                    )
                    if data_split_type == "train":
                        train_datasets.append(dataset)
                        train_dataset_names.append(dataset_name)
                    else:
                        test_datasets.append(dataset)
                    print(f"Dataset: {dataset_name} ({data_split_type}), size: {len(dataset)}")

    print(f"Combining {len(train_datasets)} datasets.")
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    balanced_sampling = bool(config.get("balanced_sampling", True)) and len(train_datasets) > 1
    balanced_sampling_mode = str(config.get("balanced_sampling_mode", "downsample_only")).strip().lower()
    if balanced_sampling:
        min_len = min(len(d) for d in train_datasets)
        if balanced_sampling_mode == "downsample_only":
            desired_total_size = len(train_datasets) * min_len
        else:
            desired_total_size = len(train_dataset)

        sampler = BalancedDistributedSampler(
            train_datasets,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed,
            desired_total_size=desired_total_size,
        )
    else:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.global_seed + 100003 * rank)
    loader = DataLoader(
        train_dataset,
        batch_size=optimizer_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        generator=loader_generator,
    )
    data_schedule = validate_training_data_schedule(
        config,
        {
            name: len(dataset)
            for name, dataset in zip(train_dataset_names, train_datasets)
        },
        steps_per_epoch=len(loader),
        epochs=args.epochs,
    )
    logger.info(
        "Training data schedule: "
        + json.dumps(data_schedule, sort_keys=True)
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    total_steps = int(
        config.get(
            "lr_schedule_total_steps",
            args.epochs * len(loader),
        )
    )
    if total_steps <= 0:
        raise ValueError(
            f"lr_schedule_total_steps must be positive, got {total_steps}"
        )
    final_lr = float(config.get('final_lr', 2e-5))
    lr_schedule = str(config.get('lr_schedule', 'linear') or 'linear').strip().lower()

    if lr_schedule in {"cosine", "cos", "cosineannealing"}:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=final_lr)
    elif lr_schedule in {"linear", "lambda"}:
        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            alpha = step / float(total_steps)
            return (final_lr / base_lr) * alpha + (1 - alpha)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError(f"Unknown lr_schedule={lr_schedule}. Use 'linear' or 'cosine'.")
    logger.info(
        "LR schedule contract: "
        f"type={lr_schedule}, total_steps={total_steps}, "
        f"base_lr={base_lr:.8g}, final_lr={final_lr:.8g}"
    )
    
    if train_steps > 0:
        if override_lr_on_resume:
            scheduler.step(max(int(train_steps) - 1, 0))
            logger.info(
                f"Scheduler re-derived from config at resumed step={train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            if os.path.isfile(latest_path) or config.get('from_checkpoint', 0):
                latest_path = latest_path if os.path.isfile(latest_path) else config.get('from_checkpoint', 0)
                latest_checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
                if "scheduler" in latest_checkpoint:
                    scheduler.load_state_dict(latest_checkpoint["scheduler"])
                    logger.info(f"Scheduler state restored from checkpoint, current lr: {scheduler.get_last_lr()[0]:.6f}")
                else:
                    for _ in range(train_steps):
                        scheduler.step()
                    logger.info(f"Scheduler fast-forwarded to step {train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}")
            else:
                for _ in range(train_steps):
                    scheduler.step()
                logger.info(f"Scheduler fast-forwarded to step {train_steps}, current lr: {scheduler.get_last_lr()[0]:.6f}")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_transport_loss = 0
    running_alignment_loss = 0
    running_context_frames = 0
    start_time = time()

    if resume_rng_state is not None:
        restore_rng_state(resume_rng_state)
        logger.info(
            f"Restored exact RNG state at epoch={start_epoch}, "
            f"batch_in_epoch={resume_batch_in_epoch}, step={train_steps}"
        )
    else:
        # Tokenizer constructors consume different RNG streams.  Reset the
        # training RNG after all construction so paired arms see the same
        # flow noise, timesteps, and stochastic conditioning.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for child_dataset in train_datasets:
            if hasattr(child_dataset, "set_epoch"):
                child_dataset.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for batch_index, batch in enumerate(loader):
            if epoch == start_epoch and batch_index < resume_batch_in_epoch:
                continue
            x, y, rel_t = batch[:3]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            rel_t = rel_t.to(device, non_blocking=True)
            active_num_cond = sample_step_context_length(
                train_context_min,
                train_context_max,
                global_seed=args.global_seed,
                epoch=epoch,
                batch_index=batch_index,
            )
            if active_num_cond != int(num_cond):
                # The dataset loads the maximum history once. Keep the most
                # recent H frames and all goal frames. One H is shared by the
                # complete optimizer batch, including all microbatches.
                history = x[:, int(num_cond) - active_num_cond : int(num_cond)]
                goals = x[:, int(num_cond) :]
                x = torch.cat((history, goals), dim=1)
            opt.zero_grad()
            optimizer_batch_actual = int(x.shape[0])
            microbatch_slices = build_microbatch_slices(
                optimizer_batch_actual,
                min(train_microbatch_size, optimizer_batch_actual),
            )
            step_metrics = {
                name: torch.zeros((), device=device, dtype=torch.float32)
                for name in ("loss", "transport", "alignment")
            }
            optimizer_samples_per_rank = 0
            repa_mode = str(config.get('repa_mode', 'none')).lower()
            repa_target = str(config.get('repa_target', 'state')).lower()
            if repa_target not in {'state', 'multilayer_teacher'}:
                raise ValueError(f"Unsupported repa_target={repa_target!r}")

            for microbatch_index, microbatch_slice in enumerate(microbatch_slices):
                x_micro = x[microbatch_slice]
                y_micro = y[microbatch_slice]
                rel_t_micro = rel_t[microbatch_slice]
                microbatch_actual = int(x_micro.shape[0])
                loss_weight = microbatch_actual / float(optimizer_batch_actual)
                is_last_microbatch = microbatch_index + 1 == len(microbatch_slices)
                sync_context = (
                    contextlib.nullcontext()
                    if is_last_microbatch
                    else model.no_sync()
                )

                # DDP requires no_sync to cover both forward and backward.
                with sync_context:
                    with torch.amp.autocast(
                        'cuda',
                        enabled=bfloat_enable,
                        dtype=torch.bfloat16,
                    ):
                        # Encode history and future frames with the frozen
                        # visual interface.
                        with torch.no_grad():
                            B, T = x_micro.shape[:2]
                            x_pix = x_micro.flatten(0, 1) * 0.5 + 0.5
                            teacher_alignment_latent = None
                            if repa_target == 'multilayer_teacher':
                                if not hasattr(tokenizer, 'encode_with_multilayer_teacher'):
                                    raise RuntimeError(
                                        "repa_target=multilayer_teacher requires "
                                        "tokenizer.encode_with_multilayer_teacher"
                                    )
                                layer_indices = tuple(
                                    int(index)
                                    for index in config.get('repa_multilayer_indices', ())
                                )
                                if not layer_indices:
                                    raise RuntimeError(
                                        "repa_multilayer_indices must be non-empty"
                                    )
                                (
                                    x_latent,
                                    teacher_alignment_latent,
                                ) = tokenizer.encode_with_multilayer_teacher(
                                    x_pix,
                                    layer_indices=layer_indices,
                                )
                                teacher_alignment_latent = (
                                    teacher_alignment_latent.unflatten(0, (B, T))
                                )
                            else:
                                x_latent = tokenizer.encode(x_pix)
                            x_latent = x_latent.unflatten(0, (B, T))

                        # Every observation supplies num_goals future targets
                        # that share the same history.
                        num_goals = T - active_num_cond
                        x_start = x_latent[:, active_num_cond:].flatten(0, 1)
                        alignment_target = x_start
                        if teacher_alignment_latent is not None:
                            alignment_target = teacher_alignment_latent[
                                :, active_num_cond:
                            ].flatten(0, 1)
                        history = x_latent[:, :active_num_cond]
                        x_cond = (
                            history.unsqueeze(1)
                            .expand(B, num_goals, *history.shape[1:])
                            .flatten(0, 1)
                        )
                        model_kwargs = dict(
                            y=y_micro.flatten(0, 1),
                            x_cond=x_cond,
                            rel_t=rel_t_micro.flatten(0, 1),
                        )
                        if model.module.uses_cached_context:
                            # Summarize each history once and reuse it for all
                            # of its goals.
                            model_kwargs["context_group_size"] = num_goals

                        # Conditional flow matching (velocity regression).
                        transport_terms = transport.training_losses(
                            model,
                            x_start,
                            model_kwargs,
                        )
                        transport_loss = transport_terms.get(
                            "transport_loss",
                            transport_terms["loss"],
                        ).mean()

                        # Optional training-only iREPA alignment branch.
                        alignment_loss = torch.zeros(
                            (),
                            device=device,
                            dtype=transport_loss.dtype,
                        )
                        if repa_mode != 'none':
                            if "projected_features" not in transport_terms:
                                raise RuntimeError(
                                    f"repa_mode={repa_mode} requires projected_features"
                                )
                            projected = transport_terms["projected_features"]
                            repa_objective = str(
                                config.get('repa_objective', 'alignment')
                            ).lower()
                            if repa_objective == 'raev2_spatial_relation':
                                if teacher_alignment_latent is None:
                                    raise RuntimeError(
                                        "raev2_spatial_relation requires a "
                                        "multi-layer teacher target"
                                    )
                                weak_x_loss = clean_latent_prediction_loss(
                                    projected,
                                    x_start,
                                )
                                spatial_relation_loss = local_spatial_relation_loss(
                                    projected,
                                    alignment_target,
                                    radius=int(config.get('spatial_relation_radius', 2)),
                                    spatial_alpha=float(config.get('repa_spatial_alpha', 0.6)),
                                    eps=float(config.get('repa_eps', 1e-6)),
                                )
                                alignment_loss = (
                                    float(config.get('weak_x_coeff', 0.5)) * weak_x_loss
                                    + float(config.get('spatial_relation_coeff', 0.1))
                                    * spatial_relation_loss
                                ).mean()
                            elif repa_objective == 'alignment':
                                alignment_loss = representation_alignment_loss(
                                    projected,
                                    alignment_target,
                                    mode=repa_mode,
                                    spatial_alpha=float(config.get('repa_spatial_alpha', 1.0)),
                                    eps=float(config.get('repa_eps', 1e-6)),
                                ).mean()
                            else:
                                raise ValueError(
                                    f"Unsupported repa_objective={repa_objective!r}"
                                )
                        loss = (
                            transport_loss
                            + float(config.get('repa_coeff', 0.0)) * alignment_loss
                        )
                        scaled_microbatch_loss = loss * loss_weight

                    if bfloat_enable:
                        scaler.scale(scaled_microbatch_loss).backward()
                    else:
                        scaled_microbatch_loss.backward()

                step_metrics["loss"] += loss.detach().float() * loss_weight
                step_metrics["transport"] += (
                    transport_loss.detach().float() * loss_weight
                )
                step_metrics["alignment"] += (
                    alignment_loss.detach().float() * loss_weight
                )
                optimizer_samples_per_rank += int(x_cond.shape[0])

            clip_val = float(config.get('grad_clip_val', 1.0))
            if not bfloat_enable:
                if clip_val and clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                opt.step()
            else:
                if clip_val and clip_val > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
                scaler.step(opt)
                scaler.update()

            update_ema(
                ema,
                model.module,
                decay=float(config.get("ema_decay", 0.9999)),
            )
            scheduler.step()

            # Log loss values:
            running_loss += step_metrics["loss"].item()
            running_transport_loss += step_metrics["transport"].item()
            running_alignment_loss += step_metrics["alignment"].item()
            running_context_frames += int(active_num_cond)
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                samples_per_sec = (
                    dist.get_world_size()
                    * optimizer_samples_per_rank
                    * steps_per_sec
                )
                # Reduce loss history over all processes:
                averages = torch.tensor(
                    [running_loss, running_transport_loss, running_alignment_loss],
                    device=device,
                ) / log_steps
                dist.all_reduce(averages, op=dist.ReduceOp.SUM)
                avg_loss, avg_transport_loss, avg_alignment_loss = (
                    averages / dist.get_world_size()
                ).tolist()
                memory_gib = torch.tensor(
                    [
                        torch.cuda.max_memory_allocated(device) / (1024**3),
                        torch.cuda.max_memory_reserved(device) / (1024**3),
                    ],
                    device=device,
                )
                dist.all_reduce(memory_gib, op=dist.ReduceOp.MAX)
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Transport: {avg_transport_loss:.4f}, "
                    f"Alignment: {avg_alignment_loss:.4f}, "
                    f"Mean Context Frames: "
                    f"{running_context_frames / log_steps:.2f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"Samples/Sec: {samples_per_sec:.2f}, "
                    f"Max Allocated GiB: {memory_gib[0].item():.2f}, "
                    f"Max Reserved GiB: {memory_gib[1].item():.2f}"
                )
                torch.cuda.reset_peak_memory_stats(device)

                # Log to wandb
                if rank == 0 and wandb.run is not None:
                    log_dict = {
                        'metrics/training_loss': avg_loss,
                        'metrics/transport_loss': avg_transport_loss,
                        'metrics/alignment_loss': avg_alignment_loss,
                        'performance/steps_per_second': steps_per_sec,
                        'performance/samples_per_second': samples_per_sec,
                        'training/epoch': epoch,
                        'training/step': train_steps,
                        'training/learning_rate': scheduler.get_last_lr()[0],
                        'training/context_frames_mean': (
                            running_context_frames / log_steps
                        ),
                    }

                    wandb.log(log_dict, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                running_transport_loss = 0
                running_alignment_loss = 0
                running_context_frames = 0
                log_steps = 0
                start_time = time()


            epoch_boundary = batch_index + 1 == len(loader)
            checkpoint_due = train_steps > 0 and (
                (
                    args.ckpt_every > 0
                    and train_steps % args.ckpt_every == 0
                )
                or (
                    args.ckpt_every_epochs > 0
                    and epoch_boundary
                    and (epoch + 1) % args.ckpt_every_epochs == 0
                )
            )

            # Save a full-state checkpoint. Every rank participates so each
            # process-local RNG stream is preserved.
            if checkpoint_due:
                final_epoch_boundary = epoch_boundary and epoch + 1 == args.epochs
                checkpoint_path = save_checkpoint_with_step(
                    model,
                    ema,
                    opt,
                    args,
                    args.epochs if final_epoch_boundary else epoch,
                    0 if final_epoch_boundary else batch_index + 1,
                    train_steps,
                    checkpoint_dir,
                    config_identity,
                    config_sha256,
                    scaler if bfloat_enable else None,
                    scheduler,
                )
                if rank == 0:
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    if epoch_boundary:
                        epoch_alias = publish_checkpoint_alias(
                            checkpoint_path,
                            f"checkpoint_epoch_{epoch + 1:04d}.pth.tar",
                        )
                        logger.info(
                            f"Published epoch checkpoint alias: {epoch_alias}"
                        )
                dist.barrier()

            eval_snapshot_due = (
                train_steps > 0
                and args.eval_snapshot_every_epochs > 0
                and epoch_boundary
                and (epoch + 1) % args.eval_snapshot_every_epochs == 0
            )
            # Epochs that already have a full-state milestone use that file for
            # paper evaluation.  Odd five-epoch boundaries get an EMA-only
            # snapshot, never a resumable training checkpoint.
            if eval_snapshot_due and not checkpoint_due:
                if rank == 0:
                    eval_snapshot_path = save_evaluation_snapshot(
                        ema,
                        epoch + 1,
                        train_steps,
                        eval_snapshot_dir,
                        config_identity,
                        config_sha256,
                    )
                    logger.info(
                        "Saved inference-only evaluation snapshot (not resumable): "
                        f"{eval_snapshot_path}"
                    )
                dist.barrier()

            evaluation_due = train_steps > 0 and (
                (
                    args.eval_every > 0
                    and train_steps % args.eval_every == 0
                )
                or (
                    args.eval_every_epochs > 0
                    and epoch_boundary
                    and (epoch + 1) % args.eval_every_epochs == 0
                )
                or (
                    epoch_boundary
                    and (epoch + 1) in args.eval_at_epochs
                )
            )
            if evaluation_due:
                eval_start_time = time()
                save_dir = os.path.join(experiment_dir, str(train_steps))
                eval_metrics = evaluate(
                    ema,
                    tokenizer,
                    transport,
                    sampler_transport,
                    test_dataset,
                    rank,
                    int(config.get("eval_batch_size", config["batch_size"])),
                    config["num_workers"],
                    latent_size,
                    device,
                    save_dir,
                    int(config.get("eval_seed", args.global_seed + train_steps)),
                    bfloat_enable,
                    num_cond,
                    transport_cfg=config.get("transport", {}),
                    max_batches=int(config.get("eval_num_batches", 1)),
                    dreamsim_cache_dir=str(
                        config.get(
                            "dreamsim_cache_dir",
                            "${NWM_MODEL_ROOT}/dreamsim",
                        )
                    ),
                )
                dist.barrier()
                eval_end_time = time()
                eval_time = eval_end_time - eval_start_time
                if rank == 0:
                    eval_metrics = {
                        "step": int(train_steps),
                        "eval_time_seconds": float(eval_time),
                        **eval_metrics,
                    }
                    metrics_path = os.path.join(
                        experiment_dir,
                        f"eval_metrics_step_{train_steps:07d}.json",
                    )
                    temporary_path = f"{metrics_path}.tmp.{os.getpid()}"
                    with open(temporary_path, "w", encoding="utf-8") as handle:
                        json.dump(eval_metrics, handle, indent=2, sort_keys=True)
                        handle.write("\n")
                    os.replace(temporary_path, metrics_path)
                    scalar_eval_metrics = {
                        key: value
                        for key, value in eval_metrics.items()
                        if isinstance(value, (int, float))
                    }
                    logger.info(
                        f"(step={train_steps:07d}) Eval metrics: "
                        f"{json.dumps(scalar_eval_metrics, sort_keys=True)}; "
                        f"per_sample_artifact={metrics_path}"
                    )
                
                # Log evaluation results to wandb
                if rank == 0 and wandb.run is not None:
                    wandb.log(
                        {
                            f"eval/{key}": value
                            for key, value in eval_metrics.items()
                            if key != "step"
                            and isinstance(value, (int, float))
                        },
                        step=train_steps,
                    )

            max_train_steps = int(args.max_train_steps)
            reached_max_steps = max_train_steps > 0 and train_steps >= max_train_steps
            if (
                termination_requested["signal"] is None
                and termination_request_file
                and os.path.isfile(termination_request_file)
            ):
                termination_requested["signal"] = "request-file"
                logger.info(
                    "Observed scheduler termination request; saving a complete "
                    "checkpoint after the current optimizer step."
                )
            local_stop = torch.tensor(
                int(termination_requested["signal"] is not None or reached_max_steps),
                device=device,
                dtype=torch.int32,
            )
            dist.all_reduce(local_stop, op=dist.ReduceOp.MAX)
            if local_stop.item():
                if not checkpoint_due:
                    checkpoint_path = save_checkpoint_with_step(
                        model,
                        ema,
                        opt,
                        args,
                        epoch,
                        batch_index + 1,
                        train_steps,
                        checkpoint_dir,
                        config_identity,
                        config_sha256,
                        scaler if bfloat_enable else None,
                        scheduler,
                    )
                if rank == 0:
                    stop_reason = (
                        f"bounded smoke max_train_steps={max_train_steps}"
                        if reached_max_steps
                        else f"signal={termination_requested['signal']}"
                    )
                    logger.info(
                        "Stop checkpoint is complete and resumable: "
                        f"{checkpoint_path}; reason={stop_reason}"
                    )
                dist.barrier()
                cleanup()
                return

        resume_batch_in_epoch = 0

    # Publish the exact final optimizer state even when the total number of
    # steps is not divisible by --ckpt-every. Downstream evaluation consumes
    # only this complete-marker-protected pointer.
    final_checkpoint_already_saved = (
        args.ckpt_every_epochs > 0
        and args.epochs % args.ckpt_every_epochs == 0
    )
    if final_checkpoint_already_saved:
        final_checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_step_{train_steps}.pth.tar"
        )
    else:
        final_checkpoint_path = save_checkpoint_with_step(
            model,
            ema,
            opt,
            args,
            args.epochs,
            0,
            train_steps,
            checkpoint_dir,
            config_identity,
            config_sha256,
            scaler if bfloat_enable else None,
            scheduler,
        )
    if rank == 0:
        publish_latest_checkpoint(final_checkpoint_path)
        final_alias = publish_checkpoint_alias(
            final_checkpoint_path, "final.pth.tar"
        )
        logger.info(f"Saved final complete checkpoint to {final_checkpoint_path}")
        logger.info(f"Published final checkpoint alias: {final_alias}")
    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


@torch.no_grad()
def evaluate(model, rae, transport, sampler_transport, test_dataloaders, rank, batch_size, num_workers, latent_size, device, save_dir, seed, bfloat_enable, num_cond, transport_cfg=None, max_batches=1, dreamsim_cache_dir="${NWM_MODEL_ROOT}/dreamsim"):
    sampler = DistributedSampler(
        test_dataloaders,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=seed
    )

    g = torch.Generator()
    g.manual_seed(int(seed))

    def seed_worker(worker_id):
        worker_seed = (int(seed) + int(worker_id)) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader = DataLoader(
        test_dataloaders,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    eval_model = DINOImageDistance(device)
    metric_sums = {
        name: torch.zeros((), device=device, dtype=torch.float64)
        for name in (
            "nwm_dino",
            "e2e_dino",
            "tokenizer_dino",
            "latent_mse",
            "nwm_pixel_mse",
            "e2e_pixel_mse",
            "tokenizer_pixel_mse",
            "nonfinite_samples",
        )
    }
    per_sample_names = (
        "nwm_dino",
        "e2e_dino",
        "tokenizer_dino",
        "latent_mse",
        "nwm_pixel_mse",
        "e2e_pixel_mse",
        "tokenizer_pixel_mse",
    )
    local_per_sample = {name: [] for name in per_sample_names}
    local_sample_keys = []
    n_samples = torch.zeros((), device=device, dtype=torch.float64)
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(int(seed) + int(rank))

    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    saved = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            x, y, rel_t = batch[:3]
            context_rel_paths = batch[3] if len(batch) >= 4 else None
        else:
            raise ValueError(f"Unexpected eval batch format: {type(batch)} with len={len(batch) if isinstance(batch, (list, tuple)) else 'NA'}")
        x = x.to(device)
        y = y.to(device)
        rel_t = rel_t.to(device).flatten(0, 1)
        with torch.amp.autocast('cuda', enabled=bfloat_enable, dtype=torch.bfloat16):
            B, T = x.shape[:2]
            num_goals = T - num_cond
            if num_goals <= 0:
                num_goals = 1
                if rank == 0:
                    print(
                        "Eval fallback: "
                        f"T={T}, num_cond={num_cond} => no goals; "
                        "using num_goals=1",
                        flush=True,
                    )
            B_flat = B * num_goals
            x_flat = x.flatten(0,1)
            x_pix = x_flat * 0.5 + 0.5
            x_latent = rae.encode(x_pix)
            x_latent = x_latent.unflatten(0, (B, T))
            init = torch.randn(
                B_flat,
                int(rae.latent_dim),
                latent_size,
                latent_size,
                device=device,
                generator=eval_generator,
            )
            tp = transport_cfg or {}
            sampling_method = str(tp.get("sampling_method", "euler"))
            num_steps = int(tp.get("num_steps", 50))
            sample_fn = sampler_transport.sample_ode(
                sampling_method=sampling_method,
                num_steps=num_steps,
                atol=1e-6,
                rtol=1e-3,
                reverse=False
            )
            x_cond = x_latent[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x_latent.shape[2:]).flatten(0, 1)
            y_target = y.flatten(0, 1)
            rel_t_target = rel_t
            mobj = model.module if hasattr(model, "module") else model
            rif_prev = getattr(mobj, "return_intermediate_features", False)
            try:
                setattr(mobj, "return_intermediate_features", False)
                memory = None
                if mobj.uses_cached_context:
                    # Samples are laid out observation-major, with all goals
                    # for one observation contiguous. Prepare the long
                    # history once per observation and expand only the compact
                    # memory state to the goal batch.
                    base_context = x_cond[::num_goals]
                    memory = mobj.expand_rollout_memory(
                        mobj.prepare_rollout_memory(base_context),
                        repeats=num_goals,
                    )
                xs = sample_fn(
                    init,
                    mobj,
                    y=y_target,
                    x_cond=x_cond,
                    rel_t=rel_t_target,
                    memory=memory,
                )
            finally:
                try:
                    setattr(mobj, "return_intermediate_features", rif_prev)
                except Exception:
                    pass
            samples_latent = xs[-1]
            xs = None
            
            samples = rae.decode(samples_latent).float()
            samples = torch.nan_to_num(samples)
            samples = samples.clamp(0,1)
            
            x_cond_pixels = rae.decode(x_cond.flatten(0, 1)).float()
            x_cond_pixels = torch.nan_to_num(x_cond_pixels).clamp(0,1).unflatten(0, (B_flat, num_cond))
            x_pix_full = x_pix.unflatten(0, (B, T))
            x_cond_raw_pixels = x_pix_full[:, :num_cond].unsqueeze(1).expand(B, num_goals, num_cond, *x_pix_full.shape[2:]).flatten(0, 1)
            if T - num_cond > 0:
                x_start_raw_pixels = x_pix_full[:, num_cond:].flatten(0, 1)
            else:
                x_start_raw_pixels = x_pix_full[:, -1]
            
            finite_by_sample = torch.isfinite(samples_latent).flatten(1).all(dim=1)
            metric_sums["nonfinite_samples"] += (~finite_by_sample).sum().double()
            if not finite_by_sample.all():
                raise RuntimeError(
                    "Held-out NWM sampling produced non-finite latent values"
                )
            if T - num_cond > 0:
                x_start_latent = x_latent[:, num_cond:].flatten(0, 1)
                x_start_pixels = rae.decode(x_start_latent).float()
            else:
                x_start_latent = x_latent[:, -1]
                x_start_pixels = rae.decode(x_start_latent).float()
            x_start_pixels = torch.nan_to_num(x_start_pixels).clamp(0,1)
            raw_target = F.interpolate(
                x_start_raw_pixels.float(),
                size=samples.shape[-2:],
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
            
            
            if rank == 0 and saved < 10:
                n_to_save = min(int(samples.shape[0]), 10 - saved)
                for i in range(n_to_save):
                    _, ax = plt.subplots(1,3,dpi=256)
                    def to_uint8_image(img3ch):
                        img = img3ch.detach().float().clamp(0.0, 1.0)
                        return (img.permute(1,2,0).cpu().numpy()*255.0).astype('uint8')
                    ax[0].imshow(to_uint8_image(x_cond_raw_pixels[i, -1]))
                    ax[0].axis('off')
                    ax[1].imshow(to_uint8_image(x_start_raw_pixels[i]))
                    ax[1].axis('off')
                    ax[2].imshow(to_uint8_image(samples[i]))
                    ax[2].axis('off')
                    plt.tight_layout()
                    plt.savefig(f'{save_dir}/{saved}.png')
                    plt.close()
                    saved += 1

            nwm_dino = eval_model(x_start_pixels, samples)
            e2e_dino = eval_model(raw_target, samples)
            tokenizer_dino = eval_model(raw_target, x_start_pixels)
            per_sample_tensors = {
                "nwm_dino": nwm_dino,
                "e2e_dino": e2e_dino,
                "tokenizer_dino": tokenizer_dino,
                "latent_mse": (
                    (samples_latent.float() - x_start_latent.float())
                    .square()
                    .flatten(1)
                    .mean(dim=1)
                ),
                "nwm_pixel_mse": (
                    (samples.float() - x_start_pixels.float())
                    .square()
                    .flatten(1)
                    .mean(dim=1)
                ),
                "e2e_pixel_mse": (
                    (samples.float() - raw_target)
                    .square()
                    .flatten(1)
                    .mean(dim=1)
                ),
                "tokenizer_pixel_mse": (
                    (x_start_pixels.float() - raw_target)
                    .square()
                    .flatten(1)
                    .mean(dim=1)
                ),
            }
            for name, values in per_sample_tensors.items():
                local_per_sample[name].extend(
                    float(value) for value in values.detach().float().cpu().tolist()
                )
            if context_rel_paths is None:
                local_sample_keys.extend(
                    f"rank{rank}:batch{batch_idx}:sample{i}"
                    for i in range(int(samples.shape[0]))
                )
            else:
                goal_paths = context_rel_paths[num_cond : num_cond + num_goals]
                local_sample_keys.extend(
                    str(goal_paths[goal_idx][batch_item])
                    for batch_item in range(B)
                    for goal_idx in range(num_goals)
                )
            metric_sums["nwm_dino"] += nwm_dino.double().sum()
            metric_sums["e2e_dino"] += e2e_dino.double().sum()
            metric_sums["tokenizer_dino"] += tokenizer_dino.double().sum()
            for name in per_sample_names[3:]:
                metric_sums[name] += per_sample_tensors[name].double().sum()
            n_samples += float(samples.shape[0])

    for value in metric_sums.values():
        dist.all_reduce(value)
    dist.all_reduce(n_samples)
    if n_samples.item() <= 0:
        raise RuntimeError("Held-out evaluator produced zero samples")
    means = {
        name: float((value / n_samples).item())
        for name, value in metric_sums.items()
        if name != "nonfinite_samples"
    }
    means["nwm_pixel_psnr"] = float(
        -10.0 * math.log10(max(means["nwm_pixel_mse"], 1e-12))
    )
    means["e2e_pixel_psnr"] = float(
        -10.0 * math.log10(max(means["e2e_pixel_mse"], 1e-12))
    )
    means["tokenizer_pixel_psnr"] = float(
        -10.0 * math.log10(max(means["tokenizer_pixel_mse"], 1e-12))
    )
    means["eval_samples"] = int(n_samples.item())
    means["nonfinite_samples"] = int(metric_sums["nonfinite_samples"].item())
    gathered_sample_keys = [None for _ in range(dist.get_world_size())]
    gathered_per_sample = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_sample_keys, local_sample_keys)
    dist.all_gather_object(gathered_per_sample, local_per_sample)
    if rank == 0:
        sample_keys = [
            key for rank_keys in gathered_sample_keys for key in rank_keys
        ]
        per_sample = {
            name: [
                value
                for rank_values in gathered_per_sample
                for value in rank_values[name]
            ]
            for name in per_sample_names
        }
        expected = int(n_samples.item())
        lengths = {"sample_keys": len(sample_keys)}
        lengths.update({name: len(values) for name, values in per_sample.items()})
        if any(length != expected for length in lengths.values()):
            raise RuntimeError(
                f"Per-sample evaluation artifact is incomplete: "
                f"expected={expected}, lengths={lengths}"
            )
        means["eval_seed"] = int(seed)
        means["sample_keys"] = sample_keys
        means["per_sample"] = per_sample
    try:
        del eval_model
    except Exception:
        pass
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    return means

def save_checkpoint_with_step(
    model,
    ema,
    opt,
    args,
    epoch,
    batch_in_epoch,
    train_steps,
    checkpoint_dir,
    resolved_config,
    config_sha256,
    scaler=None,
    scheduler=None,
):
    """Gather per-rank RNG and atomically publish a full-state checkpoint."""

    if scheduler is None:
        raise ValueError("A scheduler state is required for an exact checkpoint")
    local_rng_state = capture_rng_state()
    rng_states_by_rank = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(rng_states_by_rank, local_rng_state)

    if dist.get_rank() != 0:
        return None

    unwrapped_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        "model": unwrapped_model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": args,
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "train_steps": int(train_steps),
        "world_size": int(dist.get_world_size()),
        "rng_states_by_rank": rng_states_by_rank,
        "config": deepcopy(resolved_config),
        "config_sha256": str(config_sha256),
    }
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    step_checkpoint_path = os.path.join(
        checkpoint_dir, f"checkpoint_step_{train_steps}.pth.tar"
    )
    completed = write_complete_checkpoint(checkpoint, step_checkpoint_path)
    publish_latest_checkpoint(completed)
    return str(completed)


def save_evaluation_snapshot(
    ema,
    epoch,
    train_steps,
    eval_snapshot_dir,
    resolved_config,
    config_sha256,
):
    """Publish EMA weights for paper evaluation without resume state."""

    unwrapped_ema = ema.module if hasattr(ema, "module") else ema
    snapshot = {
        "ema": unwrapped_ema.state_dict(),
        "epoch": int(epoch),
        "train_steps": int(train_steps),
        "config": deepcopy(resolved_config),
        "config_sha256": str(config_sha256),
    }
    path = os.path.join(
        eval_snapshot_dir, f"eval_epoch_{int(epoch):04d}.pth.tar"
    )
    return str(write_complete_evaluation_snapshot(snapshot, path))

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs; defaults to config.epochs, then 300 for legacy configs.",
    )
    # parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=2000)
    parser.add_argument(
        "--ckpt-every-epochs",
        type=int,
        default=0,
        help="Also save exact checkpoints at this epoch interval; 0 disables it.",
    )
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=0,
        help="Also run held-out evaluation at this epoch interval; 0 disables it.",
    )
    parser.add_argument(
        "--eval-at-epochs",
        type=str,
        default="",
        help=(
            "Comma-separated one-indexed epochs for held-out evaluation; "
            "for example, 10,50."
        ),
    )
    parser.add_argument(
        "--eval-snapshot-every-epochs",
        type=int,
        default=0,
        help=(
            "Publish an inference-only EMA snapshot at this epoch interval; "
            "epochs with a full checkpoint reuse that checkpoint."
        ),
    )
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Positive values stop after this many optimizer steps for bounded smoke runs.",
    )
    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
