# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC BY-NC 4.0 license found in
# LICENSES/CC-BY-NC-4.0-NWM.txt. Derived from facebookresearch/nwm and
# modified by the authors; see THIRD_PARTY.md.
# Based on isolated_nwm_eval.py from facebookresearch/nwm.
import os
import sys
import json
import pickle
import argparse
import gc
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.distributed as torch_dist
import nwm.utils.distributed as dist
import lpips
try:
    from dreamsim import dreamsim
    _dreamsim_available = True
except Exception:
    _dreamsim_available = False
from torcheval.metrics import FrechetInceptionDistance
from torchmetrics.functional.image import structural_similarity_index_measure


def _list_episode_dirs(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        return []
    out = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    out.sort()
    return out


# DINO Wrapper
class DINOMetric(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Production paper evaluation uses an explicit local no-register
        # DINOv2-B/14 checkout and checkpoint.  Loading the state ourselves
        # avoids network access and does not depend on a per-job timm/HF cache.
        repo_dir = os.environ.get("DINOV2_REPO_DIR")
        checkpoint = os.environ.get("DINOV2_CHECKPOINT")
        if repo_dir and checkpoint:
            if not os.path.isfile(os.path.join(repo_dir, "hubconf.py")):
                raise FileNotFoundError(f"Invalid local DINOv2 repository: {repo_dir}")
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(f"Missing local DINOv2 checkpoint: {checkpoint}")
            self.model = torch.hub.load(
                repo_dir,
                "dinov2_vitb14",
                source="local",
                pretrained=False,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            del state
            self.model = self.model.to(device)
        else:
            import timm

            self.model = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=True,
                img_size=224,
            ).to(device)
        self.model.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def features(self, img):
        with torch.no_grad():
            img = img.to(device=self.device, dtype=torch.float32)
            img = F.interpolate(
                img,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            mean = self.mean.to(device=img.device, dtype=img.dtype)
            std = self.std.to(device=img.device, dtype=img.dtype)
            img = (img - mean) / std
            feat = self.model.forward_features(img)
            # timm returns normalized CLS + 16x16 patch tokens, whereas the
            # torch.hub API returns a dictionary. Exclude CLS for paper parity.
            if isinstance(feat, dict):
                feat = feat["x_norm_patchtokens"]
            else:
                feat = feat[:, 1:]
            return F.normalize(feat.float(), dim=-1)

    def per_sample_distance(self, img0, img1):
        feat0 = self.features(img0)
        feat1 = self.features(img1)
        return (1.0 - (feat0 * feat1).sum(dim=-1)).mean(dim=-1)

    def forward(self, img0, img1):
        return self.per_sample_distance(img0, img1).mean()


def per_sample_psnr(img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
    """Paper-aligned RGB PSNR in dB for images in ``[0, 1]``."""
    img0 = img0.float().clamp(0.0, 1.0)
    img1 = img1.float().clamp(0.0, 1.0)
    mse = (img0 - img1).square().mean(dim=(1, 2, 3))
    values = -10.0 * torch.log10(mse.clamp_min(1.0e-10))
    return torch.where(mse == 0, torch.full_like(values, 100.0), values)


def per_sample_ssim(img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
    """Paper-aligned multichannel SSIM with scikit-image's 7px window."""
    return structural_similarity_index_measure(
        img0.float().clamp(0.0, 1.0),
        img1.float().clamp(0.0, 1.0),
        gaussian_kernel=False,
        kernel_size=7,
        data_range=1.0,
        reduction="none",
    )

def get_loss_fn(loss_fn_type, secs, device):
    if loss_fn_type == 'lpips':
        general_lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

        def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
            img0 = img0.to(device=device, dtype=torch.float32)
            img1 = img1.to(device=device, dtype=torch.float32)
            img0 = img0 * 2.0 - 1.0
            img1 = img1 * 2.0 - 1.0
            dist = general_lpips_loss_fn(img0, img1)
            return dist.mean()

        return loss_fn

    elif loss_fn_type == 'dreamsim':
        if _dreamsim_available:
            dreamsim_cache_dir = os.environ.get("DREAMSIM_CACHE_DIR") or (
                os.path.join(os.environ["NWM_MODEL_ROOT"], "dreamsim")
                if os.environ.get("NWM_MODEL_ROOT")
                else "./models"
            )
            dreamsim_loss_fn, preprocess = dreamsim(
                pretrained=True,
                device=device,
                cache_dir=dreamsim_cache_dir,
            )
            to_pil = transforms.ToPILImage()
            def _preprocess_batch(x: torch.Tensor) -> torch.Tensor:
                try:
                    out = preprocess(x)
                    if isinstance(out, torch.Tensor):
                        if out.dim() == 3:
                            out = out.unsqueeze(0)
                        return out.to(device)
                except Exception:
                    pass
                out = []
                x_cpu = x.detach().cpu()
                for i in range(x_cpu.shape[0]):
                    out.append(preprocess(to_pil(x_cpu[i])).to(device))
                return torch.cat(out, dim=0)
            def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
                img0 = img0.to(dtype=torch.float32)
                img1 = img1.to(dtype=torch.float32)
                all_img0 = _preprocess_batch(img0)
                all_img1 = _preprocess_batch(img1)
                dist = dreamsim_loss_fn(all_img0, all_img1)
                return dist.mean()
            return loss_fn
        else:
            def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
                return torch.tensor(float('nan'), device=device)
            return loss_fn

    elif loss_fn_type == 'fid':
        fid_metrics = {}
        for sec in secs:
            fid_metrics[sec] = FrechetInceptionDistance(feature_dim=2048).to(device)
        return fid_metrics

    elif loss_fn_type == 'dino':
        dino_fn = DINOMetric(device)

        def loss_fn(img0: torch.Tensor, img1: torch.Tensor):
            return dino_fn(img0, img1)

        return loss_fn

    else:
        raise NotImplementedError

def _sync_fid_metrics(fid_metrics, secs, device="cuda"):
    if not dist.is_dist_avail_and_initialized():
        return fid_metrics
    # Every formal evaluation currently launches one rank. Calling
    # all_gather_object in that case serializes the complete 2048-d FID state
    # and materializes a multi-GiB CUDA staging tensor just to gather the
    # object back to itself. The local accumulators are already complete.
    if dist.get_world_size() == 1:
        return fid_metrics

    secs_list = [int(s) for s in list(secs)]

    serialized = pickle.dumps(fid_metrics)
    gathered = [None] * dist.get_world_size()

    torch_dist.barrier()
    torch_dist.all_gather_object(gathered, serialized)

    final = {s: FrechetInceptionDistance(feature_dim=2048).to(device) for s in secs_list}
    for ser in gathered:
        curr = pickle.loads(ser)
        for s in secs_list:
            final[s].merge_state([curr[s]])

    return final


def evaluate(args, dataset_name, eval_type, metric_logger, loss_fns, gt_dir, exp_dir, secs, rollout_fps):
    (lpips_fn, dreamsim_fn, fid_fn, dino_fn) = loss_fns

    gt_eps = _list_episode_dirs(gt_dir)
    exp_eps = _list_episode_dirs(exp_dir)

    match_mode = str(getattr(args, "match_mode", "auto") or "auto").lower()
    if match_mode not in {"auto", "strict", "intersection"}:
        match_mode = "auto"

    gt_set = set(gt_eps)
    exp_set = set(exp_eps)

    if match_mode == "strict":
        if gt_set != exp_set:
            missing_in_pred = sorted(gt_set - exp_set)
            extra_in_pred = sorted(exp_set - gt_set)
            raise ValueError(
                f"GT/PRED episode sets differ. missing_in_pred={len(missing_in_pred)} extra_in_pred={len(extra_in_pred)}"
            )
        eps = gt_eps
    elif match_mode == "intersection":
        eps = sorted(gt_set & exp_set)
    else:
        if gt_set == exp_set:
            eps = gt_eps
        else:
            eps = sorted(gt_set & exp_set)
    if match_mode != "strict" and gt_set != exp_set and dist.is_main_process():
        print(
            f"WARNING: match_mode={match_mode} scores only the {len(eps)} episodes "
            f"present in both folders (gt={len(gt_set)}, pred={len(exp_set)}); "
            "the result is not comparable to a full-set evaluation."
        )

    max_trajs = int(getattr(args, "max_trajs", 0) or 0)
    if max_trajs > 0:
        eps = eps[:max_trajs]

    if len(eps) == 0:
        raise ValueError(f"No matched episodes under gt_dir={gt_dir} and exp_dir={exp_dir}")

    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    if world_size > 1:
        eps = eps[global_rank::world_size]
    if len(eps) == 0:
        return

    if eval_type == 'rollout':
        eval_name = f'rollout_{rollout_fps}fps'
        valid_eps = [ep for ep in eps if os.path.isdir(os.path.join(gt_dir, ep))]
        if len(valid_eps) == 0:
            return
        sample_ep_dir = os.path.join(gt_dir, valid_eps[0])
        frame_files = [f for f in os.listdir(sample_ep_dir) if f.endswith('.png')]
        max_idx = max(int(os.path.splitext(f)[0]) for f in frame_files)
        rollout_last_idx = getattr(args, "rollout_last_idx", None)
        if rollout_last_idx is not None:
            try:
                max_idx = min(int(max_idx), int(rollout_last_idx))
            except Exception:
                pass
        max_frames = max_idx + 1

        if rollout_fps == 1:
            image_idxs = np.arange(0, max_frames)
            secs = image_idxs.copy()
        else:
            first = np.arange(0, min(4, max_frames))
            every4 = np.arange(4, max_frames, 4)
            last = np.array([max_frames - 1])
            image_idxs = np.unique(np.concatenate([first, every4, last]))
            secs = image_idxs.copy()

        if fid_fn is not None:
            fid_fn = get_loss_fn('fid', secs, device='cuda')
    elif eval_type == 'time':
        eval_name = eval_type
        image_idxs = secs.copy()
        
    total_pairs = 0
    skipped_episodes = 0
    for batch_start in tqdm(
        range(0, len(eps), args.batch_size),
        total=(len(eps) + args.batch_size - 1) // args.batch_size,
        disable=not dist.is_main_process(),
    ):
        batch_eps = eps[batch_start:batch_start + args.batch_size]

        gt_batch, exp_batch = {}, {}
        for sec in secs:
            gt_batch[sec] = []
            exp_batch[sec] = []

        batch_pairs = 0
        for ep in batch_eps:
            gt_ep_dir = os.path.join(gt_dir, ep)
            exp_ep_dir = os.path.join(exp_dir, ep)

            if not os.path.isdir(gt_ep_dir) or not os.path.isdir(exp_ep_dir):
                if match_mode == "strict":
                    raise FileNotFoundError(f"Missing episode dir. gt={gt_ep_dir} pred={exp_ep_dir}")
                skipped_episodes += 1
                continue

            ok = True
            sec_items = []
            for sec, image_idx in zip(secs, image_idxs):
                gt_sec_img_path = os.path.join(gt_ep_dir, f"{int(image_idx)}.png")
                exp_sec_img_path = os.path.join(exp_ep_dir, f"{int(image_idx)}.png")
                if not os.path.isfile(gt_sec_img_path) or not os.path.isfile(exp_sec_img_path):
                    ok = False
                    break
                sec_items.append((sec, gt_sec_img_path, exp_sec_img_path))

            if not ok:
                if match_mode == "strict":
                    raise FileNotFoundError(f"Missing frame(s) under gt={gt_ep_dir} pred={exp_ep_dir}")
                skipped_episodes += 1
                continue

            for sec, gt_sec_img_path, exp_sec_img_path in sec_items:
                gt_sec_img = transforms.ToTensor()(Image.open(gt_sec_img_path).convert("RGB")).unsqueeze(0)
                exp_sec_img = transforms.ToTensor()(Image.open(exp_sec_img_path).convert("RGB")).unsqueeze(0)

                gt_batch[sec].append(gt_sec_img)
                exp_batch[sec].append(exp_sec_img)

            batch_pairs += 1

        if batch_pairs <= 0:
            continue
        total_pairs += batch_pairs

        for sec in secs:
            if len(gt_batch[sec]) == 0:
                continue

            sec_gt_batch = torch.cat(gt_batch[sec], dim=0).to('cuda', non_blocking=True)
            sec_exp_batch = torch.cat(exp_batch[sec], dim=0).to('cuda', non_blocking=True)

            if sec_exp_batch.shape[-2:] != sec_gt_batch.shape[-2:]:
                sec_exp_batch = F.interpolate(sec_exp_batch, size=sec_gt_batch.shape[-2:], mode="bicubic", align_corners=False,antialias=True)

            n = int(sec_gt_batch.shape[0])
            if n <= 0:
                continue

            if eval_type == 'rollout':
                tag = f'frame{sec}'
            else:
                tag = f'{sec}s'

            metric_functions = {
                "lpips": lpips_fn,
                "dreamsim": dreamsim_fn,
                "dino": dino_fn,
            }
            if "psnr" in set(args.metrics):
                value = float(per_sample_psnr(sec_gt_batch, sec_exp_batch).mean().cpu())
                metric_logger.meters[
                    f'{dataset_name}_{eval_name}_psnr_{tag}'
                ].update(value, n=n)
            if "ssim" in set(args.metrics):
                value = float(per_sample_ssim(sec_gt_batch, sec_exp_batch).mean().cpu())
                metric_logger.meters[
                    f'{dataset_name}_{eval_name}_ssim_{tag}'
                ].update(value, n=n)
            for metric_name, metric_fn in metric_functions.items():
                if metric_fn is None:
                    continue
                value = float(metric_fn(sec_gt_batch, sec_exp_batch).detach().cpu().item())
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite metric {metric_name} for {dataset_name}/{eval_name}/{tag}: {value}"
                    )
                metric_logger.meters[
                    f'{dataset_name}_{eval_name}_{metric_name}_{tag}'
                ].update(value, n=n)

            fid_gt = sec_gt_batch.clamp(0.0, 1.0)
            fid_exp = sec_exp_batch.clamp(0.0, 1.0)

            if fid_fn is not None:
                fid_fn[sec].update(images=fid_gt, is_real=True)
                fid_fn[sec].update(images=fid_exp, is_real=False)
            
    if skipped_episodes:
        print(
            f"WARNING: skipped {skipped_episodes} episode(s) with missing frames "
            f"under {exp_dir} (match_mode={match_mode})."
        )
    if total_pairs == 0:
        raise ValueError(
            f"No complete GT/prediction pairs were scored under gt_dir={gt_dir} "
            f"and exp_dir={exp_dir}; check --horizons/--num_sec_eval."
        )

    if fid_fn is not None:
        fid_fn = _sync_fid_metrics(fid_fn, secs, device="cuda")

        for sec in secs:
            if eval_type == 'rollout':
                tag = f'frame{sec}'
            else:
                tag = f'{sec}s'
            metric_logger.meters[f'{dataset_name}_{eval_name}_fid_{tag}'].update(fid_fn[int(sec)].compute().item(), n=1)

def save_metric_to_disk(metric_logger, log_p):
    metric_logger.synchronize_between_processes()
    if not dist.is_main_process():
        return
    log_stats = {k: float(meter.global_avg) for k, meter in metric_logger.meters.items()}
    non_finite = {key: value for key, value in log_stats.items() if not math.isfinite(value)}
    if non_finite:
        raise ValueError(f"Refusing to save non-finite metrics: {non_finite}")
    os.makedirs(os.path.dirname(log_p), exist_ok=True)
    temporary = f"{log_p}.tmp"
    with open(temporary, 'w') as json_file:
        json.dump(log_stats, json_file, indent=4)
        json_file.flush()
        os.fsync(json_file.fileno())
    os.replace(temporary, log_p)

def main(args):
    dist.init_distributed()
    device = 'cuda'

    # Loading Datasets
    dataset_names = args.datasets.split(',')
    
    if args.horizons:
        secs = np.array([int(h) for h in args.horizons.split(',') if h.strip()])
    else:
        secs = np.array([2**i for i in range(0, args.num_sec_eval)])
    
    metric_names = set(args.metrics)

    # Initialize only the requested loss functions. This keeps the required
    # DINO comparison independent from optional network/cache-heavy metrics.
    print("Initializing Metrics...")
    lpips_fn = get_loss_fn('lpips', secs, device) if "lpips" in metric_names else None
    dreamsim_fn = get_loss_fn('dreamsim', secs, device) if "dreamsim" in metric_names else None
    dino_fn = get_loss_fn('dino', secs, device) if "dino" in metric_names else None
    errors = []

    for dataset_name in dataset_names:
        gt_dataset_dir = os.path.join(args.gt_dir, dataset_name)
        exp_dataset_dir = os.path.join(args.exp_dir, dataset_name)
        
        if 'rollout' in args.eval_types:
            for rollout_fps in args.rollout_fps_values:
                rollout_fid_fn = None
                rollout_loss_fns = None
                try:
                    metric_logger = dist.MetricLogger(delimiter="  ")
                    print("Evaluating rollout", rollout_fps, dataset_name)
                    
                    eval_name = f'rollout_{rollout_fps}fps'
                    gt_dataset_rollout_dir = os.path.join(gt_dataset_dir, eval_name)
                    exp_dataset_rollout_dir = os.path.join(exp_dataset_dir, eval_name)
                    
                    # FID needs separate instance per sec usually if using torchmetrics accum style
                    rollout_fid_fn = get_loss_fn('fid', secs, device) if "fid" in metric_names else None
                    rollout_loss_fns = (lpips_fn, dreamsim_fn, rollout_fid_fn, dino_fn)
                    
                    with torch.no_grad():
                        evaluate(args, dataset_name, 'rollout', metric_logger, rollout_loss_fns, 
                                 gt_dataset_rollout_dir, exp_dataset_rollout_dir, secs, rollout_fps)
                    
                    output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                    save_metric_to_disk(metric_logger, output_fn)
                except Exception as e:
                    errors.append(f"{dataset_name}/rollout_{rollout_fps}fps: {type(e).__name__}: {e}")
                    print(f"Error in rollout eval: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # FID keeps one feature accumulator per rollout frame.
                    # Drop those references before constructing the time-horizon
                    # accumulators; otherwise both sets coexist and can exceed
                    # 16 GiB even though either phase fits independently.
                    rollout_loss_fns = None
                    rollout_fid_fn = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if 'time' in args.eval_types:
            time_fid_fn = None
            time_loss_fns = None
            try:
                metric_logger = dist.MetricLogger(delimiter="  ")
                print("Evaluating time", dataset_name)
                eval_name = 'time'
                gt_dataset_time_dir = os.path.join(gt_dataset_dir, eval_name)
                exp_dataset_time_dir = os.path.join(exp_dataset_dir, eval_name)
                
                time_fid_fn = get_loss_fn('fid', secs, device) if "fid" in metric_names else None
                time_loss_fns = (lpips_fn, dreamsim_fn, time_fid_fn, dino_fn)
                
                with torch.no_grad():
                    evaluate(args, dataset_name, eval_name, metric_logger, time_loss_fns, 
                             gt_dataset_time_dir, exp_dataset_time_dir, secs, None)
                
                output_fn = os.path.join(args.exp_dir, f'{dataset_name}_{eval_name}.json')
                save_metric_to_disk(metric_logger, output_fn)
            except Exception as e:
                errors.append(f"{dataset_name}/time: {type(e).__name__}: {e}")
                print(f"Error in time eval: {e}")
                import traceback
                traceback.print_exc()
            finally:
                time_loss_fns = None
                time_fid_fn = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if errors:
        raise RuntimeError("Evaluation failed:\n" + "\n".join(errors))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default=32, help="batch size (decrease if OOM)")
    parser.add_argument("--eval_types", type=str, default='time,rollout', help="evaluations")
    parser.add_argument("--disable_fid", type=int, default=0, help="disable FID computation")
    parser.add_argument(
        "--metrics",
        type=str,
        default="psnr,ssim,lpips,dreamsim,fid,dino",
        help="Comma-separated subset of psnr,ssim,lpips,dreamsim,fid,dino.",
    )
    parser.add_argument("--gt_dir", type=str, default=None, help="gt directory")
    parser.add_argument("--exp_dir", type=str, default=None, help="experiment directory")
    parser.add_argument(
        "--num_sec_eval",
        type=int,
        default=5,
        help="Number of power-of-two horizons: 5 gives 1,2,4,8,16 s; 6 adds 32 s.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=None,
        help="Explicit comma-separated horizons in seconds (e.g. 4,16); overrides --num_sec_eval.",
    )
    parser.add_argument("--datasets", type=str, default=None, help="dataset name")
    parser.add_argument("--input_fps", type=int, default=4, help="input fps")
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="rollout fps values")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")
    parser.add_argument(
        "--match_mode",
        type=str,
        default="strict",
        choices=["strict", "intersection", "auto"],
        help="GT/PRED matching: strict (default) fails on any missing episode or frame.",
    )
    parser.add_argument("--max_trajs", type=int, default=0, help="Max number of trajectories to evaluate (0 means no limit)")
    parser.add_argument(
        "--rollout_last_idx",
        type=int,
        default=None,
        help="Last rollout frame index to score (default: last ground-truth frame, i.e. 63 for 16 s or 127 for 32 s at 4 fps).",
    )
    
    args = parser.parse_args()
    
    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]
    args.eval_types = args.eval_types.split(',')
    requested_metrics = [name.strip() for name in args.metrics.split(',') if name.strip()]
    allowed_metrics = {"psnr", "ssim", "lpips", "dreamsim", "fid", "dino"}
    unknown_metrics = sorted(set(requested_metrics) - allowed_metrics)
    if unknown_metrics:
        parser.error(f"Unknown metrics: {unknown_metrics}")
    if args.disable_fid:
        requested_metrics = [name for name in requested_metrics if name != "fid"]
    if not requested_metrics:
        parser.error("At least one metric must be requested")
    args.metrics = requested_metrics
    
    main(args)
