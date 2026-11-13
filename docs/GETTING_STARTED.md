# Getting started

Installation, data, model assets, training and evaluation for
[Visual Representation and History Modeling for Navigation World Models](../README.md).
Run all commands from the repository root.

## Installation

```bash
conda create -y -n nwm python=3.11
conda activate nwm
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
conda install -y ffmpeg
```

The fused GDN kernels need a CUDA GPU with Triton. To run on a machine
without Triton, set `gdn_backend: torch` in the config.

## Data

Download RECON, SACSoN and SCAND and convert them with the
[visualnav-transformer](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling)
scripts (`process_bags.py`, `process_recon.py`):

```text
$NWM_DATA_ROOT/processed/
├── recon/<trajectory>/{0.jpg, 1.jpg, ..., traj_data.pkl}
├── sacson/<trajectory>/...
└── scand/<trajectory>/...
```

The train/test splits are in `data_splits/`. Each training epoch draws 62,813
observations from every dataset (the size of SCAND) and interleaves the three
streams; every observation supplies four future targets.

## Frozen visual interfaces

Configs resolve paths from environment variables:

```bash
export NWM_DATA_ROOT=/path/to/datasets          # contains processed/{recon,sacson,scand}
export NWM_MODEL_ROOT=/path/to/models           # frozen encoders, decoders and statistics
export NWM_RUN_ROOT=/path/to/runs               # checkpoints and evaluation outputs
export PAE_SOURCE_ROOT=/path/to/PAE             # git checkout, commit c054d54 (patched below)
export VJEPA_SOURCE_ROOT=/path/to/vjepa2        # git checkout, commit 204698b
export SEMANTIC_WM_SOURCE_ROOT=/path/to/semantic-wm   # git checkout, commit 2fd7618
```

Expected layout of `$NWM_MODEL_ROOT`:

| Path | Source |
|---|---|
| `rae/dinov2-b/{decoder.pt, stats.pt}` | [`nyu-visionx/RAE-collections`](https://huggingface.co/nyu-visionx/RAE-collections) `decoders/dinov2/wReg_base/ViTXL_n08/model.pt`, `stats/dinov2/wReg_base/imagenet1k/stat.pt` |
| `rae/dinov2-l/{decoder.pt, stats.pt}` | same repository, `wReg_large` |
| `dinov2-with-registers-large/` | [`facebook/dinov2-with-registers-large`](https://huggingface.co/facebook/dinov2-with-registers-large) @ `e4c89a4` (used by PAE) |
| `pae-official/AE-models/dinov2-large.pt`, `pae-official/Latent-stats/PAE_DINOv2L.pt` | [`yuezhengrong/PAE-collections`](https://huggingface.co/yuezhengrong/PAE-collections) |
| `tokenizers/cosmos-ci16x16/` | [`nvidia/Cosmos-0.1-Tokenizer-CI16x16`](https://huggingface.co/nvidia/Cosmos-0.1-Tokenizer-CI16x16) |
| `tokenizers/vjepa2.1/vjepa2_1_vitl_dist_vitG_384.pt` | [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2) ViT-L |
| `tokenizers/semantic-wm/vjepa/adapter_vjepa_image_96.pt` | S-VAE adapter and decoder from [semantic-wm](https://github.com/chandar-lab/semantic-wm) |
| `dreamsim/` | cache directory for DreamSim (override with `DREAMSIM_CACHE_DIR`) |

Patch the PAE checkout once after cloning it:

```bash
python tools/patch_pae_source.py --source-root $PAE_SOURCE_ROOT --asset-manifest config/assets/pae.json
```

The paper's DINO metric uses the official no-register DINOv2-B/14 from
[facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
(`dinov2_vitb14_pretrain.pth`). Point `evaluate.py` to a local checkout and
checkpoint; without these variables it falls back to the timm
`vit_base_patch14_dinov2.lvd142m` weights, which give slightly different values.

```bash
export DINOV2_REPO_DIR=/path/to/dinov2
export DINOV2_CHECKPOINT=/path/to/dinov2_vitb14_pretrain.pth
```

The DINOv2-B encoder for RAE-B is fetched from the Hugging Face Hub at the
revision pinned in `RAE/configs/stage1/DINOv2-B.yaml`. File sizes and SHA-256
hashes are listed in `config/assets/`.

## Training

Every model is trained on 2 GPUs (48 observations per rank, global batch 96)
with AdamW, a linear learning-rate schedule from 2e-4 to 2e-6, gradient
clipping at 1.0, BF16, EMA and seed 42.

```bash
torchrun --nproc_per_node=2 train.py --config config/main/rae_b_balanced_gdn.yaml --global-seed 42
```

| Experiment | Configs |
|---|---|
| Study I (10 epochs) | `config/study1_interfaces/{rae_b, rae_l, pae_l, cosmos, vjepa}.yaml` |
| Main (50 epochs) | `config/main/rae_b_{global_softmax, cached_linear, balanced_gdn}.yaml`, `config/main/{pae_l, vjepa}_balanced_gdn{,_irepa}.yaml` |
| Component ablation (10 epochs) | Global-Softmax cells for RAE-B and V-JEPA reuse Study I; the remaining RAE-B, PAE-L and V-JEPA cells use the epoch-10 checkpoints of the main runs; `config/ablation/*.yaml` trains the four cells without a main counterpart |

Each config checks the expected world size, per-dataset sizes, steps per epoch
and trainable-parameter budget at start-up, so a changed data split or GPU
count fails immediately instead of silently changing the protocol.

## Evaluation

All comparisons use a 50-step Euler sampler with paired sample ids and paired
initial noise. The paper evaluates fixed sample sets, shipped in
`data_splits/eval_indices/`:

| Index | Future frames | RECON / SACSoN / SCAND samples |
|---|---|---|
| `full16` | 64 (16 s at 4 fps) | 598 / 199 / 42 |
| `full32` | 128 (32 s at 4 fps) | 150 / 112 / 37 |

Pass the same `--eval_index_dir` when writing ground truth and predictions.
Without it, every stride-8 window of every test trajectory is used, which is a
different and much larger set.

**Direct prediction and rollout.** `--eval_type time` predicts each horizon
from real history; `--eval_type rollout` appends predicted latents to the
history at 4 fps without decoding and re-encoding.

```bash
IDX=data_splits/eval_indices/full16
OUT=$NWM_RUN_ROOT/eval16

# ground-truth frames -> $OUT/gt
torchrun --standalone --nproc_per_node=1 infer.py \
  --exp config/main/rae_b_balanced_gdn.yaml --datasets sacson --eval_type time \
  --gt 1 --input_fps 4 --num_sec_eval 5 --eval_index_dir $IDX --output_dir $OUT --seed 42

# direct predictions at 1, 2, 4, 8, 16 s -> $OUT/<run>_euler_action-<mode>
torchrun --standalone --nproc_per_node=1 infer.py \
  --exp config/main/rae_b_balanced_gdn.yaml --datasets sacson --eval_type time \
  --checkpoint_path /path/to/checkpoint.pth.tar --input_fps 4 --num_sec_eval 5 \
  --eval_index_dir $IDX --sampling_method euler --num_steps 50 --output_dir $OUT --seed 42

# 4 fps rollout: repeat both commands with
#   --eval_type rollout --rollout_fps_values 4
```

`--num_sec_eval N` evaluates the power-of-two horizons 1, 2, ..., 2^(N-1) s.
For 32-second endpoints use `IDX=data_splits/eval_indices/full32`,
`--num_sec_eval 6` and add `--eval_len_traj_pred 128 --eval_min_dist_cat -128 --eval_max_dist_cat 128`.

**Decoded-image metrics.**

```bash
torchrun --standalone --nproc_per_node=1 evaluate.py \
  --datasets sacson --gt_dir <gt_dir> --exp_dir <prediction_dir> \
  --eval_types time,rollout --rollout_fps_values 4 --input_fps 4 --num_sec_eval 5 \
  --metrics dino,lpips,dreamsim,psnr,ssim,fid
```

`<gt_dir>` is `$OUT/gt` and `<prediction_dir>` the prediction folder written
above; use `--num_sec_eval 6` for the 32-second set. Matching is strict by
default: a missing episode or frame is an error rather than a smaller sample.
Rollout frames are scored up to the last ground-truth frame (frame 63 for
`full16`, 127 for `full32`).

**Reconstruction reference** `D(E(x))`:

```bash
python tools/generate_tokenizer_floor.py --config config/study1_interfaces/pae_l.yaml \
  --output-dir <out> --eval-index-dir data_splits/eval_indices/full16 --datasets sacson,recon,scand
```

It writes the 4 s and 16 s frames; score them with `evaluate.py --horizons 4,16`.

**Open-loop trajectory prediction** (two seconds, eight steps, CEM with 120
candidates, top-3, one update, latent-token scoring):

```bash
torchrun --standalone --nproc_per_node=1 planning_eval.py \
  --exp config/main/rae_b_global_softmax.yaml --checkpoint_path <ckpt> --datasets sacson \
  --output_dir <out> --batch_size 1 --num_samples 120 --topk 3 --opt_steps 1 \
  --num_repeat_eval 1 --rollout_stride 1 --score_type dino --traj_sampler curve \
  --eval_index_dir data_splits/eval_indices/full16 --num_steps 50 --seed 42
```

`--score_type dino` scores candidates by cosine distance in the
representation's own latent tokens (DINO features for RAE); this is the score
used for every planning result in the paper.

**Efficiency benchmark** (model core only; tokenizer, decoder, data loading and
communication are excluded):

```bash
python tools/benchmark_scalable_nwm.py --mode cuda \
  --scenarios 224x8,224x16,224x32,224x64,224x128 \
  --batch-size 120 --context-group-size 120 --input-dtype bfloat16 \
  --include-training 0 --output systems.json
```

Without `--variant`, the benchmark runs the three configs in `config/systems/`.
