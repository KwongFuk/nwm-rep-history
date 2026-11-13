# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the CC BY-NC 4.0 license found in
# LICENSES/CC-BY-NC-4.0-NWM.txt. Derived from facebookresearch/nwm and
# modified by the authors; see THIRD_PARTY.md.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple
from timm.models.vision_transformer import PatchEmbed
from RAE.src.stage2.models.model_utils import RMSNorm, NormAttention, SwiGLUFFN, VisionRotaryEmbeddingFast
from nwm.models.alignment import RepresentationAlignmentHead
from nwm.models.gdn import FrameGatedDeltaMixer


def modulate(x, shift, scale):
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    if shift.dim() != 3 or shift.shape[1] not in {1, x.shape[1]}:
        raise ValueError(
            "Token modulation expects [B,D], [B,1,D], or [B,N,D] "
            f"conditions, got x={tuple(x.shape)}, shift={tuple(shift.shape)}"
        )
    return x * (1 + scale) + shift


def modulate_context(x, shift, scale):
    """Modulate frame-major context tokens with target-token conditions."""

    if shift.dim() == 2:
        return modulate(x, shift, scale)
    if shift.dim() != 3 or x.shape[1] % shift.shape[1] != 0:
        raise ValueError(
            "Context modulation requires its token count to be a multiple of "
            f"the timestep-token count, got x={tuple(x.shape)}, "
            f"shift={tuple(shift.shape)}"
        )
    repeats = x.shape[1] // shift.shape[1]
    shift = shift.repeat(1, repeats, 1)
    scale = scale.repeat(1, repeats, 1)
    return x * (1 + scale) + shift


def apply_gate(x, gate):
    if gate.dim() == 2:
        gate = gate.unsqueeze(1)
    if gate.dim() != 3 or gate.shape[1] not in {1, x.shape[1]}:
        raise ValueError(
            "Token gate expects [B,D], [B,1,D], or [B,N,D], got "
            f"x={tuple(x.shape)}, gate={tuple(gate.shape)}"
        )
    return x * gate


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        gate = gate.repeat_interleave(repeat, dim=1)
    return x * gate


def _resolve_attention_layout(layout, depth, *, default, name):
    """Return an explicit per-block spatial-attention layout."""

    if layout is None:
        values = [default] * int(depth)
    elif isinstance(layout, str):
        values = [
            value.strip().lower()
            for value in layout.split(",")
            if value.strip()
        ]
        if len(values) == 1:
            values *= int(depth)
    else:
        values = [str(value).strip().lower() for value in layout]
    if len(values) != int(depth):
        raise ValueError(
            f"{name} must contain exactly {depth} entries, got {len(values)}"
        )
    supported = {"global", "window", "shifted_window"}
    unsupported = sorted(set(values) - supported)
    if unsupported:
        raise ValueError(
            f"{name} contains unsupported entries {unsupported}; "
            f"expected only {sorted(supported)}"
        )
    return tuple(values)


def _resolve_self_mixer_layout(layout, depth, *, name="self_mixer_layout"):
    """Canonicalize an explicit per-block mixed Self-attention layout."""

    if layout is None:
        return None
    if isinstance(layout, str):
        values = [value.strip().lower() for value in layout.split(",") if value.strip()]
    else:
        values = [str(value).strip().lower() for value in layout]
    if len(values) != int(depth):
        raise ValueError(
            f"{name} must contain exactly {depth} entries, got {len(values)}"
        )
    aliases = {
        "g": "gdn",
        "gdn": "gdn",
        "w": "window",
        "window": "window",
        "swa": "shifted_window",
        "shifted_window": "shifted_window",
        "f": "full",
        "full": "full",
        "global": "full",
        "softmax": "full",
    }
    unsupported = sorted({value for value in values if value not in aliases})
    if unsupported:
        raise ValueError(
            f"{name} contains unsupported entries {unsupported}; "
            f"expected only {sorted(aliases)}"
        )
    return tuple(aliases[value] for value in values)


class WindowAttention(nn.Module):
    """Exact softmax attention inside regular or shifted 2D windows.

    Q/K RoPE is applied before partitioning, so local attention retains the
    global 2D coordinates used by the original full-attention block.  The
    projection surface intentionally mirrors ``NormAttention`` so canonical
    weights can be copied between global and window ablations.  Shifted
    windows use the standard cyclic-shift mask, so opposite image boundaries
    never become artificial neighbors.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int,
        shift_size: int = 0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size"
            )
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _partition(self, tensor, grid_hw):
        batch, heads, tokens, head_dim = tensor.shape
        height, width = (int(grid_hw[0]), int(grid_hw[1]))
        if height * width != tokens:
            raise ValueError(
                f"grid_hw={grid_hw} does not match token count {tokens}"
            )
        window = min(self.window_size, height, width)
        if height % window != 0 or width % window != 0:
            raise ValueError(
                f"grid_hw={grid_hw} must be divisible by effective "
                f"window_size={window}"
            )
        windows_h = height // window
        windows_w = width // window
        tensor = tensor.transpose(1, 2).reshape(
            batch, height, width, heads, head_dim
        )
        tensor = tensor.reshape(
            batch,
            windows_h,
            window,
            windows_w,
            window,
            heads,
            head_dim,
        )
        tensor = tensor.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            batch * windows_h * windows_w,
            heads,
            window * window,
            head_dim,
        )
        metadata = (batch, height, width, windows_h, windows_w, window)
        return tensor, metadata

    @staticmethod
    def _unpartition(tensor, metadata):
        batch, height, width, windows_h, windows_w, window = metadata
        _, heads, _, head_dim = tensor.shape
        tensor = tensor.reshape(
            batch,
            windows_h,
            windows_w,
            heads,
            window,
            window,
            head_dim,
        )
        tensor = tensor.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch, height * width, heads * head_dim
        )
        return tensor

    @staticmethod
    def _shift_mask(
        *,
        height: int,
        width: int,
        window: int,
        shift: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros((1, height, width, 1), device=device)
        height_slices = (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )
        width_slices = (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )
        region = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                mask[:, height_slice, width_slice, :] = region
                region += 1
        windows_h = height // window
        windows_w = width // window
        mask = mask.reshape(
            1,
            windows_h,
            window,
            windows_w,
            window,
            1,
        )
        mask = mask.permute(0, 1, 3, 5, 2, 4).reshape(
            windows_h * windows_w,
            window * window,
        )
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        return mask.ne(0).to(dtype=dtype) * torch.finfo(dtype).min

    def forward(self, x, *, grid_hw, rope=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        height, width = (int(grid_hw[0]), int(grid_hw[1]))
        window = min(self.window_size, height, width)
        shift = self.shift_size if window < min(height, width) else 0
        if shift:
            q = torch.roll(
                q.reshape(batch, self.num_heads, height, width, self.head_dim),
                shifts=(-shift, -shift),
                dims=(2, 3),
            ).reshape(batch, self.num_heads, tokens, self.head_dim)
            k = torch.roll(
                k.reshape(batch, self.num_heads, height, width, self.head_dim),
                shifts=(-shift, -shift),
                dims=(2, 3),
            ).reshape(batch, self.num_heads, tokens, self.head_dim)
            v = torch.roll(
                v.reshape(batch, self.num_heads, height, width, self.head_dim),
                shifts=(-shift, -shift),
                dims=(2, 3),
            ).reshape(batch, self.num_heads, tokens, self.head_dim)
        q, metadata = self._partition(q.to(v.dtype), grid_hw)
        k, key_metadata = self._partition(k.to(v.dtype), grid_hw)
        v, value_metadata = self._partition(v, grid_hw)
        if metadata != key_metadata or metadata != value_metadata:
            raise RuntimeError("window partition metadata mismatch")
        attention_mask = None
        if shift:
            attention_mask = self._shift_mask(
                height=height,
                width=width,
                window=window,
                shift=shift,
                device=q.device,
                dtype=q.dtype,
            )
            attention_mask = attention_mask.repeat(batch, 1, 1).unsqueeze(1)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        output = self._unpartition(output, metadata)
        if shift:
            output = torch.roll(
                output.reshape(batch, height, width, channels),
                shifts=(shift, shift),
                dims=(1, 2),
            ).reshape(batch, tokens, channels)
        if output.shape != (batch, tokens, channels):
            raise RuntimeError(
                f"Unexpected window output shape {tuple(output.shape)}"
            )
        return self.proj_drop(self.proj(output))


class CrossLinearMemory(NamedTuple):
    key_value: torch.Tensor
    key_sum: torch.Tensor


class CrossWindowMemory(NamedTuple):
    """Reusable per-window K/V for shifted-window cross-attention."""

    key: torch.Tensor
    value: torch.Tensor
    height: int
    width: int
    window: int
    shift: int
    frames: int


class CachedLinearCrossAttention(nn.Module):
    """ELU+1 kernel linear cross-attention with a reusable context summary."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        eps: float = 1.0e-6,
        feature_scaling: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.eps = float(eps)
        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, 2 * self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        # Optional learned calibration replaces MultiheadAttention bias_k/v in
        # strictly parameter-matched SL experiments. It stays opt-in so older
        # cached-linear checkpoints retain their exact parameter surface.
        if bool(feature_scaling):
            self.q_feature_scale = nn.Parameter(torch.ones(self.dim))
            self.k_feature_scale = nn.Parameter(torch.ones(self.dim))
        else:
            self.q_feature_scale = None
            self.k_feature_scale = None

    @staticmethod
    def feature_map(tensor: torch.Tensor) -> torch.Tensor:
        return F.elu(tensor) + 1.0

    def prepare_context(self, context: torch.Tensor) -> CrossLinearMemory:
        batch, tokens, channels = context.shape
        key, value = self.kv(context).chunk(2, dim=-1)
        if self.k_feature_scale is not None:
            key = key * self.k_feature_scale
        key = key.reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.feature_map(key)
        key_value = key.transpose(-2, -1) @ value
        key_sum = key.sum(dim=-2)
        return CrossLinearMemory(key_value=key_value, key_sum=key_sum)

    def query(
        self,
        query: torch.Tensor,
        memory: CrossLinearMemory,
    ) -> torch.Tensor:
        batch, tokens, channels = query.shape
        context_batch = int(memory.key_value.shape[0])
        if context_batch <= 0 or batch % context_batch != 0:
            raise ValueError(
                f"query batch {batch} is not divisible by cached batch "
                f"{context_batch}"
            )
        group_size = batch // context_batch
        projected = self.q(query).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        if self.q_feature_scale is not None:
            projected = projected * self.q_feature_scale.reshape(
                1, self.num_heads, 1, self.head_dim
            )
        projected = self.feature_map(projected)
        projected = projected.reshape(
            context_batch,
            group_size,
            self.num_heads,
            tokens,
            self.head_dim,
        )
        numerator = torch.matmul(
            projected,
            memory.key_value.unsqueeze(1),
        )
        denominator = (
            projected * memory.key_sum.unsqueeze(1).unsqueeze(-2)
        ).sum(dim=-1, keepdim=True)
        output = numerator / denominator.clamp_min(self.eps)
        output = output.permute(0, 1, 3, 2, 4).reshape(
            batch, tokens, channels
        )
        return self.proj(output)


class CachedShiftedWindowCrossAttention(nn.Module):
    """Cached exact-softmax cross-attention in shifted spatial windows.

    Context frames share a spatial grid.  K/V are projected and partitioned
    once, then every target window attends to the corresponding shifted window
    in every context frame.  The Swin cyclic-shift mask prevents wrapped image
    boundaries from becoming artificial neighbors.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        window_size: int,
        shift_size: int,
        qkv_bias: bool = True,
        mask_layout: str = "legacy",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={num_heads}")
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        mask_layout = str(mask_layout).lower()
        if mask_layout not in SHIFTED_WINDOW_CROSS_MASKS:
            raise ValueError(
                f"shifted_window_cross_mask must be one of "
                f"{sorted(SHIFTED_WINDOW_CROSS_MASKS)}, got {mask_layout!r}"
            )
        self.mask_layout = mask_layout
        if not 0 <= int(shift_size) < int(window_size):
            raise ValueError(
                "shift_size must satisfy 0 <= shift_size < window_size"
            )
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.kv = nn.Linear(self.dim, 2 * self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    @staticmethod
    def _effective_geometry(grid_hw, window_size: int, shift_size: int):
        height, width = (int(grid_hw[0]), int(grid_hw[1]))
        if height <= 0 or width <= 0:
            raise ValueError(f"grid_hw must be positive, got {grid_hw}")
        window = min(int(window_size), height, width)
        if height % window != 0 or width % window != 0:
            raise ValueError(
                f"grid_hw={grid_hw} must be divisible by effective "
                f"window_size={window}"
            )
        shift = int(shift_size) if window < min(height, width) else 0
        if shift >= window:
            raise ValueError(
                f"effective shift_size={shift} must be smaller than window={window}"
            )
        return height, width, window, shift

    def _partition_query(self, tensor, *, height, width, window, shift):
        batch, heads, tokens, head_dim = tensor.shape
        if tokens != height * width:
            raise ValueError(
                f"query grid {(height, width)} does not match {tokens} tokens"
            )
        tensor = tensor.reshape(batch, heads, height, width, head_dim)
        if shift:
            tensor = torch.roll(tensor, shifts=(-shift, -shift), dims=(2, 3))
        windows_h, windows_w = height // window, width // window
        return tensor.reshape(
            batch,
            heads,
            windows_h,
            window,
            windows_w,
            window,
            head_dim,
        ).permute(0, 2, 4, 1, 3, 5, 6).reshape(
            batch,
            windows_h * windows_w,
            heads,
            window * window,
            head_dim,
        )

    def _partition_context(
        self,
        tensor,
        *,
        frames,
        height,
        width,
        window,
        shift,
    ):
        batch, heads, tokens, head_dim = tensor.shape
        spatial = height * width
        if tokens != frames * spatial:
            raise ValueError(
                f"context grid {frames}x{height}x{width} does not match "
                f"{tokens} tokens"
            )
        tensor = tensor.reshape(
            batch, heads, frames, height, width, head_dim
        )
        if shift:
            tensor = torch.roll(tensor, shifts=(-shift, -shift), dims=(3, 4))
        windows_h, windows_w = height // window, width // window
        return tensor.reshape(
            batch,
            heads,
            frames,
            windows_h,
            window,
            windows_w,
            window,
            head_dim,
        ).permute(0, 3, 5, 1, 2, 4, 6, 7).reshape(
            batch,
            windows_h * windows_w,
            heads,
            frames * window * window,
            head_dim,
        )

    @staticmethod
    def _unpartition_query(tensor, *, height, width, window, shift):
        batch, _, heads, _, head_dim = tensor.shape
        windows_h, windows_w = height // window, width // window
        tensor = tensor.reshape(
            batch,
            windows_h,
            windows_w,
            heads,
            window,
            window,
            head_dim,
        ).permute(0, 3, 1, 4, 2, 5, 6).reshape(
            batch, heads, height, width, head_dim
        )
        if shift:
            tensor = torch.roll(tensor, shifts=(shift, shift), dims=(2, 3))
        return tensor.permute(0, 2, 3, 1, 4).reshape(
            batch, height * width, heads * head_dim
        )

    @staticmethod
    def _apply_spatial_rope(tensor, *, frames: int, rope):
        if rope is None:
            return tensor
        batch, heads, tokens, head_dim = tensor.shape
        if tokens % int(frames) != 0:
            raise ValueError("context token count must be divisible by frames")
        spatial = tokens // int(frames)
        tensor = tensor.reshape(batch, heads, frames, spatial, head_dim)
        tensor = tensor.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, heads, spatial, head_dim
        )
        tensor = rope(tensor)
        return tensor.reshape(batch, frames, heads, spatial, head_dim).permute(
            0, 2, 1, 3, 4
        ).reshape(batch, heads, tokens, head_dim)

    def prepare_context(
        self,
        context: torch.Tensor,
        *,
        grid_hw,
        rope=None,
    ) -> CrossWindowMemory:
        batch, tokens, _ = context.shape
        height, width, window, shift = self._effective_geometry(
            grid_hw, self.window_size, self.shift_size
        )
        spatial = height * width
        if tokens % spatial != 0:
            raise ValueError(
                f"context tokens={tokens} are not divisible by spatial grid "
                f"{height}x{width}"
            )
        frames = tokens // spatial
        key, value = self.kv(context).chunk(2, dim=-1)
        key = key.reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self._apply_spatial_rope(key, frames=frames, rope=rope)
        key = self._partition_context(
            key,
            frames=frames,
            height=height,
            width=width,
            window=window,
            shift=shift,
        )
        value = self._partition_context(
            value,
            frames=frames,
            height=height,
            width=width,
            window=window,
            shift=shift,
        )
        return CrossWindowMemory(
            key=key,
            value=value,
            height=height,
            width=width,
            window=window,
            shift=shift,
            frames=frames,
        )

    def query(
        self,
        query: torch.Tensor,
        memory: CrossWindowMemory,
        *,
        grid_hw,
        rope=None,
    ) -> torch.Tensor:
        batch, tokens, channels = query.shape
        height, width, window, shift = self._effective_geometry(
            grid_hw, self.window_size, self.shift_size
        )
        if (height, width, window, shift) != (
            memory.height,
            memory.width,
            memory.window,
            memory.shift,
        ):
            raise ValueError("query geometry does not match cached context geometry")
        projected = self.q(query).reshape(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        if rope is not None:
            projected = rope(projected)
        projected = self._partition_query(
            projected.to(memory.value.dtype),
            height=height,
            width=width,
            window=window,
            shift=shift,
        )
        context_batch = int(memory.key.shape[0])
        if context_batch <= 0 or batch % context_batch != 0:
            raise ValueError(
                f"query batch {batch} is not divisible by cached context "
                f"batch {context_batch}"
            )
        group_size = batch // context_batch
        windows = projected.shape[1]
        query_tokens = projected.shape[-2]
        key_tokens = memory.key.shape[-2]
        # Goals for one observation share exactly the same K/V.  Concatenate
        # their independent query rows before SDPA so K/V stay compact and the
        # Blackwell Flash-SDPA path remains a standard 4-D launch.
        projected = projected.reshape(
            context_batch,
            group_size,
            windows,
            self.num_heads,
            query_tokens,
            self.head_dim,
        ).permute(0, 2, 3, 1, 4, 5).reshape(
            context_batch * windows,
            self.num_heads,
            group_size * query_tokens,
            self.head_dim,
        )
        key = memory.key.reshape(
            context_batch * windows,
            self.num_heads,
            key_tokens,
            self.head_dim,
        ).to(memory.value.dtype)
        value = memory.value.reshape(
            context_batch * windows,
            self.num_heads,
            key_tokens,
            self.head_dim,
        )
        attention_mask = None
        if shift:
            attention_mask = WindowAttention._shift_mask(
                height=height,
                width=width,
                window=window,
                shift=shift,
                device=projected.device,
                dtype=projected.dtype,
            ).repeat(1, 1, memory.frames)
            # Query rows are laid out goal-major (row = goal * Q + token), so
            # the per-token mask must be tiled once per goal.  The paper's
            # checkpoints were trained with ``repeat_interleave`` (row =
            # token * G + goal), which assigns the wrong window mask to most
            # rows whenever several goals share one history (group_size > 1).
            # ``legacy`` reproduces that behaviour; ``corrected`` fixes it.
            if self.mask_layout == "corrected":
                attention_mask = attention_mask.repeat(1, group_size, 1)
            else:
                attention_mask = attention_mask.repeat_interleave(
                    group_size,
                    dim=1,
                )
            attention_mask = attention_mask.repeat(
                context_batch, 1, 1
            ).unsqueeze(1)
        output = F.scaled_dot_product_attention(
            projected,
            key,
            value,
            attn_mask=attention_mask,
        )
        output = output.reshape(
            context_batch,
            windows,
            self.num_heads,
            group_size,
            query_tokens,
            self.head_dim,
        ).permute(0, 3, 1, 2, 4, 5).reshape(
            batch,
            windows,
            self.num_heads,
            query_tokens,
            self.head_dim,
        )
        output = self._unpartition_query(
            output,
            height=height,
            width=width,
            window=window,
            shift=shift,
        )
        if output.shape != (batch, tokens, channels):
            raise RuntimeError(
                f"Unexpected cross-window output shape {tuple(output.shape)}"
            )
        return self.proj(output)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for timesteps, suitable for inputs in [0, 1] or [-0.5, 0.5].
    """
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W  # stop gradient manually
        # t: (B, 1) or (B,) -> (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_proj = t * W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class ActionEmbedder(nn.Module):
    """
    Unified ActionEmbedder using GaussianFourierEmbedding for all continuous components.
    """
    def __init__(self, hidden_size, embedding_size=256):
        super().__init__()
        hsize = hidden_size // 3

        self.x_emb = GaussianFourierEmbedding(hsize, embedding_size, scale=1.0)
        self.y_emb = GaussianFourierEmbedding(hsize, embedding_size, scale=1.0)

        self.angle_emb = GaussianFourierEmbedding(hidden_size - 2 * hsize, embedding_size, scale=1.0)

    def forward(self, xya):
        # xya: [Batch, 3] -> (x, y, angle)
        x = self.x_emb(xya[..., 0])
        y = self.y_emb(xya[..., 1])
        a = self.angle_emb(xya[..., 2])
        return torch.cat([x, y, a], dim=-1)


def _build_channel_mixer(hidden_size: int, *, mlp_ratio: float) -> nn.Module:
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    return SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))


#################################################################################
#                                 Core CDiT Model                               #
#################################################################################

# History-mixer configurations used in the paper.
#   attention_mode="softmax"          Global-Softmax (original CDiT cross-attention)
#   attention_mode="cached_linear"    Cached-Linear: cached ELU+1 linear cross-attention
#   attention_mode="cross_linear_swa" Cross (L,L,L,SWA) cycles; used by Balanced GDN
#   attention_mode="cross_gdn_swa"    Cross (G,...,L,SWA) cycles; systems proxy
SHIFTED_WINDOW_CROSS_MASKS = frozenset({"legacy", "corrected"})
CACHED_ATTENTION_MODES = frozenset(
    {"cached_linear", "cross_linear_swa", "cross_gdn_swa"}
)
ATTENTION_MODES = frozenset({"softmax"}) | CACHED_ATTENTION_MODES
#   backbone_mode="dit"            one self-mixer type for all blocks
#   backbone_mode="mixed_hybrid"   explicit per-block self_mixer_layout (Balanced GDN)
#   backbone_mode="gdn_swa_hybrid" self (G,G,G,SWA) cycles; systems proxy
BACKBONE_MODES = frozenset({"dit", "mixed_hybrid", "gdn_swa_hybrid"})


class CDiTBlock(nn.Module):
    """Conditional DiT block: target self-mixer, history mixer, SwiGLU MLP."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_low_rank_adaln=False,
        use_qknorm: bool = False,
        attention_mode: str = "softmax",
        backbone_mode: str = "dit",
        self_mixer_type: str = "softmax",
        cross_mixer_type: str = "softmax",
        self_attention_type: str = "global",
        window_size: int = 8,
        linear_feature_scaling: bool = False,
        gdn_conv_kernel_size: int = 4,
        gdn_backend: str = "torch",
        gdn_cross_cache_backend: str = "scan",
        gdn_cross_query_backend: str = "torch",
        gdn_state_dtype: str = "float32",
        shifted_window_cross_mask: str = "legacy",
        **block_kwargs,
    ):
        super().__init__()
        attention_mode = str(attention_mode).lower()
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(f"Unsupported attention_mode={attention_mode!r}")
        backbone_mode = str(backbone_mode).lower()
        if backbone_mode not in BACKBONE_MODES:
            raise ValueError(f"Unsupported backbone_mode={backbone_mode!r}")
        self.attention_mode = attention_mode
        self.backbone_mode = backbone_mode
        self.self_attention_type = str(self_attention_type).lower()
        if self.self_attention_type not in {"global", "window", "shifted_window"}:
            raise ValueError(
                f"Unsupported self_attention_type={self.self_attention_type!r}"
            )
        if self.self_attention_type != "global" and backbone_mode != "dit":
            raise ValueError(
                "window self-attention layouts are only used with backbone_mode=dit"
            )
        if backbone_mode == "dit":
            self.self_mixer_type = (
                self.self_attention_type
                if self.self_attention_type in {"window", "shifted_window"}
                else "full"
            )
        else:
            self.self_mixer_type = str(self_mixer_type).lower()
        self.context_mixer_type = (
            "cached_linear"
            if attention_mode == "cached_linear"
            else str(cross_mixer_type).lower()
            if attention_mode in {"cross_linear_swa", "cross_gdn_swa"}
            else "softmax"
        )

        # Target self-mixer.
        self.norm1 = RMSNorm(hidden_size)
        if self.self_mixer_type in {"window", "shifted_window"}:
            self.attn = WindowAttention(
                hidden_size,
                num_heads=num_heads,
                window_size=int(window_size),
                shift_size=(
                    int(window_size) // 2
                    if self.self_mixer_type == "shifted_window"
                    else 0
                ),
                qkv_bias=True,
                qk_norm=bool(use_qknorm),
                use_rmsnorm=True,
            )
        elif self.self_mixer_type in {"full", "softmax"}:
            self.attn = NormAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=bool(use_qknorm),
                use_rmsnorm=True,
            )
        elif self.self_mixer_type == "gdn":
            self.attn = FrameGatedDeltaMixer(
                hidden_size,
                num_heads=num_heads,
                mixer_type="gdn",
                conv_kernel_size=int(gdn_conv_kernel_size),
                backend=str(gdn_backend),
                cross_cache_backend=str(gdn_cross_cache_backend),
                cross_query_backend=str(gdn_cross_query_backend),
                state_dtype=str(gdn_state_dtype),
            )
        else:
            raise ValueError(f"Unsupported self mixer {self.self_mixer_type!r}")

        # History (context) mixer.
        self.norm2 = RMSNorm(hidden_size)
        self.norm_cond = RMSNorm(hidden_size)
        if self.context_mixer_type == "cached_linear":
            self.cttn = CachedLinearCrossAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                feature_scaling=bool(linear_feature_scaling),
            )
        elif self.context_mixer_type == "shifted_window":
            self.cttn = CachedShiftedWindowCrossAttention(
                hidden_size,
                num_heads=num_heads,
                window_size=int(window_size),
                shift_size=int(window_size) // 2,
                qkv_bias=True,
                mask_layout=str(shifted_window_cross_mask),
            )
        elif self.context_mixer_type == "gdn":
            self.cttn = FrameGatedDeltaMixer(
                hidden_size,
                num_heads=num_heads,
                mixer_type="gdn",
                conv_kernel_size=int(gdn_conv_kernel_size),
                backend=str(gdn_backend),
                cross_cache_backend=str(gdn_cross_cache_backend),
                cross_query_backend=str(gdn_cross_query_backend),
                state_dtype=str(gdn_state_dtype),
            )
        else:
            self.cttn = nn.MultiheadAttention(
                hidden_size,
                num_heads=num_heads,
                add_bias_kv=True,
                bias=True,
                batch_first=True,
                **block_kwargs,
            )

        if use_low_rank_adaln:
            rank_dim = min(hidden_size // 3, 512)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, rank_dim, bias=True),
                nn.SiLU(),
                nn.Linear(rank_dim, 11 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 11 * hidden_size, bias=True),
            )
        self.norm3 = RMSNorm(hidden_size)
        self.mlp = _build_channel_mixer(hidden_size, mlp_ratio=mlp_ratio)

    @property
    def uses_cached_context(self) -> bool:
        return self.attention_mode in CACHED_ATTENTION_MODES

    def prepare_context_memory(self, x_cond: torch.Tensor, *, grid_hw, rope=None):
        """Summarize the (condition-independent) history once for reuse."""

        if not self.uses_cached_context:
            return None
        normalized = self.norm_cond(x_cond)
        if isinstance(self.cttn, CachedLinearCrossAttention):
            return self.cttn.prepare_context(normalized)
        if isinstance(self.cttn, CachedShiftedWindowCrossAttention):
            return self.cttn.prepare_context(normalized, grid_hw=grid_hw, rope=rope)
        return self.cttn.prepare_cross_state(normalized, grid_hw=grid_hw, rope=rope)

    def forward(self, x, c, x_cond, rope=None, *, grid_hw, context_memory=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_ca_xcond,
            scale_ca_xcond,
            shift_ca_x,
            scale_ca_x,
            gate_ca_x,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(11, dim=-1)

        # Target self-mixing.
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        if isinstance(self.attn, WindowAttention):
            attn_out = self.attn(attn_in, grid_hw=grid_hw, rope=rope)
        elif isinstance(self.attn, NormAttention):
            attn_out = self.attn(attn_in, rope=rope)
        else:
            attn_out = self.attn.forward_self(attn_in, grid_hw=grid_hw, rope=rope)
        x = x + apply_gate(attn_out, gate_msa)

        # History mixing.
        cross_query = modulate(self.norm2(x), shift_ca_x, scale_ca_x)
        if self.uses_cached_context:
            if context_memory is None:
                context_memory = self.prepare_context_memory(
                    x_cond, grid_hw=grid_hw, rope=rope
                )
            if isinstance(self.cttn, CachedLinearCrossAttention):
                cross_out = self.cttn.query(cross_query, context_memory)
            elif isinstance(self.cttn, CachedShiftedWindowCrossAttention):
                cross_out = self.cttn.query(
                    cross_query, context_memory, grid_hw=grid_hw, rope=rope
                )
            else:
                cross_out = self.cttn.query_cross_state(
                    cross_query, context_memory, grid_hw=grid_hw, rope=rope
                )
        else:
            x_cond_norm = modulate_context(
                self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond
            )
            cross_out = self.cttn(
                query=cross_query,
                key=x_cond_norm,
                value=x_cond_norm,
                need_weights=False,
            )[0]
        x = x + apply_gate(cross_out, gate_ca_x)

        x = x + apply_gate(
            self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp)),
            gate_mlp,
        )
        return x



class DDTHeadBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm: bool = False,
        use_low_rank_adaln: bool = False,
        adaln_rank: int = None,
        attention_type: str = "global",
        window_size: int = 8,
        **block_kwargs,
    ):
        super().__init__()
        self.attention_type = str(attention_type).lower()
        if self.attention_type not in {
            "global",
            "window",
            "shifted_window",
        }:
            raise ValueError(
                f"Unsupported DDT attention_type={self.attention_type!r}"
            )
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        if self.attention_type in {"window", "shifted_window"}:
            self.attn = WindowAttention(
                hidden_size,
                num_heads=num_heads,
                window_size=int(window_size),
                shift_size=(
                    int(window_size) // 2
                    if self.attention_type == "shifted_window"
                    else 0
                ),
                qkv_bias=True,
                qk_norm=bool(use_qknorm),
                use_rmsnorm=True,
            )
        else:
            self.attn = NormAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=bool(use_qknorm),
                use_rmsnorm=True,
                **block_kwargs,
            )
        self.mlp = _build_channel_mixer(hidden_size, mlp_ratio=mlp_ratio)
        if use_low_rank_adaln:
            rank = (
                int(adaln_rank)
                if adaln_rank is not None
                else min(hidden_size // 3, 512)
            )
            if rank <= 0:
                raise ValueError("adaln_rank must be positive")
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, rank, bias=True),
                nn.SiLU(),
                nn.Linear(rank, 6 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope=None,
        *,
        grid_hw=None,
    ) -> torch.Tensor:
        if c.dim() == 2:
            c = c.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_input = DDTModulate(self.norm1(x), shift_msa, scale_msa)
        if self.attention_type in {"window", "shifted_window"}:
            if grid_hw is None:
                raise ValueError("DDT window attention requires grid_hw")
            attn_output = self.attn(attn_input, grid_hw=grid_hw, rope=rope)
        else:
            attn_output = self.attn(attn_input, rope=rope)
        x = x + DDTGate(attn_output, gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        patch_size,
        out_channels,
        *,
        use_low_rank_adaln: bool = False,
        adaln_rank: int = None,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        if use_low_rank_adaln:
            rank = (
                int(adaln_rank)
                if adaln_rank is not None
                else min(hidden_size // 3, 512)
            )
            if rank <= 0:
                raise ValueError("adaln_rank must be positive")
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, rank, bias=True),
                nn.SiLU(),
                nn.Linear(rank, 2 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c):
        if c.dim() == 2:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """Conditional flow transformer (CDiT backbone + DDT head) for NWMs.

    History latents form a separate, condition-independent context stream;
    the noisy future latent forms the target stream. Action, horizon and flow
    time modulate the target blocks through adaptive normalization, so the
    history summary can be prepared once and reused across candidate actions,
    horizons and integration steps (see ``prepare_rollout_memory``).
    """

    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=1,
        in_channels=768,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        head_width=None,
        head_depth=2,
        head_num_heads=16,
        use_low_rank_adaln_head=False,
        head_adaln_rank: int = None,
        head_mlp_ratio: float = None,
        self_attention_layout=None,
        self_mixer_layout=None,
        head_attention_layout=None,
        window_size: int = 8,
        head_window_size: int = None,
        linear_feature_scaling: bool = False,
        use_qknorm: bool = False,
        attention_mode: str = "softmax",
        backbone_mode: str = "dit",
        gdn_softmax_every: int = 4,
        cross_gdn_layers_per_cycle: int = 3,
        gdn_conv_kernel_size: int = 4,
        gdn_backend: str = "torch",
        gdn_cross_cache_backend: str = "scan",
        gdn_cross_query_backend: str = "torch",
        gdn_state_dtype: str = "float32",
        shifted_window_cross_mask: str = "legacy",
        repa_mode: str = "none",
        repa_layer_depth: int = 8,
        repa_kernel_size: int = 3,
        repa_target_dim: int = None,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.head_width = head_width if head_width is not None else in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = int(depth)
        self.head_depth = head_depth
        self.head_num_heads = head_num_heads
        self.use_low_rank_adaln_head = bool(use_low_rank_adaln_head)
        self.head_adaln_rank = (
            int(head_adaln_rank) if head_adaln_rank is not None else None
        )
        self.head_mlp_ratio = (
            float(head_mlp_ratio) if head_mlp_ratio is not None else float(mlp_ratio)
        )
        self.window_size = int(window_size)
        self.head_window_size = (
            int(head_window_size) if head_window_size is not None else self.window_size
        )
        self.mlp_ratio = float(mlp_ratio)
        self.self_attention_layout = _resolve_attention_layout(
            self_attention_layout, depth, default="global", name="self_attention_layout"
        )
        self.self_mixer_layout = _resolve_self_mixer_layout(self_mixer_layout, depth)
        self.head_attention_layout = _resolve_attention_layout(
            head_attention_layout, head_depth, default="global", name="head_attention_layout"
        )

        self.attention_mode = str(attention_mode).lower()
        if self.attention_mode not in ATTENTION_MODES:
            raise ValueError(f"Unsupported attention_mode={self.attention_mode!r}")
        self.backbone_mode = str(backbone_mode).lower()
        if self.backbone_mode not in BACKBONE_MODES:
            raise ValueError(f"Unsupported backbone_mode={self.backbone_mode!r}")
        self.gdn_softmax_every = int(gdn_softmax_every)
        self.cross_gdn_layers_per_cycle = int(cross_gdn_layers_per_cycle)
        self.gdn_backend = str(gdn_backend).lower()
        if self.gdn_backend not in {"torch", "triton", "triton_fused"}:
            raise ValueError(f"Unsupported gdn_backend={self.gdn_backend!r}")
        if self.attention_mode in {"cross_linear_swa", "cross_gdn_swa"} and self.gdn_softmax_every <= 0:
            raise ValueError("Cross SWA cycles require positive gdn_softmax_every")
        if self.attention_mode == "cross_gdn_swa" and not (
            0 <= self.cross_gdn_layers_per_cycle < self.gdn_softmax_every
        ):
            raise ValueError(
                f"cross_gdn_layers_per_cycle must be in [0,{self.gdn_softmax_every - 1}]"
            )
        if self.backbone_mode == "gdn_swa_hybrid" and self.attention_mode not in {
            "cross_linear_swa",
            "cross_gdn_swa",
        }:
            raise ValueError("gdn_swa_hybrid requires a Cross Linear/GDN + SWA history mixer")
        if self.backbone_mode == "mixed_hybrid":
            if self.self_mixer_layout is None:
                raise ValueError("mixed_hybrid requires an explicit self_mixer_layout")
            if self.attention_mode not in {"cross_linear_swa", "cross_gdn_swa"}:
                raise ValueError("mixed_hybrid requires a cached Cross SWA history mixer")

        self.repa_mode = str(repa_mode).lower()
        if self.repa_mode not in {"none", "repa", "irepa"}:
            raise ValueError(f"Unsupported repa_mode={self.repa_mode!r}")
        self.repa_layer_depth = int(repa_layer_depth)
        self.repa_target_dim = (
            int(repa_target_dim) if repa_target_dim is not None else int(in_channels)
        )
        self.return_intermediate_features = self.repa_mode != "none"

        # -------------------------
        # Encoder (CDiT backbone)
        # -------------------------
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # Gaussian Fourier embeddings for flow time t in [0, 1] and horizon rel_t.
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        self.time_embedder = GaussianFourierEmbedding(hidden_size)
        self.dyn_fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dyn_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, num_patches, hidden_size),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                CDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_low_rank_adaln=False,
                    use_qknorm=bool(use_qknorm),
                    attention_mode=self.attention_mode,
                    backbone_mode=self.backbone_mode,
                    self_mixer_type=self._self_mixer_for_block(index),
                    cross_mixer_type=self._cross_mixer_for_block(index),
                    self_attention_type=self.self_attention_layout[index],
                    window_size=self.window_size,
                    linear_feature_scaling=bool(linear_feature_scaling),
                    gdn_conv_kernel_size=int(gdn_conv_kernel_size),
                    gdn_backend=self.gdn_backend,
                    gdn_cross_cache_backend=str(gdn_cross_cache_backend).lower(),
                    gdn_cross_query_backend=str(gdn_cross_query_backend).lower(),
                    gdn_state_dtype=str(gdn_state_dtype).lower(),
                    shifted_window_cross_mask=str(shifted_window_cross_mask).lower(),
                )
                for index in range(depth)
            ]
        )

        # Optional iREPA alignment branch (training only).
        if self.repa_mode != "none":
            if not 1 <= self.repa_layer_depth <= depth:
                raise ValueError(
                    f"repa_layer_depth={self.repa_layer_depth} must be in [1,{depth}]"
                )
            self.repa_projector = RepresentationAlignmentHead(
                hidden_size,
                self.repa_target_dim,
                mode=self.repa_mode,
                kernel_size=int(repa_kernel_size),
            )
        else:
            self.repa_projector = None

        half_head = hidden_size // num_heads // 2
        grid = int(self.x_embedder.num_patches ** 0.5)
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head, pt_seq_len=grid)

        # -------------------------
        # DDT head: re-embeds raw x_t (query stream) and attends with the
        # token-wise backbone features projected to head_width.
        # -------------------------
        self.x_embedder_head = PatchEmbed(input_size, patch_size, in_channels, self.head_width, bias=True)
        self.head_projector = nn.Linear(hidden_size, self.head_width, bias=True)
        # Unused projections kept for checkpoint compatibility.
        self.head_cond_projector = nn.Linear(hidden_size, self.head_width, bias=True)
        self.final_cond_proj = nn.Linear(hidden_size, self.head_width, bias=True)
        self.head_blocks = nn.ModuleList(
            [
                DDTHeadBlock(
                    self.head_width,
                    self.head_num_heads,
                    mlp_ratio=self.head_mlp_ratio,
                    use_qknorm=bool(use_qknorm),
                    use_low_rank_adaln=self.use_low_rank_adaln_head,
                    adaln_rank=self.head_adaln_rank,
                    attention_type=self.head_attention_layout[index],
                    window_size=self.head_window_size,
                )
                for index in range(self.head_depth)
            ]
        )
        half_head2 = self.head_width // self.head_num_heads // 2
        self.head_feat_rope = VisionRotaryEmbeddingFast(dim=half_head2, pt_seq_len=grid)
        self.final_layer = DDTFinalLayer(
            self.head_width,
            patch_size,
            self.out_channels,
            use_low_rank_adaln=self.use_low_rank_adaln_head,
            adaln_rank=self.head_adaln_rank,
        )

        self.initialize_weights()

    def _self_mixer_for_block(self, index: int) -> str:
        if self.backbone_mode == "mixed_hybrid":
            return self.self_mixer_layout[index]
        if self.backbone_mode == "gdn_swa_hybrid":
            return "shifted_window" if (index + 1) % self.gdn_softmax_every == 0 else "gdn"
        return "softmax"

    def _cross_mixer_for_block(self, index: int) -> str:
        if self.attention_mode not in {"cross_linear_swa", "cross_gdn_swa"}:
            return "softmax"
        if (index + 1) % self.gdn_softmax_every == 0:
            return "shifted_window"
        if (
            self.attention_mode == "cross_gdn_swa"
            and index % self.gdn_softmax_every < self.cross_gdn_layers_per_cycle
        ):
            return "gdn"
        return "cached_linear"


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        pos_embed = pos_embed.repeat(self.context_size + 1, 1, 1)
        self.pos_embed.data.copy_(pos_embed)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize head patch_embed too:
        w2 = self.x_embedder_head.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder_head.proj.bias, 0)

        # Initialize action embedding:
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            if isinstance(block.attn, FrameGatedDeltaMixer):
                block.attn.reset_gdn_parameters()
            if isinstance(block.cttn, FrameGatedDeltaMixer):
                block.cttn.reset_gdn_parameters()
        for block in self.head_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, num_patches, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        num_patches = x.shape[1]
        h = w = int(num_patches ** 0.5)
        assert h * w == num_patches, f"num_patches {num_patches} is not a square"
        # derive patch size dynamically from the last dimension
        patch_area_times_c = x.shape[2]
        if patch_area_times_c % c != 0:
            # handle swapped token layout: (N, Ctok, num_patches)
            if x.shape[2] == num_patches and x.shape[1] != num_patches:
                x = x.transpose(1, 2)
                patch_area_times_c = x.shape[2]
            # if still not divisible, raise
        # decide effective channel count
        if patch_area_times_c % c == 0:
            c_eff = c
        elif patch_area_times_c % (2 * c) == 0:
            c_eff = 2 * c
        else:
            raise AssertionError(f"channel mismatch: {patch_area_times_c} not divisible by {c} or {2*c}")
        p = int(((patch_area_times_c // c_eff)) ** 0.5)
        assert p * p * c_eff == patch_area_times_c, f"cannot factor last dim {patch_area_times_c} into p^2*c_eff with c_eff={c_eff}"

        x = x.reshape(x.shape[0], h, w, p, p, c_eff)
        x = x.permute(0, 5, 3, 4, 1, 2)  # (N, C_eff, p, p, H, W)
        imgs = x.reshape(x.shape[0], c_eff, h * p, w * p)
        if c_eff == 2 * c:
            imgs_mu, imgs_sigma = imgs.chunk(2, dim=1)
            return imgs_mu
        return imgs

    def forward(
        self,
        x,
        t,
        y,
        x_cond,
        rel_t,
        memory=None,
        context_group_size: int = 1,
        return_features=None,
    ):
        """Predict the flow velocity for noisy target latents ``x``.

        x:       [B,C,H,W] noisy future latent z_t
        t:       [B] flow time
        y:       [B,3] relative action (dx, dy, dtheta)
        x_cond:  [B,F,C,H,W] history latents
        rel_t:   [B] requested horizon
        memory:  optional per-block history summaries from
                 ``prepare_rollout_memory`` (cached history mixers only)
        """

        x_raw = x
        if x_cond is None or x_cond.ndim != 5:
            raise ValueError("x_cond must be [B,F,C,H,W]")
        context_frames = int(x_cond.shape[1])
        if not 1 <= context_frames <= int(self.context_size):
            raise ValueError(
                f"context frames {context_frames} must be in [1,{self.context_size}]"
            )
        memory = self._resolve_grouped_memory(
            x, x_cond, memory, context_group_size=context_group_size
        )
        cached_context = memory is not None and self.uses_cached_context
        x_tok, x_cond_tok, c, t_emb = self._prepare_inputs(
            x, t, y, x_cond, rel_t, embed_context=not cached_context
        )

        collect_depths = {self.repa_layer_depth} if self.repa_projector is not None else set()
        z, collected = self._process_blocks(
            x_tok, c, x_cond_tok, memory=memory, collect_depths=collect_depths
        )
        z_intermediate = collected.get(self.repa_layer_depth)

        # Fuse the time embedding into token features before the DDT head.
        z = F.silu(z + (t_emb.unsqueeze(1) if t_emb.dim() == 2 else t_emb))
        z_head = self.head_projector(z)
        x_head = self.x_embedder_head(x_raw)
        grid = int(self.x_embedder.num_patches ** 0.5)
        for blk in self.head_blocks:
            x_head = blk(x_head, z_head, rope=self.head_feat_rope, grid_hw=(grid, grid))
        x_out = self.final_layer(x_head, z_head)
        if x_out.dim() != 3:
            x_out = x_out.reshape(x_out.shape[0], self.x_embedder.num_patches, -1)
        x_out = self.unpatchify(x_out)

        if return_features is None:
            return_features = self.return_intermediate_features and self.training
        if return_features:
            if z_intermediate is None:
                raise RuntimeError("return_features=True requires repa_mode != 'none'")
            projected = self.repa_projector(z_intermediate, hw=(grid, grid))
            return x_out, z_intermediate, projected
        return x_out

    @property
    def uses_cached_context(self) -> bool:
        return self.attention_mode in CACHED_ATTENTION_MODES

    def _resolve_grouped_memory(self, x, x_cond, memory, *, context_group_size: int):
        """Build the history summary once per group of targets sharing a history."""

        context_group_size = int(context_group_size)
        if context_group_size <= 0:
            raise ValueError("context_group_size must be positive")
        if memory is None and context_group_size > 1 and self.uses_cached_context:
            if x_cond is None or x_cond.shape[0] != x.shape[0]:
                raise ValueError(
                    "Grouped context preparation requires one repeated context entry per target"
                )
            if x.shape[0] % context_group_size != 0:
                raise ValueError(
                    f"target batch {x.shape[0]} is not divisible by "
                    f"context_group_size={context_group_size}"
                )
            memory = self.expand_rollout_memory(
                self.prepare_rollout_memory(x_cond[::context_group_size]),
                repeats=context_group_size,
            )
        return memory

    def _embed_context(self, x_cond):
        frames = int(x_cond.shape[1])
        if not 1 <= frames <= int(self.context_size):
            raise ValueError(f"context frames {frames} must be in [1,{self.context_size}]")
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(
            0, (x_cond.shape[0], x_cond.shape[1])
        )
        x_cond = x_cond + self.pos_embed[:frames]
        return x_cond.flatten(1, 2)

    def prepare_rollout_memory(self, x_cond):
        """Summarize the observation history once for reuse.

        The returned per-block states (linear (M, u) summaries, window K/V or
        GDN states) are independent of action, horizon and flow time, so they
        can be shared by every candidate action and every ODE step. Returns
        ``None`` for Global-Softmax, which recomputes attention per query.
        """

        if not self.uses_cached_context:
            return None
        x_cond_tok = self._embed_context(x_cond)
        grid = int(self.x_embedder.num_patches ** 0.5)
        return [
            block.prepare_context_memory(x_cond_tok, grid_hw=(grid, grid), rope=self.feat_rope)
            for block in self.blocks
        ]

    @staticmethod
    def expand_rollout_memory(memory, *, repeats: int):
        """Share per-observation history states across ``repeats`` targets.

        Cached states broadcast over the target batch, so they are reused
        without copying.
        """

        if int(repeats) <= 0:
            raise ValueError("repeats must be positive")
        return memory

    def _prepare_inputs(self, x, t, y, x_cond, rel_t, *, embed_context=True):
        x = self.x_embedder(x) + self.pos_embed[-1:]
        if embed_context:
            if x_cond is None:
                raise ValueError("x_cond is required when context is not cached")
            x_cond = self._embed_context(x_cond)
        else:
            x_cond = None

        t_emb = self.t_embedder(t[..., None])
        y_emb = self.y_embedder(y)
        span_emb = self.time_embedder(rel_t[..., None])
        c_dyn = self.dyn_fuse(torch.cat([y_emb, span_emb], dim=-1))
        if t_emb.dim() == 3:
            if t_emb.shape[1] != x.shape[1]:
                raise ValueError(
                    "Per-token timesteps must match target patch tokens, got "
                    f"timesteps={t_emb.shape[1]}, tokens={x.shape[1]}"
                )
            c_dyn = c_dyn.unsqueeze(1)
        gate = torch.sigmoid(self.dyn_gate(t_emb))
        c = t_emb + gate * c_dyn
        return x, x_cond, c, t_emb

    def _process_blocks(self, x, c, x_cond, memory=None, *, collect_depths=()):
        if memory is not None and len(memory) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} memory states, got {len(memory)}")
        collect_depths = {int(depth) for depth in collect_depths}
        grid = int(self.x_embedder.num_patches ** 0.5)
        collected = {}
        for index, block in enumerate(self.blocks):
            x = block(
                x,
                c,
                x_cond,
                rope=self.feat_rope,
                grid_hw=(grid, grid),
                context_memory=None if memory is None else memory[index],
            )
            if index + 1 in collect_depths:
                collected[index + 1] = x
        return x, collected


    def commit_rollout_prediction(self, x_cond, clean_prediction, *, refresh_memory: bool = True):
        """Append one clean rollout prediction to the latent history.

        Called only after an ODE solve finishes; the predicted latent replaces
        the oldest history frame without decoding and re-encoding. Returns the
        new history and, for cached history mixers, its refreshed summary.
        """

        if x_cond.ndim != 5 or clean_prediction.ndim != 4:
            raise ValueError("Expected x_cond [B,T,C,H,W] and clean_prediction [B,C,H,W]")
        if x_cond.shape[0] != clean_prediction.shape[0]:
            raise ValueError("Context and prediction batch sizes must match")
        if x_cond.shape[1] < self.context_size:
            raise ValueError(
                f"Need at least {self.context_size} context frames, got {x_cond.shape[1]}"
            )
        current = x_cond[:, -self.context_size :]
        next_context = torch.cat((current[:, 1:], clean_prediction.unsqueeze(1)), dim=1)
        memory = (
            self.prepare_rollout_memory(next_context)
            if refresh_memory and self.uses_cached_context
            else None
        )
        return next_context, memory


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                #
#################################################################################

def CDiT_XL_2(**kwargs):
    kwargs.setdefault("num_heads", 16)
    return CDiT(depth=28, hidden_size=1152, patch_size=1, **kwargs)

def CDiT_L_2(**kwargs):
    kwargs.setdefault("num_heads", 16)
    return CDiT(depth=24, hidden_size=1024, patch_size=1, **kwargs)

def CDiT_B_2(**kwargs):
    kwargs.setdefault("num_heads", 12)
    return CDiT(depth=12, hidden_size=768, patch_size=1, **kwargs)

def CDiT_S_2(**kwargs):
    kwargs.setdefault("num_heads", 6)
    return CDiT(depth=12, hidden_size=384, patch_size=1, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2,
    'CDiT-L/2':  CDiT_L_2,
    'CDiT-B/2':  CDiT_B_2,
    'CDiT-S/2':  CDiT_S_2
}
