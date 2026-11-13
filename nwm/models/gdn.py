# Adapted from NVlabs/Sana @ 6298508 (SANA-WM frame-wise GDN). Modified by the
# authors. Licensed under the Apache License 2.0; see LICENSES/Apache-2.0-Sana.txt.
"""Frame-wise GDN mixer for the RAE-NWM hybrid backbone.

The recurrence and gate initialization follow SANA-WM's frame-wise Gated
Delta Network (GDN). ``backend="triton_fused"`` uses the vendored SANA-WM
Triton/autograd kernel; the Torch backend remains available for CPU tests and
numerical parity checks. The recurrence scans complete spatial frames and
returns the final (noisy target) frame.

Reference implementation:
https://github.com/NVlabs/Sana/tree/6298508fcb511762a11c42cff45b2fc9fd930325
Apache-2.0 license; see THIRD_PARTY.md.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from RAE.src.stage2.models.model_utils import RMSNorm


OUTPUT_GATE_INIT_BIAS = 1.278464542761074  # silu(x) == 1


class GDNState(NamedTuple):
    """Paired canonical GDN state cached for Cross reads."""

    state_kv: torch.Tensor
    state_z: torch.Tensor


class FrameGatedDeltaMixer(nn.Module):
    """Causal frame-wise GDN or periodic softmax with one parameter layout.

    Inputs are ordered as clean context frames followed by the noisy target
    frame.  The returned tokens correspond only to the target frame.  Keeping
    the same parameter layout for both mixer types mirrors SANA-WM's hybrid
    blocks and makes the layer schedule explicit in checkpoints.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        mixer_type: str,
        conv_kernel_size: int = 4,
        backend: str = "torch",
        cross_cache_backend: str = "scan",
        cross_query_backend: str = "torch",
        state_dtype: str = "float32",
        eps: float = 1.0e-6,
        output_rms_cap: float | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        mixer_type = str(mixer_type).lower()
        if mixer_type not in {"gdn", "softmax"}:
            raise ValueError(f"Unsupported GDN hybrid mixer_type={mixer_type!r}")
        if int(conv_kernel_size) < 0:
            raise ValueError("conv_kernel_size must be non-negative")
        backend = str(backend).lower()
        if backend not in {"torch", "triton", "triton_fused"}:
            raise ValueError(f"Unsupported GDN backend={backend!r}")
        cross_cache_backend = str(cross_cache_backend).lower()
        if cross_cache_backend not in {
            "scan",
            "final",
            "tree",
            "fused_final",
            "fused_final_raw",
        }:
            raise ValueError(
                "Unsupported Cross-GDN cache backend="
                f"{cross_cache_backend!r}"
            )
        cross_query_backend = str(cross_query_backend).lower()
        if cross_query_backend not in {"torch", "triton"}:
            raise ValueError(
                "Unsupported Cross-GDN query backend="
                f"{cross_query_backend!r}"
            )
        state_dtype = str(state_dtype).lower()
        if state_dtype not in {"float32", "bfloat16"}:
            raise ValueError(
                f"Unsupported Cross-GDN state dtype={state_dtype!r}"
            )

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.mixer_type = mixer_type
        self.conv_kernel_size = int(conv_kernel_size)
        self.backend = backend
        self.cross_cache_backend = cross_cache_backend
        self.cross_query_backend = cross_query_backend
        self.state_dtype = state_dtype
        self.eps = float(eps)
        self.output_rms_cap = (
            None if output_rms_cap is None else float(output_rms_cap)
        )
        if self.output_rms_cap is not None and self.output_rms_cap <= 0:
            raise ValueError("output_rms_cap must be positive or None")

        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        # SANA-WM normalizes Q/K over the full channel dimension before the
        # head split, rather than normalizing each head independently.
        self.q_norm = RMSNorm(self.dim)
        self.k_norm = RMSNorm(self.dim)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

        self.beta_proj = nn.Linear(self.dim, self.num_heads, bias=True)
        self.gate_proj = nn.Linear(self.dim, self.num_heads, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.full((self.num_heads,), -5.0, dtype=torch.float32))
        self.output_gate = nn.Linear(self.dim, self.dim, bias=True)

        if self.conv_kernel_size > 0:
            self.conv_k_weight = nn.Parameter(
                torch.zeros(self.dim, 1, self.conv_kernel_size)
            )
            self.conv_k_bias = nn.Parameter(torch.zeros(self.dim))
        else:
            self.register_parameter("conv_k_weight", None)
            self.register_parameter("conv_k_bias", None)

        self.reset_gdn_parameters()

    def reset_gdn_parameters(self) -> None:
        """Initialize gates near linear-attention equivalence."""

        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 5.0)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        with torch.no_grad():
            self.A_log.fill_(math.log(1.0))
            self.dt_bias.fill_(-5.0)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.constant_(self.output_gate.bias, OUTPUT_GATE_INIT_BIAS)
        if self.conv_k_weight is not None:
            with torch.no_grad():
                self.conv_k_weight.zero_()
                # Causal depthwise Conv1d: the last tap is the current frame.
                self.conv_k_weight[:, 0, -1] = 1.0
                self.conv_k_bias.zero_()

    def _stabilize_gdn_output(self, output: torch.Tensor) -> torch.Tensor:
        """Bound only pathological per-token GDN outputs by their RMS.

        The delta-rule denominator is theoretically positive but can approach
        zero on rare learned states, producing a finite yet destructive
        outlier. This guard is inactive below ``output_rms_cap`` and keeps
        the fused recurrence unchanged on the ordinary path.
        """

        if self.mixer_type != "gdn" or self.output_rms_cap is None:
            return output
        output_float = output.float()
        inverse_rms = torch.rsqrt(
            output_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        scale = torch.clamp(self.output_rms_cap * inverse_rms, max=1.0)
        return (output_float * scale).to(output.dtype)

    @staticmethod
    def _apply_spatial_rope(
        tensor: torch.Tensor,
        rope,
    ) -> torch.Tensor:
        """Apply the same spatial RoPE independently to every frame."""

        if rope is None:
            return tensor
        batch, heads, frames, spatial, head_dim = tensor.shape
        flattened = tensor.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, heads, spatial, head_dim
        )
        rotated = rope(flattened)
        return rotated.reshape(batch, frames, heads, spatial, head_dim).permute(
            0, 2, 1, 3, 4
        )

    def _causal_key_convolution(
        self,
        keys: torch.Tensor,
        *,
        frames: int,
        spatial: int,
    ) -> torch.Tensor:
        if self.conv_k_weight is None:
            return keys
        batch, tokens, channels = keys.shape
        if tokens != frames * spatial:
            raise ValueError(
                f"Key token count {tokens} does not match frames*spatial "
                f"({frames}*{spatial})"
            )
        temporal = keys.reshape(batch, frames, spatial, channels)
        temporal = temporal.permute(0, 2, 3, 1).reshape(
            batch * spatial, channels, frames
        )
        temporal = F.pad(temporal, (self.conv_kernel_size - 1, 0))
        temporal = F.conv1d(
            temporal,
            self.conv_k_weight,
            self.conv_k_bias,
            groups=channels,
        )
        return temporal.reshape(batch, spatial, channels, frames).permute(
            0, 3, 1, 2
        ).reshape(batch, tokens, channels)

    def _project_qkv(
        self,
        sequence: torch.Tensor,
        *,
        frames: int,
        spatial: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, channels = sequence.shape
        qkv = self.qkv(sequence).reshape(batch, tokens, 3, channels)
        q, k, v = qkv.unbind(dim=2)
        if self.mixer_type == "gdn":
            k = self._causal_key_convolution(k, frames=frames, spatial=spatial)
        q = self.q_norm(q).reshape(
            batch, frames, spatial, self.num_heads, self.head_dim
        )
        k = self.k_norm(k).reshape(
            batch, frames, spatial, self.num_heads, self.head_dim
        )
        v = v.reshape(batch, frames, spatial, self.num_heads, self.head_dim)
        return q, k, v

    def _project_cross_qkv(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        frames: int,
        spatial: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project target queries and context keys/values without wasted paths."""

        batch, query_tokens, channels = query.shape
        if query_tokens != spatial:
            raise ValueError(
                f"Cross-GDN query tokens={query_tokens} do not match spatial={spatial}"
            )
        if context.shape != (batch, frames * spatial, channels):
            raise ValueError(
                "Cross-GDN context shape does not match batch, frames, spatial, "
                f"and channels: {tuple(context.shape)}"
            )
        weight = self.qkv.weight
        bias = self.qkv.bias
        q = F.linear(
            query,
            weight[:channels],
            None if bias is None else bias[:channels],
        )
        k = F.linear(
            context,
            weight[channels : 2 * channels],
            None if bias is None else bias[channels : 2 * channels],
        )
        v = F.linear(
            context,
            weight[2 * channels :],
            None if bias is None else bias[2 * channels :],
        )
        if self.mixer_type == "gdn":
            k = self._causal_key_convolution(k, frames=frames, spatial=spatial)
        q = self.q_norm(q).reshape(
            batch, 1, spatial, self.num_heads, self.head_dim
        )
        k = self.k_norm(k).reshape(
            batch, frames, spatial, self.num_heads, self.head_dim
        )
        v = v.reshape(batch, frames, spatial, self.num_heads, self.head_dim)
        return q, k, v

    def _softmax_target(
        self,
        sequence: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        # Only the final-frame query is needed.  Because it is the last frame,
        # attending over the whole prefix is exactly its frame-causal row.
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        q = self._apply_spatial_rope(q, rope)
        k = self._apply_spatial_rope(k, rope)
        batch, heads, frames, spatial, head_dim = q.shape
        q_target = q[:, :, -1]
        k_all = k.reshape(batch, heads, frames * spatial, head_dim)
        v_all = v.reshape(batch, heads, frames * spatial, head_dim)
        output = F.scaled_dot_product_attention(q_target, k_all, v_all)
        return output.transpose(1, 2).reshape(batch, spatial, self.dim)

    def _gdn_target(
        self,
        sequence: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        batch, tokens, _ = sequence.shape
        frames = q.shape[1]
        spatial = q.shape[2]

        beta = torch.sigmoid(self.beta_proj(sequence)).reshape(
            batch, frames, spatial, self.num_heads
        ).permute(0, 3, 1, 2)
        frame_features = sequence.reshape(batch, frames, spatial, self.dim).mean(dim=2)
        gate = self.gate_proj(frame_features).float()
        decay = torch.exp(
            -torch.exp(self.A_log).view(1, 1, self.num_heads)
            * F.softplus(gate + self.dt_bias.view(1, 1, self.num_heads))
        ).permute(0, 2, 1)

        q = F.relu(q).permute(0, 3, 1, 4, 2)
        k = F.relu(k).permute(0, 3, 1, 4, 2)
        v = v.permute(0, 3, 1, 4, 2)
        q_rot = self._apply_spatial_rope(
            q.permute(0, 1, 2, 4, 3), rope
        ).permute(0, 1, 2, 4, 3)
        k_rot = self._apply_spatial_rope(
            k.permute(0, 1, 2, 4, 3), rope
        ).permute(0, 1, 2, 4, 3)

        key_scale = (self.head_dim * spatial) ** -0.5
        k = k * key_scale
        k_rot = k_rot * key_scale

        # The D x D state is kept in fp32, matching the numerical contract of
        # the official fused scan.  T is only context_size+1 (five here).
        state_kv = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=sequence.device,
            dtype=torch.float32,
        )
        state_z = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            device=sequence.device,
            dtype=torch.float32,
        )
        output = None
        for frame_index in range(frames):
            beta_t = beta[:, :, frame_index].float().unsqueeze(2)
            decay_t = decay[:, :, frame_index].unsqueeze(-1)
            k_t = k[:, :, frame_index].float()
            k_rot_t = k_rot[:, :, frame_index].float()
            v_t = v[:, :, frame_index].float()

            k_beta = k_t * beta_t
            k_rot_beta = k_rot_t * beta_t

            state_kv = decay_t.unsqueeze(-1) * (
                state_kv
                - torch.matmul(
                    torch.matmul(state_kv, k_rot_beta),
                    k_rot_t.transpose(-1, -2),
                )
            ) + torch.matmul(v_t * beta_t, k_rot_t.transpose(-1, -2))

            state_z = decay_t * (
                state_z
                - torch.matmul(
                    k_beta,
                    torch.matmul(k_t.transpose(-1, -2), state_z.unsqueeze(-1)),
                ).squeeze(-1)
            ) + k_beta.sum(dim=-1)

            if frame_index == frames - 1:
                q_rot_t = q_rot[:, :, frame_index].float()
                q_t = q[:, :, frame_index].float()
                numerator = torch.matmul(state_kv, q_rot_t)
                denominator = torch.matmul(
                    state_z.unsqueeze(2), q_t
                ).squeeze(2).unsqueeze(2)
                output = numerator / (denominator + self.eps)

        if output is None:
            raise RuntimeError("GDN scan produced no target output")
        return output.permute(0, 3, 1, 2).reshape(
            batch, spatial, self.dim
        ).to(sequence.dtype)

    def _gdn_target_triton_fused(
        self,
        sequence: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        """Run the exact frame-wise Self-GDN with SANA-WM's fused kernel.

        This is deliberately not dispatched to FLA's token-causal gated
        delta-rule operator: the local architecture updates one state per
        complete spatial frame and carries a paired normalization state.
        The vendored SANA-WM kernel preserves those semantics while fusing
        ReLU, RoPE, state updates, normalization, and autograd.
        """

        if not sequence.is_cuda:
            raise ValueError("Fused Self-GDN requires CUDA")
        from nwm.models.sana_fused_gdn import fused_gdn_forward_with_grad

        batch, tokens, _ = sequence.shape
        frames = q.shape[1]
        spatial = q.shape[2]
        if tokens != frames * spatial:
            raise ValueError(
                f"Self-GDN tokens={tokens} do not match frames*spatial "
                f"({frames}*{spatial})"
            )

        beta = torch.sigmoid(self.beta_proj(sequence)).reshape(
            batch,
            frames,
            spatial,
            self.num_heads,
        ).permute(0, 3, 1, 2).contiguous()
        frame_features = sequence.reshape(
            batch,
            frames,
            spatial,
            self.dim,
        ).mean(dim=2)
        gate = self.gate_proj(frame_features).float()
        decay = torch.exp(
            -torch.exp(self.A_log).view(1, 1, self.num_heads)
            * F.softplus(gate + self.dt_bias.view(1, 1, self.num_heads))
        ).permute(0, 2, 1).contiguous()

        qkv = torch.stack([q, k, v], dim=3).reshape(
            batch,
            frames * spatial,
            3,
            self.num_heads,
            self.head_dim,
        ).contiguous()
        if rope is None:
            rope_cos = torch.ones(
                frames * spatial,
                self.head_dim,
                device=sequence.device,
                dtype=torch.float32,
            )
            rope_sin = torch.zeros_like(rope_cos)
        else:
            rope_cos = rope.freqs_cos.to(
                device=sequence.device,
                dtype=torch.float32,
            )
            rope_sin = rope.freqs_sin.to(
                device=sequence.device,
                dtype=torch.float32,
            )
            if rope_cos.shape != (spatial, self.head_dim):
                raise ValueError(
                    "Self-GDN fused RoPE table has shape "
                    f"{tuple(rope_cos.shape)}, expected {(spatial, self.head_dim)}"
                )
            # VisionRotaryEmbeddingFast stores the same positive sine for
            # both elements of each pair and supplies the [-,+] signs in
            # rotate_half(). The fused kernel loads x[d^1] directly, so its
            # table must carry those signs explicitly.
            rope_sign = torch.ones_like(rope_sin)
            rope_sign[:, 0::2] = -1.0
            rope_sin = rope_sin * rope_sign
            rope_cos = rope_cos.repeat(frames, 1).contiguous()
            rope_sin = rope_sin.repeat(frames, 1).contiguous()

        output = fused_gdn_forward_with_grad(
            qkv,
            beta,
            decay,
            None,
            None,
            rope_cos,
            rope_sin,
            frames,
            spatial,
            k_scale=(self.head_dim * spatial) ** -0.5,
            norm_eps=self.eps,
            eps=self.eps,
            qk_norm=False,
        )
        return output[:, -spatial:].reshape(
            batch,
            spatial,
            self.dim,
        ).to(sequence.dtype)

    def _softmax_cross(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        q = self._apply_spatial_rope(q, rope)[:, :, 0]
        k = self._apply_spatial_rope(k, rope)
        batch, heads, frames, spatial, head_dim = k.shape
        output = F.scaled_dot_product_attention(
            q,
            k.reshape(batch, heads, frames * spatial, head_dim),
            v.reshape(batch, heads, frames * spatial, head_dim),
        )
        return output.transpose(1, 2).reshape(batch, spatial, self.dim)

    def _gdn_cross(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        """Write observed frames into a recurrent state, then read one target."""

        batch = query.shape[0]
        frames = k.shape[1]
        spatial = k.shape[2]
        beta = torch.sigmoid(self.beta_proj(context)).reshape(
            batch, frames, spatial, self.num_heads
        ).permute(0, 3, 1, 2)
        frame_features = context.reshape(
            batch, frames, spatial, self.dim
        ).mean(dim=2)
        gate = self.gate_proj(frame_features).float()
        decay = torch.exp(
            -torch.exp(self.A_log).view(1, 1, self.num_heads)
            * F.softplus(gate + self.dt_bias.view(1, 1, self.num_heads))
        ).permute(0, 2, 1)

        q = F.relu(q).permute(0, 3, 1, 2, 4)
        k = F.relu(k).permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        q_rot = self._apply_spatial_rope(q, rope).permute(0, 1, 2, 4, 3)
        k_rot = self._apply_spatial_rope(k, rope).permute(0, 1, 2, 4, 3)
        q = q.permute(0, 1, 2, 4, 3)
        k = k.permute(0, 1, 2, 4, 3)
        v = v.permute(0, 1, 2, 4, 3)

        key_scale = (self.head_dim * spatial) ** -0.5
        k = k * key_scale
        k_rot = k_rot * key_scale
        state_kv = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        state_z = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            device=query.device,
            dtype=torch.float32,
        )
        for frame_index in range(frames):
            beta_t = beta[:, :, frame_index].float().unsqueeze(2)
            decay_t = decay[:, :, frame_index].unsqueeze(-1)
            k_t = k[:, :, frame_index].float()
            k_rot_t = k_rot[:, :, frame_index].float()
            v_t = v[:, :, frame_index].float()
            k_beta = k_t * beta_t
            k_rot_beta = k_rot_t * beta_t
            state_kv = decay_t.unsqueeze(-1) * (
                state_kv
                - torch.matmul(
                    torch.matmul(state_kv, k_rot_beta),
                    k_rot_t.transpose(-1, -2),
                )
            ) + torch.matmul(v_t * beta_t, k_rot_t.transpose(-1, -2))
            state_z = decay_t * (
                state_z
                - torch.matmul(
                    k_beta,
                    torch.matmul(k_t.transpose(-1, -2), state_z.unsqueeze(-1)),
                ).squeeze(-1)
            ) + k_beta.sum(dim=-1)

        numerator = torch.matmul(state_kv, q_rot[:, :, 0].float())
        denominator = torch.matmul(
            state_z.unsqueeze(2), q[:, :, 0].float()
        ).squeeze(2).unsqueeze(2)
        output = numerator / (denominator + self.eps)
        return output.permute(0, 3, 1, 2).reshape(
            batch, spatial, self.dim
        ).to(query.dtype)

    def _gdn_cross_triton(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        """Vectorize frame transitions and run the official Triton state scan."""

        from nwm.models.gdn_triton import frame_gdn_scan

        batch = query.shape[0]
        frames = k.shape[1]
        spatial = k.shape[2]
        beta = torch.sigmoid(self.beta_proj(context)).reshape(
            batch, frames, spatial, self.num_heads
        ).permute(0, 3, 1, 2).float()
        frame_features = context.reshape(
            batch, frames, spatial, self.dim
        ).mean(dim=2)
        gate = self.gate_proj(frame_features).float()
        decay = torch.exp(
            -torch.exp(self.A_log).view(1, 1, self.num_heads)
            * F.softplus(gate + self.dt_bias.view(1, 1, self.num_heads))
        ).permute(0, 2, 1)

        q = F.relu(q).permute(0, 3, 1, 2, 4)
        k = F.relu(k).permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        q_rot = self._apply_spatial_rope(q, rope).permute(0, 1, 2, 4, 3)
        k_rot = self._apply_spatial_rope(k, rope).permute(0, 1, 2, 4, 3)
        q = q.permute(0, 1, 2, 4, 3)
        k = k.permute(0, 1, 2, 4, 3)
        v = v.permute(0, 1, 2, 4, 3)

        key_scale = (self.head_dim * spatial) ** -0.5
        k = k.float() * key_scale
        k_rot = k_rot.float() * key_scale
        v = v.float()
        beta = beta.unsqueeze(3)
        decay = decay.unsqueeze(-1).unsqueeze(-1)
        identity = torch.eye(
            self.head_dim,
            device=query.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, self.head_dim, self.head_dim)

        k_rot_beta = k_rot * beta
        w_kv = decay * (
            identity - torch.matmul(k_rot_beta, k_rot.transpose(-1, -2))
        )
        u_kv = torch.matmul(v * beta, k_rot.transpose(-1, -2))
        k_beta = k * beta
        w_z = decay * (
            identity - torch.matmul(k_beta, k.transpose(-1, -2))
        )
        u_z = k_beta.sum(dim=-1)

        batch_heads = batch * self.num_heads
        state_kv, state_z = frame_gdn_scan(
            w_kv.reshape(
                batch_heads, frames, self.head_dim, self.head_dim
            ).contiguous(),
            u_kv.reshape(
                batch_heads, frames, self.head_dim, self.head_dim
            ).contiguous(),
            w_z.reshape(
                batch_heads, frames, self.head_dim, self.head_dim
            ).contiguous(),
            u_z.reshape(
                batch_heads, frames, self.head_dim
            ).contiguous(),
        )
        final_kv = state_kv[:, -1].reshape(
            batch, self.num_heads, self.head_dim, self.head_dim
        )
        final_z = state_z[:, -1].reshape(
            batch, self.num_heads, self.head_dim
        )
        numerator = torch.matmul(final_kv, q_rot[:, :, 0].float())
        denominator = torch.matmul(
            final_z.unsqueeze(2), q[:, :, 0].float()
        ).squeeze(2).unsqueeze(2)
        output = numerator / (denominator + self.eps)
        return output.permute(0, 3, 1, 2).reshape(
            batch, spatial, self.dim
        ).to(query.dtype)

    def _gdn_cross_triton_fused(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope,
    ) -> torch.Tensor:
        """Run SANA-WM's complete autograd mega-kernel for cross-context GDN.

        Context frames provide K/V state updates. A final synthetic frame
        carries only the target Q, with beta=0 and decay=1, so querying the
        final output does not contaminate the state with the noisy target.
        """

        from nwm.models.sana_fused_gdn import fused_gdn_forward_with_grad

        batch = query.shape[0]
        context_frames = k.shape[1]
        spatial = k.shape[2]
        total_frames = context_frames + 1

        beta_context = torch.sigmoid(self.beta_proj(context)).reshape(
            batch, context_frames, spatial, self.num_heads
        ).permute(0, 3, 1, 2)
        frame_features = context.reshape(
            batch, context_frames, spatial, self.dim
        ).mean(dim=2)
        gate = self.gate_proj(frame_features).float()
        decay_context = torch.exp(
            -torch.exp(self.A_log).view(1, 1, self.num_heads)
            * F.softplus(gate + self.dt_bias.view(1, 1, self.num_heads))
        ).permute(0, 2, 1)

        zero_q_context = torch.zeros_like(k)
        zero_kv_target = torch.zeros_like(q)
        q_all = torch.cat([zero_q_context, q], dim=1)
        k_all = torch.cat([k, zero_kv_target], dim=1)
        v_all = torch.cat([v, zero_kv_target], dim=1)
        qkv = torch.stack([q_all, k_all, v_all], dim=3).reshape(
            batch,
            total_frames * spatial,
            3,
            self.num_heads,
            self.head_dim,
        ).contiguous()

        beta = torch.cat(
            [
                beta_context,
                torch.zeros(
                    batch,
                    self.num_heads,
                    1,
                    spatial,
                    device=query.device,
                    dtype=beta_context.dtype,
                ),
            ],
            dim=2,
        ).contiguous()
        decay = torch.cat(
            [
                decay_context,
                torch.ones(
                    batch,
                    self.num_heads,
                    1,
                    device=query.device,
                    dtype=decay_context.dtype,
                ),
            ],
            dim=2,
        ).contiguous()

        if rope is None:
            rope_cos = torch.ones(
                total_frames * spatial,
                self.head_dim,
                device=query.device,
                dtype=torch.float32,
            )
            rope_sin = torch.zeros_like(rope_cos)
        else:
            rope_cos = rope.freqs_cos.to(device=query.device, dtype=torch.float32)
            rope_sin = rope.freqs_sin.to(device=query.device, dtype=torch.float32)
            if rope_cos.shape != (spatial, self.head_dim):
                raise ValueError(
                    "Cross-GDN fused RoPE table has shape "
                    f"{tuple(rope_cos.shape)}, expected {(spatial, self.head_dim)}"
                )
            rope_sign = torch.ones_like(rope_sin)
            rope_sign[:, 0::2] = -1.0
            rope_sin = rope_sin * rope_sign
            rope_cos = rope_cos.repeat(total_frames, 1).contiguous()
            rope_sin = rope_sin.repeat(total_frames, 1).contiguous()

        output = fused_gdn_forward_with_grad(
            qkv,
            beta,
            decay,
            None,
            None,
            rope_cos,
            rope_sin,
            total_frames,
            spatial,
            k_scale=(self.head_dim * spatial) ** -0.5,
            norm_eps=self.eps,
            eps=self.eps,
            qk_norm=False,
        )
        return output[:, -spatial:].reshape(
            batch, spatial, self.dim
        ).to(query.dtype)

    def _prepare_cross_state_triton(
        self,
        key: torch.Tensor,
        key_rot: torch.Tensor,
        value: torch.Tensor,
        beta: torch.Tensor,
        decay: torch.Tensor,
    ) -> GDNState:
        """Prepare the reusable Cross-GDN state with the Triton scan kernel.

        The portable cached path performs one Python-level recurrence per
        context frame.  This path materializes the affine frame transitions
        in parallel and lets the autograd-enabled Triton scan apply all frame
        transitions in a single kernel launch.  Only the final state is
        exposed to repeated target queries.
        """

        if not key.is_cuda:
            raise ValueError("Triton cached Cross-GDN preparation requires CUDA")
        from nwm.models.gdn_triton import (
            frame_gdn_affine_tree_final,
            frame_gdn_final_scan,
            frame_gdn_scan,
            fused_frame_gdn_final_state,
        )

        batch, heads, frames, head_dim, _ = key.shape
        requested_state_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.state_dtype]
        inference_only = not torch.is_grad_enabled()
        if inference_only and self.cross_cache_backend == "fused_final":
            batch_heads = batch * heads
            state_kv, state_z = fused_frame_gdn_final_state(
                key.reshape(
                    batch_heads, frames, head_dim, key.shape[-1]
                ).contiguous(),
                key_rot.reshape(
                    batch_heads, frames, head_dim, key_rot.shape[-1]
                ).contiguous(),
                value.reshape(
                    batch_heads, frames, head_dim, value.shape[-1]
                ).contiguous(),
                beta.reshape(batch_heads, frames, beta.shape[-1]).contiguous(),
                decay.reshape(batch_heads, frames).contiguous(),
                state_dtype=requested_state_dtype,
            )
            return GDNState(
                state_kv=state_kv.reshape(
                    batch, heads, head_dim, head_dim
                ),
                state_z=state_z.reshape(batch, heads, head_dim),
            )

        beta = beta.float().unsqueeze(3)
        decay_matrix = decay.float().unsqueeze(-1).unsqueeze(-1)
        key = key.float()
        key_rot = key_rot.float()
        value = value.float()
        identity = torch.eye(
            head_dim,
            device=key.device,
            dtype=torch.float32,
        ).reshape(1, 1, 1, head_dim, head_dim)

        key_rot_beta = key_rot * beta
        transition_kv = decay_matrix * (
            identity - torch.matmul(key_rot_beta, key_rot.transpose(-1, -2))
        )
        update_kv = torch.matmul(value * beta, key_rot.transpose(-1, -2))
        key_beta = key * beta
        transition_z = decay_matrix * (
            identity - torch.matmul(key_beta, key.transpose(-1, -2))
        )
        update_z = key_beta.sum(dim=-1)

        batch_heads = batch * heads
        transition_kv = transition_kv.reshape(
            batch_heads, frames, head_dim, head_dim
        ).contiguous()
        update_kv = update_kv.reshape(
            batch_heads, frames, head_dim, head_dim
        ).contiguous()
        transition_z = transition_z.reshape(
            batch_heads, frames, head_dim, head_dim
        ).contiguous()
        update_z = update_z.reshape(
            batch_heads, frames, head_dim
        ).contiguous()
        if inference_only and self.cross_cache_backend in {"final", "tree"}:
            reducer = (
                frame_gdn_affine_tree_final
                if self.cross_cache_backend == "tree"
                else frame_gdn_final_scan
            )
            final_kv, final_z = reducer(
                transition_kv,
                update_kv,
                transition_z,
                update_z,
                state_dtype=requested_state_dtype,
            )
            return GDNState(
                state_kv=final_kv.reshape(
                    batch, heads, head_dim, head_dim
                ),
                state_z=final_z.reshape(batch, heads, head_dim),
            )

        state_kv, state_z = frame_gdn_scan(
            transition_kv,
            update_kv,
            transition_z,
            update_z,
        )
        return GDNState(
            state_kv=state_kv[:, -1].reshape(
                batch, heads, head_dim, head_dim
            ),
            state_z=state_z[:, -1].reshape(batch, heads, head_dim),
        )

    def prepare_cross_state(
        self,
        context: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
        rope=None,
    ) -> GDNState:
        """Write context once and return the paired canonical GDN state."""

        if self.mixer_type != "gdn":
            raise RuntimeError("prepare_cross_state requires mixer_type='gdn'")
        height, width = int(grid_hw[0]), int(grid_hw[1])
        spatial = height * width
        if context.ndim != 3 or context.shape[-1] != self.dim:
            raise ValueError(
                f"Cross-GDN context must be [B,N,{self.dim}], got "
                f"{tuple(context.shape)}"
            )
        if context.shape[1] % spatial != 0:
            raise ValueError(
                f"Cross-GDN context tokens={context.shape[1]} are not divisible "
                f"by spatial={spatial}"
            )
        batch = context.shape[0]
        frames = context.shape[1] // spatial
        channels = self.dim
        weight = self.qkv.weight
        bias = self.qkv.bias
        # One K/V-only GEMM: unlike the ordinary qkv projection this never
        # computes a context-side Q, and unlike two sliced F.linear calls it
        # reads the context activation only once through the GEMM pipeline.
        key, value = F.linear(
            context,
            weight[channels:],
            None if bias is None else bias[channels:],
        ).chunk(2, dim=-1)
        key = self._causal_key_convolution(
            key,
            frames=frames,
            spatial=spatial,
        )
        key = self.k_norm(key).reshape(
            batch,
            frames,
            spatial,
            self.num_heads,
            self.head_dim,
        )
        value = value.reshape(
            batch,
            frames,
            spatial,
            self.num_heads,
            self.head_dim,
        )
        beta = torch.sigmoid(self.beta_proj(context)).reshape(
            batch,
            frames,
            spatial,
            self.num_heads,
        )
        frame_features = context.reshape(
            batch,
            frames,
            spatial,
            self.dim,
        ).mean(dim=2)
        gate = self.gate_proj(frame_features).float()
        decay = torch.exp(
            -torch.exp(self.A_log).view(1, 1, self.num_heads)
            * F.softplus(gate + self.dt_bias.view(1, 1, self.num_heads))
        )

        if (
            self.backend in {"triton", "triton_fused"}
            and self.cross_cache_backend == "fused_final_raw"
            and not torch.is_grad_enabled()
        ):
            from nwm.models.gdn_triton import fused_raw_key_frame_gdn_final_state

            raw_key = key.reshape(batch, frames, spatial, self.dim)
            raw_value = value.reshape(batch, frames, spatial, self.dim)
            key_inv_rms = torch.rsqrt(
                raw_key.float().square().mean(dim=-1) + self.k_norm.eps
            )
            if rope is None:
                rope_cos = torch.ones(
                    spatial,
                    self.head_dim,
                    device=context.device,
                    dtype=torch.float32,
                )
                rope_sin = torch.zeros_like(rope_cos)
            else:
                rope_cos = rope.freqs_cos.to(
                    device=context.device, dtype=torch.float32
                ).contiguous()
                rope_sin = rope.freqs_sin.to(
                    device=context.device, dtype=torch.float32
                ).contiguous()
            state_dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }[self.state_dtype]
            state_kv, state_z = fused_raw_key_frame_gdn_final_state(
                raw_key.contiguous(),
                raw_value.contiguous(),
                beta.contiguous(),
                decay.contiguous(),
                key_inv_rms.contiguous(),
                self.k_norm.weight.contiguous(),
                rope_cos,
                rope_sin,
                num_heads=self.num_heads,
                key_scale=(self.head_dim * spatial) ** -0.5,
                state_dtype=state_dtype,
            )
            return GDNState(
                state_kv=state_kv.reshape(
                    batch, self.num_heads, self.head_dim, self.head_dim
                ),
                state_z=state_z.reshape(
                    batch, self.num_heads, self.head_dim
                ),
            )

        beta = beta.permute(0, 3, 1, 2)
        decay = decay.permute(0, 2, 1)

        key = F.relu(key).permute(0, 3, 1, 4, 2)
        value = value.permute(0, 3, 1, 4, 2)
        key_rot = self._apply_spatial_rope(
            key.permute(0, 1, 2, 4, 3),
            rope,
        ).permute(0, 1, 2, 4, 3)
        key_scale = (self.head_dim * spatial) ** -0.5
        key = key * key_scale
        key_rot = key_rot * key_scale

        # The complete SANA fused autograd kernel returns token outputs rather
        # than a reusable final state.  Cached Cross-GDN therefore uses the
        # dedicated autograd-enabled Triton state scan for both accelerated
        # backends instead of falling back to the Python frame loop.
        if self.backend in {"triton", "triton_fused"}:
            return self._prepare_cross_state_triton(
                key,
                key_rot,
                value,
                beta,
                decay,
            )

        state_kv = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device=context.device,
            dtype=torch.float32,
        )
        state_z = torch.zeros(
            batch,
            self.num_heads,
            self.head_dim,
            device=context.device,
            dtype=torch.float32,
        )
        for frame_index in range(frames):
            beta_t = beta[:, :, frame_index].float().unsqueeze(2)
            decay_t = decay[:, :, frame_index].unsqueeze(-1)
            key_t = key[:, :, frame_index].float()
            key_rot_t = key_rot[:, :, frame_index].float()
            value_t = value[:, :, frame_index].float()
            key_beta = key_t * beta_t
            key_rot_beta = key_rot_t * beta_t
            state_kv = decay_t.unsqueeze(-1) * (
                state_kv
                - torch.matmul(
                    torch.matmul(state_kv, key_rot_beta),
                    key_rot_t.transpose(-1, -2),
                )
            ) + torch.matmul(
                value_t * beta_t,
                key_rot_t.transpose(-1, -2),
            )
            state_z = decay_t * (
                state_z
                - torch.matmul(
                    key_beta,
                    torch.matmul(
                        key_t.transpose(-1, -2),
                        state_z.unsqueeze(-1),
                    ),
                ).squeeze(-1)
            ) + key_beta.sum(dim=-1)
        return GDNState(state_kv=state_kv, state_z=state_z)

    def query_cross_state(
        self,
        query: torch.Tensor,
        state: GDNState,
        *,
        grid_hw: tuple[int, int],
        rope=None,
    ) -> torch.Tensor:
        """Read a cached Cross-GDN state without recomputing context."""

        height, width = int(grid_hw[0]), int(grid_hw[1])
        spatial = height * width
        state_batch = int(state.state_kv.shape[0])
        query_batch = int(query.shape[0])
        if (
            query.shape[1:] != (spatial, self.dim)
            or state_batch <= 0
            or query_batch % state_batch != 0
        ):
            raise ValueError(
                "Cross-GDN query/state shape mismatch: "
                f"query={tuple(query.shape)}, state={tuple(state.state_kv.shape)}"
            )
        channels = self.dim
        bias = self.qkv.bias
        projected = F.linear(
            query,
            self.qkv.weight[:channels],
            None if bias is None else bias[:channels],
        )
        if (
            self.cross_query_backend == "triton"
            and query.is_cuda
            and not torch.is_grad_enabled()
        ):
            from nwm.models.gdn_triton import grouped_raw_query_gdn_state_read

            query_inv_rms = torch.rsqrt(
                projected.float().square().mean(dim=-1) + self.q_norm.eps
            )
            if rope is None:
                rope_cos = torch.ones(
                    spatial,
                    self.head_dim,
                    device=query.device,
                    dtype=torch.float32,
                )
                rope_sin = torch.zeros_like(rope_cos)
            else:
                rope_cos = rope.freqs_cos.to(
                    device=query.device, dtype=torch.float32
                ).contiguous()
                rope_sin = rope.freqs_sin.to(
                    device=query.device, dtype=torch.float32
                ).contiguous()
            output = grouped_raw_query_gdn_state_read(
                state.state_kv.contiguous(),
                state.state_z.contiguous(),
                projected.contiguous(),
                query_inv_rms.contiguous(),
                self.q_norm.weight.contiguous(),
                rope_cos,
                rope_sin,
                num_heads=self.num_heads,
                eps=self.eps,
            ).to(query.dtype)
            output_gate = F.silu(self.output_gate(query).float()).to(
                output.dtype
            )
            return self.proj(output * output_gate)

        projected = self.q_norm(projected).reshape(
            query_batch,
            1,
            spatial,
            self.num_heads,
            self.head_dim,
        )
        projected = F.relu(projected).permute(0, 3, 1, 2, 4)
        projected_rot = self._apply_spatial_rope(
            projected,
            rope,
        ).permute(0, 1, 2, 4, 3)
        projected = projected.permute(0, 1, 2, 4, 3)
        projected_rot = projected_rot[:, :, 0].float()
        projected = projected[:, :, 0].float()
        group_size = query_batch // state_batch
        grouped_query = projected.reshape(
            state_batch,
            group_size,
            self.num_heads,
            self.head_dim,
            spatial,
        )
        grouped_rot = projected_rot.reshape_as(grouped_query)
        numerator = torch.matmul(
            state.state_kv.float().unsqueeze(1),
            grouped_rot,
        )
        denominator = torch.matmul(
            state.state_z.float().unsqueeze(1).unsqueeze(3),
            grouped_query,
        ).squeeze(3).unsqueeze(3)
        output = (numerator / (denominator + self.eps)).reshape(
            query_batch,
            self.num_heads,
            self.head_dim,
            spatial,
        )
        output = output.permute(0, 3, 1, 2).reshape(
            query_batch,
            spatial,
            self.dim,
        ).to(query.dtype)
        output_gate = F.silu(self.output_gate(query).float()).to(output.dtype)
        return self.proj(output * output_gate)

    def forward_self(
        self,
        target: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
        rope=None,
    ) -> torch.Tensor:
        """Canonical target-only GDN Self mixer with no context path."""

        height, width = int(grid_hw[0]), int(grid_hw[1])
        spatial = height * width
        if target.shape[1:] != (spatial, self.dim):
            raise ValueError(
                f"Self-GDN target must be [B,{spatial},{self.dim}], got "
                f"{tuple(target.shape)}"
            )
        q, k, v = self._project_qkv(target, frames=1, spatial=spatial)
        if self.backend == "triton_fused":
            output = self._gdn_target_triton_fused(
                target,
                q,
                k,
                v,
                rope=rope,
            )
        else:
            output = self._gdn_target(target, q, k, v, rope=rope)
        output_gate = F.silu(self.output_gate(target).float()).to(output.dtype)
        return self.proj(output * output_gate)

    def forward_cross(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
        rope=None,
    ) -> torch.Tensor:
        """Cross-attention replacement using SANA-WM frame-wise recurrence."""

        height, width = int(grid_hw[0]), int(grid_hw[1])
        spatial = height * width
        if query.ndim != 3 or context.ndim != 3:
            raise ValueError("Cross-GDN query/context must be BNC tensors")
        if context.shape[1] % spatial != 0:
            raise ValueError(
                f"Cross-GDN context tokens={context.shape[1]} are not divisible "
                f"by spatial={spatial}"
            )
        frames = context.shape[1] // spatial
        q, k, v = self._project_cross_qkv(
            query,
            context,
            frames=frames,
            spatial=spatial,
        )
        if self.mixer_type == "softmax":
            output = self._softmax_cross(q, k, v, rope=rope)
        elif self.backend == "triton":
            output = self._gdn_cross_triton(
                query,
                context,
                q,
                k,
                v,
                rope=rope,
            )
        elif self.backend == "triton_fused":
            output = self._gdn_cross_triton_fused(
                query,
                context,
                q,
                k,
                v,
                rope=rope,
            )
        else:
            output = self._gdn_cross(
                query,
                context,
                q,
                k,
                v,
                rope=rope,
            )
        gate = F.silu(self.output_gate(query).float()).to(output.dtype)
        return self.proj(output * gate)

    def forward(
        self,
        target: torch.Tensor,
        context: torch.Tensor,
        *,
        grid_hw: tuple[int, int],
        rope=None,
    ) -> torch.Tensor:
        height, width = int(grid_hw[0]), int(grid_hw[1])
        spatial = height * width
        if target.ndim != 3 or context.ndim != 3:
            raise ValueError("GDN target/context must be BNC tensors")
        if target.shape[1] != spatial:
            raise ValueError(
                f"Target tokens={target.shape[1]} do not match grid {grid_hw}"
            )
        if context.shape[0] != target.shape[0] or context.shape[2] != target.shape[2]:
            raise ValueError("GDN target/context batch or channel mismatch")
        if context.shape[1] % spatial != 0:
            raise ValueError(
                f"Context tokens={context.shape[1]} are not divisible by spatial={spatial}"
            )
        frames = context.shape[1] // spatial + 1
        sequence = torch.cat([context, target], dim=1)
        q, k, v = self._project_qkv(sequence, frames=frames, spatial=spatial)
        if self.mixer_type == "softmax":
            output = self._softmax_target(sequence, q, k, v, rope=rope)
        elif self.backend == "triton_fused":
            output = self._gdn_target_triton_fused(
                sequence,
                q,
                k,
                v,
                rope=rope,
            )
        else:
            output = self._gdn_target(sequence, q, k, v, rope=rope)
        output = self._stabilize_gdn_output(output)
        gate = F.silu(self.output_gate(target).float()).to(output.dtype)
        return self.proj(output * gate)
