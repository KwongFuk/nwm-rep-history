# Adapted from NVlabs/Sana @ 6298508 (SANA-WM frame-wise GDN). Modified by the
# authors. Licensed under the Apache License 2.0; see LICENSES/Apache-2.0-Sana.txt.
"""Triton state scan for frame-wise GDN.

Adapted from NVlabs/Sana commit
6298508fcb511762a11c42cff45b2fc9fd930325,
``diffusion/model/ops/frame_gdn/scan_triton.py`` (Apache-2.0).
The public wrapper keeps this optional: importing the project on CPU does not
require executing or compiling a Triton kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _scan_forward_kernel(
    w_kv_ptr,
    u_kv_ptr,
    w_z_ptr,
    u_z_ptr,
    state_kv_ptr,
    state_z_ptr,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    index_bh = tl.program_id(0)
    stride_dd = D * D
    base_dd = index_bh.to(tl.int64) * T * stride_dd
    base_d = index_bh.to(tl.int64) * T * D
    offsets = tl.arange(0, BD)
    mask_d = offsets < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    state_kv = tl.zeros([BD, BD], dtype=tl.float32)
    state_z = tl.zeros([BD], dtype=tl.float32)

    for frame_index in range(0, T):
        offset_dd = base_dd + frame_index * stride_dd
        offset_d = base_d + frame_index * D
        w_kv = tl.load(
            w_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        u_kv = tl.load(
            u_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        state_kv = tl.dot(state_kv, w_kv, allow_tf32=False) + u_kv
        state_kv_output = (
            state_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :]
        )
        tl.store(
            state_kv_output,
            state_kv.to(state_kv_output.dtype.element_ty),
            mask=mask_dd,
        )

        w_z = tl.load(
            w_z_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        u_z = tl.load(
            u_z_ptr + offset_d + offsets,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        state_z = tl.sum(w_z * state_z[None, :], axis=1) + u_z
        state_z_output = state_z_ptr + offset_d + offsets
        tl.store(
            state_z_output,
            state_z.to(state_z_output.dtype.element_ty),
            mask=mask_d,
        )


@triton.jit(do_not_specialize=["T"])
def _scan_final_forward_kernel(
    w_kv_ptr,
    u_kv_ptr,
    w_z_ptr,
    u_z_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    """Apply affine frame transitions but write only the terminal state."""

    index_bh = tl.program_id(0)
    stride_dd = D * D
    base_dd = index_bh.to(tl.int64) * T * stride_dd
    base_d = index_bh.to(tl.int64) * T * D
    offsets = tl.arange(0, BD)
    mask_d = offsets < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    state_kv = tl.zeros([BD, BD], dtype=tl.float32)
    state_z = tl.zeros([BD], dtype=tl.float32)

    for frame_index in range(0, T):
        offset_dd = base_dd + frame_index * stride_dd
        offset_d = base_d + frame_index * D
        w_kv = tl.load(
            w_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        u_kv = tl.load(
            u_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        state_kv = tl.dot(state_kv, w_kv, allow_tf32=False) + u_kv

        w_z = tl.load(
            w_z_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        u_z = tl.load(
            u_z_ptr + offset_d + offsets,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        state_z = tl.sum(w_z * state_z[None, :], axis=1) + u_z

    output_dd = index_bh.to(tl.int64) * stride_dd
    output_d = index_bh.to(tl.int64) * D
    tl.store(
        final_state_kv_ptr
        + output_dd
        + offsets[:, None] * D
        + offsets[None, :],
        state_kv,
        mask=mask_dd,
    )
    tl.store(
        final_state_z_ptr + output_d + offsets,
        state_z,
        mask=mask_d,
    )


@triton.jit(do_not_specialize=["F"])
def _fused_frame_final_state_kernel(
    key_ptr,
    key_rot_ptr,
    value_ptr,
    beta_ptr,
    decay_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    F,
    D: tl.constexpr,
    S: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
):
    """Build and apply every frame update in one final-state-only kernel.

    The inputs contain only K, rotated K, V, beta, and decay.  Q, output
    gates, token outputs, affine transition tensors, and intermediate states
    are deliberately absent from the cached Cross prefill path.
    """

    index_bh = tl.program_id(0)
    offsets_d = tl.arange(0, BD)
    mask_d = offsets_d < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    state_kv = tl.zeros([BD, BD], dtype=tl.float32)
    state_z = tl.zeros([BD], dtype=tl.float32)
    frame_stride = D * S
    bh_frame_base = index_bh.to(tl.int64) * F * frame_stride
    bh_beta_base = index_bh.to(tl.int64) * F * S
    bh_decay_base = index_bh.to(tl.int64) * F

    for frame_index in range(0, F):
        correction_kv = tl.zeros([BD, BD], dtype=tl.float32)
        update_kv = tl.zeros([BD, BD], dtype=tl.float32)
        correction_z = tl.zeros([BD], dtype=tl.float32)
        update_z = tl.zeros([BD], dtype=tl.float32)
        frame_base = bh_frame_base + frame_index * frame_stride
        beta_base = bh_beta_base + frame_index * S

        for spatial_start in range(0, S, BS):
            offsets_s = spatial_start + tl.arange(0, BS)
            mask_s = offsets_s < S
            mask_ds = mask_d[:, None] & mask_s[None, :]
            feature_offsets = (
                frame_base + offsets_d[:, None] * S + offsets_s[None, :]
            )
            key = tl.load(
                key_ptr + feature_offsets,
                mask=mask_ds,
                other=0.0,
            ).to(tl.float32)
            key_rot = tl.load(
                key_rot_ptr + feature_offsets,
                mask=mask_ds,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                value_ptr + feature_offsets,
                mask=mask_ds,
                other=0.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_ptr + beta_base + offsets_s,
                mask=mask_s,
                other=0.0,
            ).to(tl.float32)
            key_rot_beta = key_rot * beta[None, :]
            key_beta = key * beta[None, :]

            state_times_key = tl.dot(
                state_kv.to(tl.bfloat16),
                key_rot_beta.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
            correction_kv += tl.dot(
                state_times_key.to(tl.bfloat16),
                tl.trans(key_rot.to(tl.bfloat16)),
                out_dtype=tl.float32,
            )
            update_kv += tl.dot(
                (value * beta[None, :]).to(tl.bfloat16),
                tl.trans(key_rot.to(tl.bfloat16)),
                out_dtype=tl.float32,
            )

            key_t_state = tl.sum(key * state_z[:, None], axis=0)
            correction_z += tl.sum(
                key_beta * key_t_state[None, :],
                axis=1,
            )
            update_z += tl.sum(key_beta, axis=1)

        decay = tl.load(decay_ptr + bh_decay_base + frame_index).to(
            tl.float32
        )
        state_kv = decay * (state_kv - correction_kv) + update_kv
        state_z = decay * (state_z - correction_z) + update_z

    output_dd = index_bh.to(tl.int64) * D * D
    output_d = index_bh.to(tl.int64) * D
    tl.store(
        final_state_kv_ptr
        + output_dd
        + offsets_d[:, None] * D
        + offsets_d[None, :],
        state_kv,
        mask=mask_dd,
    )
    tl.store(
        final_state_z_ptr + output_d + offsets_d,
        state_z,
        mask=mask_d,
    )


@triton.jit(do_not_specialize=["F"])
def _fused_raw_key_final_state_kernel(
    raw_key_ptr,
    value_ptr,
    beta_ptr,
    decay_ptr,
    key_inv_rms_ptr,
    key_norm_weight_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    final_state_kv_ptr,
    final_state_z_ptr,
    F,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    KEY_SCALE: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
):
    """Final-state Cross prefill with fused K-norm/ReLU/RoPE/layout.

    Raw K/V stay in GEMM-native ``[B,F,S,C]`` layout, beta in
    ``[B,F,S,H]``, and decay in ``[B,F,H]``.  This removes the large
    permute/contiguous copies and never materializes normalized or rotated K.
    """

    index_bh = tl.program_id(0)
    index_b = index_bh // H
    index_h = index_bh % H
    offsets_d = tl.arange(0, BD)
    offsets_pair = offsets_d ^ 1
    mask_d = offsets_d < D
    mask_pair = offsets_pair < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    rotate_sign = tl.where((offsets_d & 1) == 0, -1.0, 1.0)
    norm_weight = tl.load(
        key_norm_weight_ptr + index_h * D + offsets_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    norm_weight_pair = tl.load(
        key_norm_weight_ptr + index_h * D + offsets_pair,
        mask=mask_pair,
        other=0.0,
    ).to(tl.float32)
    state_kv = tl.zeros([BD, BD], dtype=tl.float32)
    state_z = tl.zeros([BD], dtype=tl.float32)

    for frame_index in range(0, F):
        correction_kv = tl.zeros([BD, BD], dtype=tl.float32)
        update_kv = tl.zeros([BD, BD], dtype=tl.float32)
        correction_z = tl.zeros([BD], dtype=tl.float32)
        update_z = tl.zeros([BD], dtype=tl.float32)
        token_base = (index_b * F + frame_index).to(tl.int64) * S

        for spatial_start in range(0, S, BS):
            offsets_s = spatial_start + tl.arange(0, BS)
            mask_s = offsets_s < S
            mask_ds = mask_d[:, None] & mask_s[None, :]
            mask_pair_s = mask_pair[:, None] & mask_s[None, :]
            token_offsets = token_base + offsets_s
            channel_offsets = index_h * D + offsets_d
            pair_channel_offsets = index_h * D + offsets_pair
            raw_key = tl.load(
                raw_key_ptr
                + token_offsets[None, :] * C
                + channel_offsets[:, None],
                mask=mask_ds,
                other=0.0,
            ).to(tl.float32)
            raw_pair = tl.load(
                raw_key_ptr
                + token_offsets[None, :] * C
                + pair_channel_offsets[:, None],
                mask=mask_pair_s,
                other=0.0,
            ).to(tl.float32)
            value = tl.load(
                value_ptr
                + token_offsets[None, :] * C
                + channel_offsets[:, None],
                mask=mask_ds,
                other=0.0,
            ).to(tl.float32)
            inverse_rms = tl.load(
                key_inv_rms_ptr + token_offsets,
                mask=mask_s,
                other=1.0,
            ).to(tl.float32)
            beta = tl.load(
                beta_ptr
                + token_offsets * H
                + index_h,
                mask=mask_s,
                other=0.0,
            ).to(tl.float32)
            key = tl.maximum(
                raw_key * inverse_rms[None, :] * norm_weight[:, None],
                0.0,
            ) * KEY_SCALE
            key_pair = tl.maximum(
                raw_pair
                * inverse_rms[None, :]
                * norm_weight_pair[:, None],
                0.0,
            ) * KEY_SCALE
            rope_offsets = offsets_s[None, :] * D + offsets_d[:, None]
            rope_mask = mask_s[None, :] & mask_d[:, None]
            rope_cos = tl.load(
                rope_cos_ptr + rope_offsets,
                mask=rope_mask,
                other=1.0,
            ).to(tl.float32)
            rope_sin = tl.load(
                rope_sin_ptr + rope_offsets,
                mask=rope_mask,
                other=0.0,
            ).to(tl.float32)
            key_rot = (
                key * rope_cos
                + key_pair * rotate_sign[:, None] * rope_sin
            )
            key_rot_beta = key_rot * beta[None, :]
            key_beta = key * beta[None, :]

            state_times_key = tl.dot(
                state_kv.to(tl.bfloat16),
                key_rot_beta.to(tl.bfloat16),
                out_dtype=tl.float32,
            )
            correction_kv += tl.dot(
                state_times_key.to(tl.bfloat16),
                tl.trans(key_rot.to(tl.bfloat16)),
                out_dtype=tl.float32,
            )
            update_kv += tl.dot(
                (value * beta[None, :]).to(tl.bfloat16),
                tl.trans(key_rot.to(tl.bfloat16)),
                out_dtype=tl.float32,
            )
            key_t_state = tl.sum(key * state_z[:, None], axis=0)
            correction_z += tl.sum(
                key_beta * key_t_state[None, :], axis=1
            )
            update_z += tl.sum(key_beta, axis=1)

        decay = tl.load(
            decay_ptr + (index_b * F + frame_index) * H + index_h
        ).to(tl.float32)
        state_kv = decay * (state_kv - correction_kv) + update_kv
        state_z = decay * (state_z - correction_z) + update_z

    output_dd = index_bh.to(tl.int64) * D * D
    output_d = index_bh.to(tl.int64) * D
    tl.store(
        final_state_kv_ptr
        + output_dd
        + offsets_d[:, None] * D
        + offsets_d[None, :],
        state_kv,
        mask=mask_dd,
    )
    tl.store(
        final_state_z_ptr + output_d + offsets_d,
        state_z,
        mask=mask_d,
    )


@triton.jit
def _grouped_state_read_kernel(
    state_kv_ptr,
    state_z_ptr,
    query_ptr,
    query_rot_ptr,
    output_ptr,
    GROUP_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Read one compact context state from several grouped target queries."""

    program = tl.program_id(0)
    query_batch = program // H
    head = program % H
    context_batch = query_batch // GROUP_SIZE
    offsets_d = tl.arange(0, BD)
    mask_d = offsets_d < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    state_base = (context_batch * H + head).to(tl.int64)
    state = tl.load(
        state_kv_ptr
        + state_base * D * D
        + offsets_d[:, None] * D
        + offsets_d[None, :],
        mask=mask_dd,
        other=0.0,
    ).to(tl.float32)
    state_z = tl.load(
        state_z_ptr + state_base * D + offsets_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    query_base = (query_batch * H + head).to(tl.int64) * D * S

    for spatial_start in range(0, S, BS):
        offsets_s = spatial_start + tl.arange(0, BS)
        mask_s = offsets_s < S
        mask_ds = mask_d[:, None] & mask_s[None, :]
        query_offsets = (
            query_base + offsets_d[:, None] * S + offsets_s[None, :]
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=mask_ds,
            other=0.0,
        ).to(tl.float32)
        query_rot = tl.load(
            query_rot_ptr + query_offsets,
            mask=mask_ds,
            other=0.0,
        ).to(tl.float32)
        numerator = tl.dot(
            state.to(tl.bfloat16),
            query_rot.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        denominator = tl.sum(state_z[:, None] * query, axis=0)
        output = numerator / (denominator[None, :] + EPS)
        tl.store(
            output_ptr + query_offsets,
            output,
            mask=mask_ds,
        )


@triton.jit
def _grouped_raw_query_state_read_kernel(
    state_kv_ptr,
    state_z_ptr,
    raw_query_ptr,
    query_inv_rms_ptr,
    query_norm_weight_ptr,
    rope_cos_ptr,
    rope_sin_ptr,
    output_ptr,
    GROUP_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    BD: tl.constexpr,
    BS: tl.constexpr,
    EPS: tl.constexpr,
):
    """Fuse Q RMSNorm/ReLU/RoPE/layout with a compact-state read."""

    program = tl.program_id(0)
    query_batch = program // H
    head = program % H
    context_batch = query_batch // GROUP_SIZE
    offsets_d = tl.arange(0, BD)
    offsets_pair = offsets_d ^ 1
    mask_d = offsets_d < D
    mask_pair = offsets_pair < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    rotate_sign = tl.where((offsets_d & 1) == 0, -1.0, 1.0)
    state_base = (context_batch * H + head).to(tl.int64)
    state = tl.load(
        state_kv_ptr
        + state_base * D * D
        + offsets_d[:, None] * D
        + offsets_d[None, :],
        mask=mask_dd,
        other=0.0,
    ).to(tl.float32)
    state_z = tl.load(
        state_z_ptr + state_base * D + offsets_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    norm_weight = tl.load(
        query_norm_weight_ptr + head * D + offsets_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    norm_weight_pair = tl.load(
        query_norm_weight_ptr + head * D + offsets_pair,
        mask=mask_pair,
        other=0.0,
    ).to(tl.float32)

    for spatial_start in range(0, S, BS):
        offsets_s = spatial_start + tl.arange(0, BS)
        mask_s = offsets_s < S
        mask_ds = mask_d[:, None] & mask_s[None, :]
        mask_pair_s = mask_pair[:, None] & mask_s[None, :]
        token_offsets = query_batch.to(tl.int64) * S + offsets_s
        channels = head * D + offsets_d
        pair_channels = head * D + offsets_pair
        raw_query = tl.load(
            raw_query_ptr
            + token_offsets[None, :] * C
            + channels[:, None],
            mask=mask_ds,
            other=0.0,
        ).to(tl.float32)
        raw_pair = tl.load(
            raw_query_ptr
            + token_offsets[None, :] * C
            + pair_channels[:, None],
            mask=mask_pair_s,
            other=0.0,
        ).to(tl.float32)
        inverse_rms = tl.load(
            query_inv_rms_ptr + token_offsets,
            mask=mask_s,
            other=1.0,
        ).to(tl.float32)
        query = tl.maximum(
            raw_query * inverse_rms[None, :] * norm_weight[:, None],
            0.0,
        )
        query_pair = tl.maximum(
            raw_pair
            * inverse_rms[None, :]
            * norm_weight_pair[:, None],
            0.0,
        )
        rope_offsets = offsets_s[None, :] * D + offsets_d[:, None]
        rope_mask = mask_s[None, :] & mask_d[:, None]
        rope_cos = tl.load(
            rope_cos_ptr + rope_offsets,
            mask=rope_mask,
            other=1.0,
        ).to(tl.float32)
        rope_sin = tl.load(
            rope_sin_ptr + rope_offsets,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        query_rot = (
            query * rope_cos
            + query_pair * rotate_sign[:, None] * rope_sin
        )
        numerator = tl.dot(
            state.to(tl.bfloat16),
            query_rot.to(tl.bfloat16),
            out_dtype=tl.float32,
        )
        denominator = tl.sum(state_z[:, None] * query, axis=0)
        output = numerator / (denominator[None, :] + EPS)
        tl.store(
            output_ptr
            + token_offsets[None, :] * C
            + channels[:, None],
            output,
            mask=mask_ds,
        )


@triton.jit(do_not_specialize=["T"])
def _scan_backward_kernel(
    w_kv_ptr,
    state_kv_ptr,
    w_z_ptr,
    state_z_ptr,
    grad_state_kv_ptr,
    grad_state_z_ptr,
    grad_w_kv_ptr,
    grad_u_kv_ptr,
    grad_w_z_ptr,
    grad_u_z_ptr,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    index_bh = tl.program_id(0)
    stride_dd = D * D
    base_dd = index_bh.to(tl.int64) * T * stride_dd
    base_d = index_bh.to(tl.int64) * T * D
    offsets = tl.arange(0, BD)
    mask_d = offsets < D
    mask_dd = mask_d[:, None] & mask_d[None, :]
    grad_kv = tl.zeros([BD, BD], dtype=tl.float32)
    grad_z = tl.zeros([BD], dtype=tl.float32)

    for reverse_index in range(0, T):
        frame_index = T - 1 - reverse_index
        offset_dd = base_dd + frame_index * stride_dd
        offset_d = base_d + frame_index * D
        grad_kv += tl.load(
            grad_state_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        grad_u_kv = (
            grad_u_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :]
        )
        tl.store(
            grad_u_kv,
            grad_kv.to(grad_u_kv.dtype.element_ty),
            mask=mask_dd,
        )

        previous_offset_dd = (
            base_dd + tl.maximum(frame_index - 1, 0) * stride_dd
        )
        previous_state_kv = tl.load(
            state_kv_ptr
            + previous_offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd & (frame_index > 0),
            other=0.0,
        ).to(tl.float32)
        grad_w_kv_value = tl.dot(
            tl.trans(previous_state_kv),
            grad_kv,
            allow_tf32=False,
        )
        grad_w_kv = (
            grad_w_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :]
        )
        tl.store(
            grad_w_kv,
            grad_w_kv_value.to(grad_w_kv.dtype.element_ty),
            mask=mask_dd,
        )
        w_kv = tl.load(
            w_kv_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        grad_kv = tl.dot(grad_kv, tl.trans(w_kv), allow_tf32=False)

        grad_z += tl.load(
            grad_state_z_ptr + offset_d + offsets,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        grad_u_z = grad_u_z_ptr + offset_d + offsets
        tl.store(
            grad_u_z,
            grad_z.to(grad_u_z.dtype.element_ty),
            mask=mask_d,
        )
        previous_offset_d = base_d + tl.maximum(frame_index - 1, 0) * D
        previous_state_z = tl.load(
            state_z_ptr + previous_offset_d + offsets,
            mask=mask_d & (frame_index > 0),
            other=0.0,
        ).to(tl.float32)
        grad_w_z_value = grad_z[:, None] * previous_state_z[None, :]
        grad_w_z = (
            grad_w_z_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :]
        )
        tl.store(
            grad_w_z,
            grad_w_z_value.to(grad_w_z.dtype.element_ty),
            mask=mask_dd,
        )
        w_z = tl.load(
            w_z_ptr
            + offset_dd
            + offsets[:, None] * D
            + offsets[None, :],
            mask=mask_dd,
            other=0.0,
        ).to(tl.float32)
        grad_z = tl.sum(w_z * grad_z[:, None], axis=0)


def _block_size(head_dim: int) -> int:
    block = max(16, triton.next_power_of_2(head_dim))
    if block > 128:
        raise ValueError(
            f"Cross-GDN head dimension {head_dim} exceeds Triton limit 128"
        )
    return block


def _num_warps(block: int) -> int:
    if block <= 16:
        return 1
    if block <= 32:
        return 2
    return 4


def _scan_forward(
    w_kv: torch.Tensor,
    u_kv: torch.Tensor,
    w_z: torch.Tensor,
    u_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_heads, frames, head_dim, _ = w_kv.shape
    block = _block_size(head_dim)
    state_kv = torch.empty_like(w_kv)
    state_z = torch.empty_like(u_z)
    _scan_forward_kernel[(batch_heads,)](
        w_kv,
        u_kv,
        w_z,
        u_z,
        state_kv,
        state_z,
        T=frames,
        D=head_dim,
        BD=block,
        num_warps=_num_warps(block),
        num_stages=1,
    )
    return state_kv, state_z


def frame_gdn_final_scan(
    w_kv: torch.Tensor,
    u_kv: torch.Tensor,
    w_z: torch.Tensor,
    u_z: torch.Tensor,
    *,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only the terminal affine-scan state for inference prefill."""

    if not all(tensor.is_cuda for tensor in (w_kv, u_kv, w_z, u_z)):
        raise ValueError("Triton final-state scan requires CUDA tensors")
    batch_heads, frames, head_dim, _ = w_kv.shape
    block = _block_size(head_dim)
    state_kv = torch.empty(
        batch_heads,
        head_dim,
        head_dim,
        device=w_kv.device,
        dtype=state_dtype,
    )
    state_z = torch.empty(
        batch_heads,
        head_dim,
        device=w_kv.device,
        dtype=state_dtype,
    )
    _scan_final_forward_kernel[(batch_heads,)](
        w_kv,
        u_kv,
        w_z,
        u_z,
        state_kv,
        state_z,
        T=frames,
        D=head_dim,
        BD=block,
        num_warps=_num_warps(block),
        num_stages=1,
    )
    return state_kv, state_z


def frame_gdn_affine_tree_final(
    w_kv: torch.Tensor,
    u_kv: torch.Tensor,
    w_z: torch.Tensor,
    u_z: torch.Tensor,
    *,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associatively reduce affine frame updates in ``ceil(log2(F))`` levels.

    This is the non-serial chunk/final-state alternative requested for long
    histories.  It trades O(F D^2) temporary storage and batched matrix work
    for parallelism across frames.  The production benchmark decides whether
    that trade is worthwhile versus the fused persistent serial-F kernel.
    """

    tensors = (w_kv, u_kv, w_z, u_z)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Affine-tree final-state reduction requires CUDA")
    if w_kv.shape != u_kv.shape or w_kv.shape != w_z.shape:
        raise ValueError("KV/Z transition matrices must have identical shapes")
    if w_kv.ndim != 4 or u_z.shape != w_kv.shape[:3]:
        raise ValueError("Expected matrices [BH,F,D,D] and z update [BH,F,D]")

    # Autocast would silently turn this nominally-FP32 correctness path into a
    # BF16 bridge.  P2 state compression is applied only to the final cache.
    with torch.amp.autocast("cuda", enabled=False):
        kv_w = w_kv.float()
        kv_u = u_kv.float()
        z_w = w_z.float()
        z_u = u_z.float()
        while kv_w.shape[1] > 1:
            pairs = kv_w.shape[1] // 2
            pair_tokens = pairs * 2
            left_kv_w = kv_w[:, :pair_tokens:2]
            right_kv_w = kv_w[:, 1:pair_tokens:2]
            left_kv_u = kv_u[:, :pair_tokens:2]
            right_kv_u = kv_u[:, 1:pair_tokens:2]
            next_kv_w = torch.matmul(left_kv_w, right_kv_w)
            next_kv_u = (
                torch.matmul(left_kv_u, right_kv_w) + right_kv_u
            )

            left_z_w = z_w[:, :pair_tokens:2]
            right_z_w = z_w[:, 1:pair_tokens:2]
            left_z_u = z_u[:, :pair_tokens:2]
            right_z_u = z_u[:, 1:pair_tokens:2]
            next_z_w = torch.matmul(right_z_w, left_z_w)
            next_z_u = (
                torch.matmul(
                    right_z_w,
                    left_z_u.unsqueeze(-1),
                ).squeeze(-1)
                + right_z_u
            )
            if kv_w.shape[1] % 2:
                next_kv_w = torch.cat((next_kv_w, kv_w[:, -1:]), dim=1)
                next_kv_u = torch.cat((next_kv_u, kv_u[:, -1:]), dim=1)
                next_z_w = torch.cat((next_z_w, z_w[:, -1:]), dim=1)
                next_z_u = torch.cat((next_z_u, z_u[:, -1:]), dim=1)
            kv_w, kv_u = next_kv_w, next_kv_u
            z_w, z_u = next_z_w, next_z_u
    return kv_u[:, 0].to(state_dtype), z_u[:, 0].to(state_dtype)


def fused_frame_gdn_final_state(
    key: torch.Tensor,
    key_rot: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse frame transition construction and terminal-state reduction.

    Args are contiguous tensors with shapes ``[BH,F,D,S]`` for K/Krot/V,
    ``[BH,F,S]`` for beta, and ``[BH,F]`` for decay.
    """

    if not all(
        tensor.is_cuda for tensor in (key, key_rot, value, beta, decay)
    ):
        raise ValueError("Fused frame final-state kernel requires CUDA tensors")
    if key.shape != key_rot.shape or key.shape != value.shape:
        raise ValueError("K, rotated K, and V must have identical shapes")
    if key.ndim != 4:
        raise ValueError(f"Expected K as [BH,F,D,S], got {tuple(key.shape)}")
    batch_heads, frames, head_dim, spatial = key.shape
    if beta.shape != (batch_heads, frames, spatial):
        raise ValueError("beta does not match [BH,F,S]")
    if decay.shape != (batch_heads, frames):
        raise ValueError("decay does not match [BH,F]")
    block_d = _block_size(head_dim)
    block_s = 32 if spatial >= 32 else triton.next_power_of_2(spatial)
    state_kv = torch.empty(
        batch_heads,
        head_dim,
        head_dim,
        device=key.device,
        dtype=state_dtype,
    )
    state_z = torch.empty(
        batch_heads,
        head_dim,
        device=key.device,
        dtype=state_dtype,
    )
    _fused_frame_final_state_kernel[(batch_heads,)](
        key,
        key_rot,
        value,
        beta,
        decay,
        state_kv,
        state_z,
        F=frames,
        D=head_dim,
        S=spatial,
        BD=block_d,
        BS=block_s,
        num_warps=8 if block_d >= 64 else 4,
        num_stages=1,
    )
    return state_kv, state_z


def fused_raw_key_frame_gdn_final_state(
    raw_key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    key_inv_rms: torch.Tensor,
    key_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    num_heads: int,
    key_scale: float,
    state_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse raw-K preparation and terminal recurrence without layout copies."""

    tensors = (
        raw_key,
        value,
        beta,
        decay,
        key_inv_rms,
        key_norm_weight,
        rope_cos,
        rope_sin,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Raw-key fused final-state kernel requires CUDA")
    if raw_key.shape != value.shape or raw_key.ndim != 4:
        raise ValueError("Raw K/V must match [B,F,S,C]")
    batch, frames, spatial, channels = raw_key.shape
    heads = int(num_heads)
    if channels % heads:
        raise ValueError("channels must be divisible by num_heads")
    head_dim = channels // heads
    if beta.shape != (batch, frames, spatial, heads):
        raise ValueError("beta must be [B,F,S,H]")
    if decay.shape != (batch, frames, heads):
        raise ValueError("decay must be [B,F,H]")
    if key_inv_rms.shape != (batch, frames, spatial):
        raise ValueError("key_inv_rms must be [B,F,S]")
    if key_norm_weight.numel() != channels:
        raise ValueError("key_norm_weight must have C elements")
    if rope_cos.shape != (spatial, head_dim) or rope_sin.shape != rope_cos.shape:
        raise ValueError("RoPE tables must be [S,D]")
    block_d = _block_size(head_dim)
    block_s = 32 if spatial >= 32 else triton.next_power_of_2(spatial)
    state_kv = torch.empty(
        batch * heads,
        head_dim,
        head_dim,
        device=raw_key.device,
        dtype=state_dtype,
    )
    state_z = torch.empty(
        batch * heads,
        head_dim,
        device=raw_key.device,
        dtype=state_dtype,
    )
    _fused_raw_key_final_state_kernel[(batch * heads,)](
        raw_key,
        value,
        beta,
        decay,
        key_inv_rms,
        key_norm_weight,
        rope_cos,
        rope_sin,
        state_kv,
        state_z,
        F=frames,
        H=heads,
        D=head_dim,
        S=spatial,
        C=channels,
        KEY_SCALE=float(key_scale),
        BD=block_d,
        BS=block_s,
        num_warps=8 if block_d >= 64 else 4,
        num_stages=1,
    )
    return state_kv, state_z


def grouped_gdn_state_read(
    state_kv: torch.Tensor,
    state_z: torch.Tensor,
    query: torch.Tensor,
    query_rot: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Read compact states without physically expanding them per query.

    Query tensors use ``[Bq,H,D,S]``.  Context state batch ``Bc`` may be
    smaller; queries are assumed observation-major with a constant group size
    ``Bq // Bc``.
    """

    if not all(
        tensor.is_cuda for tensor in (state_kv, state_z, query, query_rot)
    ):
        raise ValueError("Grouped GDN state read requires CUDA tensors")
    if query.shape != query_rot.shape or query.ndim != 4:
        raise ValueError("query and rotated query must match [Bq,H,D,S]")
    query_batch, heads, head_dim, spatial = query.shape
    context_batch = state_kv.shape[0]
    if context_batch <= 0 or query_batch % context_batch != 0:
        raise ValueError(
            f"query batch {query_batch} is not divisible by state batch "
            f"{context_batch}"
        )
    if state_kv.shape != (context_batch, heads, head_dim, head_dim):
        raise ValueError("state_kv shape does not match query heads/head_dim")
    if state_z.shape != (context_batch, heads, head_dim):
        raise ValueError("state_z shape does not match query heads/head_dim")
    group_size = query_batch // context_batch
    block_d = _block_size(head_dim)
    block_s = 32 if spatial >= 32 else triton.next_power_of_2(spatial)
    output = torch.empty_like(query)
    _grouped_state_read_kernel[(query_batch * heads,)](
        state_kv,
        state_z,
        query,
        query_rot,
        output,
        GROUP_SIZE=group_size,
        H=heads,
        D=head_dim,
        S=spatial,
        BD=block_d,
        BS=block_s,
        EPS=float(eps),
        num_warps=4 if block_d >= 64 else 2,
        num_stages=1,
    )
    return output


def grouped_raw_query_gdn_state_read(
    state_kv: torch.Tensor,
    state_z: torch.Tensor,
    raw_query: torch.Tensor,
    query_inv_rms: torch.Tensor,
    query_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    *,
    num_heads: int,
    eps: float,
) -> torch.Tensor:
    """Read compact states from GEMM-native Q without Q-side temporaries."""

    tensors = (
        state_kv,
        state_z,
        raw_query,
        query_inv_rms,
        query_norm_weight,
        rope_cos,
        rope_sin,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Grouped raw-query GDN read requires CUDA")
    if raw_query.ndim != 3:
        raise ValueError("raw_query must be [Bq,S,C]")
    query_batch, spatial, channels = raw_query.shape
    heads = int(num_heads)
    if channels % heads:
        raise ValueError("channels must be divisible by num_heads")
    head_dim = channels // heads
    context_batch = int(state_kv.shape[0])
    if context_batch <= 0 or query_batch % context_batch:
        raise ValueError("query batch must be divisible by state batch")
    if state_kv.shape != (context_batch, heads, head_dim, head_dim):
        raise ValueError("state_kv shape does not match raw query")
    if state_z.shape != (context_batch, heads, head_dim):
        raise ValueError("state_z shape does not match raw query")
    if query_inv_rms.shape != (query_batch, spatial):
        raise ValueError("query_inv_rms must be [Bq,S]")
    if query_norm_weight.numel() != channels:
        raise ValueError("query_norm_weight must have C elements")
    if rope_cos.shape != (spatial, head_dim) or rope_sin.shape != rope_cos.shape:
        raise ValueError("RoPE tables must be [S,D]")
    block_d = _block_size(head_dim)
    block_s = 32 if spatial >= 32 else triton.next_power_of_2(spatial)
    group_size = query_batch // context_batch
    output = torch.empty_like(raw_query)
    _grouped_raw_query_state_read_kernel[(query_batch * heads,)](
        state_kv,
        state_z,
        raw_query,
        query_inv_rms,
        query_norm_weight,
        rope_cos,
        rope_sin,
        output,
        GROUP_SIZE=group_size,
        H=heads,
        D=head_dim,
        S=spatial,
        C=channels,
        BD=block_d,
        BS=block_s,
        EPS=float(eps),
        num_warps=4 if block_d >= 64 else 2,
        num_stages=1,
    )
    return output


def _scan_backward(
    w_kv: torch.Tensor,
    state_kv: torch.Tensor,
    grad_state_kv: torch.Tensor,
    w_z: torch.Tensor,
    state_z: torch.Tensor,
    grad_state_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_heads, frames, head_dim, _ = w_kv.shape
    block = _block_size(head_dim)
    grad_w_kv = torch.empty_like(w_kv)
    grad_u_kv = torch.empty_like(w_kv)
    grad_w_z = torch.empty_like(w_z)
    grad_u_z = torch.empty_like(state_z)
    _scan_backward_kernel[(batch_heads,)](
        w_kv,
        state_kv,
        w_z,
        state_z,
        grad_state_kv,
        grad_state_z,
        grad_w_kv,
        grad_u_kv,
        grad_w_z,
        grad_u_z,
        T=frames,
        D=head_dim,
        BD=block,
        num_warps=_num_warps(block),
        num_stages=1,
    )
    return grad_w_kv, grad_u_kv, grad_w_z, grad_u_z


class _FrameGDNScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        w_kv: torch.Tensor,
        u_kv: torch.Tensor,
        w_z: torch.Tensor,
        u_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_kv, state_z = _scan_forward(
            w_kv.detach(),
            u_kv.detach(),
            w_z.detach(),
            u_z.detach(),
        )
        ctx.save_for_backward(w_kv, state_kv, w_z, state_z)
        return state_kv, state_z

    @staticmethod
    def backward(
        ctx,
        grad_state_kv: torch.Tensor,
        grad_state_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        w_kv, state_kv, w_z, state_z = ctx.saved_tensors
        return _scan_backward(
            w_kv,
            state_kv,
            grad_state_kv.contiguous(),
            w_z,
            state_z,
            grad_state_z.contiguous(),
        )


def frame_gdn_scan(
    w_kv: torch.Tensor,
    u_kv: torch.Tensor,
    w_z: torch.Tensor,
    u_z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every recurrent state using the official SANA-WM scan order."""

    if not all(tensor.is_cuda for tensor in (w_kv, u_kv, w_z, u_z)):
        raise ValueError("Triton Cross-GDN scan requires CUDA tensors")
    return _FrameGDNScan.apply(w_kv, u_kv, w_z, u_z)
