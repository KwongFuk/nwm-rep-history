"""Small training-loop utilities with CPU-testable contracts."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def validate_training_data_schedule(
    config: Mapping[str, object],
    observed_train_sizes: Mapping[str, int],
    *,
    steps_per_epoch: int,
    epochs: int,
) -> dict[str, object]:
    """Fail closed when a registered dataset or epoch budget drifts.

    The optional ``expected_*`` keys leave historical configs unchanged while
    making comparison runs explicit about what one dataset epoch means.
    """

    observed = {
        str(name): int(size) for name, size in observed_train_sizes.items()
    }
    expected_sizes_raw = config.get("expected_train_dataset_sizes")
    if expected_sizes_raw is not None:
        if not isinstance(expected_sizes_raw, Mapping):
            raise ValueError("expected_train_dataset_sizes must be a mapping")
        expected_sizes = {
            str(name): int(size) for name, size in expected_sizes_raw.items()
        }
        if observed != expected_sizes:
            raise ValueError(
                "Training dataset cardinality mismatch: "
                f"observed={observed}, expected={expected_sizes}"
            )

    steps_per_epoch = int(steps_per_epoch)
    epochs = int(epochs)
    expected_steps = config.get("expected_steps_per_epoch")
    if expected_steps is not None and steps_per_epoch != int(expected_steps):
        raise ValueError(
            "Optimizer steps per epoch mismatch: "
            f"observed={steps_per_epoch}, expected={int(expected_steps)}"
        )

    total_steps = steps_per_epoch * epochs
    expected_total = config.get("expected_total_optimizer_steps")
    if expected_total is not None and total_steps != int(expected_total):
        raise ValueError(
            "Total optimizer-step budget mismatch: "
            f"observed={total_steps}, expected={int(expected_total)}"
        )

    return {
        "observed_train_dataset_sizes": observed,
        "balanced_sampling": bool(config.get("balanced_sampling", False)),
        "balanced_sampling_mode": str(
            config.get("balanced_sampling_mode", "downsample_only")
        ),
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "total_optimizer_steps": total_steps,
    }


def parse_epoch_milestones(
    value: Any,
    *,
    total_epochs: int | None = None,
) -> tuple[int, ...]:
    """Parse a canonical, finite set of one-indexed epoch milestones."""

    if value is None:
        return ()
    if isinstance(value, str):
        if not value.strip():
            return ()
        raw_values = value.split(",")
    else:
        raw_values = list(value)

    milestones: list[int] = []
    for raw_value in raw_values:
        token = str(raw_value).strip()
        if not token:
            raise ValueError("epoch milestones cannot contain empty values")
        try:
            epoch = int(token)
        except ValueError as error:
            raise ValueError(
                f"epoch milestone must be an integer, got {token!r}"
            ) from error
        if epoch <= 0:
            raise ValueError(f"epoch milestone must be positive, got {epoch}")
        milestones.append(epoch)

    if len(set(milestones)) != len(milestones):
        raise ValueError(f"epoch milestones must be unique, got {milestones}")
    milestones.sort()
    if (
        total_epochs is not None
        and milestones
        and milestones[-1] > int(total_epochs)
    ):
        raise ValueError(
            f"epoch milestone {milestones[-1]} exceeds total epochs {total_epochs}"
        )
    return tuple(milestones)


def resolve_time_dist_shift(
    *,
    latent_channels: int,
    latent_size: int,
    transport_config: Mapping[str, Any],
) -> float:
    """Resolve the RAE dimension shift with the transport's valid lower bound."""

    if bool(transport_config.get("time_dist_shift_disable", False)):
        return 1.0
    explicit = transport_config.get("time_dist_shift")
    if explicit is not None:
        value = float(explicit)
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(
                "Explicit time_dist_shift must be finite and at least 1.0, "
                f"got {value}"
            )
        return value

    channels = int(latent_channels)
    size = int(latent_size)
    base = float(transport_config.get("time_dist_shift_base", 4096))
    if channels <= 0 or size <= 0:
        raise ValueError(
            "latent_channels and latent_size must be positive for the "
            "dimension-derived time shift"
        )
    if not math.isfinite(base) or base <= 0:
        raise ValueError(f"time_dist_shift_base must be positive, got {base}")
    return max(1.0, math.sqrt(channels * size * size / base))


def sample_step_context_length(
    minimum: int,
    maximum: int,
    *,
    global_seed: int,
    epoch: int,
    batch_index: int,
) -> int:
    """Uniformly select one history length for a complete optimizer batch.

    The selection is a pure function of the exact-resume cursor, so all DDP
    ranks use the same shape and a resumed run reproduces the uninterrupted
    history-length schedule without adding another mutable RNG stream.
    """

    minimum = int(minimum)
    maximum = int(maximum)
    if minimum <= 0 or maximum < minimum:
        raise ValueError(f"invalid context range [{minimum}, {maximum}]")
    sequence = np.random.SeedSequence(
        [int(global_seed), int(epoch), int(batch_index), 0x47444E]
    )
    generator = np.random.default_rng(sequence)
    return int(generator.integers(minimum, maximum + 1))


def validate_transition_batch_contract(config, world_size: int) -> dict[str, int]:
    """Resolve observation and transition-target batch sizes explicitly."""

    world_size = int(world_size)
    observations_per_rank = int(config["batch_size"])
    if world_size <= 0 or observations_per_rank <= 0:
        raise ValueError("world_size and batch_size must be positive")
    expected_world_size = config.get("expected_world_size")
    if (
        expected_world_size is not None
        and world_size != int(expected_world_size)
    ):
        raise ValueError(
            f"world_size={world_size} does not match "
            f"expected_world_size={expected_world_size}"
        )

    goal_counts = {
        int(dataset.get("goals_per_obs", 1))
        for dataset in config.get("datasets", {}).values()
        if dataset.get("train")
    }
    if not goal_counts:
        goal_counts = {1}
    if len(goal_counts) != 1:
        raise ValueError(
            "All training datasets must use the same goals_per_obs for an "
            f"auditable transition batch, got {sorted(goal_counts)}"
        )
    goals_per_observation = next(iter(goal_counts))
    global_observations = observations_per_rank * world_size
    expected_observations = config.get("expected_global_observation_batch")
    if (
        expected_observations is not None
        and global_observations != int(expected_observations)
    ):
        raise ValueError(
            f"global observation batch={global_observations} does not match "
            f"configured value={expected_observations}"
        )
    global_transitions = global_observations * goals_per_observation
    expected_transitions = config.get("effective_transition_batch")
    if (
        expected_transitions is not None
        and global_transitions != int(expected_transitions)
    ):
        raise ValueError(
            f"effective transition batch={global_transitions} does not match "
            f"configured value={expected_transitions}"
        )
    return {
        "world_size": world_size,
        "observations_per_rank": observations_per_rank,
        "global_observations": global_observations,
        "goals_per_observation": goals_per_observation,
        "global_transition_targets": global_transitions,
    }


def build_microbatch_slices(
    optimizer_batch_size: int,
    train_microbatch_size: int,
) -> tuple[slice, ...]:
    """Partition one optimizer batch without changing its sample set."""

    optimizer_batch_size = int(optimizer_batch_size)
    train_microbatch_size = int(train_microbatch_size)
    if optimizer_batch_size <= 0:
        raise ValueError("optimizer_batch_size must be positive")
    if train_microbatch_size <= 0:
        raise ValueError("train_microbatch_size must be positive")
    if train_microbatch_size > optimizer_batch_size:
        raise ValueError(
            "train_microbatch_size cannot exceed optimizer_batch_size"
        )
    return tuple(
        slice(start, min(start + train_microbatch_size, optimizer_batch_size))
        for start in range(0, optimizer_batch_size, train_microbatch_size)
    )
