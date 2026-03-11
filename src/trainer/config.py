from dataclasses import dataclass


@dataclass
class TrainerConfig:
    """Shared configuration for hardware, loop control, checkpointing, and logging.

    This config is inherited by both ``PretrainConfig`` and ``FinetuneConfig`` and
    covers everything that is common to both training regimes.
    """

    # Hardware / Fabric
    accelerator: str = "auto"
    devices: int | str = "auto"
    strategy: str = "auto"
    precision: str = "32-true"
    num_workers: int = 4

    # Training loop
    max_epochs: int | None = 100
    batch_size: int = 64
    val_batch_size: int = 64
    train_ratio: float = 0.8
    grad_accum_steps: int = 1
    gradient_clip_val: float | None = None

    # Validation
    val_frequency: int = 1  # run a validation epoch every N training epochs

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 1  # save a checkpoint every N epochs

    # Logging
    log_every_n_steps: int = 10  # log training metrics every N optimiser steps

    # Reproducibility
    seed: int | None = None
