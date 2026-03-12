"""Trainer configuration with Pydantic validation.

Provides TrainerConfig, the shared configuration for hardware, loop control,
checkpointing, and logging that is consumed by BaseTrainer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, BaseModel, model_validator

# Accepted values mirror what Lightning Fabric supports.
AcceleratorType = Literal["auto", "cpu", "gpu", "cuda", "mps", "tpu"]
StrategyType = Literal["auto", "ddp", "fsdp", "deepspeed", "dp"]
PrecisionType = Literal[
    "16-mixed",
    "bf16-mixed",
    "16-true",
    "bf16-true",
    "32-true",
    "64-true",
]


class TrainerConfig(BaseModel):
    """Shared configuration for hardware, loop control, checkpointing, and logging.

    This config is consumed by BaseTrainer and covers everything that is
    common to both pre-training and fine-tuning regimes. All fields are
    validated at construction time via Pydantic.

    Attributes:
        accelerator: Hardware accelerator backend.
        devices: Number of devices to use, or "auto" for automatic
            selection.
        strategy: Distributed training strategy.
        precision: Numerical precision for training.
        num_workers: Number of DataLoader worker processes.
        max_epochs: Maximum number of training epochs. None means
            unlimited (train until early stopping or manual halt).
        batch_size: Training batch size per device.
        val_batch_size: Validation batch size per device.
        train_ratio: Fraction of the dataset used for training.
            Must be in (0, 1).
        grad_accum_steps: Number of batches to accumulate gradients over
            before performing an optimiser step.
        gradient_clip_val: Maximum gradient norm for clipping.
            None disables clipping.
        val_frequency: Run a validation epoch every N training epochs.
        checkpoint_dir: Directory to save checkpoints to.
        checkpoint_frequency: Save a checkpoint every N epochs.
        log_every_n_steps: Log training metrics every N optimiser steps.
        seed: Random seed for reproducibility. None disables seeding.

    Examples:
        >>> config = TrainerConfig()
        >>> config.batch_size
        64

        >>> config = TrainerConfig(
        ...     accelerator="gpu",
        ...     devices=2,
        ...     precision="16-mixed",
        ...     max_epochs=50,
        ...     batch_size=128,
        ...     gradient_clip_val=1.0,
        ...     seed=42,
        ... )
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # Hardware / Fabric
    accelerator: AcceleratorType = Field(
        default="auto",
        description="Hardware accelerator backend.",
    )
    devices: int | Literal["auto"] = Field(
        default="auto",
        description='Number of devices, or "auto" for automatic selection.',
    )
    strategy: StrategyType = Field(
        default="auto",
        description="Distributed training strategy.",
    )
    precision: PrecisionType = Field(
        default="32-true",
        description="Numerical precision for training.",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        description="Number of DataLoader worker processes.",
    )

    # Training loop
    max_epochs: int | None = Field(
        default=100,
        gt=0,
        description="Maximum training epochs. None for unlimited.",
    )
    batch_size: int = Field(
        default=64,
        gt=0,
        description="Training batch size per device.",
    )
    val_batch_size: int = Field(
        default=64,
        gt=0,
        description="Validation batch size per device.",
    )
    train_ratio: float = Field(
        default=0.8,
        gt=0.0,
        lt=1.0,
        description="Fraction of the dataset used for training.",
    )
    grad_accum_steps: int = Field(
        default=1,
        ge=1,
        description="Number of batches to accumulate gradients over.",
    )
    gradient_clip_val: float | None = Field(
        default=None,
        gt=0.0,
        description="Max gradient norm for clipping. None disables clipping.",
    )

    # Validation
    val_frequency: int = Field(
        default=1,
        ge=1,
        description="Run validation every N training epochs.",
    )

    # Checkpointing
    checkpoint_dir: str = Field(
        default="./checkpoints",
        min_length=1,
        description="Directory to save checkpoints to.",
    )
    checkpoint_frequency: int = Field(
        default=1,
        ge=1,
        description="Save a checkpoint every N epochs.",
    )

    # Logging
    log_every_n_steps: int = Field(
        default=10,
        ge=1,
        description="Log training metrics every N optimiser steps.",
    )

    # Reproducibility
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility. None disables seeding.",
    )

    # Cross-field validators

    @model_validator(mode="after")
    def _validate_devices(self) -> TrainerConfig:
        """Ensure devices is either "auto" or a positive integer.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If devices is an integer <= 0.
        """
        if isinstance(self.devices, int) and self.devices <= 0:
            raise ValueError(f"devices must be a positive integer or 'auto', got {self.devices}")
        return self
