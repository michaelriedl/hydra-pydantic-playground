"""Base trainer built on Lightning Fabric.

This module contains the :class:`BaseTrainer` abstract class that defines
the training loop skeleton. Subclasses implement what changes between
training regimes (model, data, per-step logic) while the base provides
device placement, the epoch / validation loop, gradient accumulation,
checkpointing, metrics logging, and progress display.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import lightning as L  # noqa: N812
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lightning.fabric.loggers import Logger
from torch.optim.lr_scheduler import LRScheduler

from .config import TrainerConfig


class BaseTrainer(ABC):
    """Abstract base trainer built on Lightning Fabric.

    Defines the training algorithm skeleton via the Template Method pattern.
    Subclasses fill in the model, data, optimizer, and per-batch logic while
    the base class handles everything else.

    The epoch loop executed by :meth:`fit` is::

        for epoch in range(max_epochs):
            _train_loop()      # calls training_step() per batch
            _val_loop()        # calls validation_step() per batch (if due)
            step scheduler     # epoch-level LR update
            save checkpoint    # if due

    Subclasses must implement:
        create_model(): Return an ``nn.Module`` (not yet moved to device).
        create_dataloaders(): Return ``(train_loader, val_loader)``.
        create_optimizer(model): Return ``(optimizer, scheduler | None)``.
        training_step(model, batch, batch_idx): Forward pass for one training batch.
            Must return a ``dict`` with at least a ``"loss"`` key.
            Do not call ``backward()`` here.
        validation_step(model, batch, batch_idx): Forward pass for one validation batch.
            Returns a ``dict`` of scalar tensors that are averaged over the epoch
            and logged with a ``"val_"`` prefix.

    Note:
        Call :meth:`_build_scheduler` inside ``create_optimizer`` to create an
        LR scheduler from a string name and kwargs, keeping scheduler creation
        consistent across subclasses.
    """

    def __init__(
        self,
        config: TrainerConfig,
        loggers: Logger | list[Logger] | None = None,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.config = config
        self.fabric = L.Fabric(
            accelerator=config.accelerator,
            devices=config.devices,
            strategy=config.strategy,
            precision=config.precision,
            loggers=loggers or [],
            callbacks=callbacks or [],
        )
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.should_stop: bool = False
        self._is_launched: bool = False
        # Set during fit(), cleared afterwards; available to callbacks via trainer.model etc.
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None

    # Abstract interface — subclasses must implement these

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Instantiate and return the model (not yet moved to device)."""

    @abstractmethod
    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Return ``(train_loader, val_loader)``."""

    @abstractmethod
    def create_optimizer(
        self,
        model: nn.Module,
    ) -> tuple[Optimizer, LRScheduler | None]:
        """Return ``(optimizer, scheduler)``; scheduler may be ``None``."""

    @abstractmethod
    def training_step(
        self,
        model: nn.Module,
        batch: Any,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a single training batch.

        Args:
            model: The Fabric-wrapped model in training mode.
            batch: The batch yielded by the train DataLoader.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            A ``dict`` containing at least a ``"loss"`` key.  Additional keys
            are logged as training metrics.  Do **not** call ``backward()``.
        """

    @abstractmethod
    def validation_step(
        self,
        model: nn.Module,
        batch: Any,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for a single validation batch.

        Args:
            model: The Fabric-wrapped model in eval mode (``torch.no_grad``
                context is already active).
            batch: The batch yielded by the val DataLoader.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            A ``dict`` of scalar tensors.  Values are averaged across the
            epoch and logged with a ``"val_"`` prefix.
        """

    # Main entry point

    def fit(self, ckpt_path: str | None = None) -> None:
        """Run the full training loop.

        Args:
            ckpt_path: Optional path to a Fabric checkpoint to resume from.
                When provided, model, optimizer, scheduler, ``current_epoch``,
                and ``global_step`` are all restored before training begins.
        """
        self._ensure_launched()

        if self.config.seed is not None:
            self.fabric.seed_everything(self.config.seed)

        # Instantiate objects — model is created on the target device/dtype
        with self.fabric.init_module():
            model = self.create_model()

        train_loader, val_loader = self.create_dataloaders()
        optimizer, scheduler = self.create_optimizer(model)

        # Fabric: wrap model + optimizer for device placement, distributed,
        # and mixed-precision, then wrap DataLoaders for distributed samplers.
        model, optimizer = self.fabric.setup(model, optimizer)
        train_loader = self.fabric.setup_dataloaders(train_loader)
        val_loader = self.fabric.setup_dataloaders(val_loader)

        # Expose training objects on self so callbacks can access them.
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Build resumable state dict (Fabric serializes model/optim state dicts)
        state: dict[str, Any] = {"model": model, "optimizer": optimizer}
        if scheduler is not None:
            state["scheduler"] = scheduler

        if ckpt_path is not None:
            self._load_checkpoint(state, ckpt_path)

        self.fabric.call("on_fit_start", trainer=self)

        # ---- Epoch loop ----
        while not self.should_stop:
            self._train_loop(model, optimizer, train_loader)

            val_metrics: dict[str, torch.Tensor] = {}
            if self._should_validate:
                val_metrics = self._val_loop(model, val_loader)

            self._step_scheduler(scheduler, val_metrics)

            self.current_epoch += 1

            if self.config.max_epochs is not None and self.current_epoch >= self.config.max_epochs:
                self.should_stop = True

            if self.current_epoch % self.config.checkpoint_frequency == 0 or self.should_stop:
                self._save_checkpoint(state)

        self.should_stop = False  # reset for subsequent fit() calls

        self.fabric.call("on_fit_end", trainer=self)

        # Clear training-object references so callbacks don't hold stale state.
        self.model = None
        self.optimizer = None
        self.scheduler = None

    # Train / val loops

    def _train_loop(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
    ) -> None:
        """Run one training epoch."""
        model.train()
        self.fabric.call("on_train_epoch_start", trainer=self)

        iterable = self._progbar(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {self.current_epoch} [train]",
        )

        for batch_idx, batch in enumerate(iterable):
            self.fabric.call("on_train_batch_start", batch=batch, batch_idx=batch_idx, trainer=self)

            # Defer gradient sync during accumulation steps (speeds up DDP).
            is_accumulating = (batch_idx + 1) % self.config.grad_accum_steps != 0

            with self.fabric.no_backward_sync(model, enabled=is_accumulating):
                outputs = self.training_step(model, batch, batch_idx)
                self.fabric.backward(outputs["loss"])

            if not is_accumulating:
                if self.config.gradient_clip_val is not None:
                    self.fabric.clip_gradients(
                        model,
                        optimizer,
                        max_norm=self.config.gradient_clip_val,
                    )
                self.fabric.call("on_before_optimizer_step", trainer=self, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.config.log_every_n_steps == 0:
                    log_dict = {f"train_{k}": v.item() for k, v in outputs.items()}
                    log_dict["epoch"] = float(self.current_epoch)
                    self.fabric.log_dict(log_dict, step=self.global_step)

            self.fabric.call(
                "on_train_batch_end",
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
                trainer=self,
            )
            self._progbar_postfix(iterable, outputs, prefix="train")

        self.fabric.call("on_train_epoch_end", trainer=self)

    def _val_loop(
        self,
        model: nn.Module,
        val_loader: DataLoader,
    ) -> dict[str, torch.Tensor]:
        """Run one validation epoch.

        Returns:
            Averaged epoch metrics with a ``"val_"`` prefix, e.g.
            ``{"val_loss": tensor(0.42), "val_acc": tensor(0.87)}``.
        """
        model.eval()
        self.fabric.call("on_validation_epoch_start", trainer=self)

        accumulated: dict[str, list[torch.Tensor]] = {}

        with torch.no_grad():
            iterable = self._progbar(
                val_loader,
                total=len(val_loader),
                desc=f"Epoch {self.current_epoch} [val]",
            )
            for batch_idx, batch in enumerate(iterable):
                self.fabric.call(
                    "on_validation_batch_start", batch=batch, batch_idx=batch_idx, trainer=self
                )
                outputs = self.validation_step(model, batch, batch_idx)
                self.fabric.call(
                    "on_validation_batch_end",
                    outputs=outputs,
                    batch=batch,
                    batch_idx=batch_idx,
                    trainer=self,
                )

                for k, v in outputs.items():
                    accumulated.setdefault(k, []).append(v.detach())

                self._progbar_postfix(iterable, outputs, prefix="val")

        epoch_metrics = {f"val_{k}": torch.stack(vs).mean() for k, vs in accumulated.items()}
        self.fabric.log_dict(
            {k: v.item() for k, v in epoch_metrics.items()},
            step=self.global_step,
        )
        self.fabric.print(
            f"Epoch {self.current_epoch} | "
            + " | ".join(f"{k}={v.item():.4f}" for k, v in epoch_metrics.items())
        )

        self.fabric.call("on_validation_epoch_end", trainer=self, val_metrics=epoch_metrics)
        model.train()
        return epoch_metrics

    # Checkpointing

    def _save_checkpoint(self, state: dict[str, Any]) -> None:
        """Save a Fabric checkpoint, augmented with loop-state scalars."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        save_state = {
            **state,
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
        }
        path = os.path.join(
            self.config.checkpoint_dir,
            f"epoch-{self.current_epoch:04d}.ckpt",
        )
        self.fabric.save(path, save_state)
        self.fabric.print(f"Saved checkpoint → {path}")

    def _load_checkpoint(self, state: dict[str, Any], path: str) -> None:
        """Restore a checkpoint in-place, updating epoch / step counters."""
        remainder = self.fabric.load(path, state)
        self.global_step = remainder.pop("global_step", 0)
        self.current_epoch = remainder.pop("current_epoch", 0)
        if remainder:
            self.fabric.print(f"[WARNING] Unused checkpoint keys: {list(remainder.keys())}")
        self.fabric.print(
            f"Resumed from {path} (epoch={self.current_epoch}, step={self.global_step})"
        )

    # Helpers

    def _ensure_launched(self) -> None:
        """Call ``fabric.launch()`` at most once, even across fit/evaluate."""
        if not self._is_launched:
            self.fabric.launch()
            self._is_launched = True

    @property
    def _should_validate(self) -> bool:
        """``True`` when the current epoch should trigger a validation run."""
        return self.current_epoch % self.config.val_frequency == 0

    def _step_scheduler(
        self,
        scheduler: LRScheduler | None,
        val_metrics: dict[str, torch.Tensor],
    ) -> None:
        """Step the LR scheduler after each epoch.

        ``ReduceLROnPlateau`` requires a monitored scalar, which is taken from
        ``val_metrics["val_loss"]``.  If validation was skipped this epoch
        (because ``val_frequency > 1``), the plateau scheduler is not stepped.
        """
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor = val_metrics.get("val_loss")
            if monitor is not None:
                scheduler.step(monitor.item())
        else:
            scheduler.step()

    def _build_scheduler(
        self,
        optimizer: Optimizer,
        scheduler_type: str | None,
        scheduler_kwargs: dict,
    ) -> LRScheduler | None:
        """Build an LR scheduler by name.

        Convenience helper for subclasses to call inside ``create_optimizer``.

        Args:
            optimizer: The optimiser to attach the scheduler to.
            scheduler_type: One of ``"cosine"``, ``"step"``,
                ``"reduce_on_plateau"``, ``"linear"``, or ``None``.
            scheduler_kwargs: Keyword arguments forwarded to the scheduler
                constructor.  For ``"cosine"``, ``T_max`` defaults to
                ``config.max_epochs`` when not provided.

        Returns:
            A configured ``LRScheduler`` instance, or ``None``.
        """
        if scheduler_type is None:
            return None

        registry: dict[str, type] = {
            "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
            "step": torch.optim.lr_scheduler.StepLR,
            "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
            "linear": torch.optim.lr_scheduler.LinearLR,
        }

        if scheduler_type not in registry:
            raise ValueError(
                f"Unknown scheduler_type '{scheduler_type}'. Valid options: {list(registry.keys())}"
            )

        kwargs = dict(scheduler_kwargs)
        if scheduler_type == "cosine" and "T_max" not in kwargs:
            kwargs["T_max"] = self.config.max_epochs or 100

        return registry[scheduler_type](optimizer, **kwargs)

    def _progbar(
        self,
        iterable: Any,
        total: int,
        desc: str,
    ) -> Any:
        """Wrap an iterable with tqdm on rank 0; pass through on other ranks."""
        if self.fabric.is_global_zero:
            return tqdm(iterable, total=total, desc=desc, leave=False)
        return iterable

    @staticmethod
    def _progbar_postfix(
        prog_bar: Any,
        outputs: dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        """Update the tqdm postfix with the latest step metrics."""
        if isinstance(prog_bar, tqdm):
            prog_bar.set_postfix({f"{prefix}_{k}": f"{v.item():.4f}" for k, v in outputs.items()})
