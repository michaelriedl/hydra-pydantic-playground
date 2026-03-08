"""Example training script demonstrating Hydra + Pydantic config validation.

Loads the MLP model configuration via Hydra, validates it through the Pydantic
``MLPConfig`` class, builds the model, and runs a dummy forward pass.

Usage:
    Run with default configuration::

        python scripts/train.py

    Override hyperparameters from the command line::

        python scripts/train.py model.dropout_rate=0.2 model.activation=gelu

    Use a deeper network with batch normalization::

        python scripts/train.py model.hidden_dims='[512, 256, 128]' model.use_batch_norm=true

    Trigger a Pydantic validation error (dropout_rate must be in [0, 1))::

        python scripts/train.py model.dropout_rate=1.5
"""

from __future__ import annotations

import logging

import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from src.models.mlp import MLP, MLPConfig

log = logging.getLogger(__name__)


def _count_parameters(model: MLP) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: The MLP model to inspect.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _run_forward_pass(model: MLP, config: MLPConfig) -> None:
    """Run a dummy forward pass and log the input/output shapes.

    Args:
        model: The MLP model to run.
        config: The model configuration (used for input shape).
    """
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.in_features)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    log.info("Forward pass successful")
    log.info("  Input shape:  %s", tuple(dummy_input.shape))
    log.info("  Output shape: %s", tuple(output.shape))


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_mnist_mlp")
def train(cfg: DictConfig) -> None:
    """Entry point for the training script.

    Demonstrates the full Hydra -> Pydantic -> PyTorch pipeline:

    1. Hydra loads and composes the YAML configuration.
    2. ``hydra.utils.instantiate`` constructs the ``MLPConfig`` Pydantic model,
       triggering full validation (field constraints, custom validators, extra
       field rejection).
    3. The validated config is used to build the ``MLP`` PyTorch module.
    4. A dummy forward pass verifies the model runs correctly.

    Args:
        cfg: The composed Hydra configuration.
    """
    log.info("Resolved Hydra configuration:\n%s", OmegaConf.to_yaml(cfg))

    # --- Step 1: Instantiate and validate via Pydantic ---
    # Hydra calls MLPConfig(**kwargs) under the hood. If any field fails
    # Pydantic validation, Hydra surfaces the error with the full config path.
    log.info("Instantiating MLPConfig (Pydantic validation runs here)...")
    model_config: MLPConfig = instantiate(cfg.model, _convert_="all")

    log.info("Validated MLPConfig:")
    log.info("  in_features:    %d", model_config.in_features)
    log.info("  hidden_dims:    %s", model_config.hidden_dims)
    log.info("  out_features:   %d", model_config.out_features)
    log.info("  activation:     %s", model_config.activation.value)
    log.info("  dropout_rate:   %.3f", model_config.dropout_rate)
    log.info("  use_batch_norm: %s", model_config.use_batch_norm)
    log.info("  bias:           %s", model_config.bias)

    # --- Step 2: Build the PyTorch model ---
    log.info("Building MLP model...")
    model = MLP(model_config)
    log.info("Model architecture:\n%s", model)

    num_params = _count_parameters(model)
    log.info("Trainable parameters: %s", f"{num_params:,}")

    # --- Step 3: Run a dummy forward pass ---
    log.info("Running dummy forward pass...")
    _run_forward_pass(model, model_config)

    log.info("Done.")


if __name__ == "__main__":
    train()
