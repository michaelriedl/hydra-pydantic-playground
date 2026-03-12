# Agents

## Project overview

A playground for integrating Hydra configuration management with Pydantic validation in a PyTorch / Lightning Fabric training setup.

- Python >= 3.12
- Package manager: uv
- Dependencies: hydra-core, pydantic, lightning, torch, torchvision
- Linter: ruff (configured in pyproject.toml)

## Project layout

```
configs/                 Hydra YAML configuration files
  model/                 Per-model config overrides
scripts/                 Entrypoint scripts (Hydra @main)
src/
  models/                Model definitions (nn.Module) and their Pydantic configs
  trainer/               BaseTrainer (Lightning Fabric) and TrainerConfig
```

## Code style

- Docstrings: Google style. Use plain text throughout; do not use RST or Sphinx markup such as double backticks, :class:, :meth:, :param:, or **bold**.
- Config classes: Use Pydantic BaseModel with frozen=True and extra="forbid". Use Field() with constraints (gt, ge, lt, le, min_length, etc.) and description strings. Add model_validator for cross-field checks where needed.
- Imports: Sorted by ruff (isort profile). Use from __future__ import annotations.
- Type hints: Use modern union syntax (X | Y), not Optional or Union.
- Comments: Keep section comments short and plain (e.g. # Training loop). Do not pad with dashes, equals signs, or other decorative characters. Do not use em-dashes; use plain double hyphens (--) when a dash is needed.

## Running

- Use the venv at .venv/bin/python (not system python).
- Entry script: python scripts/train.py (Hydra CLI).
