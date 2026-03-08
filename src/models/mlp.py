"""Multi-Layer Perceptron (MLP) model with Pydantic configuration."""

from __future__ import annotations

import itertools
from enum import StrEnum

from torch import Tensor, nn
from pydantic import Field, BaseModel, model_validator


class ActivationType(StrEnum):
    """Supported activation function types for MLP layers.

    Attributes:
        RELU: Rectified Linear Unit activation.
        GELU: Gaussian Error Linear Unit activation.
        SILU: Sigmoid Linear Unit (Swish) activation.
        TANH: Hyperbolic tangent activation.
    """

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"


_ACTIVATION_REGISTRY: dict[ActivationType, type[nn.Module]] = {
    ActivationType.RELU: nn.ReLU,
    ActivationType.GELU: nn.GELU,
    ActivationType.SILU: nn.SiLU,
    ActivationType.TANH: nn.Tanh,
}


class MLPConfig(BaseModel):
    """Configuration for a Multi-Layer Perceptron (MLP) model.

    Defines the architecture and regularization hyperparameters for an MLP.
    All fields are validated at construction time via Pydantic.

    Attributes:
        in_features: Number of input features to the first layer.
        hidden_dims: Sequence of hidden layer dimensions. Must contain at least
            one element.
        out_features: Number of output features from the final layer.
        activation: Activation function applied after each hidden layer.
            Defaults to ``"relu"``.
        dropout_rate: Dropout probability applied after each hidden layer
            activation. Must be in ``[0, 1)``. Defaults to ``0.0`` (no dropout).
        use_batch_norm: Whether to apply batch normalization before the
            activation in each hidden layer. Defaults to ``False``.
        bias: Whether linear layers include a bias term. Defaults to ``True``.

    Examples:
        >>> config = MLPConfig(in_features=784, hidden_dims=[256, 128], out_features=10)
        >>> config.activation
        <ActivationType.RELU: 'relu'>

        >>> config = MLPConfig(
        ...     in_features=784,
        ...     hidden_dims=[512, 256, 128],
        ...     out_features=10,
        ...     activation="gelu",
        ...     dropout_rate=0.1,
        ...     use_batch_norm=True,
        ... )
    """

    model_config = {"frozen": True, "extra": "forbid"}

    in_features: int = Field(..., gt=0, description="Number of input features.")
    hidden_dims: list[int] = Field(
        ...,
        min_length=1,
        description="Sequence of hidden layer dimensions.",
    )
    out_features: int = Field(..., gt=0, description="Number of output features.")
    activation: ActivationType = Field(
        default=ActivationType.RELU,
        description="Activation function applied after each hidden layer.",
    )
    dropout_rate: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied after each hidden layer activation.",
    )
    use_batch_norm: bool = Field(
        default=False,
        description="Whether to apply batch normalization before the activation.",
    )
    bias: bool = Field(
        default=True,
        description="Whether linear layers include a bias term.",
    )

    @model_validator(mode="after")
    def _validate_hidden_dims_positive(self) -> MLPConfig:
        """Ensure every element in ``hidden_dims`` is a positive integer.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If any hidden dimension is not a positive integer.
        """
        for idx, dim in enumerate(self.hidden_dims):
            if dim <= 0:
                raise ValueError(f"hidden_dims[{idx}] must be a positive integer, got {dim}")
        return self


class MLP(nn.Module):
    """Multi-Layer Perceptron built from an :class:`MLPConfig`.

    Constructs a feedforward neural network with configurable hidden layers,
    activation functions, dropout, and batch normalization.

    The architecture for each hidden layer follows:
        ``Linear -> [BatchNorm] -> Activation -> [Dropout]``

    The final output layer is a plain ``Linear`` projection with no activation
    or regularization.

    Args:
        config: A validated :class:`MLPConfig` instance defining the
            architecture.

    Examples:
        >>> config = MLPConfig(in_features=784, hidden_dims=[256, 128], out_features=10)
        >>> model = MLP(config)
        >>> x = torch.randn(32, 784)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 10])
    """

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config

        activation_cls = _ACTIVATION_REGISTRY[config.activation]
        layer_dims = [config.in_features, *config.hidden_dims]

        layers: list[nn.Module] = []
        for fan_in, fan_out in itertools.pairwise(layer_dims):
            layers.append(nn.Linear(fan_in, fan_out, bias=config.bias))

            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(fan_out))

            layers.append(activation_cls())

            if config.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=config.dropout_rate))

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.out_features, bias=config.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass through the MLP.

        Args:
            x: Input tensor of shape ``(batch_size, in_features)``.

        Returns:
            Output tensor of shape ``(batch_size, out_features)``. Raw logits
            with no activation applied.
        """
        h = self.hidden_layers(x)
        return self.output_layer(h)

    def __repr__(self) -> str:
        """Return a human-readable summary of the MLP architecture.

        Returns:
            String representation including layer dimensions, activation,
            and regularization settings.
        """
        dims = [self.config.in_features, *self.config.hidden_dims, self.config.out_features]
        dim_str = " -> ".join(map(str, dims))
        return (
            f"{self.__class__.__name__}("
            f"dims=[{dim_str}], "
            f"activation={self.config.activation.value}, "
            f"dropout={self.config.dropout_rate}, "
            f"batch_norm={self.config.use_batch_norm}"
            f")"
        )
