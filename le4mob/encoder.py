"""Geography-derived location encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


@dataclass(frozen=True)
class LocationEncoderConfig:
    """Configuration for the LE4Mob location encoder."""

    positional_encoding: str = "grid"
    network: str = "fcnet"
    min_radius: float = 10.0
    max_radius: float = 10_000.0
    frequency_num: int = 32
    hidden_dim: int = 256
    output_dim: int = 128
    dropout: float = 0.5
    residual_layers: int = 4
    compute_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.positional_encoding != "grid":
            raise ValueError("LE4Mob supports positional_encoding='grid'.")
        if self.network != "fcnet":
            raise ValueError("LE4Mob supports network='fcnet'.")
        if self.frequency_num < 2:
            raise ValueError("frequency_num must be at least 2.")
        if not 0 < self.min_radius < self.max_radius:
            raise ValueError("Expected 0 < min_radius < max_radius.")
        if self.hidden_dim <= 0 or self.output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.residual_layers < 0:
            raise ValueError("residual_layers cannot be negative.")
        if self.compute_dtype not in {"float32", "float64"}:
            raise ValueError("compute_dtype must be 'float32' or 'float64'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LocationEncoderConfig:
        aliases = {
            "pe_type": "positional_encoding",
            "nn_type": "network",
            "min_lambda": "min_radius",
            "max_lambda": "max_radius",
            "dim_hidden": "hidden_dim",
            "dim_output": "output_dim",
        }
        canonical = {aliases.get(key, key): value for key, value in values.items()}
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(canonical) - known)
        if unknown:
            raise ValueError(f"Unknown location-encoder settings: {unknown}.")
        return cls(**canonical)


class GridPositionalEncoding(nn.Module):
    """Multi-scale sine/cosine encoding for two-dimensional coordinates."""

    coordinate_dim = 2

    def __init__(self, min_radius: float, max_radius: float, frequency_num: int) -> None:
        super().__init__()
        if frequency_num < 2:
            raise ValueError("frequency_num must be at least 2.")
        if not 0 < min_radius < max_radius:
            raise ValueError("Expected 0 < min_radius < max_radius.")

        exponent = torch.arange(frequency_num, dtype=torch.float64) / (frequency_num - 1)
        timescales = min_radius * torch.exp(exponent * math.log(max_radius / min_radius))
        self.register_buffer("frequencies", timescales.reciprocal(), persistent=True)
        self.min_radius = float(min_radius)
        self.max_radius = float(max_radius)
        self.frequency_num = int(frequency_num)
        self.embedding_dim = 4 * self.frequency_num

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape[-1] != self.coordinate_dim:
            raise ValueError(
                f"Expected coordinates with final dimension 2, got {tuple(coordinates.shape)}."
            )
        frequencies = self.frequencies.to(device=coordinates.device, dtype=coordinates.dtype)
        angles = coordinates.unsqueeze(-1) * frequencies
        encoded = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        return encoded.flatten(start_dim=-3)


class ResidualLayer(nn.Module):
    def __init__(self, width: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layers(inputs)


class FCNet(nn.Module):
    """Residual projection network used by the location encoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.5,
        residual_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        layers.extend(ResidualLayer(hidden_dim, dropout) for _ in range(residual_layers))
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(self.features(inputs))


class LocationEncoder(nn.Module):
    """Encode arbitrary, including previously unseen, coordinates."""

    def __init__(self, config: LocationEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.positional_encoding = GridPositionalEncoding(
            min_radius=config.min_radius,
            max_radius=config.max_radius,
            frequency_num=config.frequency_num,
        )
        self.network = FCNet(
            input_dim=self.positional_encoding.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            dropout=config.dropout,
            residual_layers=config.residual_layers,
        )
        self.to(dtype={"float32": torch.float32, "float64": torch.float64}[config.compute_dtype])

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def forward(self, coordinates: Tensor, normalize: bool = False) -> Tensor:
        parameter = next(self.network.parameters())
        coordinates = coordinates.to(device=parameter.device, dtype=parameter.dtype)
        embeddings = self.network(self.positional_encoding(coordinates))
        return functional.normalize(embeddings, p=2, dim=-1) if normalize else embeddings
