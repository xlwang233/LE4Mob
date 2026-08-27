"""Downstream models for commuter flow generation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_inputs(origin: Tensor, destinations: Tensor) -> tuple[int, int]:
    if origin.ndim != 2:
        raise ValueError("origin must have shape (batch_size, embedding_dim).")
    if destinations.ndim != 3:
        raise ValueError(
            "destinations must have shape (batch_size, num_destinations, embedding_dim)."
        )
    if origin.shape[0] != destinations.shape[0]:
        raise ValueError("origin and destinations must have the same batch size.")
    if origin.shape[1] != destinations.shape[2]:
        raise ValueError("Origin and destination embedding dimensions must match.")
    return destinations.shape[1], destinations.shape[2]


class DeepGravity(nn.Module):
    """DeepGravity-style MLP over origin, destination, and optional distance features.

    ``num_hidden_layers`` retains the convention used in the paper experiments: it
    counts the repeated hidden blocks after the first input-to-hidden block.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        num_hidden_layers: int = 6,
        dropout_rate: float = 0.5,
        add_distance: bool = True,
        distance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers cannot be negative.")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1).")
        if distance_scale <= 0:
            raise ValueError("distance_scale must be positive.")

        self.input_dim = input_dim
        self.add_distance = add_distance
        self.distance_scale = float(distance_scale)
        combined_dim = 2 * input_dim + int(add_distance)

        layers: list[nn.Module] = [
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_hidden_layers):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        origin: Tensor,
        destinations: Tensor,
        *,
        distances: Tensor | None = None,
    ) -> Tensor:
        num_destinations, embedding_dim = _validate_inputs(origin, destinations)
        if embedding_dim != self.input_dim:
            raise ValueError(
                f"Expected embeddings with dimension {self.input_dim}, got {embedding_dim}."
            )

        expanded_origin = origin.unsqueeze(1).expand(-1, num_destinations, -1)
        features = [expanded_origin, destinations]
        if self.add_distance:
            if distances is None:
                raise ValueError("distances are required when add_distance=True.")
            if distances.shape != destinations.shape[:2]:
                raise ValueError("distances must have shape (batch_size, num_destinations).")
            features.append((distances / self.distance_scale).unsqueeze(-1))
        return self.mlp(torch.cat(features, dim=-1)).squeeze(-1)

    def predict_proba(
        self,
        origin: Tensor,
        destinations: Tensor,
        *,
        distances: Tensor | None = None,
    ) -> Tensor:
        return torch.softmax(self(origin, destinations, distances=distances), dim=-1)


class Bilinear(nn.Module):
    """Learned bilinear origin-destination interaction followed by an MLP."""

    def __init__(
        self,
        input_dim: int,
        bilinear_output_dim: int,
        hidden_dim: int,
        *,
        num_hidden_layers: int = 6,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or bilinear_output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Model dimensions must be positive.")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers cannot be negative.")
        if not 0 <= dropout_rate < 1:
            raise ValueError("dropout_rate must be in [0, 1).")

        self.input_dim = input_dim
        self.bilinear = nn.Bilinear(input_dim, input_dim, bilinear_output_dim)
        layers: list[nn.Module] = [
            nn.Linear(bilinear_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ]
        for _ in range(num_hidden_layers):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        origin: Tensor,
        destinations: Tensor,
        *,
        distances: Tensor | None = None,
    ) -> Tensor:
        del distances
        num_destinations, embedding_dim = _validate_inputs(origin, destinations)
        if embedding_dim != self.input_dim:
            raise ValueError(
                f"Expected embeddings with dimension {self.input_dim}, got {embedding_dim}."
            )
        expanded_origin = origin.unsqueeze(1).expand(-1, num_destinations, -1)
        interactions = self.bilinear(expanded_origin, destinations)
        return self.mlp(interactions).squeeze(-1)

    def predict_proba(
        self,
        origin: Tensor,
        destinations: Tensor,
        *,
        distances: Tensor | None = None,
    ) -> Tensor:
        return torch.softmax(self(origin, destinations, distances=distances), dim=-1)
