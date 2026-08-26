"""Contrastive and distance-aware objectives for LE4Mob."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn



@dataclass(frozen=True)
class DistanceRegularizationConfig:
    """Settings for the distance-aware regularizer."""

    sample_ratio: float = 0.2
    weight: float = 1.0
    coordinate_system: str = "projected"
    earth_radius_m: float = 6_371_000.0

    def __post_init__(self) -> None:
        if not 0 <= self.sample_ratio <= 1:
            raise ValueError("sample_ratio must be in [0, 1].")
        if self.weight < 0:
            raise ValueError("weight cannot be negative.")
        if self.coordinate_system not in {"lon-lat", "projected"}:
            raise ValueError("coordinate_system must be 'lon-lat' or 'projected'.")
        if self.earth_radius_m <= 0:
            raise ValueError("earth_radius_m must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> DistanceRegularizationConfig:
        copied = dict(values)
        if copied.get("coordinate_system") == "lon_lat":
            copied["coordinate_system"] = "lon-lat"
        return cls(**copied)


def symmetric_contrastive_loss(logits_per_text: Tensor, logits_per_location: Tensor) -> Tensor:
    """CLIP-style symmetric cross entropy for aligned text/location pairs."""
    if logits_per_text.ndim != 2 or logits_per_location.ndim != 2:
        raise ValueError("Contrastive logits must be matrices.")
    if logits_per_location.shape != logits_per_text.transpose(0, 1).shape:
        raise ValueError("Text and location logits have incompatible shapes.")
    labels = torch.arange(logits_per_text.shape[0], device=logits_per_text.device)
    return (
        functional.cross_entropy(logits_per_text, labels)
        + functional.cross_entropy(logits_per_location, labels)
    ) / 2


def haversine_distance_m(first: Tensor, second: Tensor, earth_radius_m: float = 6_371_000.0) -> Tensor:
    """Great-circle distance for ``(longitude, latitude)`` coordinates in degrees."""
    if first.shape != second.shape or first.shape[-1] != 2:
        raise ValueError("Haversine inputs must have matching shapes ending in 2.")
    radians = torch.pi / 180.0
    lon1, lat1 = first.unbind(dim=-1)
    lon2, lat2 = second.unbind(dim=-1)
    lon1, lat1, lon2, lat2 = lon1 * radians, lat1 * radians, lon2 * radians, lat2 * radians
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    haversine = (
        torch.sin(delta_lat / 2).square()
        + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon / 2).square()
    ).clamp(0.0, 1.0)
    central_angle = 2 * torch.atan2(torch.sqrt(haversine), torch.sqrt((1 - haversine).clamp_min(0.0)))
    return earth_radius_m * central_angle


def paired_distance_preservation_loss(
    coordinates: Tensor,
    embeddings: Tensor,
    config: DistanceRegularizationConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Match embedding cosine similarity to batch-relative geographic proximity.

    This preserves the research implementation: sample batch elements without
    replacement, form disjoint random pairs, min-max normalize their physical
    distances, and regress cosine similarity onto ``1 - normalized_distance``.
    """
    if coordinates.ndim != 2 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates must have shape (batch, 2).")
    if embeddings.ndim != 2 or embeddings.shape[0] != coordinates.shape[0]:
        raise ValueError("embeddings must have shape (batch, embedding_dim).")

    sample_size = int(coordinates.shape[0] * config.sample_ratio)
    sample_size = (sample_size // 2) * 2
    if sample_size < 4 or config.weight == 0:
        return embeddings.sum() * 0.0

    indices = torch.randperm(coordinates.shape[0], device=coordinates.device, generator=generator)[
        :sample_size
    ]
    coordinate_pairs = coordinates[indices].reshape(-1, 2, 2)
    embedding_pairs = embeddings[indices].reshape(-1, 2, embeddings.shape[-1])

    if config.coordinate_system == "lon-lat":
        physical_distance = haversine_distance_m(
            coordinate_pairs[:, 0], coordinate_pairs[:, 1], config.earth_radius_m
        )
    else:
        physical_distance = torch.linalg.vector_norm(coordinate_pairs[:, 0] - coordinate_pairs[:, 1], dim=-1)

    first = functional.normalize(embedding_pairs[:, 0], p=2, dim=-1)
    second = functional.normalize(embedding_pairs[:, 1], p=2, dim=-1)
    similarity = ((first * second).sum(dim=-1) + 1.0) / 2.0

    distance_range = (physical_distance.max() - physical_distance.min()).clamp_min(1e-8)
    normalized_distance = (physical_distance - physical_distance.min()) / distance_range
    target_similarity = (1.0 - normalized_distance).to(dtype=similarity.dtype)
    return functional.mse_loss(similarity, target_similarity)


@dataclass(frozen=True)
class LossOutput:
    total: Tensor
    contrastive: Tensor
    distance: Tensor


class LE4MobLoss(nn.Module):
    def __init__(self, distance_config: DistanceRegularizationConfig) -> None:
        super().__init__()
        self.distance_config = distance_config

    def forward(
        self,
        logits_per_text: Tensor,
        logits_per_location: Tensor,
        coordinates: Tensor,
        location_embeddings: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> LossOutput:
        contrastive = symmetric_contrastive_loss(logits_per_text, logits_per_location)
        distance = paired_distance_preservation_loss(
            coordinates, location_embeddings, self.distance_config, generator=generator
        )
        total = contrastive + self.distance_config.weight * distance
        return LossOutput(total=total, contrastive=contrastive, distance=distance)
