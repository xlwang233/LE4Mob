"""Language-location contrastive model used to pre-train LE4Mob."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from le4mob.encoder import LocationEncoder, LocationEncoderConfig
from le4mob.losses import LE4MobLoss, LossOutput


@dataclass(frozen=True)
class LE4MobModelConfig:
    location_encoder: LocationEncoderConfig
    text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_encoder_revision: str | None = None
    initial_temperature: float = 0.07
    freeze_text_encoder: bool = True
    normalize_projected_text: bool = False

    def __post_init__(self) -> None:
        if not self.text_encoder_name.strip():
            raise ValueError("text_encoder_name cannot be empty.")
        if self.text_encoder_revision is not None and not self.text_encoder_revision.strip():
            raise ValueError("text_encoder_revision cannot be empty when provided.")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["location_encoder"] = self.location_encoder.to_dict()
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LE4MobModelConfig:
        copied = dict(values)
        copied["location_encoder"] = LocationEncoderConfig.from_dict(copied["location_encoder"])
        return cls(**copied)


@dataclass(frozen=True)
class LE4MobOutput:
    logits_per_text: Tensor
    logits_per_location: Tensor
    location_embeddings: Tensor


class LE4MobModel(nn.Module):
    """Align frozen language features with an inductive location encoder."""

    def __init__(self, config: LE4MobModelConfig, text_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        if text_encoder is None:
            from transformers import AutoModel

            text_encoder = AutoModel.from_pretrained(
                config.text_encoder_name,
                revision=config.text_encoder_revision,
            )
        self.text_encoder = text_encoder
        hidden_size = getattr(getattr(text_encoder, "config", SimpleNamespace()), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("The text encoder must expose config.hidden_size.")

        if config.freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False

        self.text_projection = nn.Linear(hidden_size, config.location_encoder.output_dim)
        self.location_encoder = LocationEncoder(config.location_encoder)
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / config.initial_temperature), dtype=torch.float32)
        )

    @staticmethod
    def _mean_pool(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        expanded_mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).to(hidden_states.dtype)
        return (hidden_states * expanded_mask).sum(dim=1) / expanded_mask.sum(dim=1).clamp_min(1e-9)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_states = outputs.hidden_states[-1]
        pooled = functional.normalize(self._mean_pool(hidden_states, attention_mask), p=2, dim=-1)
        projected = self.text_projection(pooled)
        if self.config.normalize_projected_text:
            projected = functional.normalize(projected, p=2, dim=-1)
        return projected

    def encode_locations(self, coordinates: Tensor, normalize: bool = False) -> Tensor:
        # The research encoder computes in float64 for stable multi-scale
        # positional encoding, then interacts with the text branch in float32.
        return self.location_encoder(coordinates, normalize=normalize).float()

    def forward(self, coordinates: Tensor, input_ids: Tensor, attention_mask: Tensor) -> LE4MobOutput:
        text_embeddings = self.encode_text(input_ids, attention_mask)
        location_embeddings = self.encode_locations(coordinates, normalize=True)
        logits_per_text = self.logit_scale.exp() * text_embeddings @ location_embeddings.transpose(0, 1)
        return LE4MobOutput(
            logits_per_text=logits_per_text,
            logits_per_location=logits_per_text.transpose(0, 1),
            location_embeddings=location_embeddings,
        )


class LE4MobPretrainingModule(nn.Module):
    """Compose the contrastive model and training objective."""

    def __init__(self, model: LE4MobModel, objective: LE4MobLoss) -> None:
        super().__init__()
        self.model = model
        self.objective = objective

    def forward(
        self,
        coordinates: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> LossOutput:
        output = self.model(coordinates, input_ids, attention_mask)
        return self.objective(
            output.logits_per_text,
            output.logits_per_location,
            coordinates,
            output.location_embeddings,
            generator=generator,
        )
