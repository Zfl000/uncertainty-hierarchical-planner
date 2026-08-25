"""Smooth, velocity-aware elliptical safety potential.

All inputs are physical states with the final dimension ordered as
``(speed, acceleration, longitudinal_position, lateral_position)``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_dynamic_elliptical_penalty(
    ego_state: torch.Tensor,
    others_state: torch.Tensor,
    *,
    base_longitudinal: float = 6.0,
    base_lateral: float = 1.8,
    time_headway: float = 1.0,
    comfortable_deceleration: float = 3.0,
    barrier_sharpness: float = 8.0,
    uncertainty_scale: float = 1.0,
    longitudinal_uncertainty: torch.Tensor | None = None,
    lateral_uncertainty: torch.Tensor | None = None,
):
    """Return the smooth penalty and its intermediate physical quantities.

    The directional closing speed treats a slower vehicle ahead and a faster
    vehicle behind symmetrically.  The longitudinal radius contains a time-gap
    term and a braking-distance term, so the margin increases continuously
    with interaction speed.  Optional standard deviations enlarge both axes.
    """
    if ego_state.shape[:-1] != others_state.shape[:-2]:
        raise ValueError(
            f"ego/others leading dimensions differ: {ego_state.shape} vs {others_state.shape}"
        )
    if ego_state.shape[-1] < 4 or others_state.shape[-1] < 4:
        raise ValueError("states must contain at least (v, a, s, y)")
    if comfortable_deceleration <= 0 or barrier_sharpness <= 0:
        raise ValueError("comfortable_deceleration and barrier_sharpness must be positive")

    ego_v = ego_state[..., 0].unsqueeze(-1)
    ego_s = ego_state[..., 2].unsqueeze(-1)
    ego_y = ego_state[..., 3].unsqueeze(-1)
    other_v = others_state[..., 0]
    delta_s = others_state[..., 2] - ego_s
    delta_y = others_state[..., 3] - ego_y

    closing_ahead = F.relu(ego_v - other_v)
    closing_behind = F.relu(other_v - ego_v)
    closing_speed = torch.where(delta_s >= 0, closing_ahead, closing_behind)

    longitudinal_radius = (
        base_longitudinal
        + time_headway * closing_speed
        + closing_speed.square() / (2.0 * comfortable_deceleration)
    )
    lateral_radius = torch.full_like(longitudinal_radius, base_lateral)
    if longitudinal_uncertainty is not None:
        longitudinal_radius = longitudinal_radius + uncertainty_scale * longitudinal_uncertainty
    if lateral_uncertainty is not None:
        lateral_radius = lateral_radius + uncertainty_scale * lateral_uncertainty

    distance = torch.sqrt(
        (delta_s / longitudinal_radius.clamp_min(1e-3)).square()
        + (delta_y / lateral_radius.clamp_min(1e-3)).square()
        + 1e-6
    )
    penalty = F.softplus(barrier_sharpness * (1.0 - distance)) / barrier_sharpness
    return penalty, distance, longitudinal_radius, closing_speed
