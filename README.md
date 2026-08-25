# Uncertainty-Aware Hierarchical Planner

This repository contains the public model architecture for a hierarchical
motion-planning system. A causal Transformer generates sparse interaction
intent anchors, and a conditional diffusion Transformer refines them into a
dense state-action trajectory. An uncertainty-aware energy module provides a
smooth dynamic safety potential and differentiable inference-time guidance.

Only architecture and self-contained utility code are released here. Training
data, raw simulator traces, model checkpoints, experiment logs, result tables,
and private deployment files are intentionally excluded. The dataset and any
additional release materials will be published after review and approval.

## Install

```bash
python -m pip install -e .
```

The package requires Python 3.8+ and PyTorch 2.0+. Select the appropriate
PyTorch wheel for your CPU or CUDA installation.

## Architecture

```python
from uhp.models import DiffusionTransformer, GPTConfig, JointContinuousGPT
from uhp import MultiMetricEnergy, UncertaintyEstimator

intent_model = JointContinuousGPT(GPTConfig(
    n_obs_ego=5,
    n_obs_others=16,
    n_out_ego=15,
    n_out_others=48,
    block_size=20,
))
trajectory_model = DiffusionTransformer(
    obs_dim=21,
    horizon=30,
    condition_dim=69,
)
```

The public code does not download or expect a dataset, checkpoint, simulator,
or vehicle platform. An application should provide its own normalized tensors,
checkpoint-loading policy, and closed-loop environment.

## Tensor interfaces

- Intent input: ego history `[B, 20, 5]`, neighboring-agent history
  `[B, 20, 16]`, and RTG `[B, 20, 1]`.
- Intent output: ego anchors `[B, 15]` and neighboring-agent anchors
  `[B, 48]`.
- Diffusion input: noisy trajectory `[B, 30, 21]`, denoising timestep `[B]`,
  and condition `[B, 69]`.
- Diffusion output: predicted clean trajectory `[B, 30, 21]`.

States use the project convention `[speed, acceleration, longitudinal position,
lateral position]`; the fifth ego channel is jerk and the neighboring-agent
block contains four state channels per tracked slot.

## Safety utilities

`uhp.safety.smooth_dynamic_elliptical_penalty` implements a Softplus barrier
with velocity-dependent longitudinal margins and optional uncertainty
enlargement. `uhp.guidance.MultiMetricEnergy` combines safety, intent/progress,
and comfort terms and exposes a trajectory-space gradient for inference-time
correction. `uhp.uncertainty.UncertaintyEstimator` computes sample dispersion
and excess-energy diagnostics.

## Reproducibility and scope

The research manuscript reports separate SUMO closed-loop and nuPlan offline
experiments. Those artifacts are not part of this public architecture-only
repository. No Apollo D-KIT adapter, physical-vehicle log, or actuator
calibration is included; deployment requires independent validation.

## License

Released under the MIT License. See [LICENSE](LICENSE).
