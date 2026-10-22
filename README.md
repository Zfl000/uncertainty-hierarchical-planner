# Uncertainty-Aware Hierarchical Planner

This repository contains the public model architecture for a hierarchical
motion-planning system. A causal Transformer generates sparse interaction
intent anchors, and a conditional diffusion Transformer refines them into a
dense state-action trajectory. An uncertainty-aware energy module provides a
smooth dynamic safety potential and differentiable inference-time guidance.

The dataset and any additional release materials will be published after review and approval.

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

The manuscript reports SUMO closed-loop evaluations and nuPlan experiments involving dataset-specific retraining followed by closed-loop simulation. For nuPlan, recorded driving data are used for offline training. During evaluation, the generated ego trajectory is executed in simulation, and the updated ego state is fed back into subsequent planning cycles. These experiments evaluate adaptation after retraining, rather than zero-shot transfer of the SUMO-trained model.

This public repository provides the model architecture. The datasets and experiment-specific artifacts required to reproduce the reported results are not included, so this repository alone does not provide a complete reproduction pipeline. The experimental protocols and configurations are described in the revised manuscript.

For the nuPlan dataset, simulation infrastructure, and metric definitions, please refer to the [official nuPlan devkit](https://github.com/motional/nuplan-devkit) and the nuPlan benchmark paper cited in the manuscript. These resources provide the underlying benchmark infrastructure; reproducing our reported results also requires the study-specific model adaptation and evaluation settings.

Release of the custom SUMO dataset is subject to approval. The dataset will be made publicly available through this repository once the required approval has been obtained.

No Apollo D-KIT adapter, physical-vehicle logs, or actuator-calibration artifacts are included. The deployment results reported in the manuscript concern software integration and computational profiling; they do not constitute physical closed-loop road validation. Vehicle deployment requires independent integration and validation.

## License

Released under the MIT License. See [LICENSE](LICENSE).
