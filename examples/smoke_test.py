"""Shape-only smoke test for the public architecture release."""

import torch

from uhp.models import DiffusionTransformer, GPTConfig, JointContinuousGPT
from uhp import MultiMetricEnergy, UncertaintyEstimator


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    intent = JointContinuousGPT(GPTConfig()).to(device).eval()
    diffusion = DiffusionTransformer(21, 30, 69).to(device).eval()

    ego = torch.zeros(2, 20, 5, device=device)
    others = torch.zeros(2, 20, 16, device=device)
    rtg = torch.zeros(2, 20, 1, device=device)
    with torch.no_grad():
        ego_anchor, others_anchor = intent(ego, others, rtg)
        condition = torch.zeros(2, 69, device=device)
        trajectory = diffusion(torch.zeros(2, 30, 21, device=device),
                               torch.zeros(2, dtype=torch.long, device=device),
                               condition)

    print("intent anchors:", tuple(ego_anchor.shape), tuple(others_anchor.shape))
    print("diffusion output:", tuple(trajectory.shape))
    print("uncertainty helper:", UncertaintyEstimator(device).__class__.__name__)
    print("energy helper:", MultiMetricEnergy.__name__)


if __name__ == "__main__":
    main()
