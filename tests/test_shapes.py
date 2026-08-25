import torch

from uhp.models import DiffusionTransformer, GPTConfig, JointContinuousGPT


def test_architecture_shapes():
    intent = JointContinuousGPT(GPTConfig()).eval()
    diffusion = DiffusionTransformer(21, 30, 69).eval()
    with torch.no_grad():
        ego, others = intent(torch.zeros(1, 20, 5), torch.zeros(1, 20, 16),
                              torch.zeros(1, 20, 1))
        out = diffusion(torch.zeros(1, 30, 21), torch.zeros(1, dtype=torch.long),
                        torch.zeros(1, 69))
    assert ego.shape == (1, 15)
    assert others.shape == (1, 48)
    assert out.shape == (1, 30, 21)
