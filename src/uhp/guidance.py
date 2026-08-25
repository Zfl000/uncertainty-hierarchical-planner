import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .safety import smooth_dynamic_elliptical_penalty


class MultiMetricEnergy(nn.Module):
    def __init__(self, device, stats_mean, stats_std):
        super().__init__()
        self.device = device
        self.register_buffer('obs_mean', stats_mean[:4])
        self.register_buffer('obs_std', stats_std[:4])

        # 论文 3.4.1 参数
        self.safe_dist = 6.0  # L_safe
        self.safe_lat = 1.8  # W_safe
        self.time_headway = 1.0
        self.comfortable_deceleration = 3.0
        self.barrier_sharpness = 8.0
        self.uncertainty_scale = 1.0
        self.target_speed = 22.0

        # 权重系数
        # Exposed for validation-only sensitivity audits.  The locked
        # configuration is (100, 15, 1); these are not changed online.
        self.w_s = float(os.getenv('TT_W_S', '100.0'))  # Safety
        self.w_p = float(os.getenv('TT_W_P', '15.0'))  # Progress
        self.w_c = float(os.getenv('TT_W_C', '1.0'))  # Comfort

    def denormalize(self, state_norm):
        return state_norm * self.obs_std + self.obs_mean

    def forward(self, x_pred, current_obs, gpt_anchors=None, use_ensemble_uncertainty=True):
        """
        x_pred: [B, 30, 21]
        current_obs: [4] (v, a, s, y)
        gpt_anchors: [B, 15] (GPT 预测的 3 个锚点，包含 v, a, s, y, jerk)
        """
        B, T, _ = x_pred.shape
        ego_phys = self.denormalize(x_pred[..., :4])

        # 1. J_safety (椭圆势场)
        others_phys = self.denormalize(x_pred[..., 5:].view(B, T, 4, 4))
        # Candidate-wise dispersion is a proxy for predicted surrounding-agent
        # uncertainty.  It is detached so guidance does not reduce uncertainty
        # by artificially collapsing the other-agent predictions.
        if use_ensemble_uncertainty and B > 1:
            other_sigma_s = others_phys[..., 2].std(dim=0, unbiased=False, keepdim=True).detach()
            other_sigma_y = others_phys[..., 3].std(dim=0, unbiased=False, keepdim=True).detach()
        else:
            other_sigma_s = other_sigma_y = None
        safety_penalty, _, _, _ = smooth_dynamic_elliptical_penalty(
            ego_phys, others_phys,
            base_longitudinal=self.safe_dist,
            base_lateral=self.safe_lat,
            time_headway=self.time_headway,
            comfortable_deceleration=self.comfortable_deceleration,
            barrier_sharpness=self.barrier_sharpness,
            uncertainty_scale=self.uncertainty_scale,
            longitudinal_uncertainty=other_sigma_s,
            lateral_uncertainty=other_sigma_y,
        )
        j_safety = safety_penalty.sum(dim=2).mean(dim=1)

        # 2. J_prog (锚点进度对齐)
        j_prog_pos = torch.tensor(0.0, device=self.device)
        j_prog_act = torch.tensor(0.0, device=self.device)

        if gpt_anchors is not None:
            # 广播处理: 如果 gpt_anchors 是 [1, 15] 但 x_pred 是 [16, ...], 需要 repeat
            if gpt_anchors.size(0) != B:
                gpt_anchors = gpt_anchors.repeat(B, 1)

            # 【修正点】：这里必须是 view(B, 3, 5) 因为包含 jerk
            gpt_intent = gpt_anchors.view(B, 3, 5)

            w_target_pos = gpt_intent[..., 2:4]  # 状态中的 S, Y
            w_target_jerk = gpt_intent[..., 4]  # 动作 Jerk

            # 底层预测值 (在第 10, 20, 30 步)
            pred_target_pos = x_pred[:, [9, 19, 29], 2:4]  # S, Y
            pred_target_jerk = x_pred[:, [9, 19, 29], 4]  # Jerk

            # 计算误差
            j_prog_pos = torch.norm(pred_target_pos - w_target_pos, dim=-1).mean(dim=1)
            j_prog_act = torch.abs(pred_target_jerk - w_target_jerk).mean(dim=1)

        j_prog_v = torch.abs(ego_phys[..., 0] - self.target_speed).mean(dim=1)

        # 综合进度能耗
        j_prog = 1.0 * j_prog_pos + 0.5 * j_prog_act + 0.1 * j_prog_v

        # 3. J_comf (舒适度 - Jerk)
        jerk = x_pred[..., 4]
        snap = jerk[:, 1:] - jerk[:, :-1]
        j_comf = (jerk ** 2).mean(dim=1) + (snap ** 2).mean(dim=1)

        return self.w_s * j_safety + self.w_p * j_prog + self.w_c * j_comf

    def get_guidance_grad(self, x_pred, current_obs, gpt_anchors, scale=1.0, grad_clip=0.2):
        with torch.enable_grad():
            x_in = x_pred.detach().requires_grad_(True)
            energy = self.forward(x_in, current_obs, gpt_anchors)
            grad = torch.autograd.grad(energy.sum(), x_in)[0]
            grad = torch.clamp(grad, -grad_clip, grad_clip)
        return grad * scale
