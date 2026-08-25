import torch


class UncertaintyEstimator:
    def __init__(self, device):
        self.device = device

    def estimate(self, trajectories, energy_costs, e_safe=30.0):
        """
        trajectories: [N_samples, Horizon, 2]
        """
        N, H, _ = trajectories.shape

        # --- 1. 计算 u_stat (轨迹点的平均标准差 - 始终为正) ---
        # 计算每个时间步的样本方差
        # var_t 形状为 [H, 2]
        var_t = torch.var(trajectories, dim=0)
        # 计算每个步点的标准差之和
        std_t = torch.sqrt(var_t.sum(dim=-1)) # [H]
        # 取全时域平均，代表采样轨迹平均偏移了几米
        u_stat = torch.mean(std_t).item()

        # --- 2. 计算 u_ecv (约束违规度 - 始终为正) ---
        violation = torch.relu(energy_costs - e_safe)
        u_ecv = torch.mean(violation).item()

        return u_stat, u_ecv
