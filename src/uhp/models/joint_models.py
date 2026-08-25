import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==========================================
# 1. 上层 GPT (JointContinuousGPT)
# ==========================================

class GPTConfig:
    def __init__(self, n_embd=256, n_layer=2, n_head=4, block_size=20,
                 n_obs_ego=5, n_obs_others=16,
                 # 修改点：n_out_ego 从 12 变为 15 (3路点 * (4状态 + 1动作))
                 n_out_ego=15,
                 n_out_others=48):
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.n_obs_ego = n_obs_ego
        self.n_obs_others = n_obs_others
        self.n_out_ego = n_out_ego
        self.n_out_others = n_out_others
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class JointContinuousGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoders
        self.ego_encoder = nn.Linear(config.n_obs_ego, config.n_embd)
        self.others_encoder = nn.Linear(config.n_obs_others, config.n_embd)
        self.rtg_encoder = nn.Linear(1, config.n_embd)

        # Fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Heads
        self.head_ego = nn.Linear(config.n_embd, config.n_out_ego)
        # [新增] Others Head
        self.head_others = nn.Linear(config.n_embd, config.n_out_others)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, ego_hist, others_hist, rtgs):
        B, T, _ = ego_hist.shape

        emb_ego = self.ego_encoder(ego_hist)
        emb_others = self.others_encoder(others_hist)
        emb_rtg = self.rtg_encoder(rtgs)

        # Fusion: Ego + Others + RTG
        emb_joint = torch.cat([emb_ego, emb_others], dim=-1)
        token_embeddings = self.fusion_mlp(emb_joint) + emb_rtg

        x = self.drop(token_embeddings + self.pos_emb[:, :T, :])

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Prediction (Use last token feature)
        feat = x[:, -1]

        pred_ego = self.head_ego(feat)  # [B, 12]
        pred_others = self.head_others(feat)  # [B, 48]

        return pred_ego, pred_others


# ==========================================
# 2. 下层 Diffusion (DiffusionTransformer)
# ==========================================
# (Diffusion 部分无需修改，沿用原代码或根据 obs_dim=21 微调)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = self.adaLN_modulation(c).chunk(6, dim=1)

        x_res = x
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_res + gate_msa.unsqueeze(1) * attn_output

        x_res = x
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_ffn.unsqueeze(1)) + shift_ffn.unsqueeze(1)
        ffn_output = self.ffn(x_norm)
        x = x_res + gate_ffn.unsqueeze(1) * ffn_output
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, obs_dim, horizon, condition_dim, hidden_size=256, depth=4, num_heads=4):
        super().__init__()
        self.horizon = horizon
        self.obs_dim = obs_dim

        self.input_proj = nn.Linear(obs_dim, hidden_size)
        self.pos_emb_seq = nn.Parameter(torch.zeros(1, horizon, hidden_size))
        nn.init.normal_(self.pos_emb_seq, std=0.02)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU()
        )
        self.cond_emb = nn.Sequential(
            nn.Linear(condition_dim, hidden_size),
            nn.GELU()
        )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_size, obs_dim)

        torch.nn.init.zeros_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, t, c):
        x = self.input_proj(x)
        x = x + self.pos_emb_seq[:, :x.shape[1], :]

        t_emb = self.time_emb(t)
        c_emb = self.cond_emb(c)
        cond_fused = t_emb + c_emb

        for block in self.blocks:
            x = block(x, cond_fused)

        x = self.final_norm(x)
        x = self.output_proj(x)
        return x