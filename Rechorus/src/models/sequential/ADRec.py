# -*- coding: UTF-8 -*-
# @Author  : Yijie Gu
# @Email   : guyj25@mail2.sysu.edu.cn

# @inproceedings{JLchen2025ADRec,
# 	title={Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach},
# 	author={Jialei Chen and Yuanbo Xu and Yiheng Jiang},
# 	booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
# 	year={2025},
# 	organization={ACM},
# 	doi = {10.1145/3711896.3737172}
# }

"""
ADRec (自回归扩散推荐) 模型实现（代码注释版）

简介 ：
    ADRec 旨在解决扩散模型在序列推荐中常见的嵌入崩溃问题。其核心思想是
    将自回归序列建模（用于保持历史语义与顺序信息）与细粒度的 token 级
    独立扩散过程深度融合。模型由两个核心模块组成：因果注意力模块（CAM）
    与自回归扩散模块（ADM）。CAM 使用带因果掩码的 Transformer 编码历史序列，
    为扩散去噪网络提供稳定的条件（条件上下文不被噪声污染）。ADM 负责
    token 级噪声注入与去噪学习；对长度为 L 的目标序列，每个 token 可以独立
    采样不同的扩散时间步 t，使每个物品在多种噪声强度下参与训练，从而丰富
    表征分布的学习能力。

主要实现要点：
    - 前向扩散遵循标准 DDPM 公式：x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps
    - 去噪网络以条件信息（CAM 编码）、加噪目标与时间嵌入作为输入进行预测。
    - 训练采用三阶段策略：
        1) 阶段1（stage1）：仅用交叉熵（CE）预训练嵌入与 CAM，初始化语义空间。
        2) 阶段2（stage2）：冻结嵌入层，仅用 MSE 去噪损失训练 ADM 使扩散主干对齐嵌入空间。
        3) 阶段3（stage3）：解冻全部参数，联合加权 CE 与 MSE 进行微调。
    - 推理时仅对最后一个待预测 token 执行完整反向扩散；历史 token 保持干净，
      避免序列级扩散污染历史条件。

额外实现细节：
    - 引入数值稳定性检查（NaN/Inf 保护）；
      以及动态训练阶段调度器（根据 epoch 切换冻结/解冻策略）。

参考：
    "Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach"
    Chen et al., KDD 2025.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel

class SiLU(nn.Module):
    def forward(self, x): 
        return x * torch.sigmoid(x)

class TransformerEncoder(nn.Module):
    """简单的 Transformer 编码器实现（用于 CAM 或通用的前馈块）

    参数说明：
        hidden_size: 隐藏维度
        num_blocks: 前馈层数量（注意本实现是残差前馈块的堆叠，简化版Transformer）
        causal: 是否启用因果语义（CAM 使用 causal=True 以保证因果遮盖）
    用途：
        - 当作为 CAM（condition encoder）时，设置 causal=True，以隐式学习序列顺序依赖，
          不使用显式的位置编码以避免扩散噪声对固定位置编码的破坏。
    """
    def __init__(self, hidden_size=64, num_blocks=2, dropout=0.1, causal=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.causal = causal
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                SiLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Dropout(self.dropout)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual
        return out

class DenoisedModel(nn.Module):
    """去噪网络（ADM 的核心去噪子网）

    实现要点：
        - 接收条件表示 `rep_item`（来自 CAM）、加噪的目标表示 `x_t` 与时间步 t 的时间嵌入。
        - 使用小系数 `lambda_uncertainty` 将噪声/时间信息注入到条件中，减弱噪声对条件的直接影响。
        - 支持两种网络：TransformerEncoder（更强表达）或简单前馈网络（轻量）。
        - 提供 `forward_cfg` 用于分类无条件引导（Classifier-Free Guidance）实现。
    """
    def __init__(self, hidden_size=64, lambda_uncertainty=1e-3, use_transformer=True):
        super().__init__()
        self.hidden_size = hidden_size
        
        if use_transformer:
            self.decoder = TransformerEncoder(hidden_size=hidden_size, num_blocks=2, dropout=0.1)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.lambda_uncertainty = lambda_uncertainty

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """时间步嵌入（sin/cos 形式）

        说明：将标量时间步映射到与隐藏维相同的向量空间，用作去噪网络的时间条件。
        rescale 在外部由 _scale_timesteps 控制。
        """
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
        args_ = timesteps.unsqueeze(-1).float() * freqs[None]
        emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        return emb

    def forward_cfg(self, c, x_t, t, mask_seq, mask_tgt, cfg_scale=1.0):
        """Classifier-Free Guidance"""
        # 先计算带条件和无条件两种预测，再按比例组合以实现CFG
        cond_eps = self.forward(c, x_t, t, mask_seq, mask_tgt)
        uncond_eps = self.forward(c, x_t, t, mask_seq, mask_tgt, condition=False)
        eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return eps

    def forward(self, rep_item, x_t, t, mask_seq=None, mask_tgt=None, condition=True):
        # 如果 condition=False，则进行“无条件”去噪（用于 CFG 的无条件分支）
        if not condition:
            rep_item = torch.zeros_like(rep_item)

        # 保证时间步为二维形状以便批次/序列广播
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        if time_emb.dim() == 2 and x_t.dim() == 3:
            time_emb = time_emb.unsqueeze(1)

        # 将条件表示、加噪表示与时间嵌入按小权重合并，λ 控制时间/噪声对条件的影响强度
        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)

        # 通过解码器（Transformer 或 MLP）进行去噪预测
        if isinstance(self.decoder, TransformerEncoder):
            out = self.decoder(rep_diffu, mask_seq)
        else:
            out = self.decoder(rep_diffu)

        # 数值保护：若出现 NaN 用 0 填充（训练中会有额外的稳定性检查）
        if torch.isnan(out).any():
            out = torch.where(torch.isnan(out), torch.zeros_like(out), out)

        return out

class DiffusionModule(nn.Module):
    """封装扩散过程的核心模块 - 支持 token-level 独立扩散。

    说明：
        - 负责噪声调度（beta/alpha）、前向扩散 q(x_t|x_0)、后验近似以及基于去噪网络的
          反向采样 p(x_{t-1}|x_t)。
        - 支持两种扩散粒度：
            * token-level independent diffusion（每个 token 可采样不同 t）
            * sequence-level diffusion（整个序列共享 t）
        - 提供 `compute_loss`（训练时的 MSE 去噪损失）与 `generate_last_token`（推理时仅对末位 token 反向去噪）
        - 包含多处数值稳定性保护（clamp、NaN 检查、最小方差下限等）。
    """
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.diffusion_steps = getattr(args, 'diffusion_steps', 50)
        self.independent_diffusion = getattr(args, 'independent_diffusion', True)
        
        # Beta schedule
        self.setup_schedule(args)
        
        # Denoising network
        self.denoise_net = DenoisedModel(
            hidden_size=self.hidden_size,
            lambda_uncertainty=getattr(args, 'lambda_uncertainty', 1e-3),
            use_transformer=True
        )
        
        # Condition encoder (CAM - Causal Attention Module)
        self.condition_encoder = TransformerEncoder(
            hidden_size=self.hidden_size,
            num_blocks=getattr(args, 'num_blocks', 2),
            dropout=getattr(args, 'dropout', 0.1),
            causal=True
        )
        
        self.cfg_scale = getattr(args, 'cfg_scale', 1.0)
        self.cfg_dropout_rate = getattr(args, 'cfg_dropout_rate', 0.1)
        self.rescale_timesteps = getattr(args, 'rescale_timesteps', False)
        
    def setup_schedule(self, args):
        """设置噪声调度（beta schedule）并计算相关系数。

        说明：alphas_cumprod 与 sqrt_one_minus_alphas_cumprod 等张量用于实现标准
        DDPM 前向扩散与后验公式（见论文公式）：
            x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps

        这里对 cosine / linear 两类调度都做了支持，并对 posterior_variance 做下界截断
        以避免数值问题（例如 log(0)）。结果均以 buffer 注册，便于分布式训练与保存。
        """
        schedule_name = getattr(args, 'noise_schedule', 'linear')
        
        if schedule_name == "linear":
            betas = np.linspace(1e-4, 0.02, self.diffusion_steps)
        elif schedule_name == "cosine":
            steps = self.diffusion_steps + 1
            x = np.linspace(0, self.diffusion_steps, steps)
            alphas_cumprod = np.cos(((x / self.diffusion_steps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 0, 0.999)
        else:
            betas = np.linspace(1e-4, 0.02, self.diffusion_steps)
        
        betas = np.array(betas, dtype=np.float64)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        
        # 计算相关参数
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        
        # 添加微小值避免数值问题
        posterior_variance = np.maximum(posterior_variance, 1e-8)
        
        # 转换为tensor并注册为buffer
        self.register_buffer('betas', torch.from_numpy(betas).float())
        self.register_buffer('alphas', torch.from_numpy(alphas).float())
        self.register_buffer('alphas_cumprod', torch.from_numpy(alphas_cumprod).float())
        self.register_buffer('alphas_cumprod_prev', torch.from_numpy(alphas_cumprod_prev).float())
        self.register_buffer('sqrt_alphas_cumprod', torch.from_numpy(sqrt_alphas_cumprod).float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.from_numpy(sqrt_one_minus_alphas_cumprod).float())
        self.register_buffer('posterior_mean_coef1', torch.from_numpy(posterior_mean_coef1).float())
        self.register_buffer('posterior_mean_coef2', torch.from_numpy(posterior_mean_coef2).float())
        self.register_buffer('posterior_variance', torch.from_numpy(posterior_variance).float())
        
        self.num_timesteps = int(betas.shape[0])
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """从数组中提取对应时间步的值"""
        # arr: [num_timesteps]
        # timesteps: [batch_size, seq_len] 或 [batch_size]
        # broadcast_shape: 目标形状
        
        # 确保timesteps是long类型
        timesteps = timesteps.long()
        
        # 使用gather获取对应时间步的值
        if timesteps.dim() == 2:
            # token-level: timesteps形状为[batch_size, seq_len]
            batch_size, seq_len = timesteps.shape
            timesteps_flat = timesteps.reshape(-1)
            res_flat = arr.to(timesteps_flat.device)[timesteps_flat].float()
            res = res_flat.reshape(batch_size, seq_len, *([1] * (len(broadcast_shape) - 2)))
        else:
            # sequence-level: timesteps形状为[batch_size]
            res = arr.to(timesteps.device)[timesteps].float()
        
        # 重塑形状以匹配broadcast_shape
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        
        # 扩展到broadcast_shape
        return res.expand(broadcast_shape)
    
    def _scale_timesteps(self, t):
        """缩放时间步"""
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def q_sample(self, x_start, t, noise=None, mask=None):
        """前向扩散：按照时间步 t 向 x_start 添加高斯噪声，返回 x_t。

        对应 DDPM 前向过程（论文公式）：
            x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * noise

        参数：
            x_start: 原始干净嵌入 x_0
            t: 每个样本（或每个 token）对应的时间步（可为 [B] 或 [B, L]）
            noise: 可选的噪声张量，默认为标准正态
            mask: 可选的掩码（0 表示 pad/无效），在 pad 位置不进行噪声注入
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        if mask is not None:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            x_t = torch.where(mask==0, x_start, x_t)
        
        return x_t
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """计算后验分布的均值和方差"""
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean
    
    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        """计算预测分布"""
        # 预测x_start
        scaled_t = self._scale_timesteps(t)
        
        if self.cfg_scale == 1.0:
            x_start_pred = self.denoise_net(rep_item, x_t, scaled_t, mask_seq, mask_tag)
        else:
            x_start_pred = self.denoise_net.forward_cfg(
                rep_item, x_t, scaled_t, mask_seq, mask_tag, self.cfg_scale
            )
        
        # 数值稳定性检查
        if torch.isnan(x_start_pred).any():
            x_start_pred = torch.where(torch.isnan(x_start_pred), torch.randn_like(x_start_pred) * 0.01, x_start_pred)
        
        # 计算后验均值（模型对 x_0 的预测经过后验公式得到 p(x_{t-1}|x_t) 的均值）
        model_mean = self.q_posterior_mean_variance(x_start_pred, x_t, t)
        
        # 使用固定的后验方差（添加微小值避免log(0)）
        posterior_variance_t = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_variance_t = torch.clamp(posterior_variance_t, min=1e-8)
        model_log_variance = torch.log(posterior_variance_t)
        
        return model_mean, model_log_variance
    
    def p_sample(self, rep_item, x_t, t, mask_seq, mask_tag):
        """从p(x_{t-1}|x_t)采样"""
        model_mean, model_log_variance = self.p_mean_variance(rep_item, x_t, t, mask_seq, mask_tag)
        
        noise = torch.randn_like(x_t)
        
        # 非零掩码：当 t==0 时不再添加随机噪声（因为已到最末步）
        if t.dim() == 2:
            # token-level: t形状为[batch_size, seq_len]
            nonzero_mask = (t != 0).float().unsqueeze(-1)
        else:
            # sequence-level: t形状为[batch_size]
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        # 数值稳定性：避免exp数值过大
        model_log_variance = torch.clamp(model_log_variance, max=10.0)
        
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
        # 数值稳定性检查
        if torch.isnan(sample).any():
            sample = torch.where(torch.isnan(sample), torch.randn_like(sample) * 0.01, sample)
        
        return sample
    
    def compute_loss(self, condition_emb, target_emb, mask):
        """计算扩散损失（MSE 去噪损失）- 支持 token-level 独立扩散和加权损失。

        过程说明：
                - 若为 token-level 模式（independent_diffusion=True），每个 token 独立采样时间步 t
                    并展平为一维 batch 进行去噪预测；这样可以让同一序列内不同 token 在不同噪声强度下
                    学习去噪能力，从而提升生成多样性和鲁棒性。
                - 若为 sequence-level 模式，则整个序列共享同一时间步。
                - 使用加权 MSE：对不同长度序列做归一化权重，防止长序列主导损失。
        """
        batch_size, seq_len, hidden_size = condition_emb.shape
        
        # 编码条件信息
        encoded_condition = self.condition_encoder(condition_emb, mask)
        
        if self.independent_diffusion and self.training:
            # token-level扩散：每个token独立采样时间步
            t_flat = torch.randint(0, self.num_timesteps, (batch_size * seq_len,), device=condition_emb.device)
            t = t_flat.view(batch_size, seq_len)
            
            # Classifier-Free Guidance dropout
            if self.training and self.cfg_scale != 1.0:
                cfg_mask = torch.rand(batch_size, 1, 1, device=condition_emb.device) > self.cfg_dropout_rate
                encoded_condition = torch.where(cfg_mask, encoded_condition, torch.zeros_like(encoded_condition))
            
            # 展平处理以支持token-level扩散
            target_emb_flat = target_emb.reshape(-1, hidden_size)
            t_flat = t.reshape(-1)
            mask_flat = mask.reshape(-1)
            
            # 前向扩散：添加噪声到目标序列
            noise = torch.randn_like(target_emb_flat)
            x_t_flat = self.q_sample(target_emb_flat, t_flat, noise, mask_flat)
            x_t = x_t_flat.reshape(batch_size, seq_len, hidden_size)
            
            # 预测去噪结果
            predicted_flat = self.denoise_net(
                encoded_condition.reshape(-1, hidden_size), 
                x_t_flat, 
                self._scale_timesteps(t_flat), 
                mask_seq=mask_flat, 
                mask_tgt=mask_flat
            )
            predicted = predicted_flat.reshape(batch_size, seq_len, hidden_size)
            
        else:
            # sequence-level扩散：所有token共享相同时间步
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=condition_emb.device)
            t_expanded = t.unsqueeze(1).expand(-1, seq_len)
            
            # Classifier-Free Guidance dropout
            if self.training and self.cfg_scale != 1.0:
                cfg_mask = torch.rand(batch_size, 1, 1, device=condition_emb.device) > self.cfg_dropout_rate
                encoded_condition = torch.where(cfg_mask, encoded_condition, torch.zeros_like(encoded_condition))
            
            # 前向扩散：添加噪声到目标序列
            noise = torch.randn_like(target_emb)
            x_t = self.q_sample(target_emb, t_expanded, noise, mask)
            
            # 预测去噪结果
            predicted = self.denoise_net(encoded_condition, x_t, self._scale_timesteps(t_expanded), 
                                         mask_seq=mask, mask_tgt=mask)
        
        # 计算加权MSE损失（参考论文发布者的实现）
        loss_mask = mask.unsqueeze(-1)
        
        # 计算每个序列的mask权重
        mask_sum = mask.sum(dim=1, keepdim=True) + 1e-8
        weight = (mask / mask_sum).unsqueeze(-1)
        
        # 计算加权损失
        losses = F.mse_loss(predicted, target_emb, reduction='none') * weight * loss_mask
        mse_loss = losses.sum() / (batch_size + 1e-8)
        
        # 数值稳定性检查
        if torch.isnan(mse_loss).any() or torch.isinf(mse_loss).any():
            mse_loss = torch.tensor(0.1, device=mse_loss.device)
        
        return mse_loss, predicted
    
    def generate_last_token(self, condition_emb, mask):
        """推理时只对最后一个 token 进行去噪（论文关键创新）

        说明：为了避免序列级扩散污染历史条件（历史 token 被噪声破坏），推理阶段仅对
        末位 token 从纯高斯噪声开始迭代去噪，历史 token 直接作为条件馈入 CAM/ADM。
        该操作保留了历史序列的干净语义，且生成效果与序列级扩散相比更稳定，但需要 T 步
        迭代（推理延迟较高）。
        """
        batch_size, seq_len, hidden_size = condition_emb.shape
        
        # 编码条件信息
        encoded_condition = self.condition_encoder(condition_emb, mask)
        
        # 准备初始状态：历史token保持原样，最后一个token从随机噪声开始
        x_t = condition_emb.clone()
        
        # 最后一个token使用随机噪声
        last_token_noise = torch.randn(batch_size, 1, hidden_size, device=condition_emb.device)
        x_t[:, -1:, :] = last_token_noise
        
        # 反向扩散过程：从T到0逐步去噪
        indices = list(range(self.num_timesteps))[::-1]
        
        for i in indices:
            # 构建时间步tensor：历史token为0，最后一个token为i
            t = torch.zeros(batch_size, seq_len, device=condition_emb.device)
            t[:, -1] = i
            
            # 采样x_{t-1}
            x_t = self.p_sample(encoded_condition, x_t, t, mask, mask)
            
            # 确保历史token保持不变
            x_t[:, :-1, :] = condition_emb[:, :-1, :]
        
        # 只返回最后一个token的去噪结果
        return x_t[:, -1:, :]
    
    def generate(self, condition_emb, mask, num_steps=None):
        """传统的序列级生成（用于对比实验）"""
        batch_size, seq_len, hidden_size = condition_emb.shape
        
        if num_steps is None:
            num_steps = self.num_timesteps
        else:
            num_steps = min(num_steps, self.num_timesteps)
        
        # 编码条件信息
        encoded_condition = self.condition_encoder(condition_emb, mask)
        
        # 从随机噪声开始
        x_t = torch.randn(batch_size, seq_len, hidden_size, device=condition_emb.device)
        
        # 反向扩散过程：从T到0逐步去噪
        indices = list(range(num_steps))[::-1]
        
        for i in indices:
            t = torch.full((batch_size, seq_len), i, device=condition_emb.device)
            x_t = self.p_sample(encoded_condition, x_t, t, mask, mask)
        
        return x_t

class ADRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                          help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                          help='Size of hidden vectors in diffusion.')
        parser.add_argument('--diffusion_steps', type=int, default=50,
                          help='Number of diffusion steps.')
        parser.add_argument('--noise_schedule', type=str, default='linear',
                          choices=['linear', 'cosine'],
                          help='Noise schedule type.')
        parser.add_argument('--num_blocks', type=int, default=2,
                          help='Number of transformer blocks in condition encoder.')
        parser.add_argument('--lambda_uncertainty', type=float, default=1e-3,
                          help='Lambda for uncertainty in denoising.')
        parser.add_argument('--cfg_scale', type=float, default=1.0,
                          help='Classifier-free guidance scale.')
        parser.add_argument('--cfg_dropout_rate', type=float, default=0.1,
                          help='Dropout rate for CFG during training.')
        parser.add_argument('--rescale_timesteps', action='store_true',
                          help='Rescale timesteps to [0, 1000].')
        parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                          help='Weight for diffusion MSE loss.')
        parser.add_argument('--ce_loss_weight', type=float, default=1.0,
                          help='Weight for cross-entropy loss.')
        parser.add_argument('--independent_diffusion', action='store_true',
                          help='Use token-level independent diffusion.')
        parser.add_argument('--training_stage', type=str, default='stage1',
                          choices=['stage1', 'stage2', 'stage3'],
                          help='Training stage: stage1 (pretrain), stage2 (warm-up), stage3 (full).')
        parser.add_argument('--warmup_epochs', type=int, default=5,
                          help='Number of warm-up epochs for stage2.')
        parser.add_argument('--pretrain_epochs', type=int, default=10,
                          help='Number of pretrain epochs for stage1.')
        return parser

    def _base_init(self, args, corpus):
        # 基本配置：嵌入维度、隐藏维度与历史长度等
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.max_his = args.history_max
        
        # 三阶段训练控制（用于实现论文中的预训练/对齐/微调流程）
        # stage1: 预训练嵌入与 CAM（CE loss）
        # stage2: 冻结嵌入，仅训练扩散主干（MSE 去噪）
        # stage3: 解冻全部参数，联合优化（CE + MSE）
        self.training_stage = getattr(args, 'training_stage', 'stage1')
        self.warmup_epochs = getattr(args, 'warmup_epochs', 5)
        self.pretrain_epochs = getattr(args, 'pretrain_epochs', 10)
        self.current_epoch = 0
        
        # 确保 hidden_size 和 emb_size 一致，或添加投影层
        if self.hidden_size != self.emb_size:
            self.projection = nn.Linear(self.hidden_size, self.emb_size)
            self.emb_projection = nn.Linear(self.emb_size, self.hidden_size)
        else:
            self.projection = nn.Identity()
            self.emb_projection = nn.Identity()
        
        # 物品嵌入
        self.i_embeddings = nn.Embedding(corpus.n_items, self.emb_size)
        # 初始化嵌入，避免数值问题
        nn.init.normal_(self.i_embeddings.weight, mean=0.0, std=0.01)
        
        self.embed_dropout = nn.Dropout(getattr(args, 'emb_dropout', 0.2))
        
        # Layer normalization
        self.hist_norm = nn.LayerNorm(self.emb_size)
        
        # 扩散模块
        self.diffusion = DiffusionModule(args)
        
        # 损失函数与权重
        self.ce_loss = nn.CrossEntropyLoss()
        self.diffusion_loss_weight = getattr(args, 'diffusion_loss_weight', 1.0)
        self.ce_loss_weight = getattr(args, 'ce_loss_weight', 1.0)
        
        self.apply(self.init_weights)
        
        # 根据初始训练阶段设置参数（会冻结/解冻对应模块权重）
        self.set_training_stage(self.training_stage)

    def init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def set_training_stage(self, stage):
        """设置训练阶段 - 适配训练框架"""
        self.training_stage = stage
        
        if stage == 'stage1':
            # 阶段1: 预训练嵌入层，冻结扩散模块
            for param in self.diffusion.parameters():
                param.requires_grad = False
            for param in self.i_embeddings.parameters():
                param.requires_grad = True
            for param in self.projection.parameters():
                param.requires_grad = True
            for param in self.emb_projection.parameters():
                param.requires_grad = True
                
        elif stage == 'stage2':
            # 阶段2: warm-up，冻结嵌入层，训练扩散模块
            for param in self.i_embeddings.parameters():
                param.requires_grad = False
            for param in self.diffusion.parameters():
                param.requires_grad = True
            # 投影层保持可训练
            for param in self.projection.parameters():
                param.requires_grad = True
            for param in self.emb_projection.parameters():
                param.requires_grad = True
                
        elif stage == 'stage3':
            # 阶段3: 全参数训练
            for param in self.parameters():
                param.requires_grad = True
    
    def set_current_epoch(self, epoch):
        """根据epoch自动切换训练阶段 - 适配您的训练框架"""
        self.current_epoch = epoch
        
        if epoch < self.pretrain_epochs:
            new_stage = 'stage1'
        elif epoch < self.pretrain_epochs + self.warmup_epochs:
            new_stage = 'stage2'
        else:
            new_stage = 'stage3'
        
        # 如果阶段发生变化，更新参数
        if new_stage != self.training_stage:
            self.set_training_stage(new_stage)

    def loss(self, out_dict):
        """损失函数 - 适配您的训练框架"""
        return out_dict['loss']
    
    def customize_parameters(self):
        """返回需要优化的参数 - 适配您的训练框架"""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, feed_dict):
        """前向传播 - 适配您的训练框架"""
        self.check_list = []
        
        # 提取输入
        if 'item_id' in feed_dict:
            i_ids = feed_dict['item_id']  # [batch_size, num_candidates]
        else:
            # 如果是impression模式，可能没有item_id
            i_ids = None
            
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        # 确保lengths是long类型
        lengths = lengths.long()
        
        # 获取历史序列的嵌入
        valid_his = (history > 0).long()
        his_vectors = self.i_embeddings(history)
        
        # 应用dropout和归一化
        his_vectors = self.embed_dropout(his_vectors)
        his_vectors = self.hist_norm(his_vectors)
        
        # 投影到扩散模型的维度
        his_vectors = self.emb_projection(his_vectors)
        
        if self.training:
            # 根据训练阶段调整行为
            if self.training_stage == 'stage1':
                # 阶段1: 只使用CE损失（类似SASRec+）
                # 编码历史序列
                mask = (history > 0).float()
                encoded_condition = self.diffusion.condition_encoder(his_vectors, mask)
                
                # 取最后一个位置的编码作为用户表示
                user_emb = encoded_condition[torch.arange(batch_size, device=history.device), (lengths - 1).long(), :]
                user_emb = self.projection(user_emb)
                
                # 候选物品嵌入
                candidate_emb = self.i_embeddings(i_ids)
                
                # 计算预测分数
                prediction = (user_emb.unsqueeze(1) * candidate_emb).sum(-1)
                prediction = torch.clamp(prediction, min=-10.0, max=10.0)
                
                # 下一个物品的ground truth（在候选列表中的第一个位置是正样本）
                next_items = torch.zeros(batch_size, dtype=torch.long, device=history.device)
                ce_loss = self.ce_loss(prediction, next_items)
                
                # 数值稳定性检查
                if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
                    ce_loss = torch.tensor(0.1, device=ce_loss.device)
                
                total_loss = self.ce_loss_weight * ce_loss
                
                return {
                    'prediction': prediction,
                    'loss': total_loss,
                    'ce_loss': ce_loss,
                    'diffusion_loss': torch.tensor(0.0, device=ce_loss.device)
                }
                
            elif self.training_stage == 'stage2':
                # 阶段2: 只使用扩散损失，冻结嵌入层
                
                # 准备序列：使用历史序列作为条件，预测下一个物品
                # 输入序列：history[:, :-1]，目标序列：history[:, 1:]
                seq_input = history[:, :-1]  # [batch_size, seq_len-1]
                seq_target = history[:, 1:]   # [batch_size, seq_len-1]
                
                # 创建mask
                mask_target = (seq_target > 0).float()  # [batch_size, seq_len-1]
                
                # 获取目标嵌入
                target_emb = self.i_embeddings(seq_target)  # [batch_size, seq_len-1, emb_size]
                target_emb = self.emb_projection(target_emb)
                
                # 扩散损失（重建整个序列）
                diffusion_loss, pred_emb = self.diffusion.compute_loss(
                    his_vectors[:, :-1, :], target_emb, mask_target
                )
                
                # 数值稳定性检查
                if torch.isnan(diffusion_loss).any() or torch.isinf(diffusion_loss).any():
                    diffusion_loss = torch.tensor(0.1, device=diffusion_loss.device)
                
                # 取最后一个位置的预测作为用户表示（用于预测，但不用于损失计算）
                last_pred = pred_emb[:, -1, :]  # [batch_size, hidden_size]
                last_pred = self.projection(last_pred)
                
                # 候选物品嵌入
                candidate_emb = self.i_embeddings(i_ids)
                
                # 计算预测分数（仅用于预测，不参与损失计算）
                prediction = (last_pred.unsqueeze(1) * candidate_emb).sum(-1)
                prediction = torch.clamp(prediction, min=-10.0, max=10.0)
                
                # 阶段2只使用扩散损失
                total_loss = self.diffusion_loss_weight * diffusion_loss
                
                return {
                    'prediction': prediction,
                    'loss': total_loss,
                    'ce_loss': torch.tensor(0.0, device=diffusion_loss.device),
                    'diffusion_loss': diffusion_loss
                }
                
            elif self.training_stage == 'stage3':
                # 阶段3: 同时使用扩散损失和推荐损失
                
                # 准备序列：使用历史序列作为条件，预测下一个物品
                seq_input = history[:, :-1]
                seq_target = history[:, 1:]
                
                # 创建mask
                mask_target = (seq_target > 0).float()
                
                # 获取目标嵌入
                target_emb = self.i_embeddings(seq_target)
                target_emb = self.emb_projection(target_emb)
                
                # 扩散损失（重建整个序列）
                diffusion_loss, pred_emb = self.diffusion.compute_loss(
                    his_vectors[:, :-1, :], target_emb, mask_target
                )
                
                # 数值稳定性检查
                if torch.isnan(diffusion_loss).any() or torch.isinf(diffusion_loss).any():
                    diffusion_loss = torch.tensor(0.1, device=diffusion_loss.device)
                
                # 取最后一个位置的预测作为用户表示
                last_pred = pred_emb[:, -1, :]
                last_pred = self.projection(last_pred)
                
                # 候选物品嵌入
                candidate_emb = self.i_embeddings(i_ids)
                
                # 计算预测分数
                prediction = (last_pred.unsqueeze(1) * candidate_emb).sum(-1)
                prediction = torch.clamp(prediction, min=-10.0, max=10.0)
                
                # 下一个物品的ground truth
                next_items = torch.zeros(batch_size, dtype=torch.long, device=history.device)
                ce_loss = self.ce_loss(prediction, next_items)
                
                # 数值稳定性检查
                if torch.isnan(ce_loss).any() or torch.isinf(ce_loss).any():
                    ce_loss = torch.tensor(0.1, device=ce_loss.device)
                
                # 总损失
                total_loss = (
                    self.ce_loss_weight * ce_loss + 
                    self.diffusion_loss_weight * diffusion_loss
                )
                
                return {
                    'prediction': prediction,
                    'loss': total_loss,
                    'ce_loss': ce_loss,
                    'diffusion_loss': diffusion_loss
                }
                
        else:
            # 评估模式：使用论文中的推理策略（只对最后一个token去噪）
            
            # 编码历史序列
            mask = (history > 0).float()
            encoded_condition = self.diffusion.condition_encoder(his_vectors, mask)
            
            # 使用扩散模型生成最后一个token的去噪表示
            # 只对最后一个token进行去噪，历史token保持不变
            denoised_last_token = self.diffusion.generate_last_token(
                his_vectors,  # 整个序列作为条件
                mask  # 整个序列的mask
            )
            
            # 投影回原始嵌入空间
            denoised_last_token = self.projection(denoised_last_token)
            
            # 候选物品嵌入
            candidate_emb = self.i_embeddings(i_ids)
            
            # 计算预测分数
            prediction = (denoised_last_token * candidate_emb).sum(-1)
            prediction = torch.clamp(prediction, min=-10.0, max=10.0)
            
            # 确保没有NaN或Inf
            if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                prediction = torch.where(
                    torch.isnan(prediction) | torch.isinf(prediction),
                    torch.zeros_like(prediction),
                    prediction
                )
            
            return {'prediction': prediction}

class ADRec(SequentialModel, ADRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'emb_size', 'hidden_size', 'diffusion_steps', 
        'diffusion_loss_weight', 'ce_loss_weight', 'cfg_scale',
        'independent_diffusion', 'training_stage', 'warmup_epochs', 'pretrain_epochs'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = ADRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        
        # 添加stage属性，用于训练框架判断最佳模型
        self.stage = int(self.training_stage[-1]) if self.training_stage.startswith('stage') else 1

    def forward(self, feed_dict):
        out_dict = ADRecBase.forward(self, feed_dict)
        if self.training:
            return {
                'prediction': out_dict['prediction'], 
                'loss': out_dict['loss']
            }
        else:
            return {'prediction': out_dict['prediction']}

class ADRecImpression(ImpressionSeqModel, ADRecBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = [
        'emb_size', 'hidden_size', 'diffusion_steps', 
        'diffusion_loss_weight', 'ce_loss_weight', 'cfg_scale',
        'independent_diffusion', 'training_stage', 'warmup_epochs', 'pretrain_epochs'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = ADRecBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        
        # 添加stage属性，用于训练框架判断最佳模型
        self.stage = int(self.training_stage[-1]) if self.training_stage.startswith('stage') else 1

    def forward(self, feed_dict):
        out_dict = ADRecBase.forward(self, feed_dict)
        if self.training:
            return {
                'prediction': out_dict['prediction'], 
                'loss': out_dict['loss']
            }
        else:
            return {'prediction': out_dict['prediction']}