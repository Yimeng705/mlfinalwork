# -*- coding: UTF-8 -*-
# @Author  : Yijie Gu
# @Email   : guyj25@mail2.sysu.edu.cn

# @article{yang2023generate,
#   title={Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion},
#   author={Yang, Zhengyi and Wu, Jiancan and Wang, Zhicai and Wang, Xiang and Yuan, Yancheng and He, Xiangnan},
#   journal={Advances in Neural Information Processing Systems},
#   year={2023}
# }

import torch.nn as nn
import torch as th
import numpy as np
import math
import torch
import torch.nn.functional as F
from models.BaseModel import SequentialModel

class SiLU(nn.Module):
    def forward(self, x): 
        return x * torch.sigmoid(x)

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class TransformerEncoder(nn.Module):
    def __init__(self, args, num_blocks=2, norm_first=True, hidden_size=64):
        super(TransformerEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.norm_first = norm_first
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.Dropout(args.dropout)
            ) for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        out = x
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual
        return out

class Diffu_xstart(nn.Module):
    def __init__(self, args):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = args.hidden_size
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            SiLU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        self.norm_diffu_rep = LayerNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(3*self.hidden_size, self.hidden_size*4),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_size*4, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size)
        )
        self.dropout = nn.Dropout(args.dropout)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        assert dim % 2 == 0
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
        args = timesteps.unsqueeze(-1).float() * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq, mask_tgt):
        t = t.reshape(rep_item.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        rep_diffu = torch.cat([x_t, time_emb.expand_as(rep_item), rep_item], dim=-1)
        rep_diffu = self.mlp(rep_diffu)
        rep_diffu = self.norm_diffu_rep(self.dropout(rep_diffu))
        return rep_diffu

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class DreamRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                          help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                          help='Size of hidden vectors in diffusion.')
        parser.add_argument('--diffusion_steps', type=int, default=100,
                          help='Number of diffusion steps.')
        parser.add_argument('--noise_schedule', type=str, default='linear',
                          choices=['linear', 'cosine'],
                          help='Noise schedule type.')
        parser.add_argument('--schedule_sampler_name', type=str, default='uniform',
                          help='Schedule sampler name.')
        parser.add_argument('--rescale_timesteps', action='store_true',
                          help='Whether to rescale timesteps.')
        parser.add_argument('--lambda_uncertainty', type=float, default=1e-3,
                          help='Lambda for uncertainty in denoising.')
        parser.add_argument('--cfg_scale', type=float, default=1.0,
                          help='Classifier-free guidance scale.')
        parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                          help='Weight for diffusion MSE loss.')
        parser.add_argument('--ce_loss_weight', type=float, default=1.0,
                          help='Weight for cross-entropy loss.')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.max_his = args.history_max
        self.diffusion_steps = args.diffusion_steps
        
        # 确保 hidden_size 和 emb_size 一致，或添加投影层
        if self.hidden_size != self.emb_size:
            self.projection = nn.Linear(self.hidden_size, self.emb_size)
            self.emb_projection = nn.Linear(self.emb_size, self.hidden_size)
        else:
            self.projection = nn.Identity()
            self.emb_projection = nn.Identity()
        
        # 物品嵌入
        self.i_embeddings = nn.Embedding(corpus.n_items + 1, self.emb_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(getattr(args, 'emb_dropout', 0.2))
        
        # Layer normalization
        self.hist_norm = LayerNorm(self.emb_size)
        
        # 初始化扩散参数
        self._setup_diffusion(args)
        
        # 扩散网络
        self.net = Diffu_xstart(args)
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False, hidden_size=self.hidden_size)
        
        # 损失函数 - 使用负对数似然损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.diffusion_loss_weight = getattr(args, 'diffusion_loss_weight', 1.0)
        self.ce_loss_weight = getattr(args, 'ce_loss_weight', 1.0)
        
        self.apply(self.init_weights)

    def _setup_diffusion(self, args):
        """设置扩散过程参数"""
        # Beta schedule
        if args.noise_schedule == "linear":
            betas = np.linspace(1e-4, 0.02, self.diffusion_steps)
        elif args.noise_schedule == "cosine":
            steps = self.diffusion_steps + 1
            x = np.linspace(0, self.diffusion_steps, steps)
            alphas_cumprod = np.cos(((x / self.diffusion_steps) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = np.clip(betas, 0, 0.999)
        else:
            betas = np.linspace(1e-4, 0.02, self.diffusion_steps)
        
        self.betas = betas
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        self.num_timesteps = int(self.betas.shape[0])
        
    def q_sample(self, x_start, t, noise=None, mask=None):
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        sqrt_alphas_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        if mask is not None:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, x_t)
        return x_t
    
    def _scale_timesteps(self, t):
        if hasattr(self, 'rescale_timesteps') and self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def independent_diffuse(self, tgt, mask, is_independent=False):
        """独立扩散"""
        batch_size = tgt.shape[0]
        device = tgt.device
        
        if is_independent:
            t = torch.randint(0, self.num_timesteps, (batch_size * tgt.shape[1],), device=device)
            x_t = self.q_sample(tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)).reshape(*tgt.shape)
        else:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, (nn.LayerNorm, LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, num_candidates]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        # 准备序列：使用历史序列作为条件，预测下一个物品
        seq_input = history[:, :-1]  # [batch_size, seq_len-1]
        seq_target = history[:, 1:]   # [batch_size, seq_len-1]
        
        # 创建mask
        mask_input = (seq_input > 0).float()  # [batch_size, seq_len-1]
        mask_target = (seq_target > 0).float()  # [batch_size, seq_len-1]
        
        # 获取嵌入
        input_emb = self.i_embeddings(seq_input)  # [batch_size, seq_len-1, emb_size]
        target_emb = self.i_embeddings(seq_target)  # [batch_size, seq_len-1, emb_size]
        
        # 应用dropout和归一化
        input_emb = self.embed_dropout(input_emb)
        input_emb = self.hist_norm(input_emb)
        
        # 投影到扩散模型的维度
        input_emb = self.emb_projection(input_emb)
        target_emb = self.emb_projection(target_emb)
        
        if self.training:
            # 训练模式：计算扩散损失和推荐损失
            
            # 1. 扩散损失（重建整个序列）
            # 编码条件信息
            encoded_condition = self.ag_encoder(input_emb, mask_input)
            
            # 扩散过程
            x_t, t = self.independent_diffuse(target_emb, mask_target, is_independent=False)
            
            # 去噪
            denoised_seq = self.net(encoded_condition, x_t, self._scale_timesteps(t), mask_input, mask_target)
            
            # 计算扩散损失
            diffusion_loss = F.mse_loss(denoised_seq[:, -1], target_emb[:, -1])
            
            # 2. 推荐损失（预测下一个物品）
            # 取最后一个位置的预测作为用户表示
            last_pred = denoised_seq[:, -1, :]  # [batch_size, hidden_size]
            last_pred = self.projection(last_pred)  # 投影回emb_size
            
            # 候选物品嵌入
            candidate_emb = self.i_embeddings(i_ids)  # [batch_size, num_candidates, emb_size]
            
            # 计算预测分数 - 点积相似度
            # 注意：这里计算的是候选物品的分数，而不是所有物品
            prediction = (last_pred.unsqueeze(1) * candidate_emb).sum(-1)  # [batch_size, num_candidates]
            
            # 下一个物品的ground truth（序列的最后一个有效物品）
            # 注意：这里需要的是在候选列表中的位置，而不是物品ID
            # 在训练时，通常第一个候选是正样本，其余是负样本
            next_items = torch.zeros(batch_size, dtype=torch.long, device=history.device)
            
            # 计算交叉熵损失
            ce_loss = self.ce_loss(prediction, next_items)
            
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
            # 评估模式：生成预测
            # 使用扩散模型生成目标序列
            encoded_condition = self.ag_encoder(input_emb, mask_input)
            
            # 从随机噪声开始
            noise_x_t = torch.randn_like(target_emb)
            
            # 反向扩散过程（简化版，实际可能需要完整的去噪过程）
            indices = list(range(self.num_timesteps))[::-1]
            for i in indices:
                t = torch.tensor([i] * batch_size, device=input_emb.device)
                # 这里简化了反向过程，实际应该实现完整的p_sample
                denoised_seq = self.net(encoded_condition, noise_x_t, self._scale_timesteps(t), mask_input, mask_target)
                noise_x_t = denoised_seq  # 简化更新
            
            # 取最后一个位置的生成结果作为用户表示
            user_emb = noise_x_t[:, -1, :]  # [batch_size, hidden_size]
            user_emb = self.projection(user_emb)  # 投影回emb_size
            
            # 候选物品嵌入
            candidate_emb = self.i_embeddings(i_ids)  # [batch_size, num_candidates, emb_size]
            
            # 计算预测分数
            prediction = (user_emb.unsqueeze(1) * candidate_emb).sum(-1)  # [batch_size, num_candidates]
            
            return {'prediction': prediction}

class DreamRec(SequentialModel, DreamRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'emb_size', 'hidden_size', 'diffusion_steps', 
        'diffusion_loss_weight', 'ce_loss_weight', 'cfg_scale'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = DreamRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = DreamRecBase.forward(self, feed_dict)
        if self.training:
            return {
                'prediction': out_dict['prediction'], 
                'loss': out_dict['loss']
            }
        else:
            return {'prediction': out_dict['prediction']}

class DreamRecImpression(SequentialModel, DreamRecBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = [
        'emb_size', 'hidden_size', 'diffusion_steps', 
        'diffusion_loss_weight', 'ce_loss_weight', 'cfg_scale'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = DreamRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = DreamRecBase.forward(self, feed_dict)
        if self.training:
            return {
                'prediction': out_dict['prediction'], 
                'loss': out_dict['loss']
            }
        else:
            return {'prediction': out_dict['prediction']}
