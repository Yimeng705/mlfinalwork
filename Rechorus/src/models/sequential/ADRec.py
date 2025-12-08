# -*- coding: UTF-8 -*-
# @Author  : Yijie Gu
# @Email   : 

""" ADRec (Auto-regressive Diffusion Recommendation)
Reference:
    "Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach"
    Chen et al., KDD'2025.
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
    def __init__(self, hidden_size=64, num_blocks=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        
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
    def __init__(self, hidden_size=64, lambda_uncertainty=1e-3):
        super().__init__()
        self.hidden_size = hidden_size
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
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
        args_ = timesteps.unsqueeze(-1).float() * freqs[None]
        emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        return emb

    def forward(self, rep_item, x_t, t, mask_seq=None, mask_tgt=None, condition=True):
        if not condition:
            rep_item = torch.zeros_like(rep_item)
            
        time_emb = self.time_embed(self.timestep_embedding(t, rep_item.size(-1)))
        
        # 扩展时间嵌入以匹配x_t的维度
        if time_emb.dim() == 2 and x_t.dim() == 3:
            time_emb = time_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)
        out = self.decoder(rep_diffu)
        return out

class DiffusionModule(nn.Module):
    """封装扩散过程的核心模块"""
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.diffusion_steps = getattr(args, 'diffusion_steps', 100)
        
        # Beta schedule
        self.setup_schedule(args)
        
        # Denoising network
        self.denoise_net = DenoisedModel(
            hidden_size=self.hidden_size,
            lambda_uncertainty=getattr(args, 'lambda_uncertainty', 1e-3)
        )
        
        # Condition encoder
        self.condition_encoder = TransformerEncoder(
            hidden_size=self.hidden_size,
            num_blocks=getattr(args, 'num_blocks', 2),
            dropout=getattr(args, 'dropout', 0.1)
        )
        
        self.cfg_scale = getattr(args, 'cfg_scale', 1.0)
        self.cfg_dropout_rate = getattr(args, 'cfg_dropout_rate', 0.1)
        
    def setup_schedule(self, args):
        """设置噪声调度"""
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
        
        self.betas = torch.tensor(betas, dtype=torch.float32)
        
        # 计算相关参数
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.tensor(np.cumprod(alphas, axis=0), dtype=torch.float32)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def _extract(self, arr, t, x_shape):
        """从数组中提取对应时间步的值"""
        batch_size = t.shape[0]
        out = torch.gather(arr, 0, t.to(arr.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, condition_emb, target_emb, mask):
        """计算扩散损失"""
        batch_size, seq_len, _ = condition_emb.shape
        
        # 编码条件信息
        encoded_condition = self.condition_encoder(condition_emb, mask)
        
        # 随机采样时间步
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=condition_emb.device)
        
        # 前向扩散：添加噪声
        noise = torch.randn_like(target_emb)
        x_t = self.q_sample(target_emb, t, noise)
        
        # Classifier-Free Guidance dropout
        if self.training and self.cfg_scale != 1.0:
            cfg_mask = torch.rand(batch_size, 1, 1, device=condition_emb.device) > self.cfg_dropout_rate
            encoded_condition = torch.where(cfg_mask, encoded_condition, torch.zeros_like(encoded_condition))
        
        # 预测去噪结果
        predicted = self.denoise_net(encoded_condition, x_t, t, mask_seq=mask, mask_tgt=mask)
        
        # 计算MSE损失（带mask）
        loss_mask = mask.unsqueeze(-1)
        mse_loss = F.mse_loss(predicted * loss_mask, target_emb * loss_mask, reduction='sum')
        mse_loss = mse_loss / (loss_mask.sum() + 1e-8)
        
        return mse_loss, predicted
    
    def generate(self, condition_emb, mask, num_steps=None):
        """反向生成过程"""
        batch_size, seq_len, hidden_size = condition_emb.shape
        
        if num_steps is None:
            num_steps = self.diffusion_steps
        else:
            # 确保生成步数不超过训练时的扩散步数
            num_steps = min(num_steps, self.diffusion_steps)
        
        # 编码条件信息
        encoded_condition = self.condition_encoder(condition_emb, mask)
        
        # 从随机噪声开始，维度与目标序列相同
        x_t = torch.randn(batch_size, seq_len, hidden_size, device=condition_emb.device)
        
        # 反向扩散过程
        for i in range(num_steps - 1, -1, -1):
            t = torch.full((batch_size,), i, device=condition_emb.device)
            
            # 预测去噪结果（使用CFG）
            if self.cfg_scale != 1.0:
                # 有条件预测
                cond_pred = self.denoise_net(encoded_condition, x_t, t, mask_seq=mask, mask_tgt=mask)
                # 无条件预测
                uncond_pred = self.denoise_net(
                    torch.zeros_like(encoded_condition), x_t, t, mask_seq=mask, mask_tgt=mask
                )
                predicted = uncond_pred + self.cfg_scale * (cond_pred - uncond_pred)
            else:
                predicted = self.denoise_net(encoded_condition, x_t, t, mask_seq=mask, mask_tgt=mask)
            
            # 更新x_t
            if i > 0:
                noise = torch.randn_like(x_t)
                sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
                sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
                x_t = (x_t - sqrt_one_minus_alpha * predicted) / sqrt_alpha
                x_t = x_t + self._extract(self.betas, t, x_t.shape) * noise
            else:
                x_t = predicted
        
        return x_t

class ADRecBase(object):
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
        parser.add_argument('--num_blocks', type=int, default=2,
                          help='Number of transformer blocks in condition encoder.')
        parser.add_argument('--lambda_uncertainty', type=float, default=1e-3,
                          help='Lambda for uncertainty in denoising.')
        parser.add_argument('--cfg_scale', type=float, default=1.0,
                          help='Classifier-free guidance scale.')
        parser.add_argument('--cfg_dropout_rate', type=float, default=0.1,
                          help='Dropout rate for CFG during training.')
        parser.add_argument('--diffusion_loss_weight', type=float, default=1.0,
                          help='Weight for diffusion MSE loss.')
        parser.add_argument('--ce_loss_weight', type=float, default=1.0,
                          help='Weight for cross-entropy loss.')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.max_his = args.history_max
        
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
        self.hist_norm = nn.LayerNorm(self.emb_size)
        
        # 扩散模块
        self.diffusion = DiffusionModule(args)
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.diffusion_loss_weight = getattr(args, 'diffusion_loss_weight', 1.0)
        self.ce_loss_weight = getattr(args, 'ce_loss_weight', 1.0)
        
        self.apply(self.init_weights)

    def init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, num_candidates]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        # 准备序列：使用历史序列作为条件，预测下一个物品
        # 输入序列：history[:, :-1]，目标序列：history[:, 1:]
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
            diffusion_loss, pred_emb = self.diffusion.compute_loss(
                input_emb, target_emb, mask_target
            )
            
            # 2. 推荐损失（预测下一个物品）
            # 取最后一个位置的预测作为用户表示
            last_pred = pred_emb[:, -1, :]  # [batch_size, hidden_size]
            last_pred = self.projection(last_pred)  # 投影回emb_size
            
            # 候选物品嵌入
            candidate_emb = self.i_embeddings(i_ids)  # [batch_size, num_candidates, emb_size]
            
            # 计算预测分数 - 点积相似度
            prediction = (last_pred.unsqueeze(1) * candidate_emb).sum(-1)  # [batch_size, num_candidates]
            
            # 下一个物品的ground truth（在候选列表中的位置，假设第一个是正样本）
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
            # 注意：这里传入的mask_target应该与目标序列长度一致
            # 生成步数不超过训练时的扩散步数
            generated_emb = self.diffusion.generate(
                input_emb, mask_target, num_steps=self.diffusion.diffusion_steps
            )
            
            # 取最后一个位置的生成结果作为用户表示
            user_emb = generated_emb[:, -1, :]  # [batch_size, hidden_size]
            user_emb = self.projection(user_emb)  # 投影回emb_size
            
            # 候选物品嵌入
            candidate_emb = self.i_embeddings(i_ids)  # [batch_size, num_candidates, emb_size]
            
            # 计算预测分数 - 点积相似度
            prediction = (user_emb.unsqueeze(1) * candidate_emb).sum(-1)  # [batch_size, num_candidates]
            
            return {'prediction': prediction}

class ADRec(SequentialModel, ADRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = [
        'emb_size', 'hidden_size', 'diffusion_steps', 
        'diffusion_loss_weight', 'ce_loss_weight', 'cfg_scale'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = ADRecBase.parse_model_args(parser)
        return SequentialModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

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
        'diffusion_loss_weight', 'ce_loss_weight', 'cfg_scale'
    ]

    @staticmethod
    def parse_model_args(parser):
        parser = ADRecBase.parse_model_args(parser)
        return ImpressionSeqModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        ImpressionSeqModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        out_dict = ADRecBase.forward(self, feed_dict)
        if self.training:
            return {
                'prediction': out_dict['prediction'], 
                'loss': out_dict['loss']
            }
        else:
            return {'prediction': out_dict['prediction']}
