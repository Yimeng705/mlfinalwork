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
        
        # Simplified transformer implementation for ReChorus compatibility
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

def create_named_schedule_sampler(name, num_timesteps):
    class UniformSampler:
        def __init__(self, T): 
            self.T = T
        def sample(self, n, device):
            t = torch.randint(0, self.T, (n,), device=device)
            return t, torch.ones_like(t, dtype=torch.float)
    
    if name == "uniform":
        return UniformSampler(num_timesteps)
    else:
        return UniformSampler(num_timesteps)

def get_named_beta_schedule(schedule_name, T=100):
    if schedule_name == "linear":
        return np.linspace(1e-4, 0.02, T)
    elif schedule_name == "cosine":
        # Simplified cosine schedule
        steps = T + 1
        x = np.linspace(0, T, steps)
        alphas_cumprod = np.cos(((x / T) + 0.008) / (1 + 0.008) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)
    else:
        return np.linspace(1e-4, 0.02, T)

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
        assert dim % 2 == 0
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
        args_ = timesteps.unsqueeze(-1).float() * freqs[None]
        emb = torch.cat([torch.cos(args_), torch.sin(args_)], dim=-1)
        return emb

    def forward(self, rep_item, x_t, t, mask_seq=None, mask_tgt=None, condition=True):
        if not condition:
            rep_item = torch.zeros_like(rep_item)
            
        t = t.reshape(x_t.shape[0], -1)
        time_emb = self.time_embed(self.timestep_embedding(t, rep_item.size(-1)))
        rep_diffu = rep_item + self.lambda_uncertainty * (x_t + time_emb)
        out = self.decoder(rep_diffu)
        return out

class ADRec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.diffusion_steps = getattr(args, 'diffusion_steps', 100)
        
        # Beta schedule
        betas = get_named_beta_schedule(getattr(args, 'noise_schedule', 'linear'), self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        
        # Alpha calculations
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        
        # Diffusion calculations
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior calculations
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        
        self.num_timesteps = int(betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(
            getattr(args, 'schedule_sampler_name', 'uniform'), 
            self.num_timesteps
        )

        self.net = DenoisedModel(
            hidden_size=self.hidden_size,
            lambda_uncertainty=getattr(args, 'lambda_uncertainty', 1e-3)
        )
        self.ag_encoder = TransformerEncoder(
            hidden_size=self.hidden_size,
            num_blocks=2,
            dropout=getattr(args, 'dropout', 0.1)
        )
        self.independent_diffusion = getattr(args, 'independent', False)
        self.cfg_scale = getattr(args, 'cfg_scale', 1.0)

    def _extract(self, arr, t, x_shape):
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr).to(t.device).float()
        out = arr.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(x_start)
            x_t = torch.where(mask_exp == 0, x_start, x_t)
            
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq, mask_tag):
        if self.cfg_scale != 1.0:
            # Classifier-Free Guidance
            cond_out = self.net(rep_item, x_t, t, mask_seq, mask_tag, condition=True)
            uncond_out = self.net(rep_item, x_t, t, mask_seq, mask_tag, condition=False)
            x_0 = uncond_out + self.cfg_scale * (cond_out - uncond_out)
        else:
            x_0 = self.net(rep_item, x_t, t, mask_seq, mask_tag)
        
        # Use fixed variance for simplicity
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = self._extract(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq, mask_tag):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq, mask_tag)
        noise = torch.randn_like(noise_x_t)
        
        nonzero_mask = (t != 0).float().view(-1, 1, 1)
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample_xt

    def independent_diffuse(self, tgt, mask, is_independent=False):
        if is_independent:
            t, weights = self.schedule_sampler.sample(tgt.shape[0] * tgt.shape[1], tgt.device)
            t = t * mask.reshape(-1).long()
            x_t = self.q_sample(tgt.reshape(-1, tgt.shape[-1]), t, mask=mask.reshape(-1)).reshape(*tgt.shape)
        else:
            t, weights = self.schedule_sampler.sample(tgt.shape[0], tgt.device)
            x_t = self.q_sample(tgt, t, mask=mask)
        return x_t, t

    def forward(self, item_rep, item_tag, mask_seq, mask_tag):
        item_rep = self.ag_encoder(item_rep, mask_seq)
        x_t, t = self.independent_diffuse(item_tag, mask_tag, self.independent_diffusion)
        
        # Apply CFG dropout during training
        if self.training and self.cfg_scale != 1.0:
            mask_cfg = torch.rand([item_rep.shape[0], 1, 1], device=item_rep.device) > 0.1
            item_rep_cond = torch.where(mask_cfg, item_rep, torch.zeros_like(item_rep))
            denoised_seq = self.net(item_rep_cond, x_t, t, mask_seq, mask_tag)
        else:
            denoised_seq = self.net(item_rep, x_t, t, mask_seq, mask_tag)
            
        # Weighted MSE loss
        eps = 1e-9
        mask_weight = mask_tag.unsqueeze(-1) / (mask_tag.sum(1, keepdim=True).unsqueeze(-1) + eps)
        losses = F.mse_loss(denoised_seq, item_tag, reduction='none') * mask_weight
        losses = losses.sum(1).mean()
        return denoised_seq, losses

    def denoise_sample(self, seq, tgt, mask_seq, mask_tag):
        seq = self.ag_encoder(seq, mask_seq)
        batch_size, seq_len, hidden_size = seq.shape
        
        # Start from random noise
        noise_x_t = torch.randn_like(tgt)
        
        # Reverse diffusion process
        for i in range(self.num_timesteps - 1, -1, -1):
            t = torch.tensor([i] * batch_size, device=seq.device)
            t_expanded = t.unsqueeze(1).repeat(1, seq_len)
            noise_x_t = self.p_sample(seq, noise_x_t, t_expanded, mask_seq, mask_tag)
            
        return noise_x_t

class ADRecBase(object):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                          help='Size of embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=64,
                          help='Size of hidden vectors.')
        parser.add_argument('--diffusion_steps', type=int, default=100,
                          help='Number of diffusion steps.')
        parser.add_argument('--noise_schedule', type=str, default='linear',
                          help='Noise schedule: linear or cosine.')
        parser.add_argument('--schedule_sampler_name', type=str, default='uniform',
                          help='Schedule sampler name.')
        parser.add_argument('--loss_lambda', type=float, default=1.0,
                          help='Weight for MSE loss.')
        parser.add_argument('--lambda_uncertainty', type=float, default=1e-3,
                          help='Lambda for uncertainty.')
        parser.add_argument('--cfg_scale', type=float, default=1.0,
                          help='Classifier-free guidance scale.')
        parser.add_argument('--independent', type=int, default=0,
                          help='Whether to use independent diffusion.')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.max_his = args.history_max
        
        # Embeddings
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        self.embed_dropout = nn.Dropout(getattr(args, 'emb_dropout', 0.2))
        
        # Layer normalization
        try:
            from utils.myutils import LayerNorm
            self.hist_norm = LayerNorm(self.emb_size)
        except Exception:
            self.hist_norm = nn.LayerNorm(self.emb_size)
        
        # ADRec module
        self.diffu = ADRec(args)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_lambda = getattr(args, 'loss_lambda', 1.0)
        
        self.apply(self.init_weights)

    def forward(self, feed_dict):
        self.check_list = []
        i_ids = feed_dict['item_id']  # [batch_size, -1]
        history = feed_dict['history_items']  # [batch_size, history_max]
        lengths = feed_dict['lengths']  # [batch_size]
        batch_size, seq_len = history.shape

        # Prepare sequence and target for per-step prediction
        # Use history[:, :-1] as input and history[:, 1:] as target
        seq = history[:, :-1]  # [batch_size, seq_len-1]
        tgt = history[:, 1:]   # [batch_size, seq_len-1]
        
        # Get embeddings
        seq_emb = self.i_embeddings(seq)
        tgt_emb = self.i_embeddings(tgt)
        seq_emb = self.embed_dropout(seq_emb)
        seq_emb = self.hist_norm(seq_emb)
        
        # Create masks
        mask_seq = (seq > 0).float()
        mask_tgt = (tgt > 0).float()

        if self.training:
            # Training: use diffusion and CE loss
            out_seq, mse_loss = self.diffu(seq_emb, tgt_emb, mask_seq, mask_tgt)
            
            # For prediction, use the last position output
            last_out = out_seq[:, -1, :]  # [batch_size, hidden_size]
            i_vectors = self.i_embeddings(i_ids)
            prediction = (last_out[:, None, :] * i_vectors).sum(-1)
            
            # Calculate CE loss for the last position
            ce_loss = self.ce_loss(prediction, tgt[:, -1])
            total_loss = ce_loss + self.loss_lambda * mse_loss
            
            return {
                'prediction': prediction, 
                'loss': total_loss,
                'ce_loss': ce_loss,
                'mse_loss': mse_loss
            }
        else:
            # Inference: denoise sample for prediction
            out_seq = self.diffu.denoise_sample(seq_emb, tgt_emb, mask_seq, mask_tgt)
            last_out = out_seq[:, -1, :]  # [batch_size, hidden_size]
            i_vectors = self.i_embeddings(i_ids)
            prediction = (last_out[:, None, :] * i_vectors).sum(-1)
            
            return {'prediction': prediction}

class ADRec(SequentialModel, ADRecBase):
    reader = 'SeqReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'hidden_size', 'diffusion_steps', 'loss_lambda']

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
            return {'prediction': out_dict['prediction'], 'loss': out_dict['loss']}
        else:
            return {'prediction': out_dict['prediction']}

class ADRecImpression(ImpressionSeqModel, ADRecBase):
    reader = 'ImpressionSeqReader'
    runner = 'ImpressionRunner'
    extra_log_args = ['emb_size', 'hidden_size', 'diffusion_steps', 'loss_lambda']

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
                'loss': out_dict['loss'],
                'u_v': out_dict.get('u_v'),
                'i_v': out_dict.get('i_v')
            }
        else:
            return {
                'prediction': out_dict['prediction'],
                'u_v': out_dict.get('u_v'),
                'i_v': out_dict.get('i_v')
            }