import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from einops import rearrange


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        batch_size = len(record_len)
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1)
            h, attn = self.att(xx, xx, xx)
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...].unsqueeze(0)
            attn = attn.permute(1, 2, 0).view(cav_num, cav_num, W, H)
            out.append(h)
        return torch.cat(out, dim=0), attn

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


class AgentWiseFusion(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super(AgentWiseFusion, self).__init__()
        inner_dim = heads * dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len)
        C, H, W = split_x[0].shape[1:]
        outs = []
        for xx in split_x:
            B = xx.shape[0]
            xx = rearrange(xx, 'b c h w -> b h w c')
            # [ (b h w d) * 3]
            qkv = self.to_qkv(xx).chunk(3, dim=-1)
            # q: (B, M, H, W, L, C)
            q, k, v = map(
                lambda t: rearrange(t, 'b h w c -> b (h w c)'), qkv)
            # attn: b, b
            att_map = torch.einsum('i c, j c  -> i j',
                                   q, k) * self.scale
            # softmax
            att_map = self.attend(att_map)
            out = torch.einsum('i j, j c ->i c',
                               att_map,
                               v)
            out = rearrange(out, 'b (h w c) -> b h w c', h=H, w=W, c=C)
            out = self.to_out(out)[0].unsqueeze(0)
            out = rearrange(out, 'b h w c -> b c h w')
            outs.append(out)
        return torch.cat(outs, dim=0), att_map

