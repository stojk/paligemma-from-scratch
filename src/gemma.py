import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def rotate_with_ROPE(sin, cos, x, head_dim):
    index = torch.arange(0,head_dim).reshape(2,-1).transpose(0,1).reshape(-1)
    revert_index = torch.arange(0,head_dim).reshape(-1,2).T.flatten()

    # make order of features to x1,x128,...,x127,x254
    # numbers are current indices 1-based
    ee = x[:,:,:,index]

    # reorder -x2,x1,...,-xd,xd-1, numers are current indices 1-based
    new_order = [i + 1 if i % 2 == 0 else i - 1 for i in range(ee.shape[-1])]
    re = ee[:, :, :, new_order]
    re[:, :, :, 0::2] = -re[:, :, :, 0::2]

    # compute the full rotated embedding
    full_rotated_mat = re * sin + ee * cos

    # revert the order of features
    full_rotated_mat = full_rotated_mat[:,:,:,revert_index]

    return full_rotated_mat


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gains = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x**2, -1, keepdim=True) + self.eps
        )
        x = x / rms * (1 + self.gains)
        return x


class GemmaAttention(nn.Module):
    def __init__(self, dim, n_heads, num_key_value_heads, rms_norm_eps):
        super(GemmaAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.num_key_value_heads = num_key_value_heads

        self.q_proj = nn.Linear(dim, dim, bias=False) # dim x dim
        self.k_proj = nn.Linear(dim, self.head_dim * num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(dim, self.head_dim * num_key_value_heads, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rmsn_pre = RMSNorm(dim, eps=rms_norm_eps)

    def forward(self, x, att_mask, sin_vec, cos_vec):
        bsz, seq_len = x.shape[0], x.shape[1]
        # apply rmsn
        x_norm = self.rmsn_pre(x)
        # apply q,k,v projections
        q = self.q_proj(x_norm)
        k = self.k_proj(x_norm)
        v = self.v_proj(x_norm)
        # reshape to bsz, seq, n_heads, head_dim and reorder to bsz, n_heads, seq, head_dim
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # rotate q and k with ROPE
        q_rot = rotate_with_ROPE(sin_vec, cos_vec, q, self.head_dim)
        k_rot = rotate_with_ROPE(sin_vec, cos_vec, k, self.head_dim)
        # compute the attention matrix (broadcasting num_key_value_heads in k to n_heads in q)
        att_mat = q_rot @ k_rot.transpose(-2, -1) / np.sqrt(self.head_dim)
        # apply causal attention mask
        att_mat_masked = att_mat + att_mask
        # compute attention scores
        att_scores = F.softmax(att_mat_masked, dim=-1)
        # compute attention output
        att_output = att_scores @ v
        # reorder axes from bsz, n_heads, seq_len, head_dim to bsz, seq_len, n_heads, head_dim
        att_output = att_output.permute(0, 2, 1, 3)
        # reshape to bsz, seq_len, dim
        att_output = att_output.reshape(att_output.shape[0], att_output.shape[1], self.dim)
        # apply output projection layer
        att_output = self.o_proj(att_output)
        
        # apply residual connection
        x = x + att_output

        return x

class GemmaMLP(nn.Module):
    def __init__(self, dim, fc_intermediate_size, rms_norm_eps):
        super(GemmaMLP, self).__init__()
        self.rmsn_post = RMSNorm(dim, eps=rms_norm_eps)
        self.act = nn.GELU(approximate="tanh")
        self.W_proj = nn.Linear(dim, fc_intermediate_size, bias=False)
        self.V_proj = nn.Linear(dim, fc_intermediate_size, bias=False)
        self.mlp_out_proj = nn.Linear(fc_intermediate_size, dim, bias=False)

    def forward(self, x):
        # apply RMSNorm
        x_norm = self.rmsn_post(x)
        # apply GeGLU MLP
        mlp_out = self.mlp_out_proj(
            self.act(self.W_proj(x_norm)) * self.V_proj(x_norm)
        )
        # apply residual connection
        x = x + mlp_out
        return x

class GemmaLayer(nn.Module):
    def __init__(self, dim, n_heads, num_key_value_heads, rms_norm_eps, fc_intermediate_size):
        super(GemmaLayer, self).__init__()
        self.attention = GemmaAttention(dim, n_heads, num_key_value_heads, rms_norm_eps)
        self.mlp = GemmaMLP(dim, fc_intermediate_size, rms_norm_eps)

    def forward(self, x, att_mask, sin_vec, cos_vec):
        x = self.attention(x, att_mask, sin_vec, cos_vec)
        x = self.mlp(x)
        return x


class Gemma(nn.Module):

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        num_key_value_heads: int,
        fc_intermediate_size: int,
        vocab_size: int,
        rms_norm_eps: float,
        max_position_embeddings: int,
        with_embedding: bool = True,
    ):
        super(Gemma, self).__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.n_layers = n_layers
        self.max_position_embeddings = max_position_embeddings

        if with_embedding:
            self.embedding_layer = nn.Embedding(vocab_size, dim)
        else:
            self.embedding_layer = None

        self.layers = nn.ModuleList(
            [
                GemmaLayer(
                    dim,
                    n_heads,
                    num_key_value_heads,
                    rms_norm_eps,
                    fc_intermediate_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_rmsn = RMSNorm(dim, eps=rms_norm_eps)
        self.out_lin_projection = nn.Linear(dim, vocab_size, bias=False)
        self.angles = torch.Tensor(
            [10000 ** (-2 * i / self.head_dim) for i in range(self.head_dim // 2)]
        )

    def forward(self, x):
        # embedding layer
        if self.embedding_layer is not None:
            x = self.embedding_layer(x)

        # normalize
        normalizer = torch.tensor(self.dim**0.5, dtype=x.dtype, device=x.device)
        x = x * normalizer

        # create attention mask
        seq_len = x.shape[1]
        att_mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0)
        att_mask = att_mask.unsqueeze(0).unsqueeze(0)
        att_mask[att_mask == False] = float("-inf")

        # create sin and cos vectors
        cos_vec = torch.Tensor(
            [
                [torch.cos(angle * m), torch.cos(angle * m)]
                for m in range(1, seq_len + 1)
                for angle in self.angles
            ],
            device=x.device,
        ).reshape(seq_len, -1)
        sin_vec = torch.Tensor(
            [
                [torch.sin(angle * m), torch.sin(angle * m)]
                for m in range(1, seq_len + 1)
                for angle in self.angles
            ],
            device=x.device,
        ).reshape(seq_len, -1)

        # apply layers
        for i in range(self.n_layers):
            x = self.layers[i].attention(
                x, att_mask, sin_vec, cos_vec)
            x = self.layers[i].mlp(x)

        # output layer
        x = self.out_rmsn(x)
        x = self.out_lin_projection(x)

        return x

    def load_hf_weights(self, weights):
        # embedding layer
        if self.embedding_layer is not None:
            self.embedding_layer.weight.data = weights['model.embed_tokens.weight']

        # decoder layers
        for i in range(0, self.n_layers):
            # self attention
            self.layers[i].attention.q_proj.weight.data = weights[f"model.layers.{i}.self_attn.q_proj.weight"]
            self.layers[i].attention.k_proj.weight.data = weights[f"model.layers.{i}.self_attn.k_proj.weight"]
            self.layers[i].attention.v_proj.weight.data = weights[f"model.layers.{i}.self_attn.v_proj.weight"]
            self.layers[i].attention.o_proj.weight.data = weights[f"model.layers.{i}.self_attn.o_proj.weight"]
            self.layers[i].attention.rmsn_pre.gains.data = weights[f"model.layers.{i}.input_layernorm.weight"]
            # MLP
            self.layers[i].mlp.W_proj.weight.data = weights[f"model.layers.{i}.mlp.gate_proj.weight"]
            self.layers[i].mlp.V_proj.weight.data = weights[f"model.layers.{i}.mlp.up_proj.weight"]
            self.layers[i].mlp.mlp_out_proj.weight.data = weights[f"model.layers.{i}.mlp.down_proj.weight"]
            self.layers[i].mlp.rmsn_post.gains.data = weights[f"model.layers.{i}.post_attention_layernorm.weight"]

        # output layer
        self.out_rmsn.gains.data = weights["model.norm.weight"]
        self.out_lin_projection.weight.data = weights["lm_head.weight"]
    