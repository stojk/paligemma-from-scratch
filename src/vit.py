import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Only differences to ViT:
#  - CLS token is not added
#  - There is no output layer (either FC or classification layer)

# We also need to change the load_weights method to match the SigLip model weights' keys


class ViT(nn.Module):

    def __init__(
        self,
        dim: int,
        n_channels: int,
        n_layers: int,
        n_heads: int,
        image_size: int,
        patch_size: int,
        fc_intermediate_size: int,
        norm_eps: float,
        out_head: Optional[str] = None,
        cls_token: bool = False,
    ):
        super(ViT, self).__init__()
        self.dim = dim
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.fc_intermediate_size = fc_intermediate_size
        self.norm_eps = norm_eps
        self.head_dim = dim // n_heads
        self.n_patches = image_size // patch_size * image_size // patch_size
        self.out_head = out_head

        self.projection_layer = nn.Conv2d(
            n_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        if cls_token:
            self.cls_token = nn.Parameter(torch.rand(1, 1, dim))
            self.pos_enc = nn.Parameter(torch.rand(1, self.n_patches + 1, dim))
        else:
            self.cls_token = None
            self.pos_enc = nn.Parameter(torch.rand(1, self.n_patches, dim))

        # create torch array for storing layers
        self.layers = nn.ModuleDict()
        for i in range(n_layers):
            ln_before = nn.LayerNorm(dim, eps=self.norm_eps)

            q_layer = nn.Linear(dim, dim)
            k_layer = nn.Linear(dim, dim)
            v_layer = nn.Linear(dim, dim)
            att_fc_out = nn.Linear(dim, dim)

            ln_after = nn.LayerNorm(dim, eps=self.norm_eps)
            fc1 = nn.Linear(dim, fc_intermediate_size)
            act1 = nn.GELU(approximate="tanh")
            fc2 = nn.Linear(fc_intermediate_size, dim)
            act2 = nn.GELU(approximate="tanh")

            self.layers[f"layer_{i}_ln_before"] = ln_before
            self.layers[f"layer_{i}_q"] = q_layer
            self.layers[f"layer_{i}_k"] = k_layer
            self.layers[f"layer_{i}_v"] = v_layer
            self.layers[f"layer_{i}_att_fc_out"] = att_fc_out
            self.layers[f"layer_{i}_ln_after"] = ln_after
            self.layers[f"layer_{i}_FN1"] = fc1
            self.layers[f"layer_{i}_act1"] = act1
            self.layers[f"layer_{i}_FN2"] = fc2
            self.layers[f"layer_{i}_act2"] = act2

        self.out_ln = nn.LayerNorm(dim, eps=self.norm_eps)
        if out_head is not None:
            raise NotImplementedError("Output head not implemented yet")
            # self.out_fc = nn.Linear(dim, dim)
            # self.out_fc = nn.Linear(dim, 1000)
            # self.out_act = nn.Tanh()

    def load_hf_weights(self, weights: dict, prefix: str = "") -> None:
        # Load the embeddings
        self.pos_enc.data = weights[f"{prefix}embeddings.position_embedding.weight"].data
        self.projection_layer.weight.data = weights[f"{prefix}embeddings.patch_embedding.weight"].data
        self.projection_layer.bias.data = weights[f"{prefix}embeddings.patch_embedding.bias"].data

        # Loop over each layer in the model
        for i in range(self.n_layers):
            # Layer normalization before self attention
            self.layers[f"layer_{i}_ln_before"].weight.data = weights[f"{prefix}encoder.layers.{i}.layer_norm1.weight"].data
            self.layers[f"layer_{i}_ln_before"].bias.data = weights[f"{prefix}encoder.layers.{i}.layer_norm1.bias"].data

            # Self attention sub-layers (Query, Key, Value)
            for part in ['q', 'k', 'v']:
                self.layers[f"layer_{i}_{part}"].weight.data = weights[f"{prefix}encoder.layers.{i}.self_attn.{part}_proj.weight"].data
                self.layers[f"layer_{i}_{part}"].bias.data = weights[f"{prefix}encoder.layers.{i}.self_attn.{part}_proj.bias"].data

            # Output projection from self attention
            self.layers[f"layer_{i}_att_fc_out"].weight.data = weights[f"{prefix}encoder.layers.{i}.self_attn.out_proj.weight"].data
            self.layers[f"layer_{i}_att_fc_out"].bias.data = weights[f"{prefix}encoder.layers.{i}.self_attn.out_proj.bias"].data

            # Layer normalization after self attention
            self.layers[f"layer_{i}_ln_after"].weight.data = weights[f"{prefix}encoder.layers.{i}.layer_norm2.weight"].data
            self.layers[f"layer_{i}_ln_after"].bias.data = weights[f"{prefix}encoder.layers.{i}.layer_norm2.bias"].data

            # Feedforward network layers
            self.layers[f"layer_{i}_FN1"].weight.data = weights[f"{prefix}encoder.layers.{i}.mlp.fc1.weight"].data
            self.layers[f"layer_{i}_FN1"].bias.data = weights[f"{prefix}encoder.layers.{i}.mlp.fc1.bias"].data
            self.layers[f"layer_{i}_FN2"].weight.data = weights[f"{prefix}encoder.layers.{i}.mlp.fc2.weight"].data
            self.layers[f"layer_{i}_FN2"].bias.data = weights[f"{prefix}encoder.layers.{i}.mlp.fc2.bias"].data

        # Load final layer normalization weights
        self.out_ln.weight.data = weights[f"{prefix}post_layernorm.weight"].data
        self.out_ln.bias.data = weights[f"{prefix}post_layernorm.bias"].data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch encoding
        x = self.projection_layer(x)
        # reshape to shape (bsz, dim, n_patches)
        x = x.reshape(x.shape[0], self.dim, -1)
        # move embedding axis as last tp (bsz, n_patches, dim)
        x = x.permute(0, 2, 1)
        # append CLS token
        if self.cls_token is not None:
            x = torch.cat([self.cls_token, x], dim=1)
        # add positional encoding
        x = x + self.pos_enc

        # perform MSA blocks
        for i in range(self.n_layers):
            # layer norm before
            ln_x = self.layers[f"layer_{i}_ln_before"](x)
            # compute query, key, values
            q = self.layers[f"layer_{i}_q"](ln_x)
            k = self.layers[f"layer_{i}_k"](ln_x)
            v = self.layers[f"layer_{i}_v"](ln_x)
            # reshape to bsz, n_patches, n_heads, head_dim and permute to bsz, n_heads, n_patches, head_dim
            q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            # compute attention matrix
            att_mat = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
            # compute attention scores
            att_scores = F.softmax(att_mat, dim=-1)
            # compute attention output
            att_output = torch.matmul(att_scores, v)
            # reorder axes from bsz, n_heads, n_patches, head_dim to bsz, n_patches, n_heads, head_dim
            att_output = att_output.permute(0, 2, 1, 3)
            # reshape to bsz, n_patches, dim
            att_output = att_output.reshape(
                att_output.shape[0], att_output.shape[1], self.dim
            )
            # fully-connected layer
            att_fc_output = self.layers[f"layer_{i}_att_fc_out"](att_output)
            # add to residual
            x = x + att_fc_output
            # layer norm after
            ln_x = self.layers[f"layer_{i}_ln_after"](x)
            # fully-connected layers
            fc_inter = self.layers[f"layer_{i}_act1"](
                self.layers[f"layer_{i}_FN1"](ln_x)
            )
            fc_out = self.layers[f"layer_{i}_FN2"](fc_inter)
            # add to residual
            x = x + fc_out

        # layer norm after
        x = self.out_ln(x)
        # fully-connected layers
        if self.out_head is not None:
            # x = self.out_fc(x)
            # x = self.out_act(x)
            raise NotImplementedError("Output head not implemented yet")
        return x
