import torch
import torch.nn as nn
from vit import ViT
from gemma import Gemma

class PaliGemma(nn.Module):
    def __init__(
        self,
        gemma_dim: int,
        gemma_n_layers: int,
        gemma_n_heads: int,
        gemma_num_key_value_heads: int,
        gemma_fc_intermediate_size: int,
        gemma_vocab_size: int,
        gemma_rms_norm_eps: float,
        gemma_max_position_embeddings: int,
        vit_dim: int,
        vit_n_channels: int,
        vit_n_layers: int,
        vit_n_heads: int,
        vit_image_size: int,
        vit_patch_size: int,
        vit_fc_intermediate_size: int,
        vit_norm_eps: float,
    ) -> None:
        super(PaliGemma, self).__init__()
        self.gemma = Gemma(
            dim = gemma_dim,
            n_layers = gemma_n_layers,
            n_heads = gemma_n_heads,
            num_key_value_heads = gemma_num_key_value_heads,
            fc_intermediate_size = gemma_fc_intermediate_size,
            vocab_size = gemma_vocab_size,
            rms_norm_eps = gemma_rms_norm_eps,
            max_position_embeddings = gemma_max_position_embeddings,
            with_embedding = False,
        )
        self.vit = ViT(
            dim=vit_dim,
            n_channels=vit_n_channels,
            n_layers=vit_n_layers,
            n_heads=vit_n_heads,
            image_size=vit_image_size,
            patch_size=vit_patch_size,
            fc_intermediate_size=vit_fc_intermediate_size,
            norm_eps=vit_norm_eps,
        )
        self.visual_embedding_to_text_embedding = torch.nn.Linear(
            vit_dim,
            gemma_dim,
            bias=True
        )
        self.text_embedding_layer = nn.Embedding(
            gemma_vocab_size,
            gemma_dim
        )

    def load_hf_weights(
        self,
        gemma_weights: dict,
        vit_weights: dict,
        multi_modal_projector_weights: dict,
    ) -> None:
        self.gemma.load_hf_weights(gemma_weights)
        self.vit.load_hf_weights(vit_weights)
        self.visual_embedding_to_text_embedding.weight.data = (
            multi_modal_projector_weights["linear.weight"]
        )
        self.visual_embedding_to_text_embedding.bias.data = (
            multi_modal_projector_weights["linear.bias"]
        )
        self.text_embedding_layer.weight.data = gemma_weights[
            "model.embed_tokens.weight"
        ]

    def forward(self, img: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:

        # get visual embeddings
        visual_emb = self.vit(img.unsqueeze(0))

        # project visual embeddings to text embeddings
        visual_emb = self.visual_embedding_to_text_embedding(visual_emb)

        # denormalize visual embeddings
        normalizer = torch.tensor(self.gemma.dim**0.5)
        visual_emb = visual_emb / normalizer

        # create text embeddings
        text_emb = self.text_embedding_layer(tokens)

        # concatenate visual and text embeddings
        multi_modal_emb = torch.cat((visual_emb, text_emb), dim=1)

        # pass through gemma
        gemma_out = self.gemma(multi_modal_emb)

        return gemma_out
