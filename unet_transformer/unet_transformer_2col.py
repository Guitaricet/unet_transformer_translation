"""
UnetTransformerModel with two columns in the encoder: vanilla transformer and U-Net
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import TransformerDecoder, TransformerEncoderLayer
from fairseq.models.fairseq_encoder import EncoderOut

from unet_transformer.unet_transformer_layer import UNetTransformerEncoderLayer
from unet_transformer.unet_transformer import UnetTransformerModel, UNetTransformerEncoder

# import modified by fairseq layers (they have different initialization)
from unet_transformer.unet_transformer import Linear, Embedding


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model("unet_transformer2col")
class UnetTransformer2ColModel(UnetTransformerModel):
    """
    UnetTransformerModel with two columns in the encoder: vanilla transformer and U-Net

    This class is a slightly modified version of TransformerModel class
    (only relpaced the encoder class and removed encoder-layerdrop and encoder_normalize_before parameters)

    Args:
        encoder (UnetTransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return UNetTransformer2ColEncoder(args, src_dict, embed_tokens)


class UNetTransformer2ColEncoder(UNetTransformerEncoder):
    """Uses both U-Net and Transformer encoder. Concatenates the last hidden states."""

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if args.encoder_layers % 2:
            raise ValueError("number of layers shoud be divisible by 2")

        # Transformer Encoder boilerplate

        embed_dim = embed_tokens.embedding_dim  # same as args.encoder_embed_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        return_all_hiddens = getattr(args, "return_all_hiddens", False)
        if return_all_hiddens:
            raise ValueError("UnetTransformer2Col does not support returning all hiddens")

        self.num_layers = args.encoder_layers
        self.dropout = args.dropout
        if getattr(args, "layer_wise_attention", False):
            raise ValueError("UNetTransformer does not support layer-wise attention")

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        # U-Net Transformer Encoder
        model_dim, ffn_hidden, n_heads = (
            embed_dim,
            args.encoder_ffn_embed_dim,
            args.encoder_attention_heads,
        )

        unet_dict = self.build_unet_stacks(args, model_dim, ffn_hidden, n_heads)

        self.input_layer = unet_dict["input_layer"]
        self.down_layers = nn.ModuleList(unet_dict["down_layers"])
        self.up_layers = nn.ModuleList(unet_dict["up_layers"])
        self.output_layer = unet_dict["output_layer"]

        assert self.num_layers == 2 + len(self.down_layers) + len(self.up_layers)

        # Vanilla transformer Encoder
        self.transformer_layers = nn.ModuleList([TransformerEncoderLayer(args) for _ in range(args.encoder_layers)])

        # More Transformer Encoder boilerplate
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward(
        self,
        src_tokens,
        src_lengths,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens)
        x = x.transpose(0, 1)  # B x T x C -> T x B x C

        # if not return_all hiddens, encoder states are expected to be an empty list
        # and we do not support encoder hiddens, but need to satisfy the interface
        encoder_states = []

        # U-Net part:
        x_unet = x
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        x_unet = self.forward_unet(x_unet, encoder_padding_mask)

        # Transformer part:
        x_transformer = x
        for layer in self.transformer_layers:
            x_transformer = layer(x_transformer, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x_transformer)

        # Combine U-Net representations and Transformer representations
        x, _ = torch.stack([x_transformer, x_unet], dim=0).max(0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=src_lengths,
        )


# Note that the number of parameters will roughly be ~1.7 of the number of parameters of BASE transformer
# because we have two encoders working in parallel and the U-Net encoder is bigger because of the convolutions
@register_model_architecture("unet_transformer2col", "unet_transformer2col_base")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 448)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1792)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
