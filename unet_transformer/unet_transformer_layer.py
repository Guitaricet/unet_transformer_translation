from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)


class UNetTransformerEncoderLayer(nn.Module):
    """
    Args:
        type_: 'up', 'down', 'same', 'none' - controls convolution and shape change
        args (argparse.Namespace): parsed command-line arguments
            input_dim - input size (int)
            model_dim - attention size (same as output size) (d_out == d_model)
            ffn_hidden - hiidden size of FFN
            Required arguments:
                type_ - (UNet-specific) 'up', 'down' or 'same'
                skip_connection - (UNet-specific) use skip-connection
                encoder_attention_heads
                attention_dropout
                dropout
            Optional arguments:
                activation_fn
                activation_dropout
    """

    def __init__(
        self,
        args,
        type_,
        input_dim=None,
        model_dim=None,
        ffn_hidden=None,
        conv_skip_connection=False,
        depthwise_conv=False,
    ):
        super().__init__()

        self.input_dim = input_dim or args.encoder_embed_dim
        self.model_dim = model_dim or args.encoder_embed_dim
        self.ffn_hidden = ffn_hidden or args.encoder_ffn_embed_dim
        self.conv_skip_connection = conv_skip_connection
        self.depthwise_conv = depthwise_conv

        self.type_ = type_
        self.conv = None

        # shrinks down input in "down" layers
        # not used in "same", but used in "up" for computing the key_padding_mask
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # use groups=1 for fast computation, use self.input_dim to be closer to the original paper
        groups = 1
        if self.depthwise_conv:
            groups = self.input_dim

        if type_ == "up":
            # double size of output
            self.conv = nn.ConvTranspose1d(
                self.input_dim, self.input_dim, kernel_size=3, stride=2, padding=1, groups=groups
            )
        elif type_ == "down":
            # half size of output
            self.conv = nn.Conv1d(
                self.input_dim, self.input_dim, kernel_size=3, padding=1, groups=groups
            )
        elif type_ == "same":
            # keep size of output the same
            self.conv = nn.Conv1d(
                self.input_dim, self.input_dim, kernel_size=3, padding=1, groups=groups
            )
        elif type_ == "none":
            raise NotImplementedError()
        else:
            raise ValueError(f"type_ should be one of: 'up', 'down', same'. Got '{type_}' instead")

        # we need conv_out to be able to keep convolution input_dim == output_dim
        # which is needed for the depthwise convolutions (gropus==input_dim == output_dim)
        self.conv_out = nn.Linear(self.input_dim, self.model_dim)
        self.conv_norm = LayerNorm(self.model_dim)

        self.self_attn_layer_norm = LayerNorm(self.model_dim)

        self.self_attn = MultiheadAttention(
            self.model_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False,
            kdim=self.input_dim,
            vdim=self.input_dim,
        )

        self.dropout = args.dropout

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        self.fc1 = Linear(self.model_dim, self.ffn_hidden)
        self.fc2 = Linear(self.ffn_hidden, self.model_dim)
        self.final_layer_norm = LayerNorm(self.model_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, key_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            key_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
                src_len = n_steps if type='same'
                src_len = n_steps if type='down' and it maxpooled inside fowrard call
                src_len = n_steps * 2 (+1) if type='up'
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
                T_tgt is the length of query, while T_src is the length of key,
                though here both query and key is x here,
                attn_mask[t_tgt, t_src] = 1 means when calculating embedding
                for t_tgt, t_src is excluded (or masked out), =0 means it is
                included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        seq_len, batch_size, embed_dim = x.shape
        input_kv = x  # used for keys and values in self-attention

        # Maybe Convolution (double or half over time axis)

        if self.conv is not None:
            residual = x  # (seq_len, batch, embed_dim)
            x = x.permute(1, 2, 0)  # (batch, embed_dim, seq_len)

            if self.type_ != "up":
                x = self.conv(x)
            else:
                # the main problem with 'up' layer is that we don't know the actual output size
                # e.g. if input seq_len=4, corresponding down layer may had the input seq_len= 8 or 9
                # key_padding_mask is supposed to have the right seq_len
                output_size = None
                if key_padding_mask is not None:
                    output_size = (batch_size, embed_dim, key_padding_mask.shape[1])

                x = self.conv(x, output_size=output_size)

            x = self.conv_out(x.transpose(1, 2)).transpose(1, 2)  # input_dim -> model_dim

            if self.type_ == "down":
                x = self.maxpool(x.contiguous())  # (batch_size, model_dim, seq_len / 2)
                # key_padding_mask = self._get_next_mask(key_padding_mask)

            x = x.permute(2, 0, 1)  # (seq_len / 2, batch, embed_dim)

            # only possible if type_=='same'
            x = residual + x if self.conv_skip_connection else x
            x = self.conv_norm(x)

        # Self-attention

        residual = x
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        _key_padding_mask = key_padding_mask
        if self.type_ == "up":
            # in this case, key_padding_mask represents the mask over post-deconvs
            # but in the attention we need a mask over pre-deconvs
            _key_padding_mask = self._get_shrinked_mask(key_padding_mask)

        x, _ = self.self_attn(
            query=x,
            key=input_kv,
            value=input_kv,
            key_padding_mask=_key_padding_mask,
            attn_mask=attn_mask,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # FFN
        if self.type_ == "down":
            key_padding_mask = self._get_shrinked_mask(key_padding_mask)

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.transpose(0, 1).unsqueeze(-1), 0.0)

        assert (
            x.shape[0] == key_padding_mask.shape[1]
        ), f"output seq_len {x.shape[0]} != padding seq_len {key_padding_mask.shape[1]}"
        assert (
            x.shape[1] == key_padding_mask.shape[0]
        ), f"output batch_size {x.shape[1]} != padding batch_size {key_padding_mask.shape[0]}"

        return x, key_padding_mask

    def _get_shrinked_mask(self, pad_mask):
        """
        Computes a mask of the post-conv inputs based on the pre-conv inputs mask

        :param pad_mask: torch.BoolTensor of shape (batch_size, seq_len)
        :param layer:
        :return:
        """
        pad_mask = pad_mask.unsqueeze(2).transpose(1, 2)
        non_pad_mask = self.maxpool((~pad_mask).float().contiguous()).bool()  # ~ is logical NOT
        pad_mask = ~non_pad_mask.squeeze(1)
        return pad_mask


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
