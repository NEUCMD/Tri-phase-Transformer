import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer, DwtEncoderLayer
from layers.selfAttention_Family import FullAttention, AttentionLayer
from layers.embed import DataEmbedding, PositionalEmbedding
from layers.dwt import Dwt
import utils.global_var


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.name = 'Transformer'
        self.n_feats = configs.enc_in

        self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 1, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, len(utils.global_var.get_value('class_names')))


    def forward(self, src):

        x1 = src
        x2 = src

        # Embedding
        enc_out = self.enc_embedding(src, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None);

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)

        return x1, x2, output

    def _get_activation_fn(self, activation):

        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise ValueError("activation should be relu/gelu, not {}".format(activation))