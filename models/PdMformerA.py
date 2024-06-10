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
        self.name = 'PdMformerA'
        self.n_feats = configs.enc_in

        self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.pos_encoder = PositionalEmbedding(2 * configs.enc_in)


        # Encoder
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        2 * configs.enc_in, configs.n_heads),
                    2 * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)

            ],
            norm_layer=torch.nn.LayerNorm(2 * configs.enc_in)
            # norm_layer=torch.nn.BatchNorm1d(configs.d_model, eps=1e-5)
        )
        # Encoder
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        2 * configs.enc_in, configs.n_heads),
                    2 * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)

            ],
            norm_layer=torch.nn.LayerNorm(2 * configs.enc_in)
            # norm_layer=torch.nn.BatchNorm1d(configs.d_model, eps=1e-5)
        )
        # Decoder
        self.decoder1 = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        2 * configs.enc_in, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        2 * configs.enc_in, configs.n_heads),
                    2 * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(2 * configs.enc_in),
            projection=nn.Linear(2 * configs.enc_in, configs.c_out, bias=True)
        )
        # Decoder
        self.decoder2 = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        2 * configs.enc_in, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        2 * configs.enc_in, configs.n_heads),
                    2 * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(2 * configs.enc_in),
            projection=nn.Linear(2 * configs.enc_in, configs.c_out, bias=True)
        )

        # Encoder
        self.encoder_predict = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        2 * configs.enc_in, configs.n_heads),
                    2 * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)

            ],
            norm_layer=torch.nn.LayerNorm(2 * configs.enc_in)
            # norm_layer=torch.nn.BatchNorm1d(configs.d_model, eps=1e-5)
        )

        # Decoder
        self.decoder_predict = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        2 * configs.enc_in, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        2 * configs.enc_in, configs.n_heads),
                    2 * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(2 * configs.enc_in),
            projection=nn.Linear(2 * configs.enc_in, configs.d_model, bias=True)
        )

        self.act = self._get_activation_fn(configs.activation)

        self.dropout = nn.Dropout(configs.dropout)

        self.fc1 = nn.Linear(configs.d_model * configs.seq_len, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, len(utils.global_var.get_value('class_names')))


    def encode(self, src, c, dwt_attention, encoder, phase):

        # with anomaly score
        # tgt = torch.cat((src, c), dim=2)

        # without anomaly score
        tgt = torch.cat((src, src), dim=2)


        src = torch.cat((src, c[:, :src.shape[1], :]), dim=2)


        src = src * math.sqrt(self.n_feats)


        src = self.pos_encoder(src)


        memory, attns = encoder(src)


        return tgt, memory

    def forward(self, src):


        src = src.to(self.device)

        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)

        tgt1, memory1 = self.encode(src, c, None, self.encoder1, 1)

        x1 = self.decoder1(torch.tensor(tgt1, dtype=torch.float32), memory1).to(self.device)


        # Phase 2 - With anomaly scores
        x1_detach = x1.detach()

        c = (x1_detach - src) ** 2

        tgt2, memory2 = self.encode(src, c, None, self.encoder2, 2)
        x2 = self.decoder2(torch.tensor(tgt2, dtype=torch.float32), memory2).to(self.device)


        # c = (x1_detach - tgt) ** 2
        c = (x2 - src) ** 2

        tgt3, memory3 = self.encode(src, c, None, self.encoder_predict, 3)
        output = self.decoder_predict(torch.tensor(tgt3, dtype=torch.float32), memory3).to(self.device)

        # Output
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        output = self.act(self.fc1(output))
        output = self.act(self.fc2(output))
        output = self.fc3(output)


        return x1, x2, output

    def _get_activation_fn(self, activation):

        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise ValueError("activation should be relu/gelu, not {}".format(activation))