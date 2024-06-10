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
        self.name = 'PdMformer'
        self.n_feats = configs.enc_in

        self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.dwt = Dwt(configs)

        self.dimension = configs.d_level + 3

        self.pos_encoder = PositionalEmbedding(self.dimension * configs.enc_in)

        # Encoder
        self.dwt_attention1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.seq_len, configs.n_heads),
                    configs.seq_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(configs.seq_len)
            # norm_layer=torch.nn.BatchNorm1d(configs.seq_len, eps=1e-5)
        )

        self.attn_r1 = nn.Linear(in_features=configs.seq_len, out_features=configs.seq_len)
        self.attn_r2 = nn.Linear(in_features=configs.seq_len, out_features=configs.seq_len)
        self.tanh = nn.Tanh()
        self.attn_r3 = nn.Linear(in_features=configs.seq_len, out_features=1)

        self.dwt_attention2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.seq_len, configs.n_heads),
                    configs.seq_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(configs.seq_len)
            # norm_layer=torch.nn.BatchNorm1d(configs.seq_len, eps=1e-5)
        )

        self.attn_ad1 = nn.Linear(in_features=configs.seq_len, out_features=configs.seq_len)
        self.attn_ad2 = nn.Linear(in_features=configs.seq_len, out_features=configs.seq_len)
        self.attn_ad3 = nn.Linear(in_features=configs.seq_len, out_features=1)

        self.dwt_attention3 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.seq_len, configs.n_heads),
                    configs.seq_len,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(configs.seq_len)
            # norm_layer=torch.nn.BatchNorm1d(configs.seq_len, eps=1e-5)
        )

        self.attn_pm1 = nn.Linear(in_features=configs.seq_len, out_features=configs.seq_len)
        self.attn_pm2 = nn.Linear(in_features=configs.seq_len, out_features=configs.seq_len)
        self.attn_pm3 = nn.Linear(in_features=configs.seq_len, out_features=1)

        # Encoder
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        self.dimension * configs.enc_in, configs.n_heads),
                    self.dimension * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)

            ],
            norm_layer=torch.nn.LayerNorm(self.dimension * configs.enc_in)
            # norm_layer=torch.nn.BatchNorm1d(configs.d_model, eps=1e-5)
        )
        # Encoder
        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        self.dimension * configs.enc_in, configs.n_heads),
                    self.dimension * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)

            ],
            norm_layer=torch.nn.LayerNorm(self.dimension * configs.enc_in)
            # norm_layer=torch.nn.BatchNorm1d(configs.d_model, eps=1e-5)
        )
        # Decoder
        self.decoder1 = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        self.dimension * configs.enc_in, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        self.dimension * configs.enc_in, configs.n_heads),
                    self.dimension * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dimension * configs.enc_in),
            projection=nn.Linear(self.dimension * configs.enc_in, configs.c_out, bias=True)
        )
        # Decoder
        self.decoder2 = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        self.dimension * configs.enc_in, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        self.dimension * configs.enc_in, configs.n_heads),
                    self.dimension * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dimension * configs.enc_in),
            projection=nn.Linear(self.dimension * configs.enc_in, configs.c_out, bias=True)
        )

        # Encoder
        self.encoder_predict = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        self.dimension * configs.enc_in, configs.n_heads),
                    self.dimension * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)

            ],
            norm_layer=torch.nn.LayerNorm(self.dimension * configs.enc_in)
            # norm_layer=torch.nn.BatchNorm1d(configs.d_model, eps=1e-5)
        )

        # Decoder
        self.decoder_predict = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, attention_dropout=configs.dropout, output_attention=False),
                        self.dimension * configs.enc_in, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        self.dimension * configs.enc_in, configs.n_heads),
                    self.dimension * configs.enc_in,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.dimension * configs.enc_in),
            projection=nn.Linear(self.dimension * configs.enc_in, configs.d_model, bias=True)
        )

        self.act = self._get_activation_fn(configs.activation)

        self.dropout = nn.Dropout(configs.dropout)

        self.fc1 = nn.Linear(configs.d_model * configs.seq_len, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, len(utils.global_var.get_value('class_names')))


    def encode(self, src, c, dwt_attention, encoder, phase):

        sp = self.dwt(src)

        # with anomaly score
        # tgt = torch.cat((tgt, c, tgt_sp), dim=2)
        # tgt = torch.cat((tgt, c), dim=2)

        # without anomaly score
        tgt = torch.cat(
            (src, src, sp.reshape([sp.shape[0], sp.shape[1], sp.shape[2] * sp.shape[3]])), dim=2)
        # tgt = torch.cat((tgt, tgt), dim=2)


        h, _ = dwt_attention(sp.permute(0, 2, 3, 1).reshape([sp.shape[0], sp.shape[2] * sp.shape[3], sp.shape[1]]))

        if phase == 1:

            z1 = self.attn_r1(h)

            z2 = self.attn_r2(sp.permute(0, 2, 3, 1).reshape([sp.shape[0], sp.shape[2] * sp.shape[3], sp.shape[1]]).to(torch.float32))

            embed_x = z1 + z2

            z3 = self.attn_r3(self.tanh(embed_x))

            z3 = z3.reshape([sp.shape[0], sp.shape[2], sp.shape[3]])

        elif phase == 2:

            z1 = self.attn_ad1(h)

            z2 = self.attn_ad2(
                sp.permute(0, 2, 3, 1).reshape([sp.shape[0], sp.shape[2] * sp.shape[3], sp.shape[1]]).to(torch.float32))

            embed_x = z1 + z2

            z3 = self.attn_ad3(self.tanh(embed_x))

            z3 = z3.reshape([sp.shape[0], sp.shape[2], sp.shape[3]])

        elif phase == 3:

            z1 = self.attn_pm1(h)

            z2 = self.attn_pm2(
                sp.permute(0, 2, 3, 1).reshape([sp.shape[0], sp.shape[2] * sp.shape[3], sp.shape[1]]).to(torch.float32))

            embed_x = z1 + z2

            z3 = self.attn_pm3(self.tanh(embed_x))

            z3 = z3.reshape([sp.shape[0], sp.shape[2], sp.shape[3]])

        batch_size = sp.size(0)

        if batch_size > 1:
            attn_w = F.softmax(z3.view(batch_size, self.n_feats, 5), dim=2)
        else:
            attn_w = self.init_variable(batch_size, self.n_feats, 5) + 1

        weighted_sp = torch.mul(attn_w.unsqueeze(1), sp)

        src = torch.cat((src, c[:, :src.shape[1], :], weighted_sp.reshape(
            [weighted_sp.shape[0], weighted_sp.shape[1], weighted_sp.shape[2] * weighted_sp.shape[3]])), dim=2)
        # src = torch.cat((src, c[:, :src.shape[1], :]), dim=2)

        src = src * math.sqrt(self.n_feats)

        src = self.pos_encoder(src)

        memory, attns = encoder(src)

        return tgt, memory, attn_w

    def forward(self, src):
        src = src.to(self.device)

        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)

        tgt1, memory1, attn_w1 = self.encode(src, c, self.dwt_attention1, self.encoder1, 1)

        x1 = self.decoder1(torch.tensor(tgt1, dtype=torch.float32), memory1).to(self.device)

        # Phase 2 - With anomaly scores
        x1_detach = x1.detach()

        c = (x1_detach - src) ** 2

        tgt2, memory2, attn_w2 = self.encode(src, c, self.dwt_attention2, self.encoder2, 2)
        x2 = self.decoder2(torch.tensor(tgt2, dtype=torch.float32), memory2).to(self.device)

        # c = (x1_detach - tgt) ** 2
        c = (x2 - src) ** 2

        tgt3, memory3, attn_w3 = self.encode(src, c, self.dwt_attention3, self.encoder_predict, 3)
        output = self.decoder_predict(torch.tensor(tgt3, dtype=torch.float32), memory3).to(self.device)


        # Output
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)

        output = self.act(self.fc1(output))
        output = self.act(self.fc2(output))
        output = self.fc3(output)

        return x1, x2, output, attn_w1, attn_w2, attn_w3

    def _get_activation_fn(self, activation):

        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise ValueError("activation should be relu/gelu, not {}".format(activation))