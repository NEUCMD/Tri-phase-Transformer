import torch
import torch.nn as nn

import utils.global_var

class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.name = 'TCN'

        self.device = torch.device('cuda:{}'.format(configs.gpu))

        self.n_class = len(utils.global_var.get_value('class_names'))
        self.n_feats = configs.enc_in
        self.d_model = configs.d_model
        self.n_layers = 4

        self.dropout = configs.dropout

        self.lstm = nn.LSTM(input_size=self.n_feats, hidden_size=self.d_model, num_layers=self.n_layers,
                            batch_first=True)

        self.relu = nn.ReLU(inplace=True)

        self.C1 = nn.Conv1d(self.n_feats, 128, 8)
        self.C2 = nn.Conv1d(128, 256, 5)
        self.C3 = nn.Conv1d(256, 128, 3)

        self.BN1 = nn.BatchNorm1d(128)
        self.BN2 = nn.BatchNorm1d(256)
        self.BN3 = nn.BatchNorm1d(128)

        self.ConvDrop = nn.Dropout(self.dropout)

        self.FC = nn.Linear(128 + self.d_model, self.n_class)

    def forward(self, src):

        batch = src.shape[0]
        h0 = torch.randn(self.n_layers, batch, self.d_model).to(self.device)
        c0 = torch.randn(self.n_layers, batch, self.d_model).to(self.device)

        output, (hn, cn) = self.lstm(src, (h0, c0))
        output = output[:,-1,:]

        output_t = src.transpose(2, 1)

        output_t = self.ConvDrop(self.relu(self.BN1(self.C1(output_t))))
        output_t = self.ConvDrop(self.relu(self.BN2(self.C2(output_t))))
        output_t = self.ConvDrop(self.relu(self.BN3(self.C3(output_t))))

        output_t = torch.mean(output_t, 2)

        x_all = torch.cat((output_t, output), dim=1)
        x_out = self.FC(x_all)

        x1 = src
        x2 = src

        return x1, x2, x_out