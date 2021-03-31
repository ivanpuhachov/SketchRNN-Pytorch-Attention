import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import SketchRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SketchRNNAttention(SketchRNN):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=2048, Nz=128, M=20, tau=1.0, dropout=0.1):
        self.encoder = EncoderAttention(enc_hidden_size, Nz, dropout=dropout).to(device)
        self.decoder = DecoderAttention(dec_hidden_size, Nz, M, tau, dropout=dropout).to(device)


class EncoderAttention(nn.Module):
    def __init__(self, enc_hidden_size=512, Nz=128, dropout=0.1, seq_len=200):
        super().__init__()
        self.input_size = 5
        self.seq_len = seq_len
        self.hidden_size = enc_hidden_size
        self.encoder_rnn = nn.LSTM(
            5, enc_hidden_size, dropout=dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*enc_hidden_size, Nz)
        self.fc_sigma = nn.Linear(2*enc_hidden_size, Nz)
        self.fc_out_weights = nn.Linear(2*enc_hidden_size, 1)

    def forward(self, inputs):
        seq_len, batch, input_size = inputs.shape
        # print("enc inputs: ", inputs.size())
        assert seq_len == self.seq_len
        outs, (hidden, cell) = self.encoder_rnn(inputs)
        h_forward, h_backward = torch.split(hidden, 1, 0)
        h = torch.cat([h_forward.squeeze(0), h_backward.squeeze(0)], 1)

        out_weights = self.fc_out_weights(outs)
        out_weighted = torch.sum(torch.mul(outs, out_weights), dim=0)
        # print("out_weighted: ",  out_weighted.shape)
        #
        # print("h: ", h.size())
        mu = self.fc_mu(out_weighted + h)
        # print("mu: ", mu.size())
        sigma_hat = self.fc_sigma(out_weighted + h)
        sigma = torch.exp(sigma_hat/2.)

        N = torch.normal(torch.zeros(mu.size(), device=device),
                         torch.ones(mu.size(), device=device))
        z = mu + sigma * N
        # print("z: ", z.shape)
        return z, mu, sigma_hat


class DecoderAttention(nn.Module):
    def __init__(self, dec_hidden_size=2048, Nz=128, M=20, tau=1.0, dropout=0.1, Ne=20):
        super().__init__()
        self.M = M
        self.Nz = Nz
        self.Ne = Ne
        self.dec_hidden_size = dec_hidden_size
        self.fc_hc = nn.Linear(Nz, 2*dec_hidden_size)
        self.decoder_rnn = nn.LSTM(Nz+5, dec_hidden_size, dropout=dropout)
        self.decoder_cell = nn.LSTMCell(input_size=Nz+5, hidden_size=self.dec_hidden_size)
        self.fc_y = nn.Linear(dec_hidden_size, 6*M+3)
        self.tau = tau
        self.out_to_emb = nn.Linear(in_features=dec_hidden_size, out_features=Ne)

    def forward(self, x, z, hidden_cell=None):
        Nmax, batch_size, input_size = x.shape
        zs = torch.stack([z] * Nmax)
        dec_input = torch.cat([x, zs], 2)

        if hidden_cell is not None:
            h, c = hidden_cell
            if len(h.shape) > 2:
                h.squeeze_(0)
                c.squeeze_(0)
        else:
            h, c = torch.zeros(size=(batch_size, self.hidden_size)), torch.zeros(size=(batch_size, self.dec_hidden_size))

        outs = list()
        for i in range(Nmax):
            h, c = self.decoder_cell(dec_input[i], (h, c))
            outs.append(h)

        # o, (h, c) = self.decoder_rnn(dec_input, hidden_cell)
        o = torch.stack(outs)
        y = self.fc_y(o)

        pi_hat, mu_x, mu_y, sigma_x_hat, sigma_y_hat, rho_xy, q_hat = torch.split(
            y, self.M, 2)
        pi = F.softmax(pi_hat, dim=2)
        sigma_x = torch.exp(sigma_x_hat) * np.sqrt(self.tau)
        sigma_y = torch.exp(sigma_y_hat) * np.sqrt(self.tau)
        rho_xy = torch.clip(torch.tanh(rho_xy), min=-1+1e-6, max=1-1e-6)
        q = F.softmax(q_hat, dim=2)
        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), (h, c)
