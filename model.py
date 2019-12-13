import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SketchRNN():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()


class Encoder(nn.Module):
    def __init__(self, enc_hidden_size=256, Nz=128, dropout=0.9):
        super().__init__()
        self.encoder_rnn = nn.LSTM(
            5, enc_hidden_size, dropout=dropout, bidirectional=True)
        self.fc_mu = nn.Linear(2*enc_hidden_size, Nz)
        self.fc_sigma = nn.Linear(2*enc_hidden_size, Nz)

    def forward(self, inputs):
        _, (hidden, cell) = self.encoder_rnn(inputs)
        h = torch.cat((hidden[0, :, :], hidden[1, :, :]), 1)

        mu = self.fc_mu(h)
        sigma_hat = self.fc_sigma(h)
        sigma = torch.exp(sigma_hat/2.)

        N = torch.normal(torch.zeros(mu.size()),
                         torch.ones(mu.size()))
        z = mu + sigma * N
        return z, mu, sigma_hat


class Decoder(nn.Module):
    def __init__(self, dec_hidden_size=256, Nz=128, M=20, dropout=0.9):
        super().__init__()
        self.M = M
        self.dec_hidden_size = dec_hidden_size
        self.fc_hc = nn.Linear(Nz, 2*dec_hidden_size)
        self.decoder_rnn = nn.LSTM(Nz+5, dec_hidden_size, dropout=dropout)
        self.fc_y = nn.Linear(dec_hidden_size, 6*M+3)

    def forward(self, s, z):
        h0, c0 = torch.split(torch.tanh(self.fc_hc(z)),
                             self.dec_hidden_size, 1)
        h0c0 = (h0.unsqueeze(0), c0.unsqueeze(0))
        z = z.unsqueeze(0)
        s = s.unsqueeze(0)

        inputs = torch.cat((s, z), 2)
        o, (h, c) = self.decoder_rnn(inputs, h0c0)
        y = self.fc_y(o)

        pi = F.softmax(y[0, :, 0:self.M-1], 1)
        mu_x = y[0, :, self.M:2*self.M-1]
        mu_y = y[0, :, 2*self.M:3*self.M-1]
        sigma_x = torch.exp(y[0, :, 3*self.M:4*self.M])
        sigma_y = torch.exp(y[0, :, 4*self.M:5*self.M])
        rho_xy = torch.tanh(y[0, :, 5*self.M:6*self.M])
        q = F.softmax(y[0, :, 6*self.M:6*self.M+3], 1)

        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q), (h, c)
