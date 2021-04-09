import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import SketchRNN, Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SketchRNNAttention(SketchRNN):
    def __init__(self, enc_hidden_size=512, dec_hidden_size=2048, Nz=128, M=20, dropout=0.1):
        self.encoder = EncoderAttention(enc_hidden_size, Nz, dropout=dropout).to(device)
        self.decoder = DecoderAttention(dec_hidden_size, Nz, M, dropout=dropout).to(device)

    def reconstruct(self, S, tau=1.0):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            Nmax = S.shape[0]
            batch_size = S.shape[1]
            assert (batch_size==1)
            s_i = torch.stack(
                [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)] * batch_size, dim=0).unsqueeze(0)
            out_points = s_i
            z, _, _ = self.encoder(S)
            z = z.unsqueeze(0)
            h0, c0 = torch.split(torch.tanh(self.decoder.fc_hc(z)),
                                 self.decoder.dec_hidden_size, 2)
            hidden_cell = (h0.contiguous(),
                           c0.contiguous())
            for i in range(Nmax):
                attention_context = self.decoder.compute_masked_attention_context(out_points)
                last_step_context = attention_context[-1].unsqueeze(0)
                lstm_input = torch.cat([s_i, z, last_step_context], 2)
                y, hidden_cell = self.decoder.lstm_prediction(inp=lstm_input, hidden_cell=hidden_cell)
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = self.decoder.extract_params(y, tau=tau)
                # (pi, mu_x, mu_y, sigma_x, sigma_y,
                #  rho_xy, q), hidden_cell = self.decoder(s_i, z, hidden_cell)
                s_i = self.sample_next(
                    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
                out_points = torch.cat([out_points, s_i], dim=0)
                if out_points[-1, 0, 4] == 1:
                    break

            out_points = out_points[1:, :, :]  # remove dummy
            return out_points


class EncoderAttention(Encoder):
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


class AttentionHead(nn.Module):
    def __init__(self, Ne=64):
        super(AttentionHead, self).__init__()
        self.Ne = Ne
        self.attention_key = nn.Linear(5, self.Ne)
        self.attention_query = nn.Linear(5, self.Ne)
        self.attention_value = nn.Linear(5, self.Ne)

    def forward(self, x):
        seq_len, batch_size, emb_size = x.shape
        keys = self.attention_key(x).permute(1, 0, 2)  # now it is (batch_size, seq_len, Ne)
        queries = self.attention_query(x).permute(1, 2, 0)  # now it is (batch_size, Ne, seq_len)
        values = self.attention_value(x).permute(1, 2, 0)  # now it is (batch_size, Ne, seq_len)

        dot_product = torch.bmm(keys, queries)
        # scale values
        dot_product /= np.sqrt(self.Ne)
        # masking attention weights
        mask = torch.tril(torch.ones((seq_len, seq_len), requires_grad=False, dtype=torch.bool), diagonal=-1)
        dot_product[:, mask] = float('-inf')
        attention_scores = torch.nn.Softmax(dim=1)(dot_product)
        context = torch.bmm(values, attention_scores).permute(2, 0, 1)
        assert (context.shape == (seq_len, batch_size, self.Ne))
        return context


class DecoderAttention(Decoder):
    def __init__(self, dec_hidden_size=2048, Nz=128, M=20, dropout=0.1, Ne=64, n_attention_heads=2):
        super().__init__()
        self.M = M
        self.Nz = Nz
        self.Ne = Ne
        self.dec_hidden_size = dec_hidden_size
        self.fc_hc = nn.Linear(Nz, 2*dec_hidden_size)
        self.decoder_rnn = nn.LSTM(5+Nz+n_attention_heads*Ne, dec_hidden_size, dropout=dropout)
        self.fc_y = nn.Linear(dec_hidden_size, 6*M+3)
        self.n_attention_heads = n_attention_heads
        self.attention_heads = nn.ModuleList([
            AttentionHead(Ne=self.Ne)
            for _ in range(self.n_attention_heads)
        ])
        self.out_to_emb = nn.Linear(in_features=dec_hidden_size, out_features=Ne)

    def compute_masked_attention_context(self, x):
        all_contexts = [head(x) for head in self.attention_heads]
        context = torch.cat(all_contexts, dim=2)
        assert (context.shape[2]==self.n_attention_heads*self.Ne)
        return context

    def forward(self, x, z, hidden_cell=None, tau=1.0):
        seq_len, batch_size, input_size = x.shape
        zs = torch.stack([z] * seq_len)
        attention_context = self.compute_masked_attention_context(x)
        lstm_input = torch.cat([x, zs, attention_context], 2)

        y, (h, c) = self.lstm_prediction(inp=lstm_input, hidden_cell=hidden_cell)

        return self.extract_params(y, tau=tau), (h, c)
