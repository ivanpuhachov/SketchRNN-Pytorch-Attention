import torch
import torch.optim as optim
from utils import ls, lp, lkl, ns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, data_loader, learning_rate=0.0001):
        self.model = model
        self.data_loader = data_loader
        self.enc_opt = optim.Adam(
            self.model.encoder.parameters(), lr=learning_rate)
        self.dec_opt = optim.Adam(
            self.model.decoder.parameters(), lr=learning_rate)

    def train(self, epoch):
        for e in range(epoch):
            for x, _ in self.data_loader:
                x = x.permute(1, 0, 2)
                self.train_on_batch(x)

    def train_on_batch(self, x):
        self.model.encoder.zero_grad()
        self.model.decoder.zero_grad()

        loss = self.loss_on_batch(x)
        loss.backward()

        # grad clip here

        self.enc_opt.step()
        self.dec_opt.step()

    def loss_on_batch(self, x):
        batch_size = x.shape[1]
        z, mu, sigma_hat = self.model.encoder(x)

        sos = torch.stack(
            [torch.Tensor([0, 0, 1, 0, 0], device=device)]*batch_size)
        zs = torch.stack([z] * batch_size)
        dec_input = torch.cat([sos, x[:-1, :, :]], 0)
        dec_input = torch.cat([dec_input, zs], 2)

        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
         q), _ = self.model.decoder(dec_input)

        Ns = ns(x)
        Ls = ls(dec_input[:, :, 0], dec_input[:, :, 1],
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, Ns)
        Lp = lp(dec_input[:, :, 2], dec_input[:, :, 3],
                dec_input[:, :, 4], q)
        Lr = Ls + Lp

        Lkl = lkl(mu, sigma_hat)
        return Lr + self.wkl * Lkl
