import torch
import torch.optim as optim
from utils import ls, lp, lkl, ns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, data_loader, learning_rate=0.0001, wkl=1.0):
        self.model = model
        self.data_loader = data_loader
        self.enc_opt = optim.Adam(
            self.model.encoder.parameters(), lr=learning_rate)
        self.dec_opt = optim.Adam(
            self.model.decoder.parameters(), lr=learning_rate)
        self.wkl = wkl

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
        z, mu, sigma_hat = self.model.encoder(x)

        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
         q), _ = self.model.decoder(x, z)

        Ns = ns(x)
        Ls = ls(x[:, :, 0], x[:, :, 1],
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, Ns)
        Lp = lp(x[:, :, 2], x[:, :, 3],
                x[:, :, 4], q)
        Lr = Ls + Lp

        Lkl = lkl(mu, sigma_hat)
        return Lr + self.wkl * Lkl
