import torch
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, learning_rate):
        self.model = model
        self.enc_opt = optim.Adam(
            self.mode.encoder.parameters(), lr=learning_rate)
        self.dec_opt = optim.Adam(
            self.mode.decoder.parameters(), lr=learning_rate)

    def train(self, X, epoch):
        for e in range(epoch):
            for x in X:
                self.train_on_batch(x)

    def train_on_batch(self, x):
        self.mode.encoder.zero_grad()
        self.mode.decoder.zero_grad()

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

        Ls = ls(dec_input[:, :, 0], dec_input[:, :, 1],
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy)
        Lp = lp(dec_input[:, :, 2], dec_input[:, :, 3], dec_input[:, :, 4], q)
        Lr = Ls + Lp

        Lkl = lkl(mu, sigma_hat)
        return Lr + self.wkl * Lkl
