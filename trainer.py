import numpy as np
import torch
import torch.optim as optim
from utils import strokes2rgb
from tqdm import tqdm
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, model, data_loader, val_loader, tb_writer, checkpoint_dir="logs/", learning_rate=0.0001, wkl=1.0, eta_min=0.0, R=0.99999, KLmin=0.2, clip_val=1.0):
        self.model = model
        self.data_loader = data_loader
        self.validation_loader = val_loader
        self.tb_writer = tb_writer
        self.enc_opt = optim.Adam(
            self.model.encoder.parameters(), lr=learning_rate)
        self.dec_opt = optim.Adam(
            self.model.decoder.parameters(), lr=learning_rate)
        assert os.path.exists(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.wkl = wkl
        self.clip_val = clip_val
        self.epoch = 0
        self.mininum_loss = 1.0  # from this loss, the trainer save models
        self.eta_min = eta_min
        self.KLmin = KLmin
        self.R = R
        self.step_per_epoch = len(
            self.data_loader.dataset) / self.data_loader.batch_size

    def train(self, epoch):
        self.save_checkpoint(path=self.checkpoint_dir + "init.pth", msg=dict())
        for e in range(epoch):
            print(f"\n - Training {e}")
            x = None
            step_in_epoch = 0
            losses = [0 for i in range(6)]
            for i, data in tqdm(enumerate(self.data_loader), total=len(self.data_loader)):
                x, lengths = data[0].to(device), data[1].to(device)
                x = x.permute(1, 0, 2)
                batch_losses = self.train_on_batch(x, step_in_epoch)
                losses = [losses[i] + batch_losses[i] for i in range(6)]
                step_in_epoch += 1
            self.epoch += 1

            losses = [losses[i] / step_in_epoch for i in range(6)]

            # Train losses plot
            print("Training losses: ", losses)
            self.tb_writer.add_scalar("train/Loss", losses[0], self.epoch)
            self.tb_writer.add_scalar("train/Ls", losses[1], self.epoch)
            self.tb_writer.add_scalar("train/Lp", losses[2], self.epoch)
            self.tb_writer.add_scalar("train/Lr", losses[3], self.epoch)
            self.tb_writer.add_scalar("train/Lkl", losses[4], self.epoch)
            self.tb_writer.add_scalar("train/weighted_Lkl", losses[5], self.epoch)
            self.tb_writer.add_scalars('train/tradeoff', {'Lkl': losses[4],
                                                                 'Lr': losses[3]}, self.epoch)

            val_losses = self.validate()
            print("Validation losses: ", val_losses)
            self.tb_writer.add_scalar("val/Loss", val_losses[0], self.epoch)
            self.tb_writer.add_scalar("val/Ls", val_losses[1], self.epoch)
            self.tb_writer.add_scalar("val/Lp", val_losses[2], self.epoch)
            self.tb_writer.add_scalar("val/Lr", val_losses[3], self.epoch)
            self.tb_writer.add_scalar("val/Lkl", val_losses[4], self.epoch)
            self.tb_writer.add_scalar("val/weighted_Lkl", val_losses[5], self.epoch)

            # Save model
            if self.mininum_loss > val_losses[0]:
                print (f"New best: {val_losses[0]}")
                self.mininum_loss = val_losses[0]
                self.save_checkpoint(path=self.checkpoint_dir+f"best_model.pth",
                                     msg={"epoch": epoch, "losses": losses, "val_losses": val_losses})

            if e % 10 == 9:
                self.save_checkpoint(path=self.checkpoint_dir + f"checkpoint_{e}.pth",
                                     msg={"epoch": epoch, "losses": losses, "val_losses": val_losses})

            # Reconstruction plots
            x = x[:, 0, :].unsqueeze(1)
            original = x
            self.tb_writer.add_text(
                'reconstruction/original', str(original), self.epoch)
            self.tb_writer.add_image(
                "reconstruction/original", strokes2rgb(original), self.epoch)

            recon = self.model.reconstruct(x)
            self.tb_writer.add_text(
                'reconstruction/prediction', str(recon), self.epoch)
            self.tb_writer.add_image(
                "reconstruction/prediction", strokes2rgb(recon), self.epoch)

            self.tb_writer.flush()
            print(f"- Done {e}")

    def train_on_batch(self, x, step_in_epoch=0):
        self.model.encoder.train()
        self.model.encoder.zero_grad()
        self.model.decoder.train()
        self.model.decoder.zero_grad()

        loss, ls, lp, lr, lkl, weighted_lkl = self.loss_on_batch(x, step_in_epoch)
        loss.backward()

        torch.nn.utils.clip_grad_value_(
            self.model.encoder.parameters(), self.clip_val)
        torch.nn.utils.clip_grad_value_(
            self.model.decoder.parameters(), self.clip_val)

        self.enc_opt.step()
        self.dec_opt.step()

        return [x.detach().cpu().item() for x in (loss, ls, lp, lr, lkl, weighted_lkl)]

    def validate(self):
        self.model.encoder.eval()
        self.model.decoder.eval()
        step_in_epoch = 0
        losses = [0 for i in range(6)]
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.validation_loader), total=len(self.validation_loader)):
                x, lengths = data[0].to(device), data[1].to(device)
                x = x.permute(1, 0, 2)
                batch_losses = self.loss_on_batch(x)
                losses = [losses[i] + batch_losses[i].detach().cpu().item() for i in range(6)]
                step_in_epoch += 1
        #  losses = [loss, Ls, Lp, Lr, Lkl, weighted_Lkl]
        losses = [losses[i] / step_in_epoch for i in range(6)]
        losses[5] = losses[4] * self.wkl
        losses[0] = losses[3] + losses[5]
        return losses



    def loss_on_batch(self, x, step_in_epoch=0):
        batch_size = x.shape[1]

        z, mu, sigma_hat = self.model.encoder(x)

        sos = torch.stack(
            [torch.tensor([0, 0, 1, 0, 0], device=device, dtype=torch.float)]*batch_size).unsqueeze(0)
        dec_input = torch.cat([sos, x[:-1, :, :]], 0)
        h0, c0 = torch.split(
            torch.tanh(
                self.model.decoder.fc_hc(z)
                ),
            self.model.decoder.dec_hidden_size, 1)
        hidden_cell = (h0.unsqueeze(0).contiguous(),
                       c0.unsqueeze(0).contiguous())

        (pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy,
         q), _ = self.model.decoder(dec_input, z, hidden_cell)

        zero_out = 1 - x[:, :, 4]
        Ls = ls(x[:, :, 0], x[:, :, 1],
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, zero_out)
        Lp = lp(x[:, :, 2], x[:, :, 3],
                x[:, :, 4], q)
        Lr = Ls + Lp
        Lkl = lkl(mu, sigma_hat, self.KLmin)

        step = step_in_epoch + self.step_per_epoch * self.epoch
        eta = 1.0 - (1.0 - self.eta_min) * self.R**step
        weighted_Lkl = self.wkl * eta * Lkl
        loss = Lr + weighted_Lkl
        return loss, Ls, Lp, Lr, Lkl, weighted_Lkl

    def save_checkpoint(self, path, msg: dict):
        print(f"Saving model to {path}")
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'encoder_opt': self.enc_opt.state_dict(),
            'decoder_opt': self.dec_opt.state_dict(),
            **msg
        }, path)

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        for k in checkpoint.keys():
            if k not in ['encoder_state_dict', 'decoder_state_dict', 'encoder_opt', 'decoder_opt']:
                print(k, checkpoint[k])
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.enc_opt.load_state_dict(checkpoint['encoder_opt'])
        self.dec_opt.load_state_dict(checkpoint['decoder_opt'])


def ls(x, y, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, zero_out):
    Nmax = x.shape[0]
    batch_size = x.shape[1]

    pdf_val = torch.sum(pi * pdf_2d_normal(x, y, mu_x, mu_y,
                                           sigma_x, sigma_y, rho_xy), dim=2)

    return -torch.sum(zero_out * torch.log(pdf_val + 1e-5)) \
        / (Nmax * batch_size)


def lp(p1, p2, p3, q):
    p = torch.cat([p1.unsqueeze(2), p2.unsqueeze(2), p3.unsqueeze(2)], dim=2)
    return -torch.sum(p*torch.log(q + 1e-4)) \
        / (q.shape[0] * q.shape[1])


def lkl(mu, sigma, KLmin=0.2):
    lkl = -torch.sum(1+sigma - mu**2 - torch.exp(sigma)) \
        / (2. * mu.shape[0] * mu.shape[1])

    KLmin = torch.tensor(KLmin, device=device)

    return torch.max(lkl, KLmin)


def pdf_2d_normal(x, y, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    norm1 = x - mu_x
    norm2 = y - mu_y
    sxsy = sigma_x * sigma_y

    z = (norm1/(sigma_x + 1e-4))**2 + (norm2/(sigma_y + 1e-4))**2 -\
        (2. * rho_xy * norm1 * norm2 / (sxsy + 1e-4))

    neg_rho = 1 - rho_xy**2
    result = torch.exp(-z/(2.*neg_rho + 1e-5))
    denom = 2. * np.pi * sxsy * torch.sqrt(neg_rho + 1e-5) + 1e-5
    result = result / denom
    return result
