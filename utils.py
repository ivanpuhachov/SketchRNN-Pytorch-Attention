import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ls(x, y, pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, Ns):
    Nmax = x.shape[0]
    batch_size = x.shape[1]

    pdf_val = sum(pi * pdf_2d_normal(x, y, mu_x, mu_y,
                                     sigma_x, sigma_y, rho_xy), dim=2)

    # make zero_out
    zero_out = torch.cat([torch.ones(Ns[0], device=device),
                          torch.zeros(Nmax - Ns[0], device=device)]).unsqueeze(1)
    for i in range(1, batch_size):
        zeros = torch.cat([torch.ones(Ns[i], device=device),
                           torch.zeros(Nmax - Ns[i], device=device)]).unsqueeze(1)
        zero_out = torch.cat([zero_out, zeros], dim=1)

    return -torch.sum(zero_out * torch.log(pdf_val + 1e-5)) \
        / float(Nmax)


def lp(p1, p2, p3, q):
    p = torch.Tensor([p1, p2, p3], device=device)
    return -torch.sum(p*torch.log(q + 1e-5)) \
        / (q.shape[0] * q.shape[1])


def lkl(mu, sigma):
    return -torch.sum(1+sigma - mu**2 - torch.exp(sigma)) \
        / (2. * mu.shape[1] * mu.shape[2])


def pdf_2d_normal(x, y, mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    norm1 = x - mu_x
    norm2 = y - mu_y
    sxsy = sigma_x * sigma_y

    z = (norm1/sigma_x)**2 + (norm2/sigma_y)**2 -\
        (2. * rho_xy * norm1 * norm2 / sxsy)

    neg_rho = 1 - rho_xy**2
    result = torch.exp(-z/(2.*neg_rho))
    denom = 2. * np.pi * sxsy * neg_rho**2
    result = result / denom
    return result


def ns(x):
    Ns = []
    for i in range(x.shape[1]):
        Ns.append(endindex(x[:, i, :]))

    return Ns


def endindex(x):
    Nmax = x.shape[0]
    for i in range(Nmax):
        if x[i, 0, 4] == 1:
            return i
    return Nmax-1
