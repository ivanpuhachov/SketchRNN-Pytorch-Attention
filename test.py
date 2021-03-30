import torch
from dataset import load_quickdraw_datasets
from model import SketchRNN
from model_attention import SketchRNNAttention
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import shutil
import numpy as np

random_seed = 23
torch.manual_seed(seed=random_seed)
np.random.seed(random_seed)
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

trainset, testset, valset = load_quickdraw_datasets("data/owl.npz")

dataloader = DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=8)

# valloader = DataLoader(valset, batch_size=100, shuffle=True, num_workers=8)
valloader = DataLoader(valset, batch_size=100, shuffle=True, num_workers=8)

d = next(iter(valloader))
sample = d[0].permute(1,0,2).to("cuda")
print("sample: ", sample.shape)

model = SketchRNNAttention(enc_hidden_size=256, dec_hidden_size=512, Nz=128, M=20, dropout=0)
print("new out: ", model.encoder(sample)[0].shape)

model = SketchRNN(enc_hidden_size=256, dec_hidden_size=512, Nz=128, M=20, dropout=0)
print("old out: ", model.encoder(sample)[0].shape)

def training():
    log_dir = 'testlogs/test_attention/'
    checkpoint_dir = log_dir + 'checkpoints/'
    print(f"tensorboard --logdir={log_dir}")

    if os.path.exists(log_dir):
        print("-- Clearing logs folder --")
        shutil.rmtree(log_dir)

    os.mkdir(log_dir)
    os.mkdir(checkpoint_dir)

    tb_writer = SummaryWriter(log_dir)

    model = SketchRNNAttention(enc_hidden_size=256, dec_hidden_size=512,
                               Nz=128, M=20, dropout=0)
    print("CUDA: ", torch.cuda.is_available())

    trainer = Trainer(model, data_loader=dataloader, val_loader=valloader, tb_writer=tb_writer,
                      checkpoint_dir=checkpoint_dir, learning_rate=0.0001, R=0.9999)

    trainer.train(epoch=10)
    tb_writer.close()

training()