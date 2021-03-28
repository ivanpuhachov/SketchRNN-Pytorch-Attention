import torch
from dataset import load_quickdraw_datasets
from model import SketchRNN
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

trainset, testset, valset = load_quickdraw_datasets("data/owl.npz")

dataloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=8)

log_dir = 'logs4/'
checkpoint_dir = log_dir + 'checkpoints/'

if os.path.exists(log_dir):
    print("-- Clearing logs folder --")
    shutil.rmtree(log_dir)

os.mkdir(log_dir)
os.mkdir(checkpoint_dir)

tb_writer = SummaryWriter(log_dir)

model = SketchRNN(enc_hidden_size=256, dec_hidden_size=512,
                  Nz=128, M=20, dropout=0.1)
print("CUDA: ", torch.cuda.is_available())

trainer = Trainer(model, dataloader, tb_writer,
                  checkpoint_dir, learning_rate=0.0001)

trainer.train(epoch=20)
tb_writer.close()
