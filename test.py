import torch
from dataset import To5vStrokes, V5Dataset
from model import SketchRNN
from trainer import Trainer
from pathlib import Path

data_path = Path.home() / 'MyDatasets/Sketches/apple/test.npy'

dataset = V5Dataset(str(data_path), To5vStrokes(max_len=200))
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=100, shuffle=True)

model = SketchRNN(enc_hidden_size=256, dec_hidden_size=256,
                  Nz=128, M=20, dropout=0.9)
trainer = Trainer(model, dataloader, learning_rate=0.0001)

trainer.train(epoch=3)
