import torch
from dataset import load_quickdraw_datasets
from model import SketchRNN
from model_attention import SketchRNNAttention
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import shutil
import argparse
from datetime import datetime
import numpy as np
import json
from shutil import copyfile


def backup_training_files(backup_dir):
    files_to_backup = ["model.py", "model_attention.py", "dataset.py", "trainer.py", "run_experiment.py"]
    for filename in files_to_backup:
        copyfile(filename, backup_dir + filename)


def create_logs_folder(folder_log):
    if os.path.exists(folder_log):
        print(f"logs folder exists {folder_log}")
        ccc = chr(np.random.randint(97, 97 + 26))
        folder_log = folder_log[:-1] + "_" + ccc + "/"
        print(f"Creating another folder: {folder_log}")
    else:
        print("creating logs folder")
    os.mkdir(folder_log)
    os.mkdir(folder_log + "checkpoints/")
    os.mkdir(folder_log + "tensorboard/")
    backup_training_files(folder_log)
    return folder_log, folder_log + "checkpoints/", folder_log + "tensorboard/"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SketchRNN!")
    parser.add_argument("--dataset", "-d", default="data/owl.npz")
    parser.add_argument("--n_epochs", "-n", type=int, default=1, help="number of training epochs")
    parser.add_argument("--batch", "-b", type=int, default=500, help="batch_size")
    parser.add_argument("--enc_h", type=int, default=256, help="encoder hidden size")
    parser.add_argument("--dec_h", type=int, default=512, help="decoder hidden size")
    parser.add_argument("--M", type=int, default=20, help="number of GMMs in the output")
    parser.add_argument("--Nz", type=int, default=128, help="latent vector size")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--R", type=float, default=0.9999, help="hyperparameter R (annealing rate)")
    parser.add_argument("--seed", type=int, default=23, help='random seed')
    parser.add_argument("--workers", type=int, default=8, help='num_workers to use')

    print("\n-- parsing args --")
    args = parser.parse_args()
    print("Received keys:")
    for key in vars(args):
        print(f"{key} \t\t {getattr(args, key)}")

    dataset_path = args.dataset
    batch_size = args.batch
    n_epochs = args.n_epochs
    learning_rate = args.lr
    random_seed = args.seed
    model_enc_h = args.enc_h
    model_dec_h = args.dec_h
    model_M = args.M
    model_Nz = args.Nz
    model_R = args.R
    num_workers = args.workers

    assert (os.path.exists(dataset_path))

    print("\n-- logs and backup  --")
    log_dir = "logs/" + datetime.now().strftime("%m.%d--%H") + f"_b{batch_size}_n{n_epochs}/"
    log_dir, checkpoint_dir, tensorboard_dir = create_logs_folder(folder_log=log_dir)
    print(f"\n\ntensorboard --logdir={tensorboard_dir}")

    with open(log_dir + "cli_args.txt", 'w') as f:
        json.dump(args.__dict__, f)

    # reproducibility
    assert torch.cuda.is_available()
    torch.manual_seed(seed=random_seed)
    np.random.seed(random_seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False

    print("\n-- datasets --")
    trainset, testset, valset = load_quickdraw_datasets(dataset_path)

    dataloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)

    tb_writer = SummaryWriter(tensorboard_dir)

    print("\n-- creating model --")

    model = SketchRNNAttention(enc_hidden_size=model_enc_h, dec_hidden_size=model_dec_h,
                               Nz=model_Nz, M=model_M, dropout=0)

    trainer = Trainer(model, data_loader=dataloader, val_loader=valloader, tb_writer=tb_writer,
                      checkpoint_dir=checkpoint_dir, learning_rate=learning_rate, R=model_R)

    print("\n-- training --")

    trainer.train(epoch=n_epochs)
    tb_writer.close()
