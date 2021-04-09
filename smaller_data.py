import numpy as np

name = "cat"

a = np.load(f"data/{name}.npz", encoding='latin1', allow_pickle=True)

np.savez_compressed(f"data/{name}_small.npz",
                    train=a['train'][:10000],
                    valid=a['valid'],
                    test=a['test'])