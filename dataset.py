import numpy as np
import torch


class To5vStrokes():
    def __init__(self, max_len=200):
        self.max_len = max_len

    def __call__(self, stroke):
        """
        Converts from stroke-3(numpy array) to stroke-5(torch tensor) format and pads to given length.
        (But does not insert special start token).
        """
        result = np.zeros((self.max_len, 5), dtype=float)
        l = len(stroke)
        assert l <= self.max_len
        result[0:l, 0:2] = stroke[:, 0:2]
        result[0:l, 3] = stroke[:, 2]
        result[0:l, 2] = 1 - result[0:l, 3]
        result[l:, 4] = 1
        return torch.from_numpy(result).float()


class V5Dataset(torch.utils.data.Dataset):
    def __init__(self, data_array, transform=None, pre_scaling=True):
        super().__init__()
        self.transform = transform
        print("Processing data: ", data_array.shape)
        self.data = [x.astype('float32') for x in data_array]
        self.true_lengths = [x.shape[0] for x in data_array]
        if pre_scaling:
            scale = self.scaling_factor(self.data)
            self.data = [self.scale_stroke(x, scale) for x in self.data]
        if transform is not None:
            for i in range(len(self.data)):
                self.data[i] = transform(self.data[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.true_lengths[index]

    def scaling_factor(self, datalist):
        data = np.concatenate([S for S in datalist])
        return np.std(data[:, 0:2])

    def scale_stroke(self, x, scale):
        x = np.float32(x)
        x[:, 0:2] /= scale
        return x


def load_quickdraw_datasets(path_to_npz):
    a = np.load(path_to_npz, encoding='latin1', allow_pickle=True)
    trainset = V5Dataset(a['train'], To5vStrokes(), pre_scaling=True)
    testset = V5Dataset(a['test'], To5vStrokes(), pre_scaling=True)
    valset = V5Dataset(a['valid'], To5vStrokes(), pre_scaling=True)

    return trainset, testset, valset