import torch
from torchvision.datasets import ImageFolder
from dataset import AlkaDataset
from cub200 import CUB200
import torchvision


def getStat(train_data):
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _, _, basename in train_loader:
        if X.shape[1] != 3:
            print(basename)
            continue
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    alka_set = CUB200('../dataset/cub200/cub200', transform=torchvision.transforms.ToTensor(), load_to_ram=False)
    train_dataset = alka_set
    print(getStat(train_dataset))

