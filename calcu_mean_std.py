import torch
from torchvision.datasets import ImageFolder
from dataset import AlkaDataset
import torchvision
from tqdm import tqdm


def getStat(train_data):
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    tqdm_epoch = tqdm(train_loader)
    for X, _, _ in tqdm_epoch:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    alka_set = AlkaDataset('../dataset/wheat_img', transform=torchvision.transforms.ToTensor(), load_to_ram=False)
    train_dataset = alka_set
    print(getStat(train_dataset))

