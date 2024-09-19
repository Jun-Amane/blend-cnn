import os
import torch
import torchvision
from torchvision.transforms import v2, InterpolationMode
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
import wandb
import pprint
from tqdm import notebook
from tqdm import tqdm
import numpy as np
import random

from ALKA import ALKA
from dataset import AlkaDataset

from util import *

training_device = "cpu"

os.environ["WANDB_MODE"] = "online"

config = {
    "model": "alka-master",
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "dropout": 0.2,
    "heads": 8,
    "batch_size": 16,
    "dataset": "AlkaSet",
    "epochs": 10,
}

def build_dataset(batch_size: int):
    data_tf = v2.Compose([
        v2.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((320, 320)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4355, 0.3777, 0.2879), (0.2653, 0.2124, 0.2194))])
    alka_set = AlkaDataset('../dataset/102flowers', transform=data_tf, load_to_ram=False)
    train_ratio = 0.8
    dataset_size = len(alka_set)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_set, val_set = random_split(alka_set, [train_size, test_size])

    train_set_len = len(train_set)
    val_set_len = len(val_set)

    print(f"Train Data Length: {train_set_len}")
    print(f"Validation Data Length: {val_set_len}")

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

    return alka_set, train_loader, val_loader


def build_model(word2idx, dropout: float):
    embeddings = load_pretrained_vectors(word2idx, "../data/crawl-300d-2M.vec")
    embeddings = torch.tensor(embeddings)
    num_classes = 102
    net_obj = ALKA(num_classes=num_classes, dropout=dropout, pretrained_embedding=embeddings, kernel_size=wandb.config.kernel_size).to(
        training_device)

    pytorch_total_params = sum(p.numel() for p in net_obj.parameters())
    pytorch_trainable_params = sum(p.numel() for p in net_obj.parameters() if p.requires_grad)
    print(f"TOTAL PARAMS OF ALKA: {pytorch_total_params}")
    print(f"TRAINABLE PARAMS OF ALKA: {pytorch_trainable_params}")

    return net_obj


def build_criterion():
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn


def build_optimizer(model, learning_rate: float, weight_decay: float):
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    return optimiser


def train_epoch(model, train_loader, criterion, optimiser, cur_epoch: int):
    train_set_len = len(train_loader.dataset)
    total_accuracy = 0.0

    model.train()
    tqdm_epoch = tqdm(train_loader)
    for images, captions, labels in tqdm_epoch:
        images = images.to(training_device)
        captions = captions.to(training_device)
        labels = labels.to(training_device)
        outputs = model(images, captions)
        loss = criterion(outputs, labels)

        # Optimizing
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        accuracy = topk_accuracy(outputs, labels)
        tqdm_epoch.set_description("Training Epoch: %d" % cur_epoch)
        tqdm_epoch.set_postfix(loss=loss.item(), accuracy=str(accuracy / 16 * 100)+'%')
        wandb.log({"train_loss": loss.item()})



def val_epoch(model, val_loader, criterion, cur_epoch: int):
    total_step_loss = 0.0
    steps_per_epoch = 0
    total_accuracy = 0.0
    total_top5_accuracy = 0.0
    val_set_len = len(val_loader.dataset)

    model.eval()
    tqdm_epoch = tqdm(val_loader)
    with torch.no_grad():
        for images, captions, labels in tqdm_epoch:
            images = images.to(training_device)
            captions = captions.to(training_device)
            labels = labels.to(training_device)
            outputs = model(images, captions)
            loss = criterion(outputs, labels)

            total_step_loss += loss.item()
            steps_per_epoch += 1
            accuracy = topk_accuracy(outputs, labels)
            top5_accuracy = topk_accuracy(outputs, labels, k=5)
            total_accuracy += accuracy
            total_top5_accuracy += top5_accuracy
            tqdm_epoch.set_description("Validation Epoch: %d" % cur_epoch)
            tqdm_epoch.set_postfix(loss=loss.item(), accuracy=str(accuracy / 16 * 100)+'%',
                                   top5_accuracy=str(top5_accuracy / 16 * 100)+'%')

        print(f"Total loss on dataset: {total_step_loss / steps_per_epoch}")
        print(f"Top-1 accuracy on dataset: {total_accuracy / val_set_len}")
        print(f"Top-5 accuracy on dataset: {total_top5_accuracy / val_set_len}")
        wandb.log({"acc@1": total_accuracy / val_set_len, "acc@5": total_top5_accuracy / val_set_len,
                   "val_loss": total_step_loss / steps_per_epoch})

        return (total_accuracy / val_set_len)




def train():
    with wandb.init(config=config):
        alka_set, train_loader, val_loader = build_dataset(batch_size=wandb.config.batch_size)
        net_obj = build_model(alka_set.word2idx, wandb.config.dropout)
        criterion = build_criterion()
        optimiser = build_optimizer(net_obj, wandb.config.learning_rate, wandb.config.weight_decay)

        epoch = wandb.config.epochs
        best_acc = 0.0
        for i in range(epoch):
            train_epoch(net_obj, train_loader, criterion, optimiser, i + 1)
            acc = val_epoch(net_obj, val_loader, criterion, i + 1)
            if acc > best_acc:
                best_acc = acc
                wandb.log({"best_acc": best_acc})

        print(f"Best Accuracy: {best_acc}")


if __name__ == "__main__":
    set_seed(999)

    # sweeps for wandb
    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'best_acc',
       'goal': 'maximize'
    }

    sweep_config['metric'] = metric

    parameters_dict = {
        'kernel_size': {
            'values': [
                [3, 4, 5, 6],
                [5, 6, 7, 8],
                [12, 14, 16, 18],
                [22, 24, 26, 28],
                [32, 34, 36, 38],
                [42, 44, 46, 48]
            ]
        }
    }

    sweep_config['parameters'] = parameters_dict

    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="alka")
    wandb.agent(sweep_id, function=train, count=6)

