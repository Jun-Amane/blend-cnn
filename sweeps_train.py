import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
import wandb
import pprint

from ALKA import ALKA
from dataset import AlkaDataset

wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project="alka",
    name=f"experiment_15-Nov-22:08",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "batchsize": 256,
        "dataset": "AlkaSet",
        "epochs": 50,
    })
# sweeps for wandb
sweep_config = {
    'method': 'random'
}
metric = {
    'name': 'val_loss',
    'goal': 'minimize'
}

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
    },
    'batch_size': {
        'values': [32, 64, 128, 256]
    },
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 1e-4
    },
    'weight_decay': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 1e-5
    },
    'text_CNN_num_classes': {
        'values': [128, 256, 512, 1024, 2048]
    },
    'dropout': {
        'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    },
}

parameters_dict.update({
    'epochs': {
        'value': 10}
})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="alka")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.text_CNN_num_classes, config.dropout)
        optimizer = build_optimizer(network=network, optimizer=config.optimizer, learning_rate=config.learning_rate, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})


def build_dataset(batch_size):
    data_tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4355, 0.3777, 0.2879), (0.2653, 0.2124, 0.2194))])

    alka_set = AlkaDataset('../dataset/102flowers', transform=data_tf)

    train_set_len = len(alka_set)

    print(f"Train Data Length: {train_set_len}")

    # Preparing the DataLoader
    train_loader = DataLoader(dataset=alka_set, batch_size=batch_size, shuffle=True)

    return train_loader


def build_network(text_CNN_num_classes, dropout):
    num_classes = 102
    net_obj = ALKA(num_classes=num_classes, dropout=dropout, text_CNN_num_classes=text_CNN_num_classes)

    return net_obj.to(device)


def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def train_epoch(net_obj, train_loader, optimizer):
    # Some training settings
    total_train_step = 0
    loss_fn = nn.CrossEntropyLoss()

    # Training
    net_obj.train()
    cumu_loss = 0
    for images, captions, labels in train_loader:
        images = images.to(device)
        captions = captions.to(device)
        labels = labels.to(device)
        outputs = net_obj(images, captions)
        loss = loss_fn(outputs, labels)

        cumu_loss += loss.item()

        # Optimizing
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        # if total_train_step % 100 == 0:
        print(f"Training Step: {total_train_step}, Loss: {loss.item()}")
        # writer.add_scalar("train_loss", loss.item(), total_train_step)
        wandb.log({"train_loss": loss.item()})

    return cumu_loss / len(train_loader)

