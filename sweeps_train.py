import os
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
from tqdm import notebook
import numpy as np

from ALKA import ALKA
from dataset import AlkaDataset

wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project="alka",
    # Track hyperparameters and run metadata
    config={
        "dataset": "AlkaSet",
        "model": "alka-master"
    }
)
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
    'fusion_dim': {
        'values': [64, 128, 256, 512, 1024]
    },
    'dropout': {
        'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    },
    'filter_sizes': {
        'values': [
            [3, 4, 5, 6],
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9],
            [7, 8, 9, 10]
        ]
    }
}

parameters_dict.update({
    'epochs': {
        'value': 30
    },
    'batch_size': {
        'value': 128
    },
    'optimizer': {
        'values': 'adam'
    }
})

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

sweep_id = wandb.sweep(sweep_config, project="alka")

training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pretrained_vectors(word2idx, fname):
    """Load pretrained vectors and create embedding layers.

    Args:
        word2idx (Dict): Vocabulary built from the corpus
        fname (str): Path to pretrained vector file

    Returns:
        embeddings (np.array): Embedding matrix with shape (N, d) where N is
            the size of word2idx and d is embedding dimension
    """

    print("Loading pretrained vectors...")
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())

    # Initialize random embeddings
    embeddings = np.random.uniform(-0.25, 0.25, (len(word2idx), d))
    embeddings[word2idx['<PAD>']] = np.zeros((d,))

    # Load pretrained vectors
    count = 0
    for line in notebook.tqdm(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        if word in word2idx:
            count += 1
            embeddings[word2idx[word]] = np.array(tokens[1:], dtype=np.float32)

    print(f"There are {count} / {len(word2idx)} pretrained vectors found.")

    return embeddings


def topk_accuracy(output, target, k=1):
    _, predicted_topk = output.topk(k, dim=1)
    acc = predicted_topk.eq(target.view(-1, 1)).sum().item()

    return acc

# Preparing the transforms
# TODO: transforms
data_tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4355, 0.3777, 0.2879), (0.2653, 0.2124, 0.2194))])

# Preparing the Dateset
# TODO: DATASET
alka_set = AlkaDataset('../dataset/102flowers', transform=data_tf)
train_ratio = 0.8
dataset_size = len(alka_set)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_set, val_set = random_split(alka_set, [train_size, test_size])

train_set_len = len(train_set)
val_set_len = len(val_set)

print(f"Train Data Length: {train_set_len}")
print(f"Validation Data Length: {val_set_len}")

# Preparing the DataLoader
batch_size = wandb.config.batch_size
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

embeddings = load_pretrained_vectors(alka_set.word2idx, "../data/crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)
num_classes = 102
loss_fn = nn.CrossEntropyLoss()


def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        # Setting up the NN
        # TODO: num_classes
        net_obj = ALKA(num_classes=num_classes, dropout=wandb.config.dropout, pretrained_embedding=embeddings,
                       fusion_dim=wandb.config.fusion_dim, filter_sizes=wandb.config.filter_sizes).to(
            training_device)
        # Loss function & Optimisation
        optimiser = torch.optim.Adam(net_obj.parameters(), lr=wandb.config.learning_rate,
                                     weight_decay=wandb.config.weight_decay)
        # Some training settings
        total_train_step = 0
        total_val_step = 0
        epoch = wandb.config.epochs
        for i in range(epoch):
            print(f"**************** Training Epoch: {i + 1} ****************")
            # Training
            net_obj.train()
            for images, captions, labels in train_loader:
                images = images.to(training_device)
                captions = captions.to(training_device)
                labels = labels.to(training_device)
                outputs = net_obj(images, captions)
                loss = loss_fn(outputs, labels)
                # Optimizing
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total_train_step += 1
                # if total_train_step % 100 == 0:
                print(f"Training Step: {total_train_step}, Loss: {loss.item()}")
                # writer.add_scalar("train_loss", loss.item(), total_train_step)
                wandb.log({"train_loss": loss.item()})
            # Validating
            total_step_loss = 0
            total_accuracy = 0
            total_top5_accuarcy = 0
            steps_per_epoch = 0
            net_obj.eval()
            print(f"**************** Validating Epoch: {i + 1} ****************")
            with torch.no_grad():
                for images, captions, labels in val_loader:
                    images = images.to(training_device)
                    captions = captions.to(training_device)
                    labels = labels.to(training_device)
                    outputs = net_obj(images, captions)
                    loss = loss_fn(outputs, labels)
                    total_step_loss += loss.item()
                    steps_per_epoch += 1
                    total_accuracy += topk_accuracy(outputs, labels)
                    total_top5_accuarcy += topk_accuracy(outputs, labels, k=5)
                total_val_step += 1
                print(f"Total Loss on Dataset: {total_step_loss / steps_per_epoch}")
                print(f"Top-1 Accuracy on Dataset: {total_accuracy / val_set_len}")
                print(f"Top-5 Accuracy on Dataset: {total_top5_accuarcy / val_set_len}")
                # writer.add_scalar("val_loss", total_step_loss, total_val_step)
                # writer.add_scalar("val_acc", total_accuracy / val_set_len, total_val_step)
                wandb.log({"acc@1": total_accuracy / val_set_len, "acc@5": total_top5_accuarcy / val_set_len,
                           "val_loss": total_step_loss / steps_per_epoch})


wandb.agent(sweep_id, train, count=5)
