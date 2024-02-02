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
import numpy as np

from ALKA import ALKA
from dataset import AlkaDataset

training_device = "cpu"

# writer = SummaryWriter("logs")
os.environ["WANDB_MODE"] = "offline"

# Wandb configs
# wandb.login()
wandb.init(
    # Set the project where this run will be logged
    project="alka",
    # Track hyperparameters and run metadata
    config={
        "model": "alka-master",
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "heads": 8,
        "batch_size": 16,
        "dataset": "AlkaSet",
        "epochs": 30,
    })


def load_pretrained_vectors(word2idx, fname):
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
    _, predicted_topk = outputs.topk(k, dim=1)
    acc = predicted_topk.eq(labels.view(-1, 1)).sum().item()

    return acc


# Preparing the transforms
data_tf = v2.Compose([
    v2.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
    v2.CenterCrop((320, 320)),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4355, 0.3777, 0.2879), (0.2653, 0.2124, 0.2194))])

# Preparing the Dateset
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

# Preparing the DataLoader
batch_size = wandb.config.batch_size
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Setting up the NN
embeddings = load_pretrained_vectors(alka_set.word2idx, "../data/crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)
num_classes = 102
net_obj = ALKA(num_classes=num_classes, dropout=wandb.config.dropout, pretrained_embedding=embeddings).to(
    training_device)

pytorch_total_params = sum(p.numel() for p in net_obj.parameters())
pytorch_trainable_params = sum(p.numel() for p in net_obj.parameters() if p.requires_grad)
print(f"TOTAL PARAMS OF ALKA: {pytorch_total_params}")
print(f"TRAINABLE PARAMS OF ALKA: {pytorch_trainable_params}")



# Loss function & Optimisation
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(net_obj.parameters(), lr=wandb.config.learning_rate,
                             weight_decay=wandb.config.weight_decay)

# Some training settings
total_train_step = 0
total_val_step = 0
best_acc = 0.0
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

        if (total_accuracy / val_set_len) > best_acc:
            best_acc = total_accuracy / val_set_len
            wandb.run.summary["best_acc"] = best_acc

    # torch.save(net_obj.state_dict(), f"Saved_{i}.pth")
    # print("Saved.")

# writer.close()
wandb.finish()
