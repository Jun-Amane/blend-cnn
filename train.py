import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
import wandb
import pprint

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
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.5,
        "batch_size": 128,
        "dataset": "AlkaSet",
        "epochs": 80,
    })

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

# Setting up the NN
# TODO: num_classes
num_classes = 102
net_obj = ALKA(num_classes=num_classes, dropout=wandb.config.dropout, vocab_size=alka_set.vocab_size).to(training_device)

# Loss function & Optimisation
loss_fn = nn.CrossEntropyLoss()
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
    for images, captions, masks, labels in train_loader:
        images = images.to(training_device)
        captions = captions.to(training_device)
        masks = masks.to(training_device)
        labels = labels.to(training_device)
        outputs = net_obj(images, captions, masks)
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

            step_accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += step_accuracy

        total_val_step += 1
        print(f"Total Loss on Dataset: {total_step_loss / steps_per_epoch}")
        print(f"Total Accuracy on Dataset: {total_accuracy / val_set_len}")
        # writer.add_scalar("val_loss", total_step_loss, total_val_step)
        # writer.add_scalar("val_acc", total_accuracy / val_set_len, total_val_step)
        wandb.log({"val_acc": total_accuracy / val_set_len, "val_loss": total_step_loss / steps_per_epoch})

    # torch.save(net_obj.state_dict(), f"Saved_{i}.pth")
    # print("Saved.")

# writer.close()
wandb.finish()
