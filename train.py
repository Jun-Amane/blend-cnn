import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ALKA import ALKA
from dataset import MultimodalDataset

training_device="cpu"

writer = SummaryWriter("logs")

# Preparing the transforms
# TODO: transforms
data_tf = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                         torchvision.transforms.ToTensor()])

# Preparing the Dateset
# TODO: DATASET
train_data = MultimodalDataset('../dataset/102flowers', transform=data_tf)
val_data = MultimodalDataset('../dataset/102flowers', transform=data_tf)

train_data_len = len(train_data)
val_data_len = len(val_data)

print(f"Train Data Length: {train_data_len}")
print(f"Validation Data Length: {val_data_len}")

# Preparing the DataLoader
batch_size=64
train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size)
val_data_loader = DataLoader(dataset=val_data, batch_size=batch_size)

# Setting up the NN
# TODO: num_classes
num_classes = 102
net_obj = ALKA(num_classes=num_classes)

# Loss function & Optimisation
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net_obj.parameters(), lr=0.001)

# Some training settings
total_train_step = 0
total_val_step = 0
epoch = 50

for i in range(epoch):
    print(f"**************** Training Epoch: {i + 1} ****************")

    # Training
    net_obj.train()
    for images, captions, labels in train_data_loader:
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

        if total_train_step % 100 == 0:
            print(f"Training Step: {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Validating
    total_step_loss = 0
    total_accuracy = 0
    net_obj.eval()
    with torch.no_grad():
        for images, text, labels in val_data_loader:
            images = images.to(training_device)
            outputs = net_obj(images, text)
            loss = loss_fn(outputs, labels)
            total_step_loss += loss.item()

            step_accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += step_accuracy

        total_val_step += 1
        print(f"Total Loss on Dataset: {total_step_loss}")
        print(f"Total Accuracy on Dataset: {total_accuracy / val_data_len}")
        writer.add_scalar("val_loss", total_step_loss, total_val_step)
        writer.add_scalar("val_acc", total_accuracy / val_data_len, total_val_step)

    # torch.save(net_obj.state_dict(), f"Saved_{i}.pth")
    # print("Saved.")

writer.close()
