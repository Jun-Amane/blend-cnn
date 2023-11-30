import os
import torch
import torchvision
from torchvision.transforms import v2
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
from tqdm import notebook
import numpy as np

from ALKA import ALKA
from dataset import AlkaDataset

training_device = "cpu"

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
    _, predicted_topk = outputs.topk(k, dim=1)
    acc = predicted_topk.eq(labels.view(-1, 1)).sum().item()

    return acc


# Preparing the transforms
# TODO: transforms
data_tf = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.4355, 0.3777, 0.2879), (0.2653, 0.2124, 0.2194))])

# Preparing the Dateset
# TODO: DATASET
alka_set = AlkaDataset('../dataset/102flowers', transform=data_tf)

val_set_len = len(alka_set)

print(f"Validation Data Length: {val_set_len}")

# Preparing the DataLoader
batch_size = 128
val_loader = DataLoader(dataset=alka_set, batch_size=batch_size, shuffle=False, num_workers=4)

# Setting up the NN
# TODO: num_classes
embeddings = load_pretrained_vectors(alka_set.word2idx, "../data/crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)
num_classes = 102
net_obj = ALKA(num_classes=num_classes, dropout=0.2, pretrained_embedding=embeddings)
net_obj.load_state_dict(torch.load('../Saved_11.pth', map_location=torch.device('cpu')))

pytorch_total_params = sum(p.numel() for p in net_obj.parameters())
pytorch_trainable_params = sum(p.numel() for p in net_obj.parameters() if p.requires_grad)
print(f"TOTAL PARAMS OF ALKA: {pytorch_total_params}")
print(f"TRAINABLE PARAMS OF ALKA: {pytorch_trainable_params}")
print(net_obj)
# Loss function & Optimisation
loss_fn = nn.CrossEntropyLoss()

# Validating
total_step_loss = 0
total_accuracy = 0
total_top5_accuarcy = 0
steps_per_epoch = 0
net_obj.eval()
print(f"**************** Validating *****************")
with torch.no_grad():
    for images, captions, labels in val_loader:
        images = images.to(training_device)
        captions = captions.to(training_device)
        labels = labels.to(training_device)
        outputs = net_obj(images, captions)
        loss = loss_fn(outputs, labels)
        total_step_loss += loss.item()
        steps_per_epoch += 1

        print(f"Step: {steps_per_epoch}, Loss: {loss.item()}")
        total_accuracy += topk_accuracy(outputs, labels)
        total_top5_accuarcy += topk_accuracy(outputs, labels, k=5)

    print(f"Total Loss on Dataset: {total_step_loss / steps_per_epoch}")
    print(f"Top-1 Accuracy on Dataset: {total_accuracy / val_set_len}")
    print(f"Top-5 Accuracy on Dataset: {total_top5_accuarcy / val_set_len}")
    # writer.add_scalar("val_loss", total_step_loss, total_val_step)
    # writer.add_scalar("val_acc", total_accuracy / val_set_len, total_val_step)


