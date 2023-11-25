import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
import wandb
import pprint
from tqdm import notebook

from dataset import AlkaDataset

class AlkaTextLSTM(nn.Module):

    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 vocab_size=None,
                 embed_dim=300):

        super(AlkaTextLSTM, self).__init__()

        # Embedding layer
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)
        else:
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=0,
                                          max_norm=5.0)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=4),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=10)
        )

        self.lstm = nn.LSTM(embed_dim, 512, batch_first=True, bidirectional=False, num_layers=1)
        self.hidden2tag = nn.Linear(512, 102)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, input_ids):

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        x_pooled = self.conv(x_reshaped)
        x_pooled = x_pooled.permute(0, 2, 1)
        _, out = self.lstm(x_pooled)
        out = self.hidden2tag(out[0])
        out = self.softmax(out.squeeze())

        return out

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# training_device = "cuda"
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
batch_size = 128
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)

# Setting up the NN
# TODO: num_classes
embeddings = load_pretrained_vectors(alka_set.word2idx, "../data/crawl-300d-2M.vec")
embeddings = torch.tensor(embeddings)
num_classes = 102
net_obj = AlkaTextLSTM(pretrained_embedding=embeddings).to(training_device)


# Loss function & Optimisation
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(net_obj.parameters(), lr=0.01,
                             weight_decay=0.001)

# Some training settings
total_train_step = 0
total_val_step = 0
epoch = 80

for i in range(epoch):
    print(f"**************** Training Epoch: {i + 1} ****************")

    # Training
    net_obj.train()
    for images, captions, labels in train_loader:
        captions = captions.to(training_device)
        labels = labels.to(training_device)
        outputs = net_obj(captions)
        loss = loss_fn(outputs, labels)

        # Optimizing
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_train_step += 1

        # if total_train_step % 100 == 0:
        print(f"Training Step: {total_train_step}, Loss: {loss.item()}")
        # writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Validating
    total_step_loss = 0
    total_accuracy = 0
    total_top5_accuarcy = 0
    steps_per_epoch = 0
    net_obj.eval()
    print(f"**************** Validating Epoch: {i + 1} ****************")
    with torch.no_grad():
        for images, captions, labels in val_loader:
            captions = captions.to(training_device)
            labels = labels.to(training_device)
            outputs = net_obj(captions)
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


    # torch.save(net_obj.state_dict(), f"Saved_{i}.pth")
    # print("Saved.")

# writer.close()
