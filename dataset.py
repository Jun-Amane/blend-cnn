import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from collections import defaultdict


class AlkaDataset(Dataset):
    def __init__(self, root_dir, transform=None, load_to_ram=False):
        self.root_dir = root_dir
        self.transform = transform
        self.load_to_ram = load_to_ram

        self.images = []

        self.classes = os.listdir(root_dir)

        self.word2idx = {}

        self.class_to_index = {class_name: i for i, class_name in enumerate(self.classes)}
        print(self.class_to_index)
        for clazz in self.classes:
            imgs = os.listdir(os.path.join(root_dir, clazz))
            for img in imgs:
                img_file = os.path.join(root_dir, clazz, img)
                if self.load_to_ram:
                    img_bin = Image.open(img_file)
                    img_bin = self.transform(img_bin)
                    self.images.append({"img": img_bin, "clazz": clazz})
                else:
                    self.images.append({"img": img_file, "clazz": clazz})



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.load_to_ram:
            image = self.images[idx]['img']
        else:
            image = Image.open(self.images[idx]['img'])
            if self.transform:
                image = self.transform(image)

        clazz = self.class_to_index[self.images[idx]['clazz']]

        return image, torch.tensor([0]), clazz

dataset = AlkaDataset(root_dir='../../dataset/wheat_img')
# img, cap, clz = dataset[0]
# img, cap1, clz = dataset[1]
# print(cap.shape)
# print(cap1.shape)
