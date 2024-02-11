import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from collections import defaultdict
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer


class AlkaDataset(Dataset):
    def __init__(self, root_dir, transform=None, load_to_ram=True):
        self.root_dir = root_dir
        self.transform = transform
        self.load_to_ram = load_to_ram
        self.image_folder = os.path.join(root_dir, '102flowers')
        self.text_folder = os.path.join(root_dir, 'text')

        self.image_files = {}
        self.classes = os.listdir(self.text_folder)
        self.tokenizer = get_tokenizer('basic_english')
        self.embedding = GloVe()

        self.tokenized_descriptions, self.class_hash_table, self.class_list, self.basename_list = self.load_descriptions()
        for basename in self.basename_list:
            if self.load_to_ram:
                cur_img = Image.open(os.path.join(self.image_folder, basename + ".jpg"))
                if self.transform:
                    cur_img = self.transform(cur_img)
                self.image_files[basename] = cur_img
            else:
                cur_img = os.path.join(self.image_folder, basename + ".jpg")
                self.image_files[basename] = cur_img


    def load_descriptions(self):
        descriptions = {}
        class_hash_table = {}
        class_list = []
        basename_list = []

        encoded_descriptions = {}

        for i in range(len(self.classes)):
            for cap_name in os.listdir(os.path.join(self.text_folder, self.classes[i])):
                basename = os.path.splitext(cap_name)[0]
                basename_list.append(basename)
                cap_path = os.path.join(self.text_folder, self.classes[i], cap_name)
                with open(cap_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    sentences = ""
                    for line in lines:
                        sentences += " " + line.strip()
                    descriptions[basename] = sentences
                    class_hash_table[basename] = self.classes[i]

                    tokenized_sent = self.tokenizer(sentences)
                    encoded_descriptions[basename] = self.embedding.get_vecs_by_tokens(tokenized_sent)

            class_list.append(self.classes[i])


        return encoded_descriptions, class_hash_table, class_list, basename_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        basename = self.basename_list[idx]
        if self.load_to_ram:
            image = self.image_files[basename]
        else:
            image = Image.open(self.image_files[basename])
            if self.transform:
                image = self.transform(image)

        class_to_index = {class_name: i for i, class_name in enumerate(self.class_list)}
        class_name = self.class_hash_table[basename]
        class_label = class_to_index[class_name]
        class_label = torch.tensor(class_label)

        descriptions = self.tokenized_descriptions[basename]

        return image, descriptions, class_label

# dataset = MultimodalDataset(root_dir='../dataset/102flowers')
# img, cap, clz = dataset[0]
# img, cap1, clz = dataset[1]
# print(cap.shape)
# print(cap1.shape)
