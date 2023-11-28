import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from collections import defaultdict


class AlkaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, '102flowers')
        self.text_folder = os.path.join(root_dir, 'text')

        self.image_files = {}
        self.classes = os.listdir(self.text_folder)

        self.tokenized_descriptions, self.class_hash_table, self.class_list, self.word2idx, self.basename_list = self.load_descriptions()
        for basename in self.basename_list:
            cur_img = Image.open(os.path.join(self.image_folder, basename + ".jpg"))
            if self.transform:
                cur_img = self.transform(cur_img)
            self.image_files[basename] = cur_img


    def load_descriptions(self):
        descriptions = {}
        class_hash_table = {}
        class_list = []
        basename_list = []

        max_len = 0
        tokenized_descriptions = {}
        tokenized_sentences = {}
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        idx = 2

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

                    tokenized_sent = word_tokenize(sentences)
                    tokenized_sentences[basename] = tokenized_sent
                    for token in tokenized_sent:
                        if token not in word2idx:
                            word2idx[token] = idx
                            idx += 1

                    max_len = max(max_len, len(tokenized_sent))
            class_list.append(self.classes[i])

        for i in range(len(self.classes)):
            for cap_name in os.listdir(os.path.join(self.text_folder, self.classes[i])):
                basename = os.path.splitext(cap_name)[0]
                tokenized_sent = tokenized_sentences[basename]
                tokenized_sent += ['<PAD>'] * (max_len - len(tokenized_sent))

                input_id = [word2idx.get(token) for token in tokenized_sent]
                tokenized_descriptions[basename] = torch.tensor(input_id)
        print(max_len)

        return tokenized_descriptions, class_hash_table, class_list, word2idx, basename_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        basename = self.basename_list[idx]
        image = self.image_files[basename]

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
