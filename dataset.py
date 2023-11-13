import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from transformers import BertModel, BertTokenizer


class AlkaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, '102flowers')
        self.text_folder = os.path.join(root_dir, 'text')

        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
        self.classes = os.listdir(self.text_folder)

        self.descriptions, self.class_hash_table, self.class_list = self.load_descriptions()

        # BERT out=768
        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def load_descriptions(self):
        descriptions = {}
        class_hash_table = {}
        class_list = []
        for i in range(len(self.classes)):
            for cap_name in os.listdir(os.path.join(self.text_folder, self.classes[i])):
                basename = os.path.splitext(cap_name)[0]
                cap_path = os.path.join(self.text_folder, self.classes[i], cap_name)
                with open(cap_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    descriptions[basename] = [line.strip() for line in lines]
                    class_hash_table[basename] = self.classes[i]
            class_list.append(self.classes[i])

        return descriptions, class_hash_table, class_list

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path)
        basename = os.path.splitext(image_name)[0]

        class_to_index = {class_name: i for i, class_name in enumerate(self.class_list)}
        class_name = self.class_hash_table[basename]
        class_label = class_to_index[class_name]
        class_label = torch.tensor(class_label)

        descriptions = self.descriptions[basename]
        tokenized_text = [self.text_tokenizer.encode(text, add_special_tokens=True) for text in descriptions]
        max_len = 128
        padded_text = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_text]
        captions_tensor = torch.tensor(padded_text)

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        return image, captions_tensor, class_label

# dataset = MultimodalDataset(root_dir='../dataset/102flowers')
# img, cap, clz = dataset[0]
# img, cap1, clz = dataset[1]
# print(cap.shape)
# print(cap1.shape)
