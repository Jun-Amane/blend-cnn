import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class AlkaDatasetEm(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, '102flowers')
        self.text_folder = os.path.join(root_dir, 'embd')

        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
        self.classes = os.listdir(self.text_folder)

        self.class_hash_table, self.class_list = self.load_descriptions()

    def load_descriptions(self):
        class_hash_table = {}
        class_list = []
        for i in range(len(self.classes)):
            for cap_name in os.listdir(os.path.join(self.text_folder, self.classes[i])):
                basename = os.path.splitext(cap_name)[0]
                class_hash_table[basename] = self.classes[i]
            class_list.append(self.classes[i])

        return class_hash_table, class_list

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
        captions_tensor = torch.load(os.path.join(self.text_folder, class_name, basename + '.pt'))

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        return image, captions_tensor, class_label
