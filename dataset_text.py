import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class AlkaDataset(Dataset):
    def __init__(self, root_dir, local_bert=False, local_bert_path='../bert-base-uncased', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_folder = os.path.join(root_dir, '102flowers')
        self.text_folder = os.path.join(root_dir, 'text')

        self.image_files = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
        self.classes = os.listdir(self.text_folder)

        self.descriptions, self.class_hash_table, self.class_list = self.load_descriptions()

        # BERT out=768
        if local_bert:
            self.text_model = BertModel.from_pretrained(local_bert_path)
            self.text_tokenizer = BertTokenizer.from_pretrained(local_bert_path)
        else:
            self.text_model = BertModel.from_pretrained("bert-base-uncased")
            self.text_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.vocab_size = self.text_tokenizer.vocab_size

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
        tokenized_text = [self.text_tokenizer.convert_tokens_to_ids(self.text_tokenizer.tokenize(self.text_tokenizer.convert_tokens_to_string(self.text_tokenizer.tokenize(text)))) for text in descriptions]

        # max_len = 128
        # padded_text = [tokens + [0] * (max_len - len(tokens)) for tokens in tokenized_text]
        padded_tokenized_texts = []
        attention_masks = []
        for input_ids in tokenized_text:
            padding_size = 128 - len(input_ids)
            input_ids = torch.tensor(input_ids)
            padded_input_ids = F.pad(input_ids, (0, padding_size), value=self.text_tokenizer.pad_token_id)
            attention_mask = torch.ones_like(padded_input_ids)
            attention_mask[padding_size:] = 0  # 将填充的部分置零

            padded_tokenized_texts.append(padded_input_ids)
            attention_masks.append(attention_mask)

        padded_tokenized_texts = torch.stack(padded_tokenized_texts, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)

        vocab_size = self.text_tokenizer.vocab_size

        # with torch.no_grad():
        #     bert_out = self.text_model(input_ids=padded_tokenized_texts, attention_mask=attention_masks)
        # text_out = bert_out.last_hidden_state

        # captions_tensor = torch.tensor(text_out)

        # 数据预处理
        if self.transform:
            image = self.transform(image)

        return padded_tokenized_texts, attention_masks, class_label

# dataset = MultimodalDataset(root_dir='../dataset/102flowers')
# img, cap, clz = dataset[0]
# img, cap1, clz = dataset[1]
# print(cap.shape)
# print(cap1.shape)
