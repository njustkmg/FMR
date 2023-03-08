import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torchvision import transforms
from torch.utils.data import DataLoader

import os
from PIL import Image
import pickle
import numpy as np

class Dataset:
    def __init__(self, index_path, data_path):
        self.data_path = data_path
        self.index_path = index_path
        self.data = []
        with open("./image_vgg19_bn.pkl", "rb") as f:
            self.image_features = pickle.load(f)
        with open(self.data_path + "/data_all.json", "r") as f:
            for row in f:
                instance = eval(row)

                texts = []
                text_num = []
                text_id = []
                for i in range(len(instance["text"])):
                    if instance["text"][i] == -1:
                        texts.append(text_id)
                        text_num.append((len(text_num), len(text_id)))
                        text_id = []
                    else:
                        text_id.append(instance["text"][i])
                instance["text_order"] = [i for (i, l) in text_num]
                instance["text"] = [texts[i] for (i, l) in text_num]
                instance["text_length"] = [l for (i, l) in text_num]

                self.data.append(instance)
        self.instances = []
        with open(self.index_path, "r") as f:
            for row in f:
                self.instances.append(int(row))

    def __getlabel__(self, index):
        label = self.data[index]["label"]
        return np.argmax(label)

    def __getitem__(self, index: int):
        # map index
        index = self.instances[index]

        image_tensor = self.image_features[index]

        # read text
        texts = self.data[index]["text"]
        text_tensor = []
        for i in range(len(texts)):
            text_tensor.append(torch.LongTensor(texts[i]))
        text_tensor = torch.cat(text_tensor, dim=0)

        # read label
        y = torch.FloatTensor(self.data[index]["label"]).view(-1)

        instance = {
            "image_tensor": image_tensor,

            "text_tensor": text_tensor,
            "text_length": self.data[index]["text_length"],
            "text_order": self.data[index]["text_order"],

            "label": y
        }
        return instance

    def __len__(self):
        return len(self.instances)


def collate_fn(x):
    # collate image
    batch_image = torch.stack([torch.from_numpy(x[i]["image_tensor"])
                             for i in range(len(x))], dim=0)

    # collate text
    batch_text_num = [(i, sum(x[i]["text_length"])) for i in range(len(x))]
    batch_text_length = [x[i]["text_length"] for i in range(len(x))]
    batch_text_order = [x[i]["text_order"] for i in range(len(x))]
    batch_text = pack_sequence([x[i]["text_tensor"] for i in range(len(x))], enforce_sorted=False)

    # collate label
    batch_label_oh = torch.cat([x[i]["label"].unsqueeze(0) for i in range(len(x))]).long()
    batch_label = torch.argmax(batch_label_oh, dim=1)

    batch_data = {
        "batch_image": batch_image,
        "batch_text": batch_text,

        "batch_label_oh": batch_label_oh,
        "batch_label": batch_label
    }
    return batch_data

