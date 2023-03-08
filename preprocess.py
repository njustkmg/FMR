import torch
from torch import nn
from torchvision import models
from torch.nn.modules.module import Module
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torchvision import transforms

import numpy as np

from ImageNet import ImageNet
from PIL import Image
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = "./wiki"

image_features = []

image_process_center = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

model = ImageNet(params={
    "base_model": "resnet34",
    "label_num": 10,
    "last_layer": "linear"
})
model = model.cuda()
model.eval()

with torch.no_grad():
    with open(data_path + "/data_all.json", "r") as f:
        for i, row in enumerate(f):
            print(i)
            instance = eval(row)

            image_path = instance["image"][0]
            image_path = data_path + "/" + image_path
            img = Image.open(image_path).convert('RGB')
            img = image_process_center(img).view(-1, 3, 224, 224).cuda()
            image_feature = model(img)
            image_features.append(image_feature.detach().cpu().numpy())
image_features = np.concatenate(image_features, axis=0)
with open("./resnet34_512_wiki.pkl", "wb") as f:
    pickle.dump(image_features, f)
