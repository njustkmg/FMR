import torch
from torch import nn
from torchvision import models
from torch.nn.modules.module import Module
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import scipy

import numpy as np
import os
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from dataset import Dataset, collate_fn
from TextNet import TextCNN
from FMR import FMR

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

data_path = "./wiki"

def dict_cuda(d):
    new_d = {}
    for k, v in d.items():
        new_d[k] = v.cuda()
    return new_d

# 测试基准模型
def test_base_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        labels = []
        preds = []
        for batch in test_loader:
            batch = dict_cuda(batch)
            labels.append(batch["batch_label"])
            pred = model(batch["batch_text"])
            preds.append(torch.argmax(pred, dim=1))
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels)
        acc = torch.mean((preds == labels).float())
        print("preds acc {}".format(acc))
    return acc

#　测试FMR模型
def test(model, test_loader):
    model.eval()
    with torch.no_grad():
        labels = []
        preds = []
        for batch in test_loader:
            batch = dict_cuda(batch)
            labels.append(batch["batch_label"])
            pred, _ = model(batch["batch_text"], batch["batch_image"])
            preds.append(torch.argmax(pred, dim=1))
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels)
        acc = torch.mean((preds == labels).float())
        print("preds acc {}".format(acc))
    return acc

# 训练基准模型
def train_base_model(model, train_loader, optimizer):
    loss_sum = 0
    model.train()
    for batch in tqdm(train_loader):
        batch = dict_cuda(batch)
        labels = batch["batch_label"]
        optimizer.zero_grad()
        logits = model(batch["batch_text"])

        loss_dict = {
            "classify_loss": F.cross_entropy(logits, labels),
        }

        loss = 0
        for k, v in loss_dict.items():
            loss += v
        loss.backward()
        optimizer.step()
        loss_sum += loss
    print(loss_sum)
    print(loss_dict)

#　训练FMR模型
def train(model, train_loader, optimizer):
    loss_sum = 0
    model.train()
    for batch in tqdm(train_loader):
        batch = dict_cuda(batch)
        images = batch["batch_image"]
        texts = batch["batch_text"]
        labels = batch["batch_label"]
        optimizer.zero_grad()
        logits, aloss = model(batch["batch_text"], batch["batch_image"])

        loss_dict = {
            "classify_loss": F.cross_entropy(logits, labels), # 分类loss，权重固定为1
            "reg_loss": aloss["reg_loss"]*0.0, # FMR论文中的lambda参数，可调，但我时直接使用0值
            "reconstruction_loss": aloss["reconstruction_loss"] # 重构固定特征z的loss权重，可调，此处固定为1
        }

        loss = 0
        for k, v in loss_dict.items():
            loss += v
        loss.backward()
        optimizer.step()
        loss_sum += loss
    print(loss_sum)
    print(loss_dict)


# 训练并输出基准模型结果
def sanity_check_compare():
    base_model = TextCNN(params={
        "embedding_path": os.path.join(data_path, "embedding.npy"),
        "feature_dim": 512,
        "last_layer": "linear"
    })
    model = nn.Sequential(base_model, nn.Linear(512, 10))
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, momentum=0.9, weight_decay=5e-4)
    train_path = os.path.join(data_path, "train_index.txt")
    vali_path = os.path.join(data_path, "vali_index.txt")
    test_path = os.path.join(data_path, "test_index.txt")
    train_dataset = Dataset(train_path, data_path)
    vali_dataset = Dataset(vali_path, data_path)
    test_dataset = Dataset(test_path, data_path)

    train_loader = DataLoader(train_dataset,
                                batch_size=32,
                                shuffle=True,
                                collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                                batch_size=64,
                                shuffle=False,
                                collate_fn=collate_fn)
    vali_loader = DataLoader(vali_dataset,
                                batch_size=256,
                                shuffle=False,
                                collate_fn=collate_fn)

    best_acc = 0
    epoch_num = 100
    feature_size = 4096
    print(f"epoch_num: {epoch_num}")
    for i in range(epoch_num):
        print("training epoch {}".format(i))
        train_base_model(model, train_loader, optimizer)
        print("vali")
        acc = test_base_model(model, vali_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./best_model.pt")

    model.load_state_dict(torch.load("./best_model.pt"))
    print("test")
    test_base_model(model, test_loader)
    print("best_vali_acc {}".format(best_acc))

# 训练并输出FMR模型结果
def sanity_check():
    base_model = TextCNN(params={
        "embedding_path": os.path.join(data_path, "embedding.npy"),
        "feature_dim": 512,
        "last_layer": "linear"
    })
    model = FMR(base_model,
                base_model_feature_size=512,
                fixed_feature_size=4096,
                output_size=10)
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, momentum=0.9, weight_decay=5e-4)
    train_path = os.path.join(data_path, "train_index.txt")
    vali_path = os.path.join(data_path, "vali_index.txt")
    test_path = os.path.join(data_path, "test_index.txt")
    train_dataset = Dataset(train_path, data_path)
    vali_dataset = Dataset(vali_path, data_path)
    test_dataset = Dataset(test_path, data_path)

    train_loader = DataLoader(train_dataset,
                                batch_size=32,
                                shuffle=True,
                                collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,
                                batch_size=64,
                                shuffle=False,
                                collate_fn=collate_fn)
    vali_loader = DataLoader(vali_dataset,
                                batch_size=256,
                                shuffle=False,
                                collate_fn=collate_fn)

    best_acc = 0
    epoch_num = 100 # 一共训练多少epoch
    feature_size = 4096
    m = 256 # 每次需要knockdown多少参数，可调
    kd_step = 1 # 每隔多少epoch进行一次knockdown，可调
    print(f"epoch_num: {epoch_num}")
    for i in range(epoch_num):
        if (i + 1) % kd_step == 0:
            model.knock_down(m)
        print("training epoch {}".format(i))
        train(model, train_loader, optimizer)
        print("vali")
        acc = test(model, vali_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "./best_model.pt")

    model.load_state_dict(torch.load("./best_model.pt"))
    print("test")
    test(model, test_loader)
    print("best_vali_acc {}".format(best_acc))

if __name__ == "__main__":
    # sanity_check()
    sanity_check_compare()

    # results
    # FMR 0.563 0.573 0.597 0.575 0.553 0.585 average = 0.57433
    # no FRM 0.567 0.569 0.559 0.585 0.541 0.538 average = 0.55983
