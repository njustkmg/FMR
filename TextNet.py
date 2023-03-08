import torch
from torch import nn
from torch.nn.modules.module import Module
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

import numpy as np

class TextCNN(Module):
    def __init__(self,
                 params):
        super(TextCNN, self).__init__()
        self.feature_dim = params["feature_dim"]
        self.last_layer = params["last_layer"]
        w = torch.from_numpy(np.load(params["embedding_path"]))
        self.padding_idx = w.size(1)
        self.embedding = nn.Embedding(
            num_embeddings=w.size(0),
            embedding_dim=w.size(1),
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            padding_idx=self.padding_idx
        )
        self.embedding.from_pretrained(w)
        self.embedding.weight.requires_grad=False
        self.input_dim = w.size(1)
        self.conv1 = nn.Conv1d(self.input_dim, 384, 15)
        self.conv1_pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(384, 512, 9)
        self.conv2_pool = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(512, 256, 7)
        self.conv3_pool = nn.MaxPool1d(2, 2)
        self.last_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, self.feature_dim)

    def forward(self, x):
        x, b_s = pad_packed_sequence(x, batch_first=True, padding_value=self.padding_idx)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1_pool(F.relu(self.conv1(x)))
        x = self.conv2_pool(F.relu(self.conv2(x)))
        x = self.conv3_pool(F.relu(self.conv3(x)))
        x = self.last_pool(x)
        x = x.view((x.size(0), 256))
        x = self.fc(x)
        if self.last_layer == "linear":
            # x = self.fcs(x)
            # y = self.last_fc(x)
            return x
        elif self.last_layer == "a_softmax":
            raise NotImplementedError()
            thetas, x_norm = self.a_softmax(x)
            return {"cos_thetas": thetas, "x_norm": x_norm}, x


class TextNet(Module):
    def __init__(self,
                 params):
        super(TextNet, self).__init__()
        self.L = params["label_num"]
        self.hidden_size = params["hidden_size"]
        self.params = params
        self.num_layers = params["num_layers"]
        self.last_layer = params["last_layer"]
        base_model = params["base_model"]

        w = torch.from_numpy(np.load(params["embedding_path"]))
        self.padding_idx = w.size(1)
        self.embedding = nn.Embedding(
            num_embeddings=w.size(0),
            embedding_dim=w.size(1),
            norm_type=2,
            scale_grad_by_freq=False,
            sparse=False,
            padding_idx=self.padding_idx
        )
        self.embedding.from_pretrained(w)

        if base_model == "rnn":
            self.net = nn.RNN(
                input_size=self.embedding.embedding_dim,
                batch_first=True,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=True,
                bidirectional=True
            )
        elif base_model == "lstm":
            self.net = nn.LSTM(
                input_size=self.embedding.embedding_dim,
                batch_first=True,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=True,
                bidirectional=True
            )
        elif base_model == "gru":
            self.net = nn.GRU(
                input_size=self.embedding.embedding_dim,
                batch_first=True,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=True,
                bidirectional=True
            )
        else:
            print("[%s] No such base model: %s" % (show_time(), args["base_model"]))
        self.d = self.hidden_size * 2

    def forward(self, x):
        if isinstance(x, PackedSequence):
            x, b_s = pad_packed_sequence(x, batch_first=True, padding_value=self.padding_idx)
            x = self.embedding(x)
            x = pack_padded_sequence(x, b_s, batch_first=True, enforce_sorted=False)
            x, _ = self.net(x)
            x, b_s = pad_packed_sequence(x, batch_first=True, padding_value=0)
            x = torch.stack([x[i, b_s[i]-1, :] for i in range(b_s.size(0))], dim=0)
            if self.last_layer == "linear":
                # x = self.fcs(x)
                # y = self.last_fc(x)
                return x
            elif self.last_layer == "a_softmax":
                thetas, x_norm = self.a_softmax(x)
                return {"cos_thetas": thetas, "x_norm": x_norm}, x
        else:
            raise NotImplementedError()

if __name__ == '__main__':
    pass
