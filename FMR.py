import torch
from torch import nn
from torchvision import models
from torch.nn.modules.module import Module
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import math
import numpy as np

class FMR(nn.Module):
    def __init__(self,
                 base_model, # 迁移的目标模型，应该为Module实例
                 base_model_feature_size, # base_model的最后输出特征的维度
                 fixed_feature_size, # 固定特征z的维度
                 output_size): #输出的维度，分类问题中为类别个数
        super(FMR, self).__init__()
        self.base_model = base_model
        self.fixed_feature_size = fixed_feature_size
        self.base_model_feature_size = base_model_feature_size
        self.output_size = output_size

        self.U_layer = nn.Linear(self.base_model_feature_size, self.fixed_feature_size, bias=True)
        self.U_mat = self.U_layer.weight
        self.W_layer = nn.Linear(self.base_model_feature_size, self.output_size, bias=True)
        self.W_mat = self.W_layer.weight
        self.V_layer = nn.Linear(self.fixed_feature_size, self.output_size, bias=False)
        self.V_mat = self.V_layer.weight

        self.mask = np.ones(self.fixed_feature_size)

    def knock_down(self, m): # 敲掉至多m个固定特征
        nonzero_idx = np.nonzero(self.mask)[0]
        if len(nonzero_idx) >= m:
            idx = np.random.choice(nonzero_idx, m, replace=False)
        else:
            idx = np.random.choice(nonzero_idx, len(nonzero_idx), replace=False)
        self.mask[idx] = 0
        print("remaining z {}".format(np.sum(self.mask)))

    def forward(self, x, z): # 输入为原始输入x以及特征z，当固定特征全部敲掉，即np.sum(self.mask)==0时(训练结束时必须全部敲掉），z的输入不起作用，可为全0
        xp = self.base_model(x)

        mask = torch.from_numpy(self.mask).float().cuda()
        mask = mask.view(1, -1)

        reg_loss = torch.mean(self.W_mat**2) \
                   + torch.mean(self.V_mat**2) \
                   + torch.mean(self.U_mat**2)

        reconstruction_loss = torch.mean(((z - self.U_layer(xp))*mask)**2) * 0.5

        x_pred = self.W_layer(xp)
        z_pred = self.V_layer(z*mask)

        return x_pred + z_pred, {
            "reg_loss": reg_loss,
            "reconstruction_loss": reconstruction_loss
        }
