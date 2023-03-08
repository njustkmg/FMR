import torch
from torch import nn
from torch.nn.modules.module import Module
from torchvision import models


class ImageNet(Module):
    def __init__(self, params):
        super(ImageNet, self).__init__()
        self.class_num = params["label_num"]
        self.base_model = params["base_model"]
        self.last_layer = params["last_layer"]
        self.L = self.class_num
        if self.base_model == "vgg16":
            self.net = nn.Sequential(
                models.vgg16(pretrained=True).features,
                nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
            )
            self.d = 512
        elif self.base_model == "vgg19_bn":
            self.net = models.vgg19_bn(pretrained=True)
            self.vgg_features = self.net.features
            self.fc_features = nn.Sequential(*list(self.net.classifier.children())[:-2])
        elif self.base_model == "resnet18":
            self.net = models.resnet18(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.d = 512
        elif self.base_model == "resnet34":
            self.net = models.resnet34(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.d = 512
        elif self.base_model == "resnet50":
            self.net = models.resnet50(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.d = 2048
        elif self.base_model == "resnet101":
            self.net = models.resnet101(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.d = 2048
        elif self.base_model == "resnet152":
            self.net = models.resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-1])
            self.d = 2048
        else:
            print("[%s] No such base model: %s" % (show_time(), args["base_model"]))
        if self.last_layer == "linear":
            pass
            # self.fcs = nn.Sequential(nn.Linear(self.d, 512), nn.Linear(512, 512))
            # self.last_fc = nn.Linear(512, self.L)
        elif self.last_layer == "a_softmax":
            # print("a_softmax")
            self.a_softmax = ASoftmax(self.d, self.L)
        else:
            raise NotImplementedError()

    def forward(self, x):
        # x.shape = (batch_size, 3, 224, 224)
        if self.base_model == "vgg19_bn":
            self.vgg_features.eval()
            self.fc_features.eval()
            with torch.no_grad():
                x = self.vgg_features(x).view(x.shape[0], -1)
                x = self.fc_features(x)
        else:
            x = self.net(x)
            x = x.view((x.size(0), -1))
        if self.last_layer == "linear":
            return x
        else:
            raise NotImplementedError()

if __name__ == '__main__':
    l = [1, 2, 2]
    x = torch.rand((6, 2, 3, 224, 224))
    args = {
        "label_num": 80,
        "sub_category_num": 20,
        "base_model": "resnet18"
    }
    M = ImageNet(args)
    x = {
        "batch_image": x,
        "batch_image_num": l
    }
    print(M(x).size())
