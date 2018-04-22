import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import resnet50


#class Model(nn.Module):
#  def __init__(self, last_conv_stride=2):
#    super(Model, self).__init__()
#    self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride)
#
#  def forward(self, x):
#    # shape [N, C, H, W]
#    x = self.base(x)
#    x = F.avg_pool2d(x, x.size()[2:])
#    # shape [N, C]
#    x = x.view(x.size(0), -1)
#
#    return x


def conv(kernel_size, stride, in_size, out_size ):
    layer = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_size),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )
    return layer

class Model(nn.Module):
    """
    CNN for Mini-Imagenet meta learning.
    """
    def __init__(self, num_classes):
        super(Model, self).__init__()
        num_filters = 32
        kernel_size = 3

        self.layer1 = conv(kernel_size, 1, 3, num_filters)
        self.layer2 = conv(kernel_size, 1, num_filters, num_filters) 
        self.layer3 = conv(kernel_size, 1, num_filters, num_filters)
        self.layer4 = conv(kernel_size, 1, num_filters, num_filters)

        self.final = nn.Linear(num_filters * 16 * 8, num_classes)
        #self.final = nn.Linear(num_filters * 5 * 5, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size(0), -1)
        return self.final(x)

    def embedding(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x.view(x.size(0), -1)

