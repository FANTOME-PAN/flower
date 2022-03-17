from torch import nn
import torch.nn.functional as F


class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            # nn.AvgPool2d(7),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# a block for resnet architecture
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Class for ResNet architecture - Standard Architecture

    Resnet with 18, 34, 50, 101 and 152 layers can be declared.

    We used ResNet 34 for our experiments.

    """

    def __init__(self, block, num_blocks, n):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def ResNet18(n):
        return ResNet(BasicBlock, [2, 2, 2, 2], n)

    def ResNet34(n):
        return ResNet(BasicBlock, [3, 4, 6, 3], n)

    def ResNet50(n):
        return ResNet(Bottleneck, [3, 4, 6, 3], n)

    def ResNet101(n):
        return ResNet(Bottleneck, [3, 4, 23, 3], n)

    def ResNet152(n):
        return ResNet(Bottleneck, [3, 8, 36, 3], n)


def build_model():
    return MobileNet(10)
# import torch.nn as nn
# import math
#
#
# def conv_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm2d(oup),
#         nn.ReLU6(inplace=True)
#     )
#
#
# def make_divisible(x, divisible_by=8):
#     import numpy as np
#     return int(np.ceil(x * 1. / divisible_by) * divisible_by)
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, oup, stride, expand_ratio):
#         super(InvertedResidual, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         hidden_dim = int(inp * expand_ratio)
#         self.use_res_connect = self.stride == 1 and inp == oup
#
#         if expand_ratio == 1:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # dw
#                 nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.ReLU6(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.use_res_connect:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNet(nn.Module):
#     def __init__(self, n_class=10, input_size=32, width_mult=1.):
#         super(MobileNet, self).__init__()
#         block = InvertedResidual
#         input_channel = 32
#         last_channel = 1280
#         interverted_residual_setting = [
#             # t, c, n, s
#             [1, 16, 1, 1],
#             [6, 24, 2, 2],
#             [6, 32, 3, 2],
#             [6, 64, 4, 2],
#             [6, 96, 3, 1],
#             [6, 160, 3, 2],
#             [6, 320, 1, 1],
#         ]
#
#         # building first layer
#         assert input_size % 32 == 0
#         # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
#         self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
#         # building inverted residual blocks
#         for t, c, n, s in interverted_residual_setting:
#             output_channel = make_divisible(c * width_mult) if t > 1 else c
#             for i in range(n):
#                 if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
#                 else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
#                 input_channel = output_channel
#         # building last several layers
#         self.features.append(conv_1x1_bn(input_channel, self.last_channel))
#         # make it nn.Sequential
#         self.features = nn.Sequential(*self.features)
#
#         # building classifier
#         self.classifier = nn.Linear(self.last_channel, n_class)
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.mean(3).mean(2)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
