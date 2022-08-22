import math
from collections import OrderedDict
from typing import Any, Sequence, Tuple

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import Tensor, hub

from .ibn import IBN


class Bottleneck_IBN(M.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        baseWidth: int,
        cardinality: int,
        stride: int = 1,
        downsample: M.Module = None,
        ibn: str = None
    ):

        super(Bottleneck_IBN, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality
        self.conv1 = M.Conv2d(
            inplanes,
            D*C,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = M.BatchNorm2d(D*C) if ibn != 'a' else IBN(D*C)
        self.conv2 = M.Conv2d(
            D*C,
            D*C,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=C,
            bias=False)
        self.bn2 = M.BatchNorm2d(D*C)
        self.conv3 = M.Conv2d(
            D*C,
            planes * 4,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = M.BatchNorm2d(planes * 4)
        self.relu = M.ReLU()

        self.downsample = downsample if downsample is not None else M.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt_IBN(M.Module):

    def __init__(self,
                 baseWidth: int,
                 cardinality: int,
                 layers: int,
                 ibn_cfg: Tuple[str, str, str, str] = ('a', 'a', 'a', None),
                 num_classes=1000):
        super(ResNeXt_IBN, self).__init__()
        block = Bottleneck_IBN

        self.cardinality = cardinality
        self.baseWidth = baseWidth
        self.num_classes = num_classes
        self.inplanes = 64
        self.output_size = 64

        self.conv1 = M.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = M.BatchNorm2d(64)
        self.relu = M.ReLU()
        self.maxpool1 = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = M.AvgPool2d(7)
        self.fc = M.Linear(512 * block.expansion, num_classes)

        M.init.normal_(self.conv1.weight, 0, math.sqrt(2. / (7 * 7 * 64)))
        for m in self.modules():
            if isinstance(m, M.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                M.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, M.BatchNorm2d) or isinstance(m, M.InstanceNorm):
                M.init.ones_(m.weight)
                M.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = M.Sequential(
                M.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                M.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth,
                            self.cardinality, stride, downsample, ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth,
                                self.cardinality, 1, None, ibn))

        return M.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = F.flatten(x, 1)
        x = self.fc(x)

        return x


def resnext50_ibn_a(pretrained=False, baseWidth=4, cardinality=32):
    return ResNeXt_IBN(baseWidth, cardinality, [
        3, 4, 6, 3], ('a', 'a', 'a', None))


def resnext101_ibn_a(pretrained=False, baseWidth=4, cardinality=32):
    return ResNeXt_IBN(baseWidth, cardinality, [
        3, 4, 23, 3], ('a', 'a', 'a', None))


def resnext152_ibn_a(pretrained=False, baseWidth=4, cardinality=32):
    return ResNeXt_IBN(baseWidth, cardinality, [
        3, 8, 36, 3], ('a', 'a', 'a', None))
