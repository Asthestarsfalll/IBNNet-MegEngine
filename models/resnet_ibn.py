import math
import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import hub
from .ibn import IBN


class BasicBlock_IBN(M.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = M.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = M.BatchNorm2d(planes)
        self.relu = M.ReLU()
        self.conv2 = M.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=False)
        self.bn2 = M.BatchNorm2d(planes)
        self.IN = M.InstanceNorm(
            planes, affine=True) if ibn == 'b' else M.Identity()
        self.downsample = downsample if downsample is not None else M.Identity()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)

        out += residual
        out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(M.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = M.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = M.BatchNorm2d(planes)
        self.conv2 = M.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = M.BatchNorm2d(planes)
        self.conv3 = M.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = M.BatchNorm2d(planes * self.expansion)
        self.IN = M.InstanceNorm(
            planes * 4, affine=True) if ibn == 'b' else M.Identity()
        self.relu = M.ReLU()
        self.downsample = downsample if downsample is not None else M.Identity()

        self.stride = stride

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
        out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(M.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = M.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = M.InstanceNorm(64, affine=True)
        else:
            self.bn1 = M.BatchNorm2d(64)
        self.relu = M.ReLU()
        self.maxpool = M.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = M.AvgPool2d(7)
        self.fc = M.Linear(512 * block.expansion, num_classes)

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
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return M.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = F.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_ibn_a(pretrained=False, **kwargs):
    return ResNet_IBN(block=BasicBlock_IBN,
                      layers=[2, 2, 2, 2],
                      ibn_cfg=('a', 'a', 'a', None),
                      **kwargs)


def resnet34_ibn_a(pretrained=False, **kwargs):
    return ResNet_IBN(block=BasicBlock_IBN,
                      layers=[3, 4, 6, 3],
                      ibn_cfg=('a', 'a', 'a', None),
                      **kwargs)

@hub.pretrained(
    "https://studio.brainpp.com/api/v1/activities/3/missions/85/files/6a5be360-5b58-4dcd-bc5a-da7307f7e7ce"
)
def resnet50_ibn_a(pretrained=False, **kwargs):
    return ResNet_IBN(block=Bottleneck_IBN,
                      layers=[3, 4, 6, 3],
                      ibn_cfg=('a', 'a', 'a', None),
                      **kwargs)


def resnet101_ibn_a(pretrained=False, **kwargs):
    return ResNet_IBN(block=Bottleneck_IBN,
                      layers=[3, 4, 23, 3],
                      ibn_cfg=('a', 'a', 'a', None),
                      **kwargs)


def resnet152_ibn_a(pretrained=False, **kwargs):
    return ResNet_IBN(block=Bottleneck_IBN,
                      layers=[3, 8, 36, 3],
                      ibn_cfg=('a', 'a', 'a', None),
                      **kwargs)


def resnet18_ibn_b(pretrained=False, **kwargs):
    return ResNet_IBN(block=BasicBlock_IBN,
                      layers=[2, 2, 2, 2],
                      ibn_cfg=('b', 'b', None, None),
                      **kwargs)


def resnet34_ibn_b(pretrained=False, **kwargs):
    return ResNet_IBN(block=BasicBlock_IBN,
                      layers=[3, 4, 6, 3],
                      ibn_cfg=('b', 'b', None, None),
                      **kwargs)


def resnet50_ibn_b(pretrained=False, **kwargs):
    return ResNet_IBN(block=Bottleneck_IBN,
                      layers=[3, 4, 6, 3],
                      ibn_cfg=('b', 'b', None, None),
                      **kwargs)


def resnet101_ibn_b(pretrained=False, **kwargs):
    return ResNet_IBN(block=Bottleneck_IBN,
                      layers=[3, 4, 23, 3],
                      ibn_cfg=('b', 'b', None, None),
                      **kwargs)


def resnet152_ibn_b(pretrained=False, **kwargs):
    return ResNet_IBN(block=Bottleneck_IBN,
                      layers=[3, 8, 36, 3],
                      ibn_cfg=('b', 'b', None, None),
                      **kwargs)
