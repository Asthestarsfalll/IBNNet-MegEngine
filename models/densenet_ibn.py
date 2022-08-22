from collections import OrderedDict
from typing import Any, Tuple

import megengine as mge
import megengine.functional as F
import megengine.module as M
from megengine import Tensor, hub


class IBN(M.Module):

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * (1-ratio))
        self.BN = M.BatchNorm2d(self.half)
        self.IN = M.InstanceNorm(planes - self.half)

    def forward(self, x):
        split = F.split(x, [self.half], 1)
        out1 = self.BN(split[0])
        out2 = self.IN(split[1])
        return F.concat([out1, out2], 1)


class DenseLayer(M.Sequential):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float = 0.0,
        ibn: bool = False,
    ):
        hidden_dim = bn_size * growth_rate
        layers = OrderedDict([
            ("norm1", IBN(num_input_features, 0.4)
             if ibn else M.BatchNorm2d(num_input_features)),
            ("relu1", M.ReLU()),
            ("conv1", M.Conv2d(
                num_input_features,
                hidden_dim,
                kernel_size=1,
                stride=1,
                bias=False)),
            ("norm2", M.BatchNorm2d(hidden_dim)),
            ("relu2", M.ReLU()),
            ("conv2", M.Conv2d(
                hidden_dim,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)),
            ("drop", M.Dropout(drop_rate)),
        ])
        super(DenseLayer, self).__init__(layers)

    def forward(self, inp):
        new_features = super().forward(inp)
        return F.concat([inp, new_features], axis=1)


class DenseBlock(M.Sequential):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        ibn: bool = False,
    ):
        layers = []
        for i in range(num_layers):
            if ibn and i % 3 == 0:
                ly = DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, True)
            else:
                ly = DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, False)
            layers.append(("denselayer{}".format(i + 1), ly))

        super(DenseBlock, self).__init__(OrderedDict(layers))


class Transition(M.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int):
        layer_dict = OrderedDict([
            ('norm', M.BatchNorm2d(num_input_features)),
            ('relu', M.ReLU()),
            ('conv', M.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False)),
            ('pool', M.AvgPool2d(kernel_size=2, stride=2))
        ])
        super(Transition, self).__init__(layer_dict)


class DenseNet_IBN(M.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ):

        super(DenseNet_IBN, self).__init__()

        # First convolution
        features = [
            ('conv0', M.Conv2d(
                in_channels=3,
                out_channels=num_init_features,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)),
            ('norm0', M.BatchNorm2d(num_init_features)),
            ('relu0', M.ReLU()),
            ('pool0', M.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            ibn = True
            if i >= 3:
                ibn = False
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                ibn=ibn,
            )
            features.append((f'denseblock{i + 1}', block))
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features,
                                   num_output_features=num_features // 2)
                features.append((f'transition{i + 1}', trans))
                num_features = num_features // 2

        # Final batch norm
        features.append(('norm5', M.BatchNorm2d(num_features)))
        # Linear layer
        self.classifier = M.Linear(num_features, num_classes)

        # make features
        self.features = M.Sequential(OrderedDict(features))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, M.Conv2d):
            M.init.msra_normal_(m.weight)
        elif isinstance(m, M.BatchNorm2d):
            M.init.ones_(m.weight)
            M.init.zeros_(m.bias)
        elif isinstance(m, M.Linear):
            M.init.zeros_(m.bias)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, )
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = F.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121_ibn_a(pretrained=False, **kwargs):
    return DenseNet_IBN(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                        **kwargs)


def densenet169_ibn_a(pretrained=False, **kwargs):
    return DenseNet_IBN(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                        **kwargs)


def densenet201_ibn_a(pretrained=False, **kwargs):
    return DenseNet_IBN(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                        **kwargs)


def densenet161_ibn_a(pretrained=False, **kwargs):
    return DenseNet_IBN(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                        **kwargs)
