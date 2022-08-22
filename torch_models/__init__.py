from __future__ import absolute_import



from .resnet_ibn import resnet18_ibn_a as torch_resnet18_ibn_a, resnet34_ibn_a as torch_resnet34_ibn_a, \
                        resnet50_ibn_a as torch_resnet50_ibn_a, resnet101_ibn_a as torch_resnet101_ibn_a, \
                        resnet152_ibn_a as torch_resnet152_ibn_a, resnet18_ibn_b as torch_resnet18_ibn_b, \
                        resnet34_ibn_b as torch_resnet34_ibn_b, resnet50_ibn_b as torch_resnet50_ibn_b, \
                        resnet101_ibn_b as torch_resnet101_ibn_b, resnet152_ibn_b as torch_resnet152_ibn_b
from .densenet_ibn import densenet121_ibn_a as torch_densenet121_ibn_a, densenet169_ibn_a as torch_densenet169_ibn_a, \
                            densenet201_ibn_a as torch_densenet201_ibn_a, densenet161_ibn_a as torch_densenet161_ibn_a
from .resnext_ibn import resnext50_ibn_a as torch_resnext50_ibn_a, resnext101_ibn_a as torch_resnext101_ibn_a, \
                            resnext152_ibn_a as torch_resnext152_ibn_a  
from .se_resnet_ibn import se_resnet50_ibn_a as torch_se_resnet50_ibn_a, se_resnet101_ibn_a as torch_se_resnet101_ibn_a, \
                            se_resnet152_ibn_a as torch_se_resnet152_ibn_a

model_urls = {
    'densenet121_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet121_ibn_a-e4af5cc1.pth',
    'densenet169_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet169_ibn_a-9f32c161.pth',
    'resnet18_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'resnet34_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet18_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth',
    'resnet34_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
    'resnext101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth',
    'se_resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}
