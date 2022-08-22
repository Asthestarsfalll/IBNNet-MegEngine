import megengine as mge
import megengine.module as M
import megengine.functional as F

class IBN(M.Module):

    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = M.InstanceNorm(self.half)
        self.BN = M.BatchNorm2d(planes - self.half)

    def forward(self, x):
        split = F.split(x, [self.half], 1)
        out1 = self.IN(split[0])
        out2 = self.BN(split[1])
        return F.concat([out1, out2], 1)


class SELayer(M.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = M.AdaptiveAvgPool2d(1)
        self.fc = M.Sequential(
            M.Linear(channel, int(channel//reduction), bias=False),
            M.ReLU(),
            M.Linear(int(channel//reduction), channel, bias=False),
            M.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x)
        y = F.flatten(y, 1)
        y = self.fc(y).reshape(b, c, 1, 1)
        return x * y
