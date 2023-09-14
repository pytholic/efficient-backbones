import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchreid


# * Utility functions
def make_divisible(x, divisible_by=8):
    import numpy as np

    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


def conv_bn(
    in_channels,
    out_channels,
    stride,
    activation=nn.ReLU,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        activation(inplace=True),
    )


def conv_1x1_bn(
    in_channels,
    out_channels,
    activation=nn.ReLU,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        activation(inplace=True),
    )


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
        # return x * y.expand_as(x) # ! issue with ncnn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(LinearBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(num_features=out_c)
        )

    def forward(self, x):
        return self.layers(x)

class GDC(nn.Module):
    """ Global Depthwise Convolution block """

    def __init__(self, embedding_size):
        super(GDC, self).__init__()
        self.layers = nn.Sequential(
            LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(512, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size))

    def forward(self, x):
        return self.layers(x)

class GNAP(nn.Module):
    """Global Norm-Aware Pooling block"""

    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c, affine=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn2 = nn.BatchNorm1d(in_c, affine=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = torch.norm(x, 2, 1, True)
        x_norm_mean = torch.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature

class Bottleneck(nn.Module):
    """
    Mobilenetv3 bottleneck block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        exp,
        se=False,
        nl="relu",
    ):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        padding = (kernel_size - 1) // 2
        activation = None

        if nl == "relu":
            activation = nn.ReLU
        elif nl == "hswish":
            activation = nn.Hardswish
        else:
            raise NotImplementedError

        SELayer = SEModule if se else nn.Identity

        self.bottleneck = nn.Sequential(
            # Expand
            nn.Conv2d(
                in_channels, exp, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(exp),
            activation(inplace=True),
            # Depthwise conv
            nn.Conv2d(
                exp,
                exp,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=exp,
                bias=False,
            ),
            nn.BatchNorm2d(exp),
            SELayer(exp),
            activation(inplace=True),
            # Pointwise comv
            nn.Conv2d(
                exp,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.skip:
            return x + self.bottleneck(x)
        else:
            return self.bottleneck(x)


class MobileNetV3(nn.Module):
    def __init__(
        self,
        num_classes=1,  # 1000
        input_size=256,  # 224
        dropout=0.2,
        mode="small",
        width_factor=0.5,  # 1.0
        embedding_size=512,  # 1280
        global_layer="GDC",
        **kwargs,
    ):

        super().__init__()
        input_channels = 16

        if mode == "large":
            # refer to Table 1 in paper
            cfg = [
                # k, exp, c,  se, nl,  s,
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hswish", 2],
                [3, 200, 80, False, "hswish", 1],
                [3, 184, 80, False, "hswish", 1],
                [3, 184, 80, False, "hswish", 1],
                [3, 480, 112, True, "hswish", 1],
                [3, 672, 112, True, "hswish", 1],
                [5, 672, 160, True, "hswish", 2],
                [5, 960, 160, True, "hswish", 1],
                [5, 960, 160, True, "hswish", 1],
            ]

        elif mode == "small":
            # refer to Table 2 in paper
            cfg = [
                # k, exp, c,  se, nl,  s,
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hswish", 2],
                [5, 240, 40, True, "hswish", 1],
                [5, 240, 40, True, "hswish", 1],
                [5, 120, 48, True, "hswish", 1],
                [5, 144, 48, True, "hswish", 1],
                [5, 288, 96, True, "hswish", 2], # stride was 2
                [5, 576, 96, True, "hswish", 1],
                [5, 576, 96, True, "hswish", 1],
            ]
            
        else:
            raise NotImplementedError

        # * Building first layer
        # assert input_size % 32 == 0
        embedding_size = (
            make_divisible(embedding_size * width_factor)
            if width_factor > 1.0
            else embedding_size
        )

        self.features = nn.ModuleList(
            conv_bn(
                in_channels=3,
                out_channels=input_channels,
                stride=1, # ? stride was 2
                activation=nn.Hardswish,
            )
        )

        # * Building model blocks
        for k, exp, c, se, nl, s in cfg:
            out_channels = make_divisible(c * width_factor)
            exp_channels = make_divisible(exp * width_factor)
            self.features.append(
                Bottleneck(
                    input_channels, out_channels, k, s, exp_channels, se, nl
                )
            )
            input_channels = out_channels

        # * Building last layers
        if mode == "large":
            last_conv = make_divisible(960 * width_factor)
            self.features.append(
                conv_1x1_bn(input_channels, last_conv, activation=nn.Hardswish)
            )
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(
                nn.Conv2d(
                    last_conv, embedding_size, kernel_size=1, stride=1, padding=0
                )
            )
            self.features.append(nn.Hardswish(inplace=True))

        if mode == "small":
            last_conv = make_divisible(576 * width_factor)
            self.features.append(
                conv_1x1_bn(input_channels, last_conv, activation=nn.Hardswish)
            )
            # self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, embedding_size, kernel_size=1, stride=1, padding=0))
            self.features.append(nn.BatchNorm2d(num_features=embedding_size))
            self.features.append(nn.Hardswish(inplace=True))

        else:
            NotImplementedError

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        if global_layer == "GDC":
            self.global_layer = GDC(embedding_size)
        elif global_layer == "GNAP":
            self.global_layer = GNAP(embedding_size)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(embedding_size, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.global_layer(x)
        # x = x.squeeze()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()


if __name__ == "__main__":
    net = MobileNetV3(
        input_size=256,
        num_classes=1,
        mode="small",
        embedding_size=512,
        width_factor=0.5,
    )
    net.eval()
    print("mobilenetv3:\n", net)

    x = (1, 3, 256, 128)
    x = torch.randn(x)
    out = net(x)

    print(out.shape)

    params, flops = torchreid.utils.compute_model_complexity(
        net, (1, 3, 256, 128)
    )

    print("Total params: %.2fM" % (params / 1000000.0))
    print("Total flops: %.2fM" % (flops / 1000000.0))
