import torch
import torch.nn as nn

import torchreid

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

class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        relu=True,
        bn=True,
    ):
        super(BasicConv, self).__init__()
        if padding > 0:
            self.pad = nn.ZeroPad2d(padding)
        else:
            self.pad = None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=not bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, 1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        return out


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ConvDW(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding=1):
        super(ConvDW, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pointwise_conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class SlimNet(nn.Module):
    def __init__(
        self,
        base_channel=16,
        dropout=0.2,
        emb_dims=512,
        num_classes=1,
        **kwargs
    ):
        super(SlimNet, self).__init__()

        self.conv_bn1 = ConvBN(3, base_channel, stride=2)
        self.conv_dw1 = ConvDW(base_channel, base_channel * 2, stride=1)
        self.conv_dw2 = ConvDW(base_channel * 2, base_channel * 2, stride=2)
        self.conv_dw3 = ConvDW(base_channel * 2, base_channel * 2, stride=1)
        self.conv_dw4 = ConvDW(base_channel * 2, base_channel * 4, stride=2)
        self.conv_dw5 = ConvDW(base_channel * 4, base_channel * 4, stride=1)
        self.conv_dw6 = ConvDW(base_channel * 4, base_channel * 4, stride=1)
        self.header_0 = ConvDW(base_channel * 4, base_channel * 4, stride=1)
        self.conv_dw7 = ConvDW(base_channel * 4, base_channel * 8, stride=2)
        self.conv_dw8 = ConvDW(base_channel * 8, base_channel * 8, stride=1)
        self.header_1 = ConvDW(base_channel * 8, base_channel * 8, stride=1)
        self.conv_dw9 = ConvDW(base_channel * 8, base_channel * 16, stride=2)
        self.header_2 = ConvDW(base_channel * 16, base_channel * 16, stride=1)
        self.extras = nn.Sequential(
            nn.Conv2d(
                base_channel * 16, base_channel * 4, kernel_size=1, padding=0
            ),
            nn.ReLU(inplace=True),
            SeparableConv(
                base_channel * 4,
                emb_dims,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            self.conv_bn1,
            self.conv_dw1,
            self.conv_dw2,
            self.conv_dw3,
            self.conv_dw4,
            self.conv_dw5,
            self.conv_dw6,
            self.header_0,
            self.conv_dw7,
            self.conv_dw8,
            self.header_1,
            self.conv_dw9,
            self.header_2,
        )

        # self.global_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.global_layer = GNAP(emb_dims)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # refer to paper section 6
            nn.Linear(emb_dims, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        header_0 = self.net[:8](x)
        header_1 = self.net[8:12](header_0)
        header_2 = self.net[12:](header_1)

        out = self.extras(header_2)

        # Apply global average pooling to get [1, 512] output
        out = self.global_layer(out)
        # out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

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
    net = SlimNet(base_channel=16)
    net = net.eval()
    print("slimnet:\n", net)

    x = (1, 3, 256, 128)
    x = torch.randn(x)
    out = net(x)

    print(out.shape)

    params, flops = torchreid.utils.compute_model_complexity(
        net, (1, 3, 256, 128)
    )

    print("Total params: %.2fM" % (params / 1000000.0))
    print("Total flops: %.2fM" % (flops / 1000000.0))
