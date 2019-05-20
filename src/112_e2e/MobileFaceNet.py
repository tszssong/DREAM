import torch.nn as nn
import math

__all__ = ['mobilefacenet']


def _make_divisible(v, divisor, min_value=None):
    
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_7x7_bn_noRelu(inp, oup, stride=1, num_group=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 7, stride, 0, groups=num_group, bias=False),
        nn.BatchNorm2d(oup)
    )

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    def __init__(self, num_classes=1000, input_size=112, width_mult=1., \
                 fc_type = "GDC", emb_size = 128, end2end=False):
        super(MobileFaceNet, self).__init__()
        self.end2end = end2end
        self.fc_type = fc_type
        self.num_fmap_end = 512
        self.emb_size = emb_size
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [2,  64, 2, 1], #res_2
            [2, 128, 1, 2], #deconv_23
            [2, 128, 4, 1], #res_3
            [2, 256, 1, 2], #deconv_34
            [2, 256, 6, 1], #res_4
            [4, 256, 1, 2], #deconv_45
            [2, 256, 3, 1], #res_5
        ]

        # building first layer
        assert input_size % 16 == 0
        input_channel = 64
        layers = [conv_3x3_bn(3, input_channel, 2)] #conv_1
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = c
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        layers.append(conv_1x1_bn(256, self.num_fmap_end))
        self.features = nn.Sequential(*layers)
        # building last several layers
        if fc_type == "GDC": 
            self.conv = conv_7x7_bn_noRelu(self.num_fmap_end, self.num_fmap_end, 1, self.num_fmap_end)  
            self.fc = nn.Linear(self.num_fmap_end, self.emb_size)        
            self.bn = nn.BatchNorm1d(self.emb_size,  affine=False)
            self.classifier = nn.Linear(self.emb_size, num_classes)
        elif fc_type == "GNAP":
            self.bn1 = nn.BatchNorm2d(self.num_fmap_end, affine=False)
            # not completed!
            # self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        self._initialize_weights()

    # def forward(self, x, yaw):
    def forward(self, x):
        x = self.features(x)
        if self.fc_type == "GDC": 
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = self.bn(x)
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilefacenet(pretrained=False,**kwargs):
    if pretrained:
        print("[ds]Error: not supported yet!")
    return MobileFaceNet(**kwargs)
