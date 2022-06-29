from __future__ import absolute_import

import torch.nn as nn


__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3', 'AlexNetV4', 'AlexNetV5', 'QAlexNetV5', 'MobileNetV1', 'resnet18']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))


class MobileNetV1(nn.Module):
    def __init__(self, quantized=False):
        super(MobileNetV1, self).__init__()

        # first_layer_bit_width = 8
        # weight_bit_width = 4
        # last_layer_bit_width = 8
        # activation_bit_width = 4

        self.quantized = quantized

        def conv_bn(inp, oup, stride):
            if not self.quantized:
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )

            else:
                raise NotImplementedError

        def conv_dw(inp, oup, stride):
            if not self.quantized:
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )

            else:
                raise NotImplementedError

        def last_conv_dw(inp, oup, stride):
            if not self.quantized:
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
                )

            else:
                raise NotImplementedError

        if not self.quantized:
            self.features = nn.Sequential(
                conv_bn(3, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                last_conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                last_conv_dw(256, 256, 1)
            )
            print("Initializing floating point version of MobileNet for Siamese Tracker.")

        else:
            raise NotImplementedError
            # print("Initializing quantized version of MobileNet for Siamese Tracker.")

    def forward(self, x):
        return self.features(x)

import math

import torch.nn as nn
import torch


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# DPR: based on AlexNet, with all filters 3x3 (maybe more like VGG))
class AlexNetV4(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV4, self).__init__()
        print("Initializing modified AlexNet for SiamTrack backbone.")
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 3),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 96, 3),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


# DPR: based on AlexNet, with all filters 3x3 (maybe more like VGG)); AND squeezed to fit FINN & ZCU104
class AlexNetV5(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV5, self).__init__()
        print("Initializing modified and squeezed AlexNet for SiamTrack backbone.")
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            _BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            _BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            _BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1),
            _BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1),
            _BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1))


import brevitas.nn as qnn


# DPR: based on AlexNet, with all filters 3x3 (maybe more like VGG)); AND squeezed to fit FINN & ZCU104; AND quantized
class QAlexNetV5(_AlexNet):
    output_stride = 8

    def __init__(self, weights_bitwidth=4, activation_bitwidth=4):
        super(QAlexNetV5, self).__init__()
        print("Initializing modified and squeezed QUANTIZED AlexNet for SiamTrack backbone.")
        self.conv1 = nn.Sequential(
            qnn.QuantConv2d(3, 64, 3, weight_bit_width=8),
            _BatchNorm2d(64),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
            nn.MaxPool2d(2, 2),
            qnn.QuantConv2d(64, 64, 3, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(64),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            qnn.QuantConv2d(64, 128, 3, 1, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(128),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(128),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True))
        self.conv4 = nn.Sequential(
            qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(128),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True))
        self.conv5 = nn.Sequential(
            qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=8))


# DPR: based on AlexNet, with all filters 3x3 (maybe more like VGG)); AND squeezed firmly to fit FINN & ZCU104;
# AND quantized
class QAlexNetV7(_AlexNet):
    output_stride = 8

    def __init__(self, weights_bitwidth=4, activation_bitwidth=4):
        super(QAlexNetV7, self).__init__()
        print("Initializing modified and squeezed QUANTIZED AlexNet V7 for SiamTrack backbone.")
        self.conv1 = nn.Sequential(
            qnn.QuantConv2d(3, 64, 3, weight_bit_width=8),
            _BatchNorm2d(64),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
            nn.MaxPool2d(2, 2),
            qnn.QuantConv2d(64, 64, 3, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(64),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
            nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(
            qnn.QuantConv2d(64, 128, 3, 1, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(128),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
            nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(
            qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(128),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True))
        self.conv4 = nn.Sequential(
            qnn.QuantConv2d(128, 64, 3, 1, weight_bit_width=weights_bitwidth),
            _BatchNorm2d(64),
            qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True))
        self.conv5 = nn.Sequential(
            qnn.QuantConv2d(64, 64, 3, 1, weight_bit_width=8))


# # DPR: based on AlexNet, with all filters 3x3 (maybe more like VGG)); AND squeezed to fit FINN & ZCU104; AND quantized;
# # AND ??????? to try to meet real time constraints
# class QAlexNetV6(_AlexNet):
#     output_stride = 8
#
#     def __init__(self, weights_bitwidth=4, activation_bitwidth=4):
#         super(QAlexNetV6, self).__init__()
#         print("Initializing modified and squeezed QUANTIZED AlexNet for SiamTrack backbone.")
#         self.conv1 = nn.Sequential(
#             qnn.QuantConv2d(3, 64, 3, weight_bit_width=8, bias=False),
#             _BatchNorm2d(64),
#             qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
#             nn.MaxPool2d(2, 2),
#             qnn.QuantConv2d(64, 64, 3, weight_bit_width=weights_bitwidth, bias=False),
#             _BatchNorm2d(64),
#             qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
#             nn.MaxPool2d(2, 2))
#         self.conv2 = nn.Sequential(
#             qnn.QuantConv2d(64, 128, 3, 1, weight_bit_width=weights_bitwidth, bias=False),
#             _BatchNorm2d(128),
#             qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True),
#             nn.MaxPool2d(2, 2))
#         self.conv3 = nn.Sequential(
#             qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=weights_bitwidth, bias=False),
#             _BatchNorm2d(128),
#             qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True))
#         self.conv4 = nn.Sequential(
#             qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=weights_bitwidth, bias=False),
#             _BatchNorm2d(128),
#             qnn.QuantReLU(bit_width=activation_bitwidth, return_quant_tensor=True))
#         self.conv5 = nn.Sequential(
#             qnn.QuantConv2d(128, 128, 3, 1, weight_bit_width=8, bias=False))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, used_layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 15x15, 7x7
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 7x7, 3x3
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)

        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        p4 = self.layer4(p3)
        # out = [x_, p1, p2, p3, p4]
        # out = [out[i] for i in self.used_layers]
        # if len(out) == 1:
        #     return out[0]
        # else:
        #     return out

        return p4


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    print("Initialized ResNet18 as SiamFC backbone.")
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet18(used_layers=[1, 2, 3, 4])
    print(net)
    net = net.cuda()

    var = torch.FloatTensor(1, 3, 127, 127).cuda()
    # var = Variable(var)
    o = net(var)
    print(o.shape)

    print('*************')
    var = torch.FloatTensor(1, 3, 255, 255).cuda()
    # var = Variable(var)
    net(var)
    o = net(var)
    print(o.shape)
