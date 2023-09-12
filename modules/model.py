import math

import torch
import torch.nn as nn

from modules.WeightModule import ChannelAttention, SpatialAttention


def conv1x1x1(in_channels, out_channels, stride=(1, 1, 1)):
    """1x1x1 convolution"""
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def conv1x3x3(in_channels, out_channels, s_stride=1):
    """1x3x3 convolution"""
    return nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                     stride=(1, s_stride, s_stride), bias=False)


def convTx1x1(in_channels, out_channels, t_size=2, t_stride=2, t_padding=0):
    """Tx1x1 convolution"""
    return nn.Conv3d(in_channels, out_channels, kernel_size=(t_size, 1, 1), stride=(t_stride, 1, 1),
                     padding=(t_padding, 0, 0), bias=False)


def conv_layer_1x1x1(in_channels, out_channels, stride=(1, 1, 1), relu=True):
    layers = [conv1x1x1(in_channels, out_channels, stride=stride),
              nn.BatchNorm3d(out_channels)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_layer_1x3x3(in_channels, out_channels, stride=1, relu=True):
    layers = [conv1x3x3(in_channels, out_channels, s_stride=stride),
              nn.BatchNorm3d(out_channels)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def conv_layer_Tx1x1(in_channels, out_channels, t_size=2, t_stride=2, relu=True):
    layers = [convTx1x1(in_channels, out_channels, t_size, t_stride),
              nn.BatchNorm3d(out_channels)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class MyPool(nn.Module):

    def __init__(self):
        super(MyPool, self).__init__()

        self.down_avg = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.down_max = nn.AdaptiveMaxPool3d((1, 1, 1))

    def forward(self, x):
        u = self.down_avg(x)
        v = self.down_max(x)
        return torch.add(u, v)
        # return torch.cat((u, v), 1)


class MyFc(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.0):
        super(MyFc, self).__init__()

        self.fc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(in_channels, out_channels, bias=False),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# Spatiotemporal Pyramid Attention
class AttModule(nn.Module):

    def __init__(self, planes):
        super(AttModule, self).__init__()

        self.ca_1 = ChannelAttention(planes)
        self.ca_2 = ChannelAttention(planes)
        self.ca_3 = ChannelAttention(planes)
        self.ca_4 = ChannelAttention(planes)
        self.ca_T = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x_t1: n c/8 8 h w
        # x_t2: n c/4 4 h w
        # x_t3: n c/2 2 h w
        # x_t4: n c   1 h w
        x_t1, x_t2, x_t3, x_t4 = x[:]
        x_size = [x_t1.size(), x_t2.size(), x_t3.size(), x_t4.size()]

        # x_t4: n c h w (remove T)
        [x_t1, x_t2, x_t3, x_t4] = [i.view(x_t4.size()).squeeze(2) for i in [x_t1, x_t2, x_t3, x_t4]]

        x_t = torch.cat([x_t1, x_t2, x_t3, x_t4], 1)

        x_t1_ca = self.ca_1(x_t1)
        x_t2_ca = self.ca_2(x_t2)
        x_t3_ca = self.ca_3(x_t3)
        x_t4_ca = self.ca_4(x_t4)
        x_t_ca = self.ca_T(x_t)

        x_sa = self.sa(x_t)

        x_ca = torch.cat((x_t1_ca, x_t2_ca, x_t3_ca, x_t4_ca), 1) + x_t_ca
        att_vector = self.softmax(x_ca)  # x_t4: n c*4 1 1 1

        weighted_x = att_vector * x_t
        weighted_x = weighted_x * x_sa

        x_out = torch.chunk(weighted_x, 4, dim=1)  # tuple

        out = []
        for i in range(4):
            out.append(x_out[i].unsqueeze(2).view(x_size[i]))

        return tuple(out)


class MyBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1):
        super().__init__()

        self.downsample = True if stride != 1 else False

        self.relu = nn.ReLU(inplace=True)

        self.conv_t1_c1 = conv_layer_1x1x1(inplanes, planes // 8)
        self.conv_t1_c2 = conv_layer_1x3x3(planes // 8, planes // 8, stride=stride)
        self.conv_t1_c3 = conv_layer_1x1x1(planes // 8, planes, relu=False)

        self.conv_t2_c1 = conv_layer_1x1x1(inplanes, planes // 4)
        self.conv_t2_c2 = conv_layer_1x3x3(planes // 4, planes // 4, stride=stride)
        self.conv_t2_c3 = conv_layer_1x1x1(planes // 4, planes, relu=False)

        self.conv_t3_c1 = conv_layer_1x1x1(inplanes, planes // 2)
        self.conv_t3_c2 = conv_layer_1x3x3(planes // 2, planes // 2, stride=stride)
        self.conv_t3_c3 = conv_layer_1x1x1(planes // 2, planes, relu=False)

        self.conv_t4_c1 = conv_layer_1x1x1(inplanes, planes)
        self.conv_t4_c2 = conv_layer_1x3x3(planes, planes, stride=stride)
        self.conv_t4_c3 = conv_layer_1x1x1(planes, planes, relu=False)

        self.att_module = AttModule(planes)

        if self.downsample:
            self.conv_t1_d1 = conv_layer_1x1x1(inplanes, planes, stride=(1, 2, 2), relu=False)
            self.conv_t2_d1 = conv_layer_1x1x1(inplanes, planes, stride=(1, 2, 2), relu=False)
            self.conv_t3_d1 = conv_layer_1x1x1(inplanes, planes, stride=(1, 2, 2), relu=False)
            self.conv_t4_d1 = conv_layer_1x1x1(inplanes, planes, stride=(1, 2, 2), relu=False)
            # nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x_t1_in, x_t2_in, x_t3_in, x_t4_in = x[:]

        x_t1 = self.conv_t1_c1(x_t1_in)
        x_t2 = self.conv_t2_c1(x_t2_in)
        x_t3 = self.conv_t3_c1(x_t3_in)
        x_t4 = self.conv_t4_c1(x_t4_in)

        x_t1 = self.conv_t1_c2(x_t1)
        x_t2 = self.conv_t2_c2(x_t2)
        x_t3 = self.conv_t3_c2(x_t3)
        x_t4 = self.conv_t4_c2(x_t4)

        x_t1, x_t2, x_t3, x_t4 = self.att_module(tuple([x_t1, x_t2, x_t3, x_t4]))

        x_t1 = self.conv_t1_c3(x_t1)
        x_t2 = self.conv_t2_c3(x_t2)
        x_t3 = self.conv_t3_c3(x_t3)
        x_t4 = self.conv_t4_c3(x_t4)

        # if self.stride != 1: downsample
        if self.downsample:
            x_t1_in = self.conv_t1_d1(x_t1_in)
            x_t2_in = self.conv_t2_d1(x_t2_in)
            x_t3_in = self.conv_t3_d1(x_t3_in)
            x_t4_in = self.conv_t4_d1(x_t4_in)

        # Res
        x_t1 += x_t1_in
        x_t2 += x_t2_in
        x_t3 += x_t3_in
        x_t4 += x_t4_in

        x_T = [x_t1, x_t2, x_t3, x_t4]
        x_T = tuple([self.relu(i) for i in x_T])

        return x_T


# ÕûÌåµÄÍøÂç
class MyModel(nn.Module):

    def __init__(self,
                 block,
                 num_blocks=None,
                 block_inplanes=None,
                 n_input_channels=3,
                 conv_t_size=2,
                 conv_t_stride=2):
        super(MyModel, self).__init__()

        if num_blocks == None:
            num_blocks = [2, 2, 2, 2]

        if block_inplanes == None:
            block_inplanes = [64, 128, 256, 512]

        self.inplanes = block_inplanes[0]

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.inplanes,
                               kernel_size=(1, 7, 7),
                               stride=(1, 2, 2),
                               padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv_t2 = conv_layer_Tx1x1(self.inplanes, self.inplanes, conv_t_size, conv_t_stride)
        self.conv_t3 = conv_layer_Tx1x1(self.inplanes, self.inplanes, conv_t_size, conv_t_stride)
        self.conv_t4 = conv_layer_Tx1x1(self.inplanes, self.inplanes, conv_t_size, conv_t_stride)

        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       num_blocks[2],
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       num_blocks[3],
                                       stride=2)

        self.pool = MyPool()
        self.fc = MyFc(block_inplanes[3] * 4, 1, 0.5)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def _make_layer(self, block, planes, num_blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_t1 = x
        x_t2 = self.conv_t2(x_t1)
        x_t3 = self.conv_t3(x_t2)
        x_t4 = self.conv_t4(x_t3)

        x_T = tuple([x_t1, x_t2, x_t3, x_t4])
        x_T = self.layer1(x_T)
        x_T = self.layer2(x_T)
        x_T = self.layer3(x_T)
        x_T = self.layer4(x_T)

        x_T = [self.pool(i) for i in x_T]
        x_T = torch.cat([i.view(i.size(0), -1) for i in x_T], 1)

        x_T = self.fc(x_T)

        return x_T

def mynet10():
    planes = get_planes()
    model = MyModel(MyBlock,
                    num_blocks=[1, 1, 1, 1],
                    block_inplanes=planes)
    return model

def mynet18():
    planes = get_planes()
    model = MyModel(MyBlock,
                    num_blocks=[2, 2, 2, 2],
                    block_inplanes=planes)
    return model


def mynet34():
    planes = get_planes()
    model = MyModel(MyBlock,
                    num_blocks=[3, 4, 6, 3],
                    block_inplanes=planes)
    return model


def get_planes():
    return [32, 64, 128, 256]


if __name__ == '__main__':
    from torchsummary import summary
    my_model = mynet18()
    summary(my_model, (3, 8, 256, 256), depth=10)
