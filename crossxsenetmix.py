import math
import torch
import torch.nn as nn
from torchvision.models import ResNet
import numpy as np
import torch.nn.functional as F
import pdb

eps = np.finfo(float).eps
device = torch.device("cuda:0" if torch.cuda.is_available() > 0 else "cpu")

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


##################################### Loss functions
class RegularLoss(nn.Module):

    def __init__(self, gamma=0, part_features=None, nparts=1):
        """
        :param bs: batch size
        :param ncrops: number of crops used at constructing dataset
        """
        super(RegularLoss, self).__init__()
        self.register_buffer('part_features', part_features)
        self.nparts = nparts
        self.gamma = gamma
        # self.batchsize = bs
        # self.ncrops = ncrops

    def forward(self, x):
        assert isinstance(x, list), "parts features should be presented in a list"
        corr_matrix = torch.zeros(self.nparts, self.nparts)
        # x = [torch.div(xx, xx.norm(dim=1, keepdim=True)) for xx in x]
        for i in range(self.nparts):
            x[i] = x[i].squeeze()
            x[i] = torch.div(x[i], x[i].norm(dim=1, keepdim=True))

        for i in range(self.nparts):
            for j in range(self.nparts):
                corr_matrix[i, j] = torch.mean(torch.mm(x[i], x[j].t()))
                if i == j:
                    corr_matrix[i, j] = 1.0 - corr_matrix[i, j]

        return torch.mul(torch.sum(torch.triu(corr_matrix)), self.gamma).to(device)


##################################### Squeeze-and-Excitation modules
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MELayer(nn.Module):
    def __init__(self, channel, reduction=16, nparts=1):
        super(MELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.nparts = nparts
        parts = list()
        for part in range(self.nparts):
            parts.append(nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
            ))
        self.parts = nn.Sequential(*parts)
        self.dresponse = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        meouts = list()
        for i in range(self.nparts):
            meouts.append(x * self.parts[i](y).view(b, c, 1, 1))

        y = self.dresponse(y).view(b, c, 1, 1)
        return x * y, meouts


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, meflag=False, nparts=1):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.meflag = meflag
        if self.meflag:
            self.se = MELayer(planes * 4, reduction=reduction, nparts=nparts)
        else:
            self.se = SELayer(planes * 4, reduction=reduction)
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

        if self.meflag:
            out, parts = self.se(out)

            out += residual
            out = self.relu(out)

            for i in range(len(parts)):
                parts[i] = self.relu(parts[i] + residual)

            return out, parts
        else:
            out = self.se(out)
            out += residual
            out = self.relu(out)
            return out

###################################### ResNet framework
class SeNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, rd=[16, 16, 16, 16], nparts=1, meflag=False):
        """
        :param rd: reductions in SENet
        :param meflag: Ture for crossx senet, Flase for default senet

        """
        super(SeNet, self).__init__()

        self.inplanes = 64
        self.meflag = meflag
        self.rd = rd
        self.nparts = nparts
        if not self.meflag:
            assert self.nparts == 1

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], reduction=self.rd[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=self.rd[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, reduction=self.rd[2], meflag=meflag, nparts=nparts)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, reduction=self.rd[3], meflag=meflag, nparts=nparts)
        self.adpavgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_ulti = nn.Linear(512 * block.expansion * nparts, num_classes)
        if self.nparts > 1:
            self.adpmaxpool = nn.AdaptiveMaxPool2d(1)
            self.fc_plty = nn.Linear(256 * block.expansion * nparts, num_classes)
            self.fc_cmbn = nn.Linear(256 * block.expansion * nparts, num_classes)

            # for the last convolutional layer
            self.conv2_1 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
            self.conv2_2 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
            # self.bn2_1 = nn.BatchNorm2d(256 * block.expansion)
            # self.bn2_2 = nn.BatchNorm2d(256 * block.expansion)

            # for the penultimate layer
            self.conv3_1 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
            self.conv3_2 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
            self.bn3_1 = nn.BatchNorm2d(256 * block.expansion)
            self.bn3_2 = nn.BatchNorm2d(256 * block.expansion)

            if nparts == 3:
                self.conv2_3 = nn.Conv2d(512 * block.expansion, 256 * block.expansion, kernel_size=1, bias=False)
                self.conv3_3 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=3, padding=1, bias=False)
                self.bn3_3 = nn.BatchNorm2d(256 * block.expansion)

        # initializing params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, reduction=16, meflag=False, nparts=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, reduction))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == blocks - 1 and meflag is True:
                layers.append(block(self.inplanes, planes, reduction=reduction, meflag=meflag, nparts=nparts))
            else:
                layers.append(block(self.inplanes, planes, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        if self.meflag:
            x, plty_parts = self.layer3(x)
            _, ulti_parts = self.layer4(x)

            cmbn_ftres = list()
            for i in range(self.nparts):
                # pdb.set_trace()
                if i == 0:
                    ulti_parts_iplt = F.interpolate(self.conv2_1(ulti_parts[i]), 28)
                    cmbn_ftres.append(self.adpavgpool(self.bn3_1(self.conv3_1(torch.add(plty_parts[i], ulti_parts_iplt)))))
                elif i == 1:
                    ulti_parts_iplt = F.interpolate(self.conv2_2(ulti_parts[i]), 28)
                    cmbn_ftres.append(self.adpavgpool(self.bn3_2(self.conv3_2(torch.add(plty_parts[i], ulti_parts_iplt)))))
                elif i == 2:
                    ulti_parts_iplt = F.interpolate(self.conv2_3(ulti_parts[i]), 28)
                    cmbn_ftres.append(self.adpavgpool(self.bn3_3(self.conv3_3(torch.add(plty_parts[i], ulti_parts_iplt)))))

                plty_parts[i] = self.adpmaxpool(plty_parts[i])
                ulti_parts[i] = self.adpavgpool(ulti_parts[i])

            # for the penultimate layer
            xp = torch.cat(plty_parts, 1)
            xp = xp.view(xp.size(0), -1)
            xp = self.fc_plty(xp)

            # for the final layer
            xf = torch.cat(ulti_parts, 1)
            xf = xf.view(xf.size(0), -1)
            xf = self.fc_ulti(xf)

            # for the combined feature
            xc = torch.cat(cmbn_ftres, 1)
            xc = xc.view(xc.size(0), -1)
            xc = self.fc_cmbn(xc)

            return xf, xp, xc, ulti_parts, plty_parts, cmbn_ftres
        else:
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.adpavgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_ulti(x)
            return x



########################################## Models
def senet50(num_classes=200, nparts=1, **kwargs):
    
    if nparts > 1:
        # resnet with osme
        kwargs.setdefault('meflag', True)
    else:
        # the normal resnet
        kwargs.setdefault('meflag', False)

    rd = [16, 32, 64, 128]

    if kwargs['meflag']:
        model = SeNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, rd=rd, nparts=nparts, meflag=True)
    else:
        # vanilla senet
        model = SeNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, rd=rd, nparts=nparts, meflag=False)
    
    return model
