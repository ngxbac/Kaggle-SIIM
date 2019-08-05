import torch
import torch.nn.functional as F
import torch.nn as nn


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z


class SpatialAttention2d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention2d, self).__init__()
        self.squeeze = nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z

# phalanx
class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention2d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.satt(x) + self.catt(x)


# Seutao
class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel//reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel//reduction), channel),
                                                nn.Sigmoid())

        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())

    def forward(self, x):
        bahs, chs, _, _ = x.size()

        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)

        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        return torch.add(chn_se, 1, spa_se)


# Center block
class FPAv2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FPAv2, self).__init__()
        self.glob = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                  nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

        self.down2_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=5, stride=2, padding=2, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down2_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=5, padding=2, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.down3_1 = nn.Sequential(nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=2, padding=1, bias=False),
                                     nn.BatchNorm2d(input_dim),
                                     nn.ELU(True))
        self.down3_2 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(output_dim),
                                     nn.ELU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.ELU(True))

    def forward(self, x):
        # x shape: 512, 16, 16
        x_glob = self.glob(x)  # 256, 1, 1
        x_glob = F.upsample(x_glob, scale_factor=16, mode='bilinear', align_corners=True)  # 256, 16, 16

        d2 = self.down2_1(x)  # 512, 8, 8
        d3 = self.down3_1(d2)  # 512, 4, 4

        d2 = self.down2_2(d2)  # 256, 8, 8
        d3 = self.down3_2(d3)  # 256, 4, 4

        d3 = F.upsample(d3, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 8, 8
        d2 = d2 + d3

        d2 = F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=True)  # 256, 16, 16
        x = self.conv1(x)  # 256, 16, 16
        x = x * d2

        x = x + x_glob

        return x


class DecoderAtt(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super(DecoderAtt, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1, bias=False)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(True)
        self.s_att = SpatialAttention2d(n_out)
        self.c_att = GAB(n_out, 16)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)

        cat_p = torch.cat([up_p, x_p], 1)
        cat_p = self.relu(self.bn(cat_p))
        s = self.s_att(cat_p)
        c = self.c_att(cat_p)
        return s + c


def conv3x3(input_dim, output_dim, rate=1):
    return nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=3, dilation=rate, padding=rate, bias=False),
                         nn.BatchNorm2d(output_dim),
                         nn.ELU(True))


class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = conv3x3(in_channels, channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.s_att = SpatialAttention2d(out_channels)
        self.c_att = GAB(out_channels, 16)

    def forward(self, x, e=None):
        x = F.upsample(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        if e is not None:
            x = torch.cat([x, e], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        s = self.s_att(x)
        c = self.c_att(x)
        output = s + c
        return output