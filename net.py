import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from thop import profile
from davit import CrossSpatialBlock,CrossChannelBlock, CrossSpatialBlock1, CrossChannelBlock1,CrossSpatialBlock2, CrossChannelBlock2
from einops.layers.torch import Rearrange


class channel_att(nn.Module):
    def __init__(self,channels):
        super(channel_att, self).__init__()

        self.att = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(channels, channels, 1, bias=False),
                                 nn.Sigmoid()
                                )
    def forward(self,x):
        att = self.att(x)
        out = x*att
        return out


class block1(nn.Module):
    def __init__(self, channels):
        super(block1, self).__init__()
        self.attenion = channel_att(channels)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, 1, 1),
                                    nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.attenion(x)
        out = x1+x2
        return out

class spectal_att(nn.Module):
    def __init__(self,channels):
        super(spectal_att, self).__init__()
        self.att = nn.Sequential(
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(channels, channels//4, 1, 1, 0),
                                    nn.Conv2d(channels//4, channels, 1, 1, 0),
                                    nn.Sigmoid()
                                    )
    def forward(self,x):
        att = self.att(x)
        out = x*att
        return out


class spa_att(nn.Module):
    def __init__(self,channels):
        super(spa_att, self).__init__()
        self.att = nn.Sequential(
                                 nn.Conv2d(channels, channels//4, 1, 1, 0),
                                 nn.Conv2d(channels//4, channels, 1, 1, 0),
                                 nn.Sigmoid()
                                )
    def forward(self,x):
        att = self.att(x)
        out = x*att
        return out


class spa_block(nn.Module):
    def __init__(self, channels):
        super(spa_block, self).__init__()
        self.attenion = spa_att(channels)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, 1, 1),
                                  )
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.attenion(x)
        out = x1+x2
        return out

class spe_block(nn.Module):
    def __init__(self, channels):
        super(spe_block, self).__init__()
        self.attenion = spectal_att(channels)
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, 1, 1),
                                  )
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.attenion(x)
        out = x1+x2
        return out


class transition1(nn.Module):
    def __init__(self, channels):
        super(transition1, self).__init__()
        self.conv1 =nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
                )
    def forward(self,c3,c4,c5):
        c5_ = self.conv1(c5)
        c5_ = F.interpolate(c5_,scale_factor=2,mode='bilinear')
        p4_ = self.conv2(c5_+c4)
        p4 = F.interpolate(p4_,scale_factor=2,mode='bilinear')
        p3 = p4 + c3
        return p3, p4_, c5

class fearture_extractor(nn.Module):
    def __init__(self, channels):
        super(fearture_extractor, self).__init__()
        self.blockpa1 = nn.Sequential(spa_block(channels=channels))
        self.downpa1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.blockpa2 = nn.Sequential(spa_block(channels=channels))
        self.downpa2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.focalpa = spa_block(channels=channels)
        self.transitiona = transition1(channels)
    def forward(self, x):
        pac3 = self.blockpa1(x)
        pac3_ = self.downpa1(pac3)
        pac4 = self.blockpa2(pac3_)
        pac4_ = self.downpa2(pac4)
        pac5 = self.focalpa(pac4_)
        pac3, pac4, pac5 = self.transitiona(pac3, pac4, pac5)
        return pac3, pac4, pac5

class fearture_extractor1(nn.Module):
    def __init__(self, channels):
        super(fearture_extractor1, self).__init__()
        self.blockpa1 = nn.Sequential(spe_block(channels=channels))
        self.downpa1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.blockpa2 = nn.Sequential(spe_block(channels=channels))
        self.downpa2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.focalpa = spe_block(channels=channels)
        self.transitiona = transition1(channels)
    def forward(self, x):
        pac3 = self.blockpa1(x)
        pac3_ = self.downpa1(pac3)
        pac4 = self.blockpa2(pac3_)
        pac4_ = self.downpa2(pac4)
        pac5 = self.focalpa(pac4_)
        pac3, pac4, pac5 = self.transitiona(pac3, pac4, pac5)
        return pac3, pac4, pac5




class Lrms(nn.Module):
    def __init__(self, channels):
        super(Lrms, self).__init__()
        self.conv1 = spe_block(channels)
        self.conv2 = spe_block(channels)
        self.conv3 = spe_block(channels)
    def forward(self, x):
        x1 = self.conv1(x)
        x1_ = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x2 = self.conv2(x1_)
        x2_ = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3 = self.conv3(x2_)
        return x1,x2,x3

class Extrctor(nn.Module):
    def __init__(self, channels):
        super(Extrctor, self).__init__()
        #self.pa = AttBlock1(dim=channels)

        self.pa = fearture_extractor(channels)
        self.pe = fearture_extractor1(channels)
    def forward(self,x,y):
        pac3,pac4,pac5 = self.pa(x)
        #pac3 = self.pa(x)
        pec3, pec4, pec5 = self.pe(y)
        return pac3,pac4,pac5, pec3, pec4,pec5
class DualAggregation(nn.Module):
    def __init__(self, channels):
        super(DualAggregation, self).__init__()
        self.before_RGp = Rearrange('b c h w -> b (h w) c')
        self.before_RGm = Rearrange('b c h w -> b (h w) c')
        self.CrossSpatial = CrossSpatialBlock(dim=channels, ffn=False)
        self.CrossChannel = CrossChannelBlock(dim=channels, ffn=False)
        self.reduce = nn.Sequential(nn.Conv2d(channels*2, channels, 1, 1, 0),nn.ReLU())
    def forward(self, p, m):
        B_,c,H,W = p.shape
        p = self.before_RGp(p)
        m = self.before_RGm(m)
        pf = self.CrossSpatial(p, m, [H,W])
        mf = self.CrossChannel(p, m, [H,W])
        pf = rearrange(pf, "b (h w) c -> b c h w", h=H, w=W)
        mf = rearrange(mf, "b (h w) c -> b c h w", h=H, w=W)
        f = torch.cat([pf, mf], dim=1)
        f = self.reduce(f)
        return f

class DualAggregation1(nn.Module):
    def __init__(self, channels):
        super(DualAggregation1, self).__init__()
        self.CrossSpatial = CrossSpatialBlock2(dim=channels, GFN=True)
        self.CrossChannel = CrossChannelBlock2(dim=channels, GFN=True)
        self.reduce = nn.Sequential(nn.Conv2d(channels*2, channels, 1, 1, 0),nn.ReLU())
    def forward(self, p, m):
        B_,c,H,W = p.shape
        p = rearrange(p, "b c h w -> b (h w) c", h=H, w=W)
        m = rearrange(m, "b c h w -> b (h w) c", h=H, w=W)
        pf = self.CrossSpatial(p, m, [H,W])
        mf = self.CrossChannel(p, m, [H,W])
        f = torch.cat([pf, mf], dim=1)
        f = self.reduce(f)
        return f


class DualAggregation2(nn.Module):
    def __init__(self, channels):
        super(DualAggregation2, self).__init__()
        self.CrossSpatial = CrossSpatialBlock1(dim=channels, ffn=True)
        self.CrossChannel = CrossChannelBlock1(dim=channels, ffn=True)
        self.reduce = nn.Sequential(nn.Conv2d(channels*2, channels, 1, 1, 0),nn.ReLU())
    def forward(self, p, m):
        B_,c,H,W = p.shape
        p = rearrange(p, "b c h w -> b (h w) c", h=H, w=W)
        m = rearrange(m, "b c h w -> b (h w) c", h=H, w=W)
        pf = self.CrossSpatial(p, m, [H,W])
        mf = self.CrossChannel(p, m, [H,W])
        f = torch.cat([pf, mf], dim=1)
        f = self.reduce(f)
        return f











class net(nn.Module):
    def __init__(self, pan_channels, ms_channels, channels):
        super(net, self).__init__()

        self.pre_pan = nn.Sequential(nn.Conv2d(pan_channels, channels, 1, 1, 0),nn.ReLU())
        self.pre_lrms = nn.Sequential(nn.Conv2d(ms_channels, channels, 1, 1, 0),nn.ReLU())

        self.extrctor1 = Extrctor(channels)
        self.DualAggregation1=DualAggregation1(channels)



        self.reconstruct = nn.Sequential(nn.Conv2d(channels*3, channels*2, 1, 1, 0),
                                         nn.ReLU(),
                                         nn.Conv2d(channels * 2, channels, 1, 1, 0),
                                         nn.ReLU(),
                                         nn.Conv2d(channels, ms_channels, 1, 1, 0)
                                         )



    def forward(self, lrms, pan):

        ms_ = F.interpolate(lrms, scale_factor=4, mode='bilinear')

        p1 = self.pre_pan(pan)
        m1 = self.pre_lrms(ms_)

        pac3,pac4,pac5, pec3, pec4,pec5 = self.extrctor1(m1,p1 )
        f1 = self.DualAggregation1(pac3,pec3)


        pac3,pac4,pac5, pec3, pec4,pec5 = self.extrctor1(pac3, pec3)
        f2 = self.DualAggregation1(pac3,pec3)



        pac3,pac4,pac5, pec3, pec4,pec5 = self.extrctor1(pac3, pec3)
        f3 = self.DualAggregation1(pac3, pec3)



        f = torch.cat([f1,f2,f3],dim=1)



        f = self.reconstruct(f)+ms_

        return f




