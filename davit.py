""" Written by Mingyu """
import logging
from copy import deepcopy
import itertools
from typing import Tuple
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple,trunc_normal_
from einops.layers.torch import Rearrange
from dat_arch import SGFN,EGFN





class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=96,
            overlapped=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        if patch_size[0] == 4:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(7, 7),
                stride=patch_size,
                padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=to_2tuple(kernel),
                stride=patch_size,
                padding=to_2tuple(pad))
            self.norm = nn.LayerNorm(in_chans)

    def forward(self, x, size):
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B,
                          H,
                          W,
                          C).permute(0, 3, 1, 2).contiguous()

        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize


class ChannelAttention(nn.Module):
    r""" Channel based self attention.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of the groups.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SpatialBlock(nn.Module):
    r""" Spatial-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads=8, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class CrossWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, kv_bias=True,q_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)
        self.q = nn.Linear(dim, dim, bias=q_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,y):
        B_, N, C = x.shape
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x






class CrossSpatialBlock(nn.Module):
    r""" Spatial-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads=8, window_size=8,
                 mlp_ratio=4., kv_bias=True,q_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.cpex = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.cpey = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)


        self.attnx = CrossWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            kv_bias=kv_bias,q_bias=q_bias)

        self.attny = CrossWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            kv_bias=kv_bias, q_bias=q_bias)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2x = norm_layer(dim)
            self.norm2y = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlpx = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
            self.mlpy = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x,y, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcutx = self.cpex[0](x, size)
        shortcuty = self.cpey[0](y, size)

        x = self.norm1x(shortcutx)
        y = self.norm1y(shortcuty)

        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size


        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = y.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        y_windows = window_partition(y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        attn_windowsx = self.attnx(x_windows,y_windows)
        attn_windowsy = self.attny(y_windows,x_windows)

        attn_windowsx = attn_windowsx.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        attn_windowsy = attn_windowsy.view(-1,
                                           self.window_size,
                                           self.window_size,
                                           C)

        x = window_reverse(attn_windowsx, self.window_size, Hp, Wp)
        y = window_reverse(attn_windowsy, self.window_size, Hp, Wp)


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            y = y[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        x = shortcutx + self.drop_pathx(x)
        y = shortcuty + self.drop_pathy(y)

        x = self.cpex[1](x, size)
        y = self.cpey[1](y, size)


        if self.ffn:
            x = x + self.drop_pathx(self.mlpx(self.norm2x(x)))
            y = y + self.drop_pathy(self.mlpy(self.norm2y(y)))
        out = x+ y
        return out

class CrossChannelAttention(nn.Module):
    r""" Channel based self attention.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of the groups.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x,y):
        B, N, C = x.shape

        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class CrossChannelBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()

        self.cpex = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.cpey = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.ffn = ffn
        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)

        self.attnx = CrossChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.attny = CrossChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2x = norm_layer(dim)
            self.norm2y = norm_layer(dim)

            mlp_hidden_dim = int(dim * mlp_ratio)

            self.mlpx = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)
            self.mlpy = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, y,size):

        x = self.cpex[0](x, size)
        y = self.cpey[0](y, size)

        curx = self.norm1x(x)
        cury = self.norm1y(y)

        curx = self.attnx(curx,cury)
        cury = self.attny(cury,curx)

        x = x + self.drop_pathx(curx)
        y = y + self.drop_pathy(cury)


        x = self.cpex[1](x, size)
        y = self.cpey[1](y, size)

        if self.ffn:
            x = x + self.drop_pathx(self.mlpx(self.norm2x(x)))
            y = y + self.drop_pathy(self.mlpy(self.norm2y(y)))
        out = x + y
        return out







class CrossSpatialBlock1(nn.Module):
    r""" Spatial-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads=8, window_size=8,
                 mlp_ratio=4., kv_bias=True,q_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.cpex = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.cpey = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)


        self.attnx = CrossWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            kv_bias=kv_bias,q_bias=q_bias)

        self.attny = CrossWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            kv_bias=kv_bias, q_bias=q_bias)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.channel_interactionx = nn.Sequential(
        #     nn.Conv2d(dim, dim // 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim //4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )
        #
        # self.channel_interactiony = nn.Sequential(
        #     nn.Conv2d(dim, dim // 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim //4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.Spatialinteraction = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1),
            #nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim*2, kernel_size=1),
            nn.Sigmoid()
        )

        self.before_Rf = Rearrange('b c h w -> b (h w) c')
        if self.ffn:
                self.norm2 = norm_layer(dim)
                mlp_hidden_dim = int(dim * mlp_ratio)
                self.mlp = Mlp(
                    in_features=dim,
                    hidden_features=mlp_hidden_dim,
                    act_layer=act_layer)

    def forward(self, x,y, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcutx = self.cpex[0](x, size)
        shortcuty = self.cpey[0](y, size)

        x = self.norm1x(shortcutx)
        y = self.norm1y(shortcuty)

        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size


        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = y.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        y_windows = window_partition(y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        attn_windowsx = self.attnx(x_windows,y_windows)
        attn_windowsy = self.attny(y_windows,x_windows)

        attn_windowsx = attn_windowsx.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        attn_windowsy = attn_windowsy.view(-1,
                                           self.window_size,
                                           self.window_size,
                                           C)

        x = window_reverse(attn_windowsx, self.window_size, Hp, Wp)
        y = window_reverse(attn_windowsy, self.window_size, Hp, Wp)


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            y = y[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        x = shortcutx + self.drop_pathx(x)
        y = shortcuty + self.drop_pathy(y)

        x = self.cpex[1](x, size)
        y = self.cpey[1](y, size)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        y = rearrange(y, "b (h w) c -> b c h w", h=H, w=W)
        #
        # s_x = self.channel_interactionx(x)
        # s_y = self.channel_interactiony(y)
        # f = s_x*y + s_y*x

        s = torch.cat([x,y],dim=1)
        s = self.Spatialinteraction(s)
        s_x, s_y = s.chunk(2, dim=1)
        f = s_x*y + s_y*x


        if self.ffn:
            f = self.before_Rf(f)
            f = f + self.drop_pathy(self.mlp(self.norm2(f)))
            f = rearrange(f, "b (h w) c -> b c h w", h=H, w=W)

        return f



class CrossChannelBlock1(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True):
        super().__init__()

        self.cpex = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.cpey = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.ffn = ffn
        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)

        self.attnx = CrossChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.attny = CrossChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()



        # self.channel_interactionx = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim * 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim * 4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )
        #
        # self.channel_interactiony = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim * 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim * 4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.Channelinteraction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            #nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.before_Rf = Rearrange('b c h w -> b (h w) c')


        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)


    def forward(self, x, y,size):
        H, W = size
        x = self.cpex[0](x, size)
        y = self.cpey[0](y, size)

        curx = self.norm1x(x)
        cury = self.norm1y(y)

        curx = self.attnx(curx,cury)
        cury = self.attny(cury,curx)

        x = x + self.drop_pathx(curx)
        y = y + self.drop_pathy(cury)


        x = self.cpex[1](x, size)
        y = self.cpey[1](y, size)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        y = rearrange(y, "b (h w) c -> b c h w", h=H, w=W)

        # c_x = self.channel_interactionx(x)
        # c_y = self.channel_interactiony(y)
        # f = c_x * y + c_y * x

        s = torch.cat([x, y], dim=1)
        s = self.Channelinteraction(s)
        s_x, s_y = s.chunk(2, dim=1)
        f = s_x * y + s_y * x



        if self.ffn:
            f = self.before_Rf(f)
            f = f + self.drop_pathy(self.mlp(self.norm2(f)))
            f = rearrange(f, "b (h w) c -> b c h w", h=H, w=W)


        return f


class CrossSpatialBlock2(nn.Module):
    r""" Spatial-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads=8, window_size=8,
                 mlp_ratio=4., kv_bias=True,q_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 GFN=True):
        super().__init__()
        self.dim = dim
        #self.ffn = ffn
        self.GFN = GFN
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.cpex = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.cpey = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)


        self.attnx = CrossWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            kv_bias=kv_bias,q_bias=q_bias)

        self.attny = CrossWindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            kv_bias=kv_bias, q_bias=q_bias)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.channel_interactionx = nn.Sequential(
        #     nn.Conv2d(dim, dim // 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim //4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )
        #
        # self.channel_interactiony = nn.Sequential(
        #     nn.Conv2d(dim, dim // 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim //4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.Spatialinteraction = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1),
            #nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim*2, kernel_size=1),
            nn.Sigmoid()
        )

        self.before_Rf = Rearrange('b c h w -> b (h w) c')
        # if self.ffn:
        #         self.norm2 = norm_layer(dim)
        #         mlp_hidden_dim = int(dim * mlp_ratio)
        #         self.mlp = Mlp(
        #             in_features=dim,
        #             hidden_features=mlp_hidden_dim,
        #             act_layer=act_layer)
        if self.GFN:
            self.norm2 = norm_layer(dim)
            self.SGFN = SGFN(in_features=dim, hidden_features=dim * 4)

    def forward(self, x,y, size):
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcutx = self.cpex[0](x, size)
        shortcuty = self.cpey[0](y, size)

        x = self.norm1x(shortcutx)
        y = self.norm1y(shortcuty)

        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size


        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = y.shape

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        y_windows = window_partition(y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        attn_windowsx = self.attnx(x_windows,y_windows)
        attn_windowsy = self.attny(y_windows,x_windows)

        attn_windowsx = attn_windowsx.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        attn_windowsy = attn_windowsy.view(-1,
                                           self.window_size,
                                           self.window_size,
                                           C)

        x = window_reverse(attn_windowsx, self.window_size, Hp, Wp)
        y = window_reverse(attn_windowsy, self.window_size, Hp, Wp)


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
            y = y[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        x = shortcutx + self.drop_pathx(x)
        y = shortcuty + self.drop_pathy(y)

        x = self.cpex[1](x, size)
        y = self.cpey[1](y, size)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        y = rearrange(y, "b (h w) c -> b c h w", h=H, w=W)
        #
        # s_x = self.channel_interactionx(x)
        # s_y = self.channel_interactiony(y)
        # f = s_x*y + s_y*x

        s = torch.cat([x,y],dim=1)
        s = self.Spatialinteraction(s)
        s_x, s_y = s.chunk(2, dim=1)
        f = s_x*y + s_y*x

        f = self.before_Rf(f)
        # if self.ffn:
        #     f = f + self.drop_pathy(self.mlp(self.norm2(f)))

        if self.GFN:
            f = f + self.norm2(self.SGFN(f,H, W))
            f = rearrange(f, "b (h w) c -> b c h w", h=H, w=W)

        return f



class CrossChannelBlock2(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 GFN=True):
        super().__init__()

        self.cpex = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.cpey = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        #self.ffn = ffn
        self.GFN = GFN
        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)

        self.attnx = CrossChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.attny = CrossChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)

        self.drop_pathx = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_pathy = DropPath(drop_path) if drop_path > 0. else nn.Identity()



        # self.channel_interactionx = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim * 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim * 4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )
        #
        # self.channel_interactiony = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(dim, dim * 4, kernel_size=1),
        #     nn.GELU(),
        #     nn.Conv2d(dim * 4, dim, kernel_size=1),
        #     nn.Sigmoid()
        # )

        self.Channelinteraction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            #nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.Sigmoid()
        )

        self.before_Rf = Rearrange('b c h w -> b (h w) c')


        # if self.ffn:
        #     self.norm2 = norm_layer(dim)
        #     mlp_hidden_dim = int(dim * mlp_ratio)
        #     self.mlp = Mlp(
        #         in_features=dim,
        #         hidden_features=mlp_hidden_dim,
        #         act_layer=act_layer)

        if self.GFN:
            self.norm2 = norm_layer(dim)
            self.EGFN = EGFN(in_features=dim, hidden_features=dim * 4)

    def forward(self, x, y,size):
        H, W = size
        x = self.cpex[0](x, size)
        y = self.cpey[0](y, size)

        curx = self.norm1x(x)
        cury = self.norm1y(y)

        curx = self.attnx(curx,cury)
        cury = self.attny(cury,curx)

        x = x + self.drop_pathx(curx)
        y = y + self.drop_pathy(cury)


        x = self.cpex[1](x, size)
        y = self.cpey[1](y, size)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        y = rearrange(y, "b (h w) c -> b c h w", h=H, w=W)

        # c_x = self.channel_interactionx(x)
        # c_y = self.channel_interactiony(y)
        # f = c_x * y + c_y * x

        s = torch.cat([x, y], dim=1)
        s = self.Channelinteraction(s)
        s_x, s_y = s.chunk(2, dim=1)
        f = s_x * y + s_y * x

        f = self.before_Rf(f)

        # if self.ffn:
        #     f = f + self.drop_pathy(self.mlp(self.norm2(f)))
        #     f = rearrange(f, "b (h w) c -> b c h w", h=H, w=W)

        if self.EGFN:
            f = f + self.norm2(self.EGFN(f,H, W))
            f = rearrange(f, "b (h w) c -> b c h w", h=H, w=W)

        return f


