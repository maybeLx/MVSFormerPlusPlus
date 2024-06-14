import copy
import pdb
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino.layers.attention import get_attention_type, MemEffAttention, FlashAttention2, CrossLinearAttention
from models.dino.layers.block import CrossBlock
from models.position_encoding import *
from models.dino.layers.mlp import Mlp
from models.dino.layers.swiglu_ffn import SwiGLU

sys.path.append("..")


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, norm_type='IN', **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        if norm_type == 'IN':
            self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        elif norm_type == 'BN':
            self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            pad: int = 1,
            dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FPNEncoder(nn.Module):
    def __init__(self, feat_chs, norm_type='BN'):
        super(FPNEncoder, self).__init__()
        self.conv00 = Conv2d(3, feat_chs[0], 7, 1, padding=3, norm_type=norm_type)
        self.conv01 = Conv2d(feat_chs[0], feat_chs[0], 5, 1, padding=2, norm_type=norm_type)

        self.downsample1 = Conv2d(feat_chs[0], feat_chs[1], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv10 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)
        self.conv11 = Conv2d(feat_chs[1], feat_chs[1], 3, 1, padding=1, norm_type=norm_type)

        self.downsample2 = Conv2d(feat_chs[1], feat_chs[2], 5, stride=2, padding=2, norm_type=norm_type)
        self.conv20 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)
        self.conv21 = Conv2d(feat_chs[2], feat_chs[2], 3, 1, padding=1, norm_type=norm_type)

        self.downsample3 = Conv2d(feat_chs[2], feat_chs[3], 3, stride=2, padding=1, norm_type=norm_type)
        self.conv30 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)
        self.conv31 = Conv2d(feat_chs[3], feat_chs[3], 3, 1, padding=1, norm_type=norm_type)

    def forward(self, x):
        conv00 = self.conv00(x)
        conv01 = self.conv01(conv00)
        down_conv0 = self.downsample1(conv01)
        conv10 = self.conv10(down_conv0)
        conv11 = self.conv11(conv10)
        down_conv1 = self.downsample2(conv11)
        conv20 = self.conv20(down_conv1)
        conv21 = self.conv21(conv20)
        down_conv2 = self.downsample3(conv21)
        conv30 = self.conv30(down_conv2)
        conv31 = self.conv31(conv30)

        return [conv01, conv11, conv21, conv31]


class FPNDecoder(nn.Module):
    def __init__(self, feat_chs):
        super(FPNDecoder, self).__init__()
        final_ch = feat_chs[-1]
        self.out0 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[3], kernel_size=1), nn.BatchNorm2d(feat_chs[3]), Swish())

        self.inner1 = nn.Conv2d(feat_chs[2], final_ch, 1)
        self.out1 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[2], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[2]), Swish())

        self.inner2 = nn.Conv2d(feat_chs[1], final_ch, 1)
        self.out2 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[1], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[1]), Swish())

        self.inner3 = nn.Conv2d(feat_chs[0], final_ch, 1)
        self.out3 = nn.Sequential(nn.Conv2d(final_ch, feat_chs[0], kernel_size=3, padding=1), nn.BatchNorm2d(feat_chs[0]), Swish())

    def forward(self, conv01, conv11, conv21, conv31):
        intra_feat = conv31
        out0 = self.out0(intra_feat)

        intra_feat = F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv21)
        out1 = self.out1(intra_feat)

        intra_feat = F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv11)
        out2 = self.out2(intra_feat)

        intra_feat = F.interpolate(intra_feat.to(torch.float32), scale_factor=2, mode="bilinear", align_corners=True) + self.inner3(conv01)
        out3 = self.out3(intra_feat)

        return [out0, out1, out2, out3]


class CrossVITDecoder(nn.Module):
    def __init__(self, args):
        super(CrossVITDecoder, self).__init__()
        self.dino_cfg = args['dino_cfg']
        self.decoder_cfg = args['dino_cfg']['decoder_cfg']
        attention_class = get_attention_type(self.decoder_cfg['attention_type'])

        ffn_type = self.decoder_cfg.get("ffn_type", "ffn")
        if ffn_type == "ffn":
            ffn_class = Mlp
        elif ffn_type == "glu":
            ffn_class = SwiGLU
        else:
            raise NotImplementedError(f"Unknown FFN...{ffn_type}")

        self.self_cross_types = self.decoder_cfg.get("self_cross_types", None)

        if self.self_cross_types is not None:
            self_attn_class = get_attention_type(self.self_cross_types[0])
            cross_attn_class = get_attention_type(self.self_cross_types[1])
        else:
            self_attn_class = attention_class
            cross_attn_class = attention_class

        self.self_attn_blocks = nn.ModuleList()
        self.cross_attn_blocks = nn.ModuleList()

        self.no_combine_norm = self.decoder_cfg.get("no_combine_norm", False)
        if not self.no_combine_norm:
            self.norm_layers = nn.ModuleList()

        self.prev_values = nn.ParameterList()
        for _ in range(self.dino_cfg['cross_interval_layers'] - 1):
            self.self_attn_blocks.append(CrossBlock(dim=self.decoder_cfg['d_model'], num_heads=self.decoder_cfg['nhead'],
                                                    attn_class=self_attn_class, ffn_layer=ffn_class, **self.decoder_cfg))
            if not self.no_combine_norm:
                self.norm_layers.append(nn.LayerNorm(self.decoder_cfg['d_model'], eps=1e-6))
            self.prev_values.append(nn.Parameter(torch.tensor(self.decoder_cfg['prev_values']), requires_grad=True))
        for _ in range(self.dino_cfg['cross_interval_layers']):
            self.cross_attn_blocks.append(CrossBlock(dim=self.decoder_cfg['d_model'], num_heads=self.decoder_cfg['nhead'],
                                                     attn_class=cross_attn_class, ffn_layer=ffn_class, **self.decoder_cfg))

        ch, vit_ch = args['out_ch'], args['vit_ch']
        self.proj = nn.Sequential(nn.Conv2d(vit_ch, ch * 4, 3, stride=1, padding=1),
                                  nn.BatchNorm2d(ch * 4), nn.SiLU())
        self.upsampler0 = nn.Sequential(nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(ch * 2), nn.SiLU())
        self.upsampler1 = nn.Sequential(nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
                                        nn.BatchNorm2d(ch), nn.SiLU())

    def forward(self, x, Fmats=None, vit_shape=None):  # [B,V,HW,C]*N, Fmats:[B,V-1,HW,HW]
        B, V, H, W, C = vit_shape
        # x:[B,V,HW,C]*N

        src_feat = None
        ref_feat_list, src_feat_list = [], []

        # [B,V,HW,C]*N
        for v in range(V):
            if v == 0:  # self-attention (ref)
                for i in range(len(self.self_attn_blocks) + 1):
                    if i == 0:
                        ref_feat_list.append(x[i][:, v])
                    else:
                        attn_inputs = {'x': ref_feat_list[-1]}
                        pre_ref_feat = self.self_attn_blocks[i - 1](**attn_inputs)
                        new_ref_feat = self.prev_values[i - 1] * pre_ref_feat + x[i][:, v]  # AAS
                        if not self.no_combine_norm:
                            new_ref_feat = self.norm_layers[i - 1](new_ref_feat)
                        ref_feat_list.append(new_ref_feat)
            else:  # cross-attention (src)
                for i in range(len(self.cross_attn_blocks)):
                    if i == 0:
                        attn_inputs = {'x': x[i][:, v], 'key': ref_feat_list[i], 'value': ref_feat_list[i]}
                    else:
                        query = self.prev_values[i - 1] * src_feat + x[i][:, v]
                        if not self.no_combine_norm:
                            query = self.norm_layers[i - 1](query)
                        attn_inputs = {'x': query, 'key': ref_feat_list[i], 'value': ref_feat_list[i]}

                    src_feat = self.cross_attn_blocks[i](**attn_inputs)
                src_feat_list.append(src_feat.unsqueeze(1))

        src_feat = torch.cat(src_feat_list, dim=1)  # [B,V-1,HW,C]
        x = torch.cat([ref_feat_list[-1].unsqueeze(1), src_feat.reshape(B, V - 1, H * W, C)], dim=1)  # [B,V,HW,C]
        x = x.reshape(B * V, H, W, C).permute(0, 3, 1, 2)  # [BV,C,H,W]

        x = self.proj(x)
        x = self.upsampler0(x)
        x = self.upsampler1(x)

        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer

        self.conv1 = Conv3d(in_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)
        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        if in_channels != base_channels:
            self.inner = nn.Conv3d(in_channels, base_channels, 1, 1)
        else:
            self.inner = nn.Identity()

        if self.last_layer:
            self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x, *kwargs):
        import torch.utils.checkpoint as cp
        x = cp.checkpoint(self.forward_once, x)
        return x

    def forward_once(self, x, *kwargs):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x


class CostRegNet2D(nn.Module):
    def __init__(self, in_channels, base_channel=8):
        super(CostRegNet2D, self).__init__()
        self.conv1 = Conv3d(in_channels, base_channel * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = Conv3d(base_channel * 2, base_channel * 2, padding=1)

        self.conv3 = Conv3d(base_channel * 2, base_channel * 4, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv4 = Conv3d(base_channel * 4, base_channel * 4, padding=1)

        self.conv5 = Conv3d(base_channel * 4, base_channel * 8, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv6 = Conv3d(base_channel * 8, base_channel * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x, *kwargs):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x


class CostRegNet3D(nn.Module):
    def __init__(self, in_channels, base_channel=8, log_var=False):
        super(CostRegNet3D, self).__init__()
        self.log_var = log_var
        self.conv1 = Conv3d(in_channels, base_channel * 2, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv2 = Conv3d(base_channel * 2, base_channel * 2, padding=1)

        self.conv3 = Conv3d(base_channel * 2, base_channel * 4, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv4 = Conv3d(base_channel * 4, base_channel * 4, padding=1)

        self.conv5 = Conv3d(base_channel * 4, base_channel * 8, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.conv6 = Conv3d(base_channel * 8, base_channel * 8, padding=1)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 8, base_channel * 4, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 4),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 4, base_channel * 2, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel * 2),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(base_channel * 2, base_channel, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=(1, 2, 2), bias=False),
            nn.BatchNorm3d(base_channel),
            nn.ReLU(inplace=True))

        if in_channels != base_channel:
            self.inner = nn.Conv3d(in_channels, base_channel, 1, 1)
        else:
            self.inner = nn.Identity()

        self.prob = nn.Conv3d(base_channel, 2 if self.log_var else 1, 1, stride=1, padding=0)

    def forward(self, x, *kwargs):
        import torch.utils.checkpoint as cp

        x = cp.checkpoint(self.forward_once, x)
        return x

    def forward_once(self, x, *kwargs):
        conv0 = x
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = self.inner(conv0) + self.conv11(x)
        x = self.prob(x)

        return x


class FFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop: float = 0.0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class FlashAttnBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            ffn_bias: bool = True,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values=1.0,
            act_layer=nn.GELU,
            attention_type="FLASH2",
            **kwargs
    ) -> None:
        super().__init__()

        if attention_type == "FLASH2":
            self.attn = FlashAttention2(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        elif attention_type == "FLASH1":
            self.attn = MemEffAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        elif attention_type == "Linear":
            self.attn = CrossLinearAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        else:
            raise NotImplementedError(f"Unkown Attention Type {attention_type}")
            # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=drop)
        self.gamma1 = nn.Parameter(torch.tensor(init_values), requires_grad=True)
        self.norm1 = nn.LayerNorm(dim)  # use LN here

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.gamma2 = nn.Parameter(torch.tensor(init_values), requires_grad=True)
        self.norm2 = nn.LayerNorm(dim)  # use LN here
        self.post_norm = kwargs.get("post_norm", True)

    def forward(self, x, positions=None):  # x:[B,C,D,H,W]
        # reshape x:[B,C,D,H,W] to [B,L(HWD),C]
        b, c, d, h, w = x.shape
        x = rearrange(x, "b c d h w -> b (h w d) c")
        if self.post_norm:  # post-norm
            x = self.norm1(x + self.gamma1 * self.attn(x, positions=positions))
            x = self.norm2(x + self.gamma2 * self.ffn(x))
        else:
            x = x + self.gamma1 * self.attn(self.norm1(x), positions=positions)
            x = x + self.gamma2 * self.ffn(self.norm2(x))
        # reshape x:[B,L(HWD),C] back to [B,C,D,H,W]
        x = rearrange(x, "b (h w d) c -> b c d h w", h=h, w=w, d=d)

        return x


class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class PureTransformerCostReg(nn.Module):
    def __init__(self, in_channels, base_channel=8, mid_channel=64, num_heads=8, mlp_ratio=4,
                 layer_num=6, drop=0.0, attn_drop=0.0, position_encoding=True, attention_type="FLASH2",
                 down_rate=4, **kwargs):
        super(PureTransformerCostReg, self).__init__()
        self.attention_type = attention_type
        self.down_rate = down_rate
        self.use_pe_proj = kwargs.get("use_pe_proj", True)
        if position_encoding and self.use_pe_proj:
            self.pe_proj = nn.Conv3d(base_channel * 3, base_channel, 1, 1, bias=False)
        else:
            self.pe_proj = nn.Identity()

        self.down = nn.Sequential(
            nn.Conv3d(in_channels, mid_channel, kernel_size=down_rate, stride=down_rate),
            LayerNorm3D(mid_channel, eps=1e-6)
        )
        self.attention_layers = nn.ModuleList()
        for _ in range(layer_num):
            self.attention_layers.append(FlashAttnBlock(mid_channel, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop,
                                                        attention_type=attention_type, **kwargs))

        self.up = nn.Sequential(
            nn.ConvTranspose3d(mid_channel, base_channel, kernel_size=down_rate, stride=down_rate),
            LayerNorm3D(base_channel, eps=1e-6)
        )

        self.prob = nn.Conv3d(base_channel, 1, 1, stride=1, padding=0)

    def forward(self, x, position3d=None):
        # x: [B,C,D,H,W]
        if position3d is not None:
            if self.use_pe_proj:
                x = x + self.pe_proj(PositionEncoding3D(position3d, x.shape[1]))  # position encoding
            else:
                x = x + self.pe_proj(PositionEncoding3D(position3d, x.shape[1] // 3))  # position encoding
        x = self.down(x)  # [B,C,D//4,H//4,W//4]

        for layer in self.attention_layers:
            x = layer(x)

        x = self.up(x)
        x = self.prob(x)  # [B,C,D,H,W]

        return x


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def conf_regression(p, n=4):
    ndepths = p.size(1)
    with torch.no_grad():
        # photometric confidence
        if n % 2 == 1:
            prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2, n // 2]),
                                                (n, 1, 1), stride=1, padding=0).squeeze(1)
        else:
            prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2 - 1, n // 2]),
                                                (n, 1, 1), stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(p.detach(), depth_values=torch.arange(ndepths, device=p.device, dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=ndepths - 1)
        conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1))
    return conf.squeeze(1)


def init_range(cur_depth, ndepths, device, dtype, H, W):
    if len(cur_depth.shape) == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B, )
        new_interval = new_interval[:, None, None]  # B H W
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                         requires_grad=False).reshape(1, -1) * new_interval.squeeze(1))  # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B, D, H, W)
    else:
        cur_depth_min = cur_depth[..., 0]  # (B,H,W)
        cur_depth_max = cur_depth[..., -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepths - 1)  # (B,H,W)
        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepths, device=device, dtype=dtype,
                                                                         requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))  # (B, D, H, W)
    return depth_range_samples


def init_inverse_range(cur_depth, ndepths, device, dtype, H, W):
    if len(cur_depth.shape) == 2:
        inverse_depth_min = 1. / cur_depth[:, 0]  # (B,)
        inverse_depth_max = 1. / cur_depth[:, -1]
        itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
        inverse_depth_hypo = inverse_depth_max[:, None, None, None] + (inverse_depth_min - inverse_depth_max)[:, None, None, None] * itv
    else:
        inverse_depth_min = 1. / cur_depth[..., 0]  # (B,H,W)
        inverse_depth_max = 1. / cur_depth[..., -1]
        itv = torch.arange(0, ndepths, device=device, dtype=dtype, requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H, W) / (ndepths - 1)  # 1 D H W
        inverse_depth_hypo = inverse_depth_max[:, None, :, :] + (inverse_depth_min - inverse_depth_max)[:, None, :, :] * itv

    return 1. / inverse_depth_hypo


def schedule_inverse_range(depth, depth_hypo, ndepths, split_itv, H, W, shift=False):
    last_depth_itv = 1. / depth_hypo[:, 2, :, :] - 1. / depth_hypo[:, 1, :, :]
    inverse_min_depth = 1 / depth + split_itv * last_depth_itv  # B H W
    inverse_max_depth = 1 / depth - split_itv * last_depth_itv  # B H W

    if shift:  # shift is used to prevent negative depth prediction. 0.002 is set when the max depth range is 500
        is_neg = (inverse_max_depth < 0.002).float()
        inverse_max_depth = inverse_max_depth - (inverse_max_depth - 0.002) * is_neg
        inverse_min_depth = inverse_min_depth - (inverse_max_depth - 0.002) * is_neg

    # cur_depth_min, (B, H, W)
    # cur_depth_max: (B, H, W)
    itv = torch.arange(0, ndepths, device=inverse_min_depth.device, dtype=inverse_min_depth.dtype,
                       requires_grad=False).reshape(1, -1, 1, 1).repeat(1, 1, H // 2, W // 2) / (ndepths - 1)  # 1 D H W

    inverse_depth_hypo = inverse_max_depth[:, None, :, :] + (inverse_min_depth - inverse_max_depth)[:, None, :, :] * itv  # B D H W
    inverse_depth_hypo = F.interpolate(inverse_depth_hypo.unsqueeze(1), [ndepths, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return 1. / inverse_depth_hypo


def schedule_range(cur_depth, ndepth, depth_inteval_pixel, H, W):
    # shape, (B, H, W)
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    if len(depth_inteval_pixel.shape) != 3:
        depth_inteval_pixel = depth_inteval_pixel[:, None, None]
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_min = torch.clamp_min(cur_depth_min, 0.001)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                                                                     requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))
    depth_range_samples = F.interpolate(depth_range_samples.unsqueeze(1), [ndepth, H, W], mode='trilinear', align_corners=True).squeeze(1)
    return depth_range_samples


if __name__ == '__main__':
    import torchsummary

    model = PureTransformerCostReg(in_channels=64).cuda()
    x = torch.rand(1, 64, 32, 64, 80).cuda()
    # x = torch.rand(1, 64, 32, 144, 160).cuda()
    # tt = model(x)
    # pdb.set_trace()
    sum = 0
    for name, param in model.named_parameters():
        num = 1
        for size in param.shape:
            num *= size
        sum += num
        print("{:30s} : {}".format(name, param.shape))
    print("total param num {}".format(sum))
