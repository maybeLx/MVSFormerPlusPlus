import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from models.dino.layers.block import CrossBlock
from models.dino.layers.attention import get_attention_type
from models.dino.layers.mlp import Mlp
from models.dino.layers.swiglu_ffn import SwiGLU

'''
- We provide two different positional encoding methods as shown below.
- You can easily switch different pos-enc in the __init__() function of FMT.
- In our experiments, PositionEncodingSuperGule usually cost more GPU memory.
'''
from .position_encoding import *


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with



class FMT(nn.Module):
    def __init__(self, attention_type="FLASH2", d_model=64, nhead=4, layer_names=["self", "cross"], **kwargs):
        super(FMT, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.layer_names = layer_names
        self.attention_type = attention_type
        attention_class = get_attention_type(attention_type)

        ffn_type = kwargs.get("ffn_type", "ffn")
        if ffn_type == "ffn":
            ffn_class = Mlp
        elif ffn_type == "glu":
            ffn_class = SwiGLU
        else:
            raise NotImplementedError(f"Unknown FFN...{ffn_type}")

        self_cross_types = kwargs.get("self_cross_types", None)
        if self_cross_types is not None:
            self_attn_class = get_attention_type(self_cross_types[0])
            cross_attn_class = get_attention_type(self_cross_types[1])
            self.layers = nn.ModuleList()
            for layer_name in self.layer_names:
                if layer_name == 'self':
                    self.layers.append(CrossBlock(dim=d_model, num_heads=nhead, attn_class=attention_class, ffn_layer=ffn_class,
                                                  attention_type=self_attn_class, **kwargs))
                elif layer_name == 'cross':
                    self.layers.append(CrossBlock(dim=d_model, num_heads=nhead, attn_class=attention_class, ffn_layer=ffn_class,
                                                  attention_type=cross_attn_class, **kwargs))
                else:
                    raise NotImplementedError(f"Unknown attention {layer_name}!")

        else:
            encoder_layer = CrossBlock(dim=d_model, num_heads=nhead, attn_class=attention_class, ffn_layer=ffn_class,
                                       attention_type=attention_type, **kwargs)
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

        self.pos_encoding = PositionEncodingSineNorm(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ref_feature=None, src_feature=None, feat="ref", attn_bias=None):
        """
        Args:
            ref_feature(torch.Tensor): [N, C, H, W]
            src_feature(torch.Tensor): [N, C, H, W]
        """

        assert ref_feature is not None

        if feat == "ref":  # only self attention layer

            assert self.d_model == ref_feature.size(1)
            _, _, H, W = ref_feature.shape

            ref_feature = einops.rearrange(self.pos_encoding(ref_feature), 'n c h w -> n (h w) c').contiguous()

            ref_feature_list = []
            for layer, name in zip(self.layers, self.layer_names):  # every self attention layer
                if name == 'self':
                    attn_inputs = {'x': ref_feature}
                    # if self.attention_type == 'BRA':
                    #     attn_inputs['H'], attn_inputs['W'] = H, W

                    ref_feature = layer(**attn_inputs)
                    ref_feature_list.append(einops.rearrange(ref_feature, 'n (h w) c -> n c h w', h=H).contiguous())
            return ref_feature_list

        elif feat == "src":

            assert self.d_model == ref_feature[0].size(1)
            _, _, H, W = ref_feature[0].shape

            ref_feature = [einops.rearrange(_, 'n c h w -> n (h w) c').contiguous() for _ in ref_feature]
            src_feature = self.pos_encoding(src_feature)
            src_feature = einops.rearrange(src_feature, 'n c h w -> n (h w) c').contiguous()

            for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
                if name == 'self':
                    attn_inputs = {'x': src_feature}
                elif name == 'cross':
                    if len(ref_feature) == len(self.layers):
                        ref_idx = i
                    else:
                        ref_idx = i // 2
                    attn_inputs = {'x': src_feature, 'key': ref_feature[ref_idx], 'value': ref_feature[ref_idx]}
                else:
                    raise KeyError

                # if self.attention_type == 'BRA':
                #     attn_inputs['H'], attn_inputs['W'] = H, W
                attn_inputs['attn_bias'] = attn_bias

                src_feature = layer(**attn_inputs)

            return einops.rearrange(src_feature, 'n (h w) c -> n c h w', h=H).contiguous()
        else:
            raise ValueError("Wrong feature name")


class FMT_with_pathway(nn.Module):
    def __init__(self, base_channel=8, **kwargs):
        super(FMT_with_pathway, self).__init__()

        self.FMT = FMT(**kwargs)

        self.dim_reduction_1 = nn.Conv2d(base_channel * 8, base_channel * 4, 1, bias=False)
        self.dim_reduction_2 = nn.Conv2d(base_channel * 4, base_channel * 2, 1, bias=False)
        self.dim_reduction_3 = nn.Conv2d(base_channel * 2, base_channel, 1, bias=False)

        self.smooth_1 = nn.Conv2d(base_channel * 4, base_channel * 4, 3, padding=1, bias=False)
        self.smooth_2 = nn.Conv2d(base_channel * 2, base_channel * 2, 3, padding=1, bias=False)
        self.smooth_3 = nn.Conv2d(base_channel * 1, base_channel * 1, 3, padding=1, bias=False)

    def _upsample_add(self, x, y):
        """_upsample_add. Upsample and add two feature maps.

        :param x: top feature map to be upsampled.
        :param y: lateral feature map.
        """

        _, _, H, W = y.size()
        return F.interpolate(x.to(torch.float32), size=(H, W), mode='bilinear') + y

    def forward(self, features):
        """forward.

        :param features: multiple views and multiple stages features
        """

        # features:{stage1:[B,V,C,H,W], stage2....stage4}

        B, V, C, H, W = features['stage1'].shape
        return_stage1_feats = []
        return_stage2_feats = []
        return_stage3_feats = []
        return_stage4_feats = []
        ref_feat_list = []

        # if Fmats is not None and self.RPE_size > 0:
        #     epi_dis = get_epi_distance(Fmats, H, W, normalize=True)
        #     epi_dis = torch.round(epi_dis * (self.RPE_size - 1)).long()  # 0~1 --> 0~RPE_size
        #     attn_bias = self.rpe_embedding(epi_dis).squeeze(-1)  # [B,V-1,HW,HW,1]->[B,V-1,HW,HW]
        # else:
        #     attn_bias = None

        for nview_idx in range(V):
            if nview_idx == 0:  # ref view
                ref_feat_list = self.FMT.forward(features['stage1'][:, nview_idx], feat="ref")
                return_stage1_feats.append(ref_feat_list[-1])
            else:  # src view
                # if attn_bias is None:
                return_stage1_feats.append(self.FMT.forward(ref_feat_list, features['stage1'][:, nview_idx], feat="src"))
                # else:
                #     return_stage1_feats.append(self.FMT.forward(ref_feat_list, features['stage1'][:, nview_idx], feat="src", attn_bias=attn_bias[:, nview_idx - 1]))
            return_stage2_feats.append(self.smooth_1(self._upsample_add(self.dim_reduction_1(return_stage1_feats[-1]), features['stage2'][:, nview_idx])))
            return_stage3_feats.append(self.smooth_2(self._upsample_add(self.dim_reduction_2(return_stage2_feats[-1]), features['stage3'][:, nview_idx])))
            return_stage4_feats.append(self.smooth_3(self._upsample_add(self.dim_reduction_3(return_stage3_feats[-1]), features['stage4'][:, nview_idx])))

        features = {
            'stage1': torch.stack(return_stage1_feats, dim=1),
            'stage2': torch.stack(return_stage2_feats, dim=1),
            'stage3': torch.stack(return_stage3_feats, dim=1),
            'stage4': torch.stack(return_stage4_feats, dim=1),
        }

        return features