from models.cost_volume import *
from models.dino.dinov2 import vit_small, vit_base, vit_large, vit_giant2
import os
# from models.topk_sampling import schedule_range_topk
from models.position_encoding import get_position_3d
from models.FMT import FMT_with_pathway
import pdb
Align_Corners_Range = False


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


class DINOv2MVSNet(nn.Module):
    def __init__(self, args):
        super(DINOv2MVSNet, self).__init__()
        self.args = args
        self.ndepths = args['ndepths']
        self.depth_interals_ratio = args['depth_interals_ratio']
        self.inverse_depth = args.get('inverse_depth', False)
        self.use_pe3d = args.get('use_pe3d', False)
        self.cost_reg_type = args.get("cost_reg_type", ["Normal", "Normal", "Normal", "Normal"])

        self.encoder = FPNEncoder(feat_chs=args['feat_chs'])
        self.decoder = FPNDecoder(feat_chs=args['feat_chs'])

        self.vit_args = args
        self.freeze_vit = self.vit_args['freeze_vit']
        dino_cfg = args.get("dino_cfg", {})
        self.vit = vit_base(img_size=518, patch_size=14, init_values=1.0, block_chunks=0, ffn_layer="mlp",
                            **dino_cfg)

        self.decoder_vit = CrossVITDecoder(self.vit_args)
        self.FMT_module = FMT_with_pathway(**args.get("FMT_config"))

        if os.path.exists(self.vit_args['vit_path']):
            state_dict = torch.load(self.vit_args['vit_path'], map_location='cpu')
            from utils import torch_init_model
            torch_init_model(self.vit, state_dict, key='model')
        else:
            print('!!!No weight in', self.vit_args['vit_path'], 'testing should neglect this.')

        self.fusions = nn.ModuleList([StageNet(args, self.ndepths[i], i) for i in range(len(self.ndepths))])

    def vit_forward(self, vit_imgs, B, V, vit_h, vit_w, Fmats=None):
        if self.freeze_vit:
            with torch.no_grad():
                vit_out = self.vit.forward_interval_features(vit_imgs)
        else:
            vit_out = self.vit.forward_interval_features(vit_imgs)

        vit_out = [v.reshape(B, V, -1, self.vit.embed_dim) for v in vit_out]
        vit_shape = [B, V, vit_h // self.vit.patch_size, vit_w // self.vit.patch_size, self.vit.embed_dim]
        vit_feat = self.decoder_vit.forward(vit_out, Fmats=Fmats, vit_shape=vit_shape)

        return vit_feat

    def forward(self, imgs, proj_matrices, depth_values, tmp=[5., 5., 5., 1.]):
        B, V, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]
        depth_interval = depth_values[:, 1] - depth_values[:, 0]
        # dinov2 patchsize=14,  0.5 * 14/16
        vit_h, vit_w = int(H * self.vit_args['rescale'] // 14 * 14), int(W * self.vit_args['rescale'] // 14 * 14)

        # feature encode
        if not self.training:
            vit_imgs = F.interpolate(imgs.reshape(B * V, 3, H, W), (vit_h, vit_w), mode='bicubic',
                                     align_corners=Align_Corners_Range)
            vit_feat = self.vit_forward(vit_imgs, B, V, vit_h, vit_w)
            conv31_h, conv31_w = H // 8, W // 8
            if vit_feat.shape[2] != conv31_h or vit_feat.shape[3] != conv31_w:
                vit_feat = F.interpolate(vit_feat, size=(conv31_h, conv31_w), mode='bilinear',
                                         align_corners=Align_Corners_Range)

            feat1s, feat2s, feat3s, feat4s = [], [], [], []
            for vi in range(V):
                img_v = imgs[:, vi]
                conv01, conv11, conv21, conv31 = self.encoder(img_v)
                conv31 = conv31 + vit_feat[vi].unsqueeze(0)
                feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)
                feat1s.append(feat1)
                feat2s.append(feat2)
                feat3s.append(feat3)
                feat4s.append(feat4)

            features = {'stage1': torch.stack(feat1s, dim=1),
                        'stage2': torch.stack(feat2s, dim=1),
                        'stage3': torch.stack(feat3s, dim=1),
                        'stage4': torch.stack(feat4s, dim=1)}

        else:
            imgs = imgs.reshape(B * V, 3, H, W)
            conv01, conv11, conv21, conv31 = self.encoder(imgs)
            vit_imgs = F.interpolate(imgs, (vit_h, vit_w), mode='bicubic', align_corners=Align_Corners_Range)

            vit_feat = self.vit_forward(vit_imgs, B, V, vit_h, vit_w)
            if vit_feat.shape[2] != conv31.shape[2] or vit_feat.shape[3] != conv31.shape[3]:
                vit_feat = F.interpolate(vit_feat, size=(conv31.shape[2], conv31.shape[3]), mode='bilinear',
                                         align_corners=Align_Corners_Range)

            conv31 = conv31 + vit_feat
            feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)

            features = {'stage1': feat1.reshape(B, V, feat1.shape[1], feat1.shape[2], feat1.shape[3]),
                        'stage2': feat2.reshape(B, V, feat2.shape[1], feat2.shape[2], feat2.shape[3]),
                        'stage3': feat3.reshape(B, V, feat3.shape[1], feat3.shape[2], feat3.shape[3]),
                        'stage4': feat4.reshape(B, V, feat4.shape[1], feat4.shape[2], feat4.shape[3])}
        features = self.FMT_module.forward(features)


        outputs = {}
        outputs_stage = {}
        height_min, height_max = None, None
        width_min, width_max = None, None

        prob_maps = torch.zeros([B, H, W], dtype=torch.float32, device=imgs.device)
        valid_count = 0
        for stage_idx in range(len(self.ndepths)):
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]  # [B,V,2,4,4]
            features_stage = features['stage{}'.format(stage_idx + 1)]
            B, V, C, H, W = features_stage.shape
            # init range
            if stage_idx == 0:
                if self.inverse_depth:
                    depth_samples = init_inverse_range(depth_values, self.ndepths[stage_idx], imgs.device, imgs.dtype,
                                                       H, W)
                else:
                    depth_samples = init_range(depth_values, self.ndepths[stage_idx], imgs.device,
                                                             imgs.dtype, H, W)
            else:
                if self.inverse_depth:
                    depth_samples = schedule_inverse_range(outputs_stage['depth'].detach(),
                                                               outputs_stage['depth_values'],
                                                               self.ndepths[stage_idx],
                                                               self.depth_interals_ratio[stage_idx], H, W)  # B D H W
                else:
                    depth_samples = schedule_range(outputs_stage['depth'].detach(),
                                                   self.ndepths[stage_idx],
                                                   self.depth_interals_ratio[stage_idx] * depth_interval,
                                                   H, W)

            # get 3D PE
            if self.cost_reg_type[stage_idx] != "Normal" and self.use_pe3d:
                K = proj_matrices_stage[:, 0, 1, :3, :3]  # [B,3,3] stage1获取全局最小最大h,w,后续根据stage1的全局空间进行归一化
                position3d, height_min, height_max, width_min, width_max = get_position_3d(
                    B, H, W, K, depth_samples,
                    depth_min=depth_values.min(), depth_max=depth_values.max(),
                    height_min=height_min, height_max=height_max,
                    width_min=width_min, width_max=width_max,
                    normalize=True
                )
            else:
                position3d = None

            outputs_stage = self.fusions[stage_idx].forward(features_stage, proj_matrices_stage, depth_samples,
                                                            tmp=tmp[stage_idx], position3d=position3d)
            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            depth_conf = outputs_stage['photometric_confidence']
            if depth_conf.shape[1] != prob_maps.shape[1] or depth_conf.shape[2] != prob_maps.shape[2]:
                depth_conf = F.interpolate(depth_conf.unsqueeze(1), [prob_maps.shape[1], prob_maps.shape[2]],
                                           mode="nearest").squeeze(1)
            prob_maps += depth_conf
            valid_count += 1
            outputs.update(outputs_stage)

        outputs['refined_depth'] = outputs_stage['depth']
        if valid_count > 0:
            outputs['photometric_confidence'] = prob_maps / valid_count

        return outputs
