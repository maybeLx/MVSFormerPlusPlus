from models.cost_volume import *
from models.dino.dinov2 import vit_small, vit_base, vit_large, vit_giant2
import os
from models.position_encoding import get_position_3d
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


class CasMVSNet(nn.Module):
    def __init__(self, args):
        super(CasMVSNet, self).__init__()
        self.args = args
        self.ndepths = args['ndepths']
        self.depth_interals_ratio = args['depth_interals_ratio']
        self.inverse_depth = args.get('inverse_depth', False)
        self.use_pe3d = args.get('use_pe3d', False)
        self.cost_reg_type = args.get("cost_reg_type", ["Normal", "Normal", "Normal", "Normal"])

        self.encoder = FPNEncoder(feat_chs=args['feat_chs'])
        self.decoder = FPNDecoder(feat_chs=args['feat_chs'])

        self.fusions = nn.ModuleList([StageNet(args, self.ndepths[i], i) for i in range(len(self.ndepths))])

    def forward(self, imgs, proj_matrices, depth_values, tmp=[5., 5., 5., 1.]):
        B, V, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[3], imgs.shape[4]
        depth_interval = depth_values[:, 1] - depth_values[:, 0]

        # feature encode
        if not self.training:
            feat1s, feat2s, feat3s, feat4s = [], [], [], []
            for vi in range(V):
                img_v = imgs[:, vi]
                conv01, conv11, conv21, conv31 = self.encoder(img_v)
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
            feat1, feat2, feat3, feat4 = self.decoder.forward(conv01, conv11, conv21, conv31)

            features = {'stage1': feat1.reshape(B, V, feat1.shape[1], feat1.shape[2], feat1.shape[3]),
                        'stage2': feat2.reshape(B, V, feat2.shape[1], feat2.shape[2], feat2.shape[3]),
                        'stage3': feat3.reshape(B, V, feat3.shape[1], feat3.shape[2], feat3.shape[3]),
                        'stage4': feat4.reshape(B, V, feat4.shape[1], feat4.shape[2], feat4.shape[3])}

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
                    depth_samples = init_inverse_range(depth_values, self.ndepths[stage_idx], imgs.device,
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
        print(f"depth min {outputs_stage['depth'].min()}, max {outputs_stage['depth'].max()}")

        if valid_count > 0:
            outputs['photometric_confidence'] = prob_maps / valid_count

        return outputs
